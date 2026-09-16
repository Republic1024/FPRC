"""L1 statistical core for Wu's Space.

This file implements the L1 layer pieces:

* ``CellStats``: a state-agnostic rolling conditional-return engine.
* ``OPS``: pure rank-pair-to-state operator functions.
* ``GridField``: a thin wrapper combining one operator with one ``CellStats``.

The contract is narrow:

* ``state`` is a one-dimensional array of integer state IDs, with NaN allowed
  to mean "no valid state for this asset today".
* ``fwd_ret`` is a one-dimensional array of realized forward excess returns.
* ``update`` applies one time step of exponential decay, then aggregates valid
  observations into the per-state sufficient statistics.
* ``score`` maps each requested state ID to an empirical-Bayes shrunk expected
  return and a confidence score.

No date logic lives here.  The backtest loop is responsible for calling
``update`` only when the corresponding forward-return window is fully visible.
"""

from __future__ import annotations

import numpy as np


class CellStats:
    """Rolling conditional-return statistics over an arbitrary state space.

    ``CellStats`` is the reusable statistical engine under every L1 field.  It
    maintains only sufficient statistics by state:

    * ``s_r``: exponentially decayed sum of returns;
    * ``s_r2``: exponentially decayed sum of squared returns;
    * ``s_n``: exponentially decayed sample count.

    It makes no assumption about the semantic meaning of the integer states.
    That separation is important: operators map factor ranks to state IDs,
    while this class estimates conditional forward returns for those IDs.
    """

    def __init__(
        self,
        n_states: int,
        n0: float = 500.0,
        min_n: float = 100,
        halflife: float = 252,
    ) -> None:
        """Create an empty rolling statistics table.

        Parameters
        ----------
        n_states:
            Total number of discrete states.  Valid state IDs are
            ``0..n_states-1``.

        n0:
            Empirical-Bayes shrinkage strength.  A state's raw mean receives
            weight ``n / (n + n0)``.  Larger values force sparse states closer
            to the global historical mean.

        min_n:
            Minimum decayed sample count required before confidence can become
            positive.  Sparse states may still get a shrunk score, but their
            confidence is zero.

        halflife:
            Exponential decay half-life measured in update steps, normally
            trading days.  ``decay = 0.5 ** (1 / halflife)``.
        """

        if not isinstance(n_states, int) or n_states <= 0:
            raise ValueError(f"n_states must be a positive integer, got {n_states!r}.")
        if not np.isfinite(n0) or n0 < 0:
            raise ValueError(f"n0 must be a finite non-negative number, got {n0!r}.")
        if not np.isfinite(min_n) or min_n < 0:
            raise ValueError(
                f"min_n must be a finite non-negative number, got {min_n!r}."
            )
        if not np.isfinite(halflife) or halflife <= 0:
            raise ValueError(
                f"halflife must be a finite positive number, got {halflife!r}."
            )

        self.n_states = n_states
        self.n0 = float(n0)
        self.min_n = float(min_n)
        self.halflife = float(halflife)
        self.decay = float(0.5 ** (1.0 / self.halflife))

        self.s_r = np.zeros(self.n_states, dtype=float)
        self.s_r2 = np.zeros(self.n_states, dtype=float)
        self.s_n = np.zeros(self.n_states, dtype=float)

    def update(self, state: np.ndarray, fwd_ret: np.ndarray) -> None:
        """Update per-state statistics with one cross-section of observations.

        Parameters
        ----------
        state:
            One-dimensional array of state IDs.  Valid values are integers in
            ``0..n_states-1``.  NaN means the observation is ignored.

        fwd_ret:
            One-dimensional array of realized forward excess returns aligned
            with ``state``.  NaN means the observation is ignored.

        Notes
        -----
        The update is fully vectorized over observations.  It first applies the
        time decay to all internal arrays, then uses ``np.add.at`` to aggregate
        returns, squared returns, and counts into their state buckets.
        """

        state_arr, ret_arr = self._as_aligned_1d_arrays(state, fwd_ret)

        # Decay happens once per update call, even if today's cross-section has
        # no valid observations.  This keeps time passage explicit.
        self.s_r *= self.decay
        self.s_r2 *= self.decay
        self.s_n *= self.decay

        valid = np.isfinite(state_arr) & np.isfinite(ret_arr)
        if not np.any(valid):
            return

        state_idx = self._validate_and_cast_state_ids(state_arr[valid])
        returns = ret_arr[valid]

        np.add.at(self.s_r, state_idx, returns)
        np.add.at(self.s_r2, state_idx, returns * returns)
        np.add.at(self.s_n, state_idx, 1.0)

    def score(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map state IDs to shrunk expected returns and confidence scores.

        Parameters
        ----------
        state:
            One-dimensional array of state IDs, possibly containing NaN.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(z_score, conf)`` with the same shape as ``state``.

            ``z_score`` is the empirical-Bayes shrunk conditional expected
            return for each state.  For NaN states, it is NaN.

            ``conf`` is zero for NaN states and for states whose decayed sample
            count is below ``min_n``.  Otherwise it is a smooth sigmoid of the
            absolute t-statistic, centered around ``|t| = 2``.
        """

        state_arr = self._as_1d_float_array(state, name="state")

        mu_shrunk, conf_by_state = self._state_level_scores()

        z_score = np.full(state_arr.shape, np.nan, dtype=float)
        conf = np.zeros(state_arr.shape, dtype=float)

        valid = np.isfinite(state_arr)
        if not np.any(valid):
            return z_score, conf

        state_idx = self._validate_and_cast_state_ids(state_arr[valid])
        z_score[valid] = mu_shrunk[state_idx]
        conf[valid] = conf_by_state[state_idx]
        return z_score, conf

    def _state_level_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-state shrunk means and confidence values."""

        n = self.s_n
        sr = self.s_r
        sr2 = self.s_r2

        mu_cell = np.divide(sr, n, out=np.zeros_like(sr), where=n > 0)

        total_n = n.sum()
        mu_global = sr.sum() / total_n if total_n > 0 else 0.0

        shrink_weight = np.divide(
            n,
            n + self.n0,
            out=np.zeros_like(n),
            where=(n + self.n0) > 0,
        )
        mu_shrunk = shrink_weight * mu_cell + (1.0 - shrink_weight) * mu_global

        second_moment = np.divide(sr2, n, out=np.zeros_like(sr2), where=n > 0)
        variance = np.maximum(second_moment - mu_cell * mu_cell, 0.0)

        # A tiny floor prevents division by zero for states with no dispersion.
        # Confidence is still gated by min_n, so this cannot promote sparse
        # cells by itself.
        standard_error = np.sqrt(np.maximum(variance, 1e-12) / np.maximum(n, 1.0))
        t_stat = np.divide(
            mu_cell - mu_global,
            standard_error,
            out=np.zeros_like(mu_cell),
            where=standard_error > 0,
        )

        sigmoid_conf = 1.0 / (1.0 + np.exp(2.0 - np.abs(t_stat)))
        conf_by_state = np.where(n >= self.min_n, sigmoid_conf, 0.0)
        return mu_shrunk, conf_by_state

    def _validate_and_cast_state_ids(self, state_values: np.ndarray) -> np.ndarray:
        """Validate finite state values and return integer indices."""

        if state_values.size == 0:
            return state_values.astype(np.int64)

        if not np.all(np.equal(state_values, np.floor(state_values))):
            bad = state_values[~np.equal(state_values, np.floor(state_values))][:5]
            raise ValueError(f"state IDs must be integer-valued; examples: {bad!r}.")

        state_idx = state_values.astype(np.int64)
        if np.any((state_idx < 0) | (state_idx >= self.n_states)):
            bad = state_idx[(state_idx < 0) | (state_idx >= self.n_states)][:5]
            raise ValueError(
                "state IDs out of range "
                f"[0, {self.n_states - 1}]; examples: {bad!r}."
            )

        return state_idx

    @staticmethod
    def _as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
        """Convert an input to a one-dimensional float array."""

        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        return arr

    @classmethod
    def _as_aligned_1d_arrays(
        cls,
        state: np.ndarray,
        fwd_ret: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert and validate aligned state/return arrays."""

        state_arr = cls._as_1d_float_array(state, name="state")
        ret_arr = cls._as_1d_float_array(fwd_ret, name="fwd_ret")
        if state_arr.shape != ret_arr.shape:
            raise ValueError(
                "state and fwd_ret must have the same shape; "
                f"got {state_arr.shape} and {ret_arr.shape}."
            )
        return state_arr, ret_arr


def op_fold(ra: np.ndarray, rb: np.ndarray, n_bins: int) -> np.ndarray:
    """Cartesian Fold operator.

    Input ranks are expected to be in ``1..n_bins`` with NaN allowed.  The
    output state IDs are in ``0..n_bins**2-1``:

        state = (ra - 1) * n_bins + (rb - 1)

    NaN ranks naturally propagate to NaN states.
    """

    return (ra - 1.0) * n_bins + (rb - 1.0)


def op_resonance(ra: np.ndarray, rb: np.ndarray, n_bins: int) -> np.ndarray:
    """Diagonal resonance operator, Wu's Space ``oplus`` prior.

    Output state IDs are diagonal distances in ``0..n_bins-1``:

        state = abs(ra - rb)
    """

    del n_bins
    return np.abs(ra - rb)


def op_conservation(ra: np.ndarray, rb: np.ndarray, n_bins: int) -> np.ndarray:
    """Anti-diagonal conservation operator, Wu's Space ``ominus`` prior.

    Output state IDs are anti-diagonal distances in ``0..n_bins-1``:

        state = abs(ra + rb - (n_bins + 1))
    """

    return np.abs(ra + rb - (n_bins + 1.0))


OPS = {
    "fold": op_fold,
    "resonance": op_resonance,
    "conservation": op_conservation,
}


class GridField:
    """One L1 factor field: operator state mapping plus ``CellStats`` engine.

    ``GridField`` does not own any date or portfolio logic.  It receives two
    already-binned rank arrays ``ra`` and ``rb`` from L0, converts them into
    state IDs through its configured operator, then delegates all statistics to
    ``CellStats``.
    """

    def __init__(
        self,
        op: str,
        n_bins: int,
        *,
        n0: float = 500.0,
        min_n: float = 100,
        halflife: float = 252,
    ) -> None:
        """Create a grid field for one rank-pair operator.

        Parameters
        ----------
        op:
            Operator name.  Must be one of ``"fold"``, ``"resonance"``, or
            ``"conservation"``.

        n_bins:
            Number of rank bins emitted by L0.  Valid input rank values are
            ``1..n_bins``.  The operator output state space is
            ``n_bins ** 2`` for Fold and ``n_bins`` for the two distance
            operators.

        n0 / min_n / halflife:
            Passed directly into the underlying ``CellStats`` instance.
        """

        if op not in OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {sorted(OPS)}.")
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}.")

        self.op = op
        self.n_bins = n_bins
        self.operator = OPS[op]

        n_states = n_bins**2 if op == "fold" else n_bins
        self.stats = CellStats(
            n_states=n_states,
            n0=n0,
            min_n=min_n,
            halflife=halflife,
        )

    @property
    def n_states(self) -> int:
        """Number of discrete states produced by this field."""

        return self.stats.n_states

    def state(self, ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
        """Map two L0 rank arrays into one state-ID array.

        ``ra`` and ``rb`` must be aligned one-dimensional arrays.  Finite
        values must be integer rank labels in ``1..n_bins``; NaN is allowed and
        propagates into the output state array.
        """

        ra_arr, rb_arr = self._as_aligned_rank_arrays(ra, rb)
        return self.operator(ra_arr, rb_arr, self.n_bins)

    def update(self, ra: np.ndarray, rb: np.ndarray, fwd_ret: np.ndarray) -> None:
        """Update field statistics from two rank arrays and forward returns."""

        state_id = self.state(ra, rb)
        self.stats.update(state_id, fwd_ret)

    def score(self, ra: np.ndarray, rb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(z_score, conf)`` for the current rank-pair states."""

        state_id = self.state(ra, rb)
        return self.stats.score(state_id)

    def _as_aligned_rank_arrays(
        self,
        ra: np.ndarray,
        rb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert, shape-check, and validate two L0 rank arrays."""

        ra_arr = CellStats._as_1d_float_array(ra, name="ra")
        rb_arr = CellStats._as_1d_float_array(rb, name="rb")
        if ra_arr.shape != rb_arr.shape:
            raise ValueError(
                f"ra and rb must have the same shape; got {ra_arr.shape} "
                f"and {rb_arr.shape}."
            )

        self._validate_rank_values(ra_arr, name="ra")
        self._validate_rank_values(rb_arr, name="rb")
        return ra_arr, rb_arr

    def _validate_rank_values(self, ranks: np.ndarray, *, name: str) -> None:
        """Validate rank values while allowing NaN missing states."""

        invalid_nonfinite = ~np.isnan(ranks) & ~np.isfinite(ranks)
        if np.any(invalid_nonfinite):
            bad = ranks[invalid_nonfinite][:5]
            raise ValueError(f"{name} ranks must be finite or NaN; examples: {bad!r}.")

        valid = np.isfinite(ranks)
        if not np.any(valid):
            return

        valid_ranks = ranks[valid]
        integer_valued = np.equal(valid_ranks, np.floor(valid_ranks))
        if not np.all(integer_valued):
            bad = valid_ranks[~integer_valued][:5]
            raise ValueError(f"{name} ranks must be integer-valued; examples: {bad!r}.")

        in_range = (valid_ranks >= 1) & (valid_ranks <= self.n_bins)
        if not np.all(in_range):
            bad = valid_ranks[~in_range][:5]
            raise ValueError(
                f"{name} ranks must be in [1, {self.n_bins}]; examples: {bad!r}."
            )


__all__ = [
    "CellStats",
    "GridField",
    "OPS",
    "op_conservation",
    "op_fold",
    "op_resonance",
]
