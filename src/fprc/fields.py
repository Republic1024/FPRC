"""Field declarations and time-safe Fold scores."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._cellstats import CellStats


@dataclass(frozen=True)
class FieldSpec:
    """An AB field; local_features augments only this field's residual learner.

    score_column optionally supplies externally computed PIT field scores.
    The caller is responsible for the external column's historical honesty.
    """

    a: str
    b: str
    name: str | None = None
    local_features: tuple[str, ...] = ()
    score_column: str | None = None

    @property
    def key(self) -> str:
        return self.name or f"{self.a}__{self.b}"


def normalize_fields(fields):
    if not fields:
        raise ValueError("fields must contain at least one AB field.")
    out = []
    for value in fields:
        if isinstance(value, FieldSpec):
            spec = value
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            spec = FieldSpec(*value)
        else:
            raise TypeError("Each field must be FieldSpec or an (A, B) pair.")
        if not all(isinstance(x, str) and x for x in (spec.a, spec.b, spec.key)):
            raise ValueError("Field names and parent columns must be nonempty strings.")
        if spec.a == spec.b:
            raise ValueError("A field requires two different parent columns.")
        if isinstance(spec.local_features, str):
            raise TypeError("local_features must be a sequence, not a string.")
        if spec.score_column is not None and not isinstance(spec.score_column, str):
            raise TypeError("score_column must be a column name.")
        out.append(spec)
    if len({s.key for s in out}) != len(out):
        raise ValueError("Field names must be unique.")
    return tuple(out)


def fold_states(a_raw, b_raw, dates, n_bins):
    # Exact L0 convention: same-date rank midpoints, left-sided edges.
    edges = np.arange(1, n_bins, dtype=float) / n_bins
    def bins(values):
        s = pd.Series(np.asarray(values, dtype=float))
        group = s.groupby(dates, sort=False)
        midpoint = (group.rank(method="average") - 0.5) / group.transform("count")
        return np.searchsorted(edges, midpoint.to_numpy(), side="left")
    a, b = bins(a_raw), bins(b_raw)
    states = (a * n_bins + b).astype(float)
    states[~(np.isfinite(a_raw) & np.isfinite(b_raw))] = np.nan
    return states


def historical_fields(specs, raw, ranks, y, dates, label_end, as_of, config):
    """Score each date using labels whose label_end is strictly before it.

    Tables decay once per observed signal date. Labels arriving on/before the
    prediction date are not admitted until a strictly later signal timestamp.
    A final checkpoint admits the labels available strictly before as_of.
    """
    n = len(y)
    result = np.full((n, len(specs)), np.nan)
    states, tables = [], []
    for m, spec in enumerate(specs):
        if spec.score_column is not None:
            result[:, m] = raw[spec.score_column].to_numpy(float)
            states.append(None)
            tables.append(None)
        else:
            states.append(fold_states(raw[spec.a], raw[spec.b], dates, config["n_bins"]))
            tables.append(CellStats(config["n_bins"] ** 2, **config["stats"]))

    eligible = np.flatnonzero(np.isfinite(y) & ~pd.isna(label_end))
    pending = eligible[np.argsort(label_end[eligible], kind="stable")]
    ends = label_end[pending]
    unique, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    cursor = 0
    for date, start, count in zip(unique, starts, counts):
        stop = int(np.searchsorted(ends, date, side="left"))
        add = pending[cursor:stop]
        today = slice(start, start + count)
        for m, table in enumerate(tables):
            if table is None:
                continue
            table.update(states[m][add], y[add])
            score, confidence = table.score(states[m][today])
            result[today, m] = score * confidence
        cursor = stop

    stop = int(np.searchsorted(ends, as_of, side="left"))
    add = pending[cursor:stop]
    for m, table in enumerate(tables):
        if table is not None:
            table.update(states[m][add], y[add])
    return result, tables


def snapshot_fields(specs, raw, dates, tables, n_bins):
    columns = []
    for spec, table in zip(specs, tables):
        if spec.score_column is not None:
            columns.append(raw[spec.score_column].to_numpy(float))
        else:
            score, conf = table.score(fold_states(raw[spec.a], raw[spec.b], dates, n_bins))
            columns.append(score * conf)
    return np.column_stack(columns)
