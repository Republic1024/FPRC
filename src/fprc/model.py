"""Reusable FPRC / Pure Gain estimator with purged forward cross-fitting."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor

from .features import build_features
from .fields import historical_fields, normalize_fields, snapshot_fields


def _names(values, name):
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of column names, not a string.")
    out = tuple(values)
    if not all(isinstance(x, str) and x for x in out) or len(set(out)) != len(out):
        raise ValueError(f"{name} must contain unique nonempty column names.")
    return out


class FPRCRegressor(RegressorMixin, BaseEstimator):
    """Equal-field residual ensemble followed by a fresh residual closure.

    Parameters are configuration only. Use ``fit(df, target='r')`` to train.
    Every local model learns r-F_m. ``pure_gain=True`` doubles its prediction,
    never its training label; closure learns r-B using historical OOF B.

    Prediction is for complete future cross-sections. Training-period
    predictions are available in ``oof_predictions_``, with NaN warm-up rows.
    See README for timing, external score columns, P/Q and missing-value rules.
    """

    def __init__(
        self, fields, all_columns=None, *, pure_gain=True,
        apex=None, q_features=(), closure_features=None,
        date_col="date", entity_col="symbol", label_end_col=None,
        label_horizon=5, refit_frequency="Y", n_bins=10,
        n0=500.0, min_cell_count=100.0, halflife=252.0,
        min_cross_section=20, min_train_rows=100, min_train_dates=20,
        device="cpu", n_jobs=4, random_state=2718,
        xgb_params=None, early_stopping_rounds=40, validation_fraction=0.2,
    ):
        self.fields = fields
        self.all_columns = all_columns
        self.pure_gain = pure_gain
        self.apex = apex
        self.q_features = q_features
        self.closure_features = closure_features
        self.date_col = date_col
        self.entity_col = entity_col
        self.label_end_col = label_end_col
        self.label_horizon = label_horizon
        self.refit_frequency = refit_frequency
        self.n_bins = n_bins
        self.n0 = n0
        self.min_cell_count = min_cell_count
        self.halflife = halflife
        self.min_cross_section = min_cross_section
        self.min_train_rows = min_train_rows
        self.min_train_dates = min_train_dates
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

    def _configure(self, target_name):
        self.fields_ = normalize_fields(self.fields)
        if not isinstance(self.pure_gain, (bool, np.bool_)):
            raise TypeError("pure_gain must be True (gain 2) or False (gain 1).")
        self.gain_ = 2.0 if self.pure_gain else 1.0
        self.parent_columns_ = tuple(dict.fromkeys(
            c for field in self.fields_ for c in (field.a, field.b)
        ))
        self.closure_columns_ = (
            self.parent_columns_ if self.closure_features is None
            else _names(self.closure_features, "closure_features")
        )
        if not self.closure_columns_:
            raise ValueError("closure_features cannot be empty.")
        q = _names(self.q_features, "q_features")
        extras = [c for field in self.fields_ for c in _names(field.local_features, "local_features")]
        if self.apex is not None and (not isinstance(self.apex, str) or not self.apex):
            raise ValueError("apex must be a column name or None.")
        if self.apex in self.parent_columns_:
            raise ValueError("Shared apex must differ from the AB parent factors.")
        if set(q) & set(self.parent_columns_ + ((self.apex,) if self.apex else ())):
            raise ValueError("Q factors must differ from parents and apex.")
        self.used_columns_ = tuple(dict.fromkeys(
            list(self.parent_columns_) + list(self.closure_columns_) + extras
            + ([self.apex] if self.apex else []) + list(q)
        ))
        self.all_columns_ = (self.used_columns_ if self.all_columns is None
                             else _names(self.all_columns, "all_columns"))
        missing = set(self.used_columns_) - set(self.all_columns_)
        if missing:
            raise ValueError(f"Used factors absent from all_columns: {sorted(missing)}")
        forbidden = {target_name, self.date_col, self.entity_col, self.label_end_col} - {None}
        score_names = {s.score_column for s in self.fields_ if s.score_column is not None}
        if set(self.all_columns_) & (forbidden | score_names):
            raise ValueError("Factor inputs cannot include target, metadata or field score columns.")
        if score_names & forbidden:
            raise ValueError("score_column cannot be the target or metadata.")
        self.unused_columns_ = tuple(c for c in self.all_columns_ if c not in self.used_columns_)
        self.q_columns_ = q
        if self.refit_frequency not in ("Y", "Q", "M"):
            raise ValueError("refit_frequency must be 'Y', 'Q', or 'M'.")
        for name in ("n_bins", "min_cross_section", "min_train_rows", "min_train_dates", "n_jobs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.label_horizon, int) or self.label_horizon < 0:
            raise ValueError("label_horizon must be a nonnegative number of observed dates.")
        if self.early_stopping_rounds is not None and (
            not isinstance(self.early_stopping_rounds, int) or self.early_stopping_rounds < 1
        ):
            raise ValueError("early_stopping_rounds must be None or a positive integer.")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie strictly between 0 and 0.5.")
        # Validate CellStats configuration even when all anchors are external.
        from ._cellstats import CellStats
        CellStats(self.n_bins ** 2, self.n0, self.min_cell_count, self.halflife)
        self.model_params_ = dict(
            objective="reg:squarederror", learning_rate=0.035, n_estimators=600,
            max_depth=5, min_child_weight=2000.0, reg_lambda=10.0,
            max_bin=63, subsample=1.0, colsample_bytree=1.0,
            tree_method="hist", device=self.device, random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        supplied = dict(self.xgb_params or {})
        blocked = {"objective", "device", "random_state", "n_jobs", "early_stopping_rounds", "callbacks"}
        if blocked & supplied.keys():
            raise ValueError(f"Use top-level configuration for {sorted(blocked & supplied.keys())}.")
        self.model_params_.update(supplied)
        if int(self.model_params_["n_estimators"]) < 1:
            raise ValueError("n_estimators must be positive.")

    def _panel(self, frame, *, training):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise TypeError("data must be a nonempty pandas DataFrame.")
        if not frame.columns.is_unique:
            raise ValueError("DataFrame column names must be unique.")
        work = frame.copy()
        # Named date/symbol index levels may be used instead of columns.
        for name in (self.date_col, self.entity_col):
            if name is not None and name not in work.columns and name in work.index.names:
                work[name] = work.index.get_level_values(name).to_numpy()
        needed = set(self.used_columns_) | {self.date_col}
        if self.entity_col is not None:
            needed.add(self.entity_col)
        needed.update(s.score_column for s in self.fields_ if s.score_column is not None)
        if training and self.label_end_col is not None:
            needed.add(self.label_end_col)
        missing = needed - set(work.columns)
        if missing:
            raise ValueError(f"Missing required data columns: {sorted(missing)}")
        dates = pd.to_datetime(work[self.date_col], errors="raise")
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError("date must be nonmissing, timezone-naive timestamps.")
        work[self.date_col] = dates.to_numpy()
        if self.entity_col is not None:
            if work[self.entity_col].isna().any():
                raise ValueError("Entity identifiers must not be missing.")
            if work.duplicated([self.date_col, self.entity_col]).any():
                raise ValueError("Duplicate date/entity rows; merge tables explicitly before fit.")
        for name in needed - {self.date_col, self.entity_col, self.label_end_col}:
            try:
                work[name] = pd.to_numeric(work[name], errors="raise").astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Factor/score {name!r} must be numeric.") from exc
            if np.isinf(work[name].to_numpy()).any():
                raise ValueError(f"{name!r} contains infinity; replace it with a valid value or NaN.")
        order = np.argsort(work[self.date_col].to_numpy(), kind="stable")
        return work.iloc[order].reset_index(drop=True), order

    def _label_ends(self, work, dates):
        if self.label_end_col is not None:
            end = pd.to_datetime(work[self.label_end_col], errors="raise")
            if end.dt.tz is not None:
                raise ValueError("label_end timestamps must be timezone-naive.")
            ends = end.to_numpy(dtype="datetime64[ns]")
        else:
            unique, codes = np.unique(dates, return_inverse=True)
            dest = codes + self.label_horizon
            ends = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
            valid = dest < len(unique)
            ends[valid] = unique[dest[valid]]
        if np.any((~pd.isna(ends)) & (ends < dates)):
            raise ValueError("label_end cannot precede its signal date.")
        return ends

    def _features(self, work, dates):
        return build_features(
            work, dates, self.fields_, self.parent_columns_, self.closure_columns_,
            self.apex, self.q_columns_, self.min_cross_section,
        )

    def _learn(self, x, target, gate, dates, ends, cutoff, stage, field):
        eligible = gate & np.isfinite(target) & (dates < cutoff) & (ends < cutoff)
        unique = np.unique(dates[eligible])
        log = dict(stage=stage, field=field, cutoff=str(pd.Timestamp(cutoff)),
                   train_rows=int(eligible.sum()), train_dates=len(unique),
                   n_features=x.shape[1], trees=0, dev_rows=0, early_stopped=False)
        if eligible.sum() < self.min_train_rows or len(unique) < self.min_train_dates:
            log["status"] = "warmup"
            self._audit.append(log)
            return None
        log.update(status="fitted", max_train_date=str(pd.Timestamp(dates[eligible].max())),
                   max_label_end=str(pd.Timestamp(ends[eligible].max())))
        date_codes = np.searchsorted(np.unique(dates), dates)
        anchor = date_codes[eligible].max()
        weights = np.power(0.5, (anchor - date_codes) / self.halflife)
        params = self.model_params_.copy()
        # The development block is entirely before deployment; purge labels
        # again at its boundary before choosing tree count, then refit.
        ndev = max(1, int(np.ceil(len(unique) * self.validation_fraction)))
        dev_start = unique[-ndev]
        tr = eligible & (dates < dev_start) & (ends < dev_start)
        va = eligible & (dates >= dev_start)
        if (self.early_stopping_rounds is not None
                and tr.sum() >= self.min_train_rows
                and len(np.unique(dates[tr])) >= self.min_train_dates
                and va.sum() >= self.min_train_rows):
            selection = XGBRegressor(**params, early_stopping_rounds=self.early_stopping_rounds)
            selection.fit(x[tr], target[tr], sample_weight=weights[tr],
                          eval_set=[(x[va], target[va])], verbose=False)
            params["n_estimators"] = int(selection.best_iteration) + 1
            log.update(dev_rows=int(va.sum()), early_stopped=True,
                       dev_start=str(pd.Timestamp(dev_start)),
                       selection_max_label_end=str(pd.Timestamp(ends[tr].max())))
        model = XGBRegressor(**params)
        model.fit(x[eligible], target[eligible], sample_weight=weights[eligible], verbose=False)
        log["trees"] = int(params["n_estimators"])
        self._audit.append(log)
        return model

    def fit(self, data, y=None, *, target=None, as_of=None):
        """Train on one long panel; y is a column name or aligned 1-D values.

        ``target='r'`` and ``y='r'`` are equivalent. Dates/labels are purged at
        every boundary. ``as_of`` defaults to 1ns after the final signal date;
        only labels with label_end < as_of can train the deployable models.
        """
        if hasattr(self, "is_fitted_"):
            del self.is_fitted_
        if target is not None and y is not None:
            raise ValueError("Supply either y or target, not both.")
        label = target if target is not None else y
        if label is None:
            raise ValueError("Supply target='r' or a target vector y.")
        name = label if isinstance(label, str) else getattr(label, "name", None)
        self._configure(name)
        if isinstance(label, str):
            if label not in data.columns:
                raise ValueError(f"Target column {label!r} is absent.")
            values = data[label].to_numpy(dtype=float)
        else:
            if isinstance(label, pd.Series) and not label.index.equals(data.index):
                raise ValueError("y Series index must exactly match data.index; no implicit join.")
            values = np.asarray(label, dtype=float)
        if values.shape != (len(data),) or np.isinf(values).any():
            raise ValueError("y must be a finite-or-NaN 1-D vector matching data rows.")
        work, order = self._panel(data, training=True)
        values = values[order]
        dates = work[self.date_col].to_numpy(dtype="datetime64[ns]")
        self.as_of_ = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(dates.max()) + pd.Timedelta(1, "ns")
        if pd.isna(self.as_of_) or self.as_of_.tz is not None or self.as_of_ <= pd.Timestamp(dates.max()):
            raise ValueError("as_of must be timezone-naive and strictly after all input signal dates.")
        cutoff = self.as_of_.to_datetime64()
        ends = self._label_ends(work, dates)
        ranks, xs, gates, xshared, shared_gate, names = self._features(work, dates)
        fields, tables = historical_fields(
            self.fields_, work, ranks, values, dates, ends, cutoff,
            {"n_bins": self.n_bins, "stats": dict(n0=self.n0, min_n=self.min_cell_count, halflife=self.halflife)},
        )
        for m in range(len(gates)):
            gates[m] = gates[m] & np.isfinite(fields[:, m])
        self.local_feature_names_ = dict(zip([s.key for s in self.fields_], names))
        self.closure_feature_names_ = tuple(f"rank:{c}" for c in self.closure_columns_)
        self._audit = []
        period = pd.DatetimeIndex(dates).to_period(self.refit_frequency)
        blocks = [(dates[period == p].min(), period == p) for p in period.unique()]
        local_oof = np.full_like(fields, np.nan)
        for boundary, deploy in blocks:
            for m, spec in enumerate(self.fields_):
                model = self._learn(xs[m], values - fields[:, m], gates[m], dates, ends,
                                    boundary, "local_oof", spec.key)
                take = deploy & gates[m]
                if model is not None and take.any():
                    local_oof[take, m] = model.predict(xs[m][take])
        # Equal weighting is strict: a missing expert is not silently dropped.
        base = fields.mean(axis=1) + self.gain_ * local_oof.mean(axis=1)
        close_gate = shared_gate & np.isfinite(base)
        closure_oof = np.full(len(data), np.nan)
        for boundary, deploy in blocks:
            model = self._learn(xshared, values - base, close_gate, dates, ends,
                                boundary, "closure_oof", "all_parents")
            take = deploy & close_gate
            if model is not None and take.any():
                closure_oof[take] = model.predict(xshared[take])
        self.local_models_ = []
        for m, spec in enumerate(self.fields_):
            model = self._learn(xs[m], values - fields[:, m], gates[m], dates, ends,
                                cutoff, "local_final", spec.key)
            if model is None:
                raise ValueError(f"Insufficient mature history for field {spec.key}; add dates or lower explicit minima.")
            self.local_models_.append(model)
        self.closure_model_ = self._learn(xshared, values - base, close_gate, dates, ends,
                                         cutoff, "closure_final", "all_parents")
        if self.closure_model_ is None:
            raise ValueError("Insufficient OOF backbone history for closure; add periods or choose monthly/quarterly refits.")
        self.field_tables_ = tables
        output = self._components(fields, local_oof, closure_oof)
        output[self.date_col] = dates
        if self.entity_col is not None:
            output[self.entity_col] = work[self.entity_col].to_numpy()
        self.oof_predictions_ = self._restore(output, order, data.index)
        self.training_audit_ = pd.DataFrame(self._audit)
        self.feature_names_in_ = np.array(self.used_columns_, dtype=object)
        self.n_features_in_ = len(self.used_columns_)
        self.training_summary_ = dict(
            rows=len(data), dates=len(np.unique(dates)), fields=len(self.fields_), gain=self.gain_,
            as_of=str(self.as_of_), mature_labels=int((np.isfinite(values) & (ends < cutoff)).sum()),
            oof_rows=int(np.isfinite(output.signal).sum()),
            warmup_or_missing_rows=int((~np.isfinite(output.signal)).sum()),
            externally_supplied_fields=[s.key for s in self.fields_ if s.score_column is not None],
            unused_columns=list(self.unused_columns_),
        )
        self.is_fitted_ = True
        del self._audit
        return self

    def _components(self, fields, local, closure):
        out = {}
        for m, spec in enumerate(self.fields_):
            out[f"field:{spec.key}"] = fields[:, m]
            out[f"local:{spec.key}"] = local[:, m]
        out["field_mean"] = fields.mean(axis=1)
        out["local_mean"] = local.mean(axis=1)
        out["backbone"] = out["field_mean"] + self.gain_ * out["local_mean"]
        out["closure"] = closure
        out["signal"] = out["backbone"] + closure
        return pd.DataFrame(out)

    @staticmethod
    def _restore(frame, order, index):
        result = frame.iloc[np.argsort(order)].copy()
        result.index = index
        return result

    def predict_components(self, data):
        """Return F_m, local predictions, B, closure and final signal in input order.

        No y or future labels are read. All dates must be >= model.as_of_.
        Generated fields are a frozen checkpoint; refit when new labels mature.
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("Call fit before predict.")
        work, order = self._panel(data, training=False)
        dates = work[self.date_col].to_numpy(dtype="datetime64[ns]")
        if np.any(dates < self.as_of_.to_datetime64()):
            raise ValueError("predict accepts future dates only; use oof_predictions_ for training-period evaluation.")
        _, xs, gates, xshared, shared_gate, _ = self._features(work, dates)
        fields = snapshot_fields(self.fields_, work, dates, self.field_tables_, self.n_bins)
        local = np.full_like(fields, np.nan)
        for m, model in enumerate(self.local_models_):
            good = gates[m] & np.isfinite(fields[:, m])
            if good.any():
                local[good, m] = model.predict(xs[m][good])
        valid = shared_gate & np.isfinite(fields).all(axis=1) & np.isfinite(local).all(axis=1)
        closure = np.full(len(work), np.nan)
        if valid.any():
            closure[valid] = self.closure_model_.predict(xshared[valid])
        return self._restore(self._components(fields, local, closure), order, data.index)

    def predict(self, data):
        """Return a Series named signal, with the caller's row index."""
        return self.predict_components(data)["signal"]

    def save(self, path):
        """Save a fitted model using joblib; load only files you trust."""
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        """Load a trusted joblib file produced by save, using compatible versions."""
        model = joblib.load(path)
        if not isinstance(model, cls) or not getattr(model, "is_fitted_", False):
            raise TypeError("File does not contain a fitted FPRC model.")
        return model
