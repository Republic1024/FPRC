"""Behavioral checks: leakage, target routing, algebra, persistence, packaging."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from fprc import FPRC, FieldSpec, old_three_fields
from fprc.fields import fold_states


def panel():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2023-01-02", periods=150)
    n = len(dates) * 28
    df = pd.DataFrame({"date": np.repeat(dates, 28), "symbol": np.tile(np.arange(28), len(dates))})
    for col in "abcdepq":
        df[col] = rng.normal(size=n)
    df["r"] = 0.03 * df.a * df.b + 0.02 * df.c + 0.01 * df.p + rng.normal(0, 0.03, n)
    df["label_end"] = df.date + pd.offsets.BDay(2)
    df.loc[df.index % 9 == 0, "q"] = np.nan
    return df


def estimator(**overrides):
    args = dict(
        fields=[FieldSpec("a", "b", "ab"), FieldSpec("c", "d", "cd")],
        all_columns=list("abcdepq"), apex="p", q_features=["q"],
        label_end_col="label_end", refit_frequency="M", pure_gain=True,
        n_bins=3, n0=5, min_cell_count=2, min_cross_section=10,
        min_train_dates=8, min_train_rows=40, n_jobs=1,
        xgb_params={"n_estimators": 7, "max_depth": 2, "min_child_weight": 1},
        early_stopping_rounds=None,
    )
    args.update(overrides)
    return FPRC(**args)


class FPRCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = panel()
        cls.boundary = sorted(cls.data.date.unique())[120]
        cls.train = cls.data[cls.data.date < cls.boundary].copy()
        cls.test = cls.data[cls.data.date >= cls.boundary].copy()
        cls.model = estimator().fit(cls.train, target="r", as_of=cls.boundary)

    def test_gain_formula_and_same_local_models(self):
        base = estimator(pure_gain=False).fit(self.train, target="r", as_of=self.boundary)
        a = self.model.predict_components(self.test)
        b = base.predict_components(self.test)
        np.testing.assert_array_equal(a.local_mean, b.local_mean)
        np.testing.assert_array_equal(a.field_mean, b.field_mean)
        np.testing.assert_allclose(a.backbone, 2 * b.backbone - b.field_mean, atol=1e-12)
        np.testing.assert_allclose(a.signal, a.backbone + a.closure)
        self.assertEqual(len(self.model.local_models_), 2)

    def test_future_labels_do_not_change_historical_predictions(self):
        changed = self.train.copy()
        cut = pd.Timestamp("2023-05-01")
        changed.loc[changed.date >= cut, "r"] += 100
        model = estimator().fit(changed, target="r", as_of=self.boundary)
        before = self.train.date < cut
        np.testing.assert_array_equal(
            self.model.oof_predictions_.loc[before, "signal"],
            model.oof_predictions_.loc[before, "signal"],
        )
        np.testing.assert_array_equal(
            self.model.oof_predictions_.loc[before, "field:ab"],
            model.oof_predictions_.loc[before, "field:ab"],
        )

    def test_model_training_boundaries(self):
        audit = self.model.training_audit_.query("status == 'fitted'")
        self.assertTrue((pd.to_datetime(audit.max_label_end) < pd.to_datetime(audit.cutoff)).all())
        self.assertTrue((pd.to_datetime(audit.max_train_date) < pd.to_datetime(audit.cutoff)).all())
        self.assertGreater(self.model.oof_predictions_.signal.isna().sum(), 0)
        self.assertGreater(self.model.oof_predictions_.signal.notna().sum(), 0)
        with self.assertRaisesRegex(ValueError, "future dates"):
            self.model.predict(self.train)

    def test_closure_trains_on_oof_backbone(self):
        class RecordingFPRC(FPRC):
            def _learn(inner, x, target, gate, dates, ends, cutoff, stage, field):
                if stage == "closure_final":
                    inner.recorded_closure_target = target.copy()
                    inner.recorded_gate = gate.copy()
                return super()._learn(x, target, gate, dates, ends, cutoff, stage, field)
        model = RecordingFPRC(**estimator().get_params(deep=False)).fit(
            self.train, target="r", as_of=self.boundary
        )
        np.testing.assert_allclose(
            model.recorded_closure_target,
            self.train.r.to_numpy() - model.oof_predictions_.backbone.to_numpy(), equal_nan=True,
        )

    def test_predict_ignores_target_and_preserves_order(self):
        original = self.model.predict(self.test)
        shuffled = self.test.sample(frac=1, random_state=5).drop(columns=["r", "label_end"])
        predicted = self.model.predict(shuffled)
        self.assertTrue(predicted.index.equals(shuffled.index))
        np.testing.assert_allclose(predicted.sort_index(), original.sort_index(), atol=1e-7)
        self.assertTrue(predicted.notna().all())  # Q NaNs do not gate rows.

    def test_field_identity_and_symmetry(self):
        specs = list(reversed(estimator().fields))
        swapped = estimator(fields=specs).fit(self.train, target="r", as_of=self.boundary)
        np.testing.assert_allclose(swapped.predict(self.test), self.model.predict(self.test), atol=1e-6)

    def test_external_fields_and_mixed_mode(self):
        train, test = self.train.copy(), self.test.copy()
        train["external"] = 0.002 * train.a
        test["external"] = 0.002 * test.a
        model = estimator(fields=[FieldSpec("a", "b", "ab", score_column="external"), ("c", "d")]).fit(
            train, target="r", as_of=self.boundary
        )
        np.testing.assert_array_equal(model.oof_predictions_["field:ab"], train.external)
        np.testing.assert_array_equal(model.predict_components(test)["field:ab"], test.external)

    def test_persistence(self):
        # All temporary artifacts stay within the new package directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = self.model.save(Path(folder) / "model.joblib")
            loaded = FPRC.load(path)
            np.testing.assert_array_equal(self.model.predict(self.test), loaded.predict(self.test))

    def test_input_failures_and_clone(self):
        cloned = clone(estimator())
        self.assertEqual(cloned.pure_gain, True)
        with self.assertRaisesRegex(ValueError, "target"):
            estimator(all_columns=list("abcdepq") + ["r"]).fit(self.train, target="r")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            estimator().fit(pd.concat([self.train, self.train.iloc[:1]]), target="r")
        with self.assertRaisesRegex(ValueError, "index"):
            estimator().fit(self.train, y=self.train.r.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.model.predict(self.test.drop(columns="a"))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            estimator().fit(self.train.iloc[:28 * 5], target="r")
        with self.assertRaisesRegex(ValueError, "precede"):
            bad = self.train.assign(label_end=self.train.date - pd.Timedelta(days=1))
            estimator().fit(bad, target="r")

    def test_simple_pairs_horizon_and_multiindex(self):
        config = estimator(apex=None, q_features=(), all_columns=list("abcd"), label_end_col=None,
                           label_horizon=2, fields=[("a", "b"), ("c", "d")])
        train = self.train.drop(columns="label_end").set_index(["date", "symbol"])
        model = config.fit(train, target="r", as_of=self.boundary)
        test = self.test.set_index(["date", "symbol"])
        self.assertTrue(model.predict(test).index.equals(test.index))
        self.assertEqual(len(model.local_feature_names_["a__b"]), 2)

    def test_preset_and_fold_ties(self):
        specs = old_three_fields()
        self.assertEqual(specs[1].a, "pb_rank")
        d = np.repeat(np.datetime64("2023-01-01"), 5)
        v = np.arange(5, dtype=float)
        np.testing.assert_array_equal(fold_states(v, v, d, 5), [0, 6, 12, 18, 24])
        ties = np.array([1, 1, 2, 3, np.nan])
        states = fold_states(ties, ties, d, 5)
        self.assertEqual(states[0], states[1])
        self.assertTrue(np.isnan(states[-1]))


if __name__ == "__main__":
    unittest.main()
