"""Run after pip install .; no repo data needed. Small CPU/GPU smoke example."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fprc import FPRC, FieldSpec


def make_panel(n_dates=220, n_stocks=48, seed=23):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    n = n_dates * n_stocks
    data = pd.DataFrame({
        "date": np.repeat(dates, n_stocks),
        "symbol": np.tile([f"S{i:03d}" for i in range(n_stocks)], n_dates),
    })
    for name in ["a", "b", "c", "d", "e", "f", "p", "q"]:
        data[name] = rng.normal(size=n)
    data["r"] = (0.007 * data.a * data.b - 0.004 * data.c
                 + 0.003 * data.e * data.f + 0.003 * data.p
                 + rng.normal(0, 0.02, n))
    data["label_end"] = np.repeat(dates + pd.offsets.BDay(3), n_stocks)
    data.loc[rng.random(n) < 0.12, "q"] = np.nan
    return data


def run(device):
    df = make_panel()
    split = sorted(df.date.unique())[175]
    train, test = df[df.date < split].copy(), df[df.date >= split].copy()
    model = FPRC(
        fields=[FieldSpec("a", "b", "ab"), ("c", "d"), ("e", "f")],
        all_columns=list("abcdefpq"), apex="p", q_features=["q"],
        pure_gain=True, label_end_col="label_end", refit_frequency="M",
        n_bins=5, n0=20, min_cell_count=5, device=device,
        min_train_dates=15, min_train_rows=100,
        xgb_params={"n_estimators": 35, "max_depth": 3, "min_child_weight": 5},
        early_stopping_rounds=8,
    ).fit(train, target="r", as_of=split)
    # Predict without any future label or label-end column.
    parts = model.predict_components(test.drop(columns=["r", "label_end"]))
    np.testing.assert_allclose(parts.backbone, parts.field_mean + 2 * parts.local_mean)
    np.testing.assert_allclose(parts.signal, parts.backbone + parts.closure)
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    model.save(output / f"pure_gain_{device}.joblib")
    restored = FPRC.load(output / f"pure_gain_{device}.joblib")
    np.testing.assert_allclose(restored.predict(test), parts.signal)
    parts.to_csv(output / f"predictions_{device}.csv", index=True)
    model.training_audit_.to_csv(output / f"training_audit_{device}.csv", index=False)
    print(model.training_summary_)
    print(parts[["field_mean", "local_mean", "backbone", "closure", "signal"]].head())
    print(f"future_rows={len(parts)}, finite_predictions={parts.signal.notna().sum()}")
    print("Formula and save/load audits: PASS. Synthetic outputs are not a financial backtest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run(parser.parse_args().device)
