"""Explicit old-three-field recipe for YOUR already PIT-clean long DataFrame.

Call train_old_three(train_df). No automatic download, selection or trading.
"""
from fprc import FPRC, old_three_fields


def train_old_three(train_df, *, as_of=None, device="cuda", score_columns=None):
    factors = [
        "momentum_20d", "reversal_5d", "pb_rank", "price_ma_ratio",
        "volume_change_5d", "turnover_rank", "volatility_20d",
        "netprofit_yoy", "ocf_to_profit", "profit_stability",
    ]
    return FPRC(
        fields=old_three_fields(score_columns=score_columns), all_columns=factors,
        apex="turnover_rank",
        q_features=["volatility_20d", "netprofit_yoy", "ocf_to_profit", "profit_stability"],
        pure_gain=True, label_end_col="label_end", refit_frequency="Y",
        device=device,
    ).fit(train_df, target="r", as_of=as_of)
