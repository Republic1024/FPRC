"""Explicit legacy field declarations; no automatic factor/axis selection."""
from .fields import FieldSpec


def old_three_fields(*, score_columns=None):
    """Return the old three AB fields, optionally with three PIT score columns.

    This declares factor identities only. No data, forecasts or fitted models
    are bundled, and this is not a claim of reproducing the old trading book.
    """
    scores = [None] * 3 if score_columns is None else list(score_columns)
    if len(scores) != 3:
        raise ValueError("score_columns must have exactly three entries.")
    return [
        FieldSpec("momentum_20d", "reversal_5d", "momentum_reversal", score_column=scores[0]),
        FieldSpec("pb_rank", "momentum_20d", "value_momentum", score_column=scores[1]),
        FieldSpec("price_ma_ratio", "volume_change_5d", "breakout_volume", score_column=scores[2]),
    ]
