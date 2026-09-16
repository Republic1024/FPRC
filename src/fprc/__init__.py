"""Field-Preserving Residual Closure. No workspace data are imported."""
from .fields import FieldSpec
from .model import FPRCRegressor
from .presets import old_three_fields

FPRC = FPRCRegressor
__version__ = "0.1.0"
__all__ = ["FPRC", "FPRCRegressor", "FieldSpec", "old_three_fields"]
