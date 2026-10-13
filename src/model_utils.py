"""Shared model utilities (importable so joblib pickles resolve in any script)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrated:
    """Wraps a fitted classifier with an isotonic calibration map fit on a holdout."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, X, y):
        p = np.clip(self.base_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        self.ir.fit(p, y)
        return self

    def predict_proba(self, X):
        p = np.clip(self.base_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        pc = np.clip(self.ir.predict(p), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - pc, pc])
