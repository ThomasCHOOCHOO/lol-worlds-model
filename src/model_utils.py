"""Shared model utilities (importable so joblib pickles resolve in any script)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Prediction clip bounds. The isotonic map itself must never output exactly
# 0 or 1: one September 2026 game was mapped to 1.000 and lost, contributing
# ~0.03 to the honest log-loss (a single game dominating the average).
# Justification for 0.001: the calibration fit slices hold ~2-3k games, so the
# finest empirical resolution is ~1/3000; 0.001 sits just above that floor --
# we never claim more confidence than ~1-in-1000 from the fit data. It also
# caps the worst-case per-game log-loss at -ln(0.001) ~= 6.91, so no single
# game can ever again contribute more than ~0.006 to a ~1,100-game average.
PRED_MIN = 0.001
PRED_MAX = 0.999


class IsotonicCalibrated:
    """Wraps a fitted classifier with an isotonic calibration map fit on a holdout.

    Map outputs are bounded to [PRED_MIN, PRED_MAX] (see above): the old
    y_min=0.0/y_max=1.0 let the shipped v1.3.2 map emit exactly 0.0.
    """

    def __init__(self, base_model, y_min=PRED_MIN, y_max=PRED_MAX):
        self.base_model = base_model
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=y_min, y_max=y_max)

    def fit(self, X, y):
        p = np.clip(self.base_model.predict_proba(X)[:, 1], PRED_MIN, PRED_MAX)
        self.ir.fit(p, y)
        return self

    def predict_proba(self, X):
        p = np.clip(self.base_model.predict_proba(X)[:, 1], PRED_MIN, PRED_MAX)
        pc = np.clip(self.ir.predict(p), PRED_MIN, PRED_MAX)
        return np.column_stack([1 - pc, pc])


def coarsen_isotonic(ir, p_fit, min_count=100):
    """Reduce an isotonic map to fewer, wider steps.

    Keeps a subset of the fitted threshold points such that every retained
    segment covers >= min_count fit games. Returns (xs, ys) defining the
    same piecewise-linear interpolation the app uses.

    Why: isotonic regression on ~2k games grows thin steps (the shipped
    v1.3.2 map has a step 0.3pp wide) whose values rest on a handful of
    games. With min_count=100, every step's value is estimated from >= 100
    games, capping the binomial standard error of any step at
    sqrt(0.25/100) = 0.05. The kept points are a subset of the fitted
    thresholds, so monotonicity is preserved exactly.
    """
    xs_all = np.asarray(ir.X_thresholds_).ravel()
    ys_all = np.asarray(ir.y_thresholds_).ravel()
    p_fit = np.asarray(p_fit)
    n = len(p_fit)
    if len(xs_all) <= 2 or n < min_count:
        return [float(xs_all[0]), float(xs_all[-1])], [float(ys_all[0]), float(ys_all[-1])]
    # Count fit games per threshold segment [xs[i], xs[i+1]); last segment
    # includes its right endpoint.
    counts = np.zeros(len(xs_all) - 1, dtype=int)
    idx = np.searchsorted(xs_all, p_fit, side="right") - 1
    idx = np.clip(idx, 0, len(counts) - 1)
    np.add.at(counts, idx, 1)
    keep = [0]
    acc = 0
    for i, c in enumerate(counts):
        acc += int(c)
        if acc >= min_count:
            keep.append(i + 1)
            acc = 0
    if keep[-1] != len(xs_all) - 1:
        # Trailing remainder (< min_count) merges into the previous segment:
        # drop the last cut so the final segment absorbs it.
        if len(keep) > 1:
            keep.pop()
        keep.append(len(xs_all) - 1)
    xs = [float(xs_all[i]) for i in keep]
    ys = [float(ys_all[i]) for i in keep]
    return xs, ys


def apply_isotonic_map(p, xs, ys):
    """Piecewise-linear isotonic map, the same math the app uses (np.interp
    after clipping to the map's input range). Inputs/outputs are clipped to
    [PRED_MIN, PRED_MAX]."""
    p = np.clip(np.asarray(p, dtype=float), PRED_MIN, PRED_MAX)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    p = np.clip(p, x[0], x[-1])
    return np.clip(np.interp(p, x, y), PRED_MIN, PRED_MAX)
