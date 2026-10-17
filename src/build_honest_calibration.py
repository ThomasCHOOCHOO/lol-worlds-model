"""Build model/honest_calibration.json from the EXISTING trained bundle.

No retraining: loads model/model.joblib (base model fitted on pre-2026-07-01
games), fits the isotonic map ONLY on 2026-07-01..2026-08-31, coarsens it,
grades it on the untouched 2026-09-01..2026-10-01 slice, and writes the
exact schema the app toggle consumes:
  {xs, ys, fitted_on, n, honest_eval{logloss, ece, n}}

Does NOT touch model.joblib, model/metrics.json, or
model/phone_artifacts.json (the shipped map stays as-is; the app keeps
both maps and the toggle switches between them). train.py main() calls the
same helper, so every future retrain regenerates this file too.

Usage: .venv/bin/python src/build_honest_calibration.py
"""
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import honest_map_from_bundle  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")
PROC = os.path.join(BASE, "data", "processed")


def main():
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    df = pd.read_csv(os.path.join(PROC, "features.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    honest_json, report = honest_map_from_bundle(bundle, df, "full")
    out_path = os.path.join(MODEL_DIR, "honest_calibration.json")
    with open(out_path, "w") as f:
        json.dump(honest_json, f, indent=2)

    print(f"wrote {out_path}")
    print(f"  fitted_on: {honest_json['fitted_on']}  n_fit={honest_json['n']}  "
          f"steps={report['n_steps_coarsened']} (uncoarsened: "
          f"{report['n_steps_uncoarsened']})")
    print(f"  HONEST (untouched Sep games, n={report['n_eval']}): "
          f"logloss={report['honest_coarsened']['logloss']:.4f} "
          f"ece={report['honest_coarsened']['ece']:.4f}")
    print(f"  honest, uncoarsened: logloss={report['honest_uncoarsened']['logloss']:.4f} "
          f"ece={report['honest_uncoarsened']['ece']:.4f}")
    print(f"  raw, uncalibrated on eval: logloss="
          f"{report['raw_uncalibrated_on_eval']['logloss']:.4f} "
          f"ece={report['raw_uncalibrated_on_eval']['ece']:.4f}")
    print(f"  shipped-style IN-SAMPLE (full tail, do not quote as quality): "
          f"logloss={report['shipped_style_in_sample_on_full_tail']['logloss']:.4f} "
          f"ece={report['shipped_style_in_sample_on_full_tail']['ece']:.4f}")
    print("  shipped map in phone_artifacts.json untouched; app keeps both maps.")


if __name__ == "__main__":
    main()
