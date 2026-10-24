from pathlib import Path
import argparse
import csv
import json
import hashlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("predictions", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with args.predictions.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
results = []
for lam in (1.0, 0.95):
    for p in (1, 5):
        selected = sorted(
            [r for r in rows if int(r["p"]) == p and float(r["lambda"]) == lam],
            key=lambda r: int(r["sample_index"]),
        )
        assert len(selected) == 201
        residual = np.array(
            [
                float(r["observed_output"])
                - float(r["fixed_parameter_one_step_prediction"])
                for r in selected
            ]
        )
        assert np.isfinite(residual).all()
        centered = residual - residual.mean()
        denominator = float(centered @ centered)
        results.append(
            {
                "p": p,
                "lambda": lam,
                "n": len(selected),
                "mean_residual": float(residual.mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "acf": {
                    str(lag): float(centered[lag:] @ centered[:-lag] / denominator)
                    for lag in range(1, 11)
                },
            }
        )
report = {
    "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
    "definition": "Demeaned sample autocorrelation: sum e[t] e[t-lag] / sum e[t]^2.",
    "interpretation": "Descriptive check of residual serial dependence; not a test of the physical noise model.",
    "results": results,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
