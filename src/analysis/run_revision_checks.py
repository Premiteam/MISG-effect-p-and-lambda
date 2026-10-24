from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
SEED = 20260926
BOOTSTRAP_REPLICATES = 10000


def validation_selection_stability() -> list[dict[str, object]]:
    source = SOURCE / "balanced_full_factorial_results.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    keys = (
        "p=1|lambda=1.0",
        "p=5|lambda=1.0",
        "p=1|lambda=0.95",
        "p=5|lambda=0.95",
    )
    rows: list[dict[str, object]] = []
    for key in keys:
        cell = data["cells"][key]
        candidates = cell["val_candidates"]
        risks = np.asarray(
            [candidate["val"]["system_risks"] for candidate in candidates],
            dtype=float,
        )
        n_systems = risks.shape[1]
        sampled = rng.integers(0, n_systems, size=(BOOTSTRAP_REPLICATES, n_systems))
        bootstrap_medians = np.median(risks[:, sampled], axis=2)
        winners = np.argmin(bootstrap_medians, axis=0)
        candidate_gamma = np.asarray([candidate["gamma"] for candidate in candidates])
        selected_gamma = float(cell["gamma"])
        rows.append(
            {
                "p": int(cell["p"]),
                "lambda": float(cell["lambda"]),
                "selected_gamma": selected_gamma,
                "shortlisted_gamma": ";".join(str(value) for value in candidate_gamma),
                "bootstrap_reselection_rate": float(
                    np.mean(candidate_gamma[winners] == selected_gamma)
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
            }
        )
    return rows


def threshold_failure_difference() -> list[dict[str, object]]:
    source = SOURCE / "trajectory_level_results.jsonl"
    models = ("B_p1_lambda095", "D_p5_lambda095")
    selected: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row["dataset"] == "test"
                and row["condition"] == "baseline"
                and row["model"] in models
                and row["trajectory_index"] in (3, 4)
            ):
                selected.append(row)

    failures: dict[str, np.ndarray] = {}
    for model in models:
        values = []
        for system_index in range(201):
            matched = [
                row
                for row in selected
                if row["model"] == model and row["system_index"] == system_index
            ]
            if len(matched) != 2:
                raise ValueError(f"Expected two trajectories for system {system_index}")
            values.append(int(all(row["hit_10_12000"] is None for row in matched)))
        failures[model] = np.asarray(values, dtype=int)

    paired_difference = failures[models[1]] - failures[models[0]]
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(
        0, paired_difference.size, size=(BOOTSTRAP_REPLICATES, paired_difference.size)
    )
    bootstrap = paired_difference[sampled].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    p1_only = int(np.sum(paired_difference < 0))
    p5_only = int(np.sum(paired_difference > 0))
    discordant = p1_only + p5_only
    smaller = min(p1_only, p5_only)
    exact_mcnemar_p = min(
        1.0,
        2.0
        * sum(math.comb(discordant, k) for k in range(smaller + 1))
        / (2.0**discordant),
    )
    return [
        {
            "n_systems": int(paired_difference.size),
            "p1_failures": int(failures[models[0]].sum()),
            "p1_failure_rate": float(failures[models[0]].mean()),
            "p5_failures": int(failures[models[1]].sum()),
            "p5_failure_rate": float(failures[models[1]].mean()),
            "paired_difference_p5_minus_p1": float(paired_difference.mean()),
            "paired_difference_ci95_low": float(low),
            "paired_difference_ci95_high": float(high),
            "p1_only_failure": p1_only,
            "p5_only_failure": p5_only,
            "same_status": int(np.sum(paired_difference == 0)),
            "exact_two_sided_mcnemar_p": exact_mcnemar_p,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
        }
    ]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    selection_rows = validation_selection_stability()
    threshold_rows = threshold_failure_difference()
    write_csv(selection_rows, ROOT / "validation_selection_stability.csv")
    write_csv(threshold_rows, ROOT / "threshold_failure_difference.csv")
    summary = {
        "validation_selection_stability": selection_rows,
        "threshold_failure_difference": threshold_rows,
    }
    (ROOT / "revision_checks.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
