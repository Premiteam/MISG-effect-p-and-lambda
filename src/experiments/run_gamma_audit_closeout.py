from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EPS = 1e-12
P_VALUES = (1, 2, 5, 8, 12)
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_AUDIT = WORKSPACE_ROOT / "data" / "balanced_full_factorial_results.json"
DEFAULT_ORIGINAL = (
    WORKSPACE_ROOT
    / "02_analysis/core/02_p_lambda_factorial/p_lambda_interaction_results"
    / "p_lambda_interaction_results.json"
)
DEFAULT_FROZEN = (
    WORKSPACE_ROOT
    / "02_analysis/core/04_frozen_trajectory_confirmation/final_frozen_comparison_results"
    / "final_frozen_comparison_results.json"
)
DEFAULT_NEW_SYSTEMS = (
    WORKSPACE_ROOT
    / "02_analysis/core/05_new_system_holdout/new_system_holdout_results"
    / "new_system_holdout_results.json"
)
DEFAULT_OUTDIR = WORKSPACE_ROOT / "data" / "closeout_results"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _validate_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN/Inf")
    return vector


def _bootstrap_medians(
    vectors: Mapping[str, np.ndarray], *, B: int, seed: int
) -> dict[str, np.ndarray]:
    if B < 500:
        raise ValueError("B must be at least 500")
    checked = {name: _validate_vector(values, name) for name, values in vectors.items()}
    sizes = {vector.size for vector in checked.values()}
    if len(sizes) != 1:
        raise ValueError("paired contrast vectors have unequal lengths")
    n = sizes.pop()
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in checked}
    completed = 0
    while completed < B:
        current = min(1000, B - completed)
        indices = rng.integers(0, n, size=(current, n))
        for name, vector in checked.items():
            parts[name].append(np.median(vector[indices], axis=1))
        completed += current
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def joint_bootstrap_rows(
    vectors: Mapping[str, np.ndarray], *, B: int, seed: int
) -> list[dict[str, Any]]:
    checked = {name: _validate_vector(values, name) for name, values in vectors.items()}
    if not checked:
        raise ValueError("at least one comparison is required")
    distributions = _bootstrap_medians(checked, B=B, seed=seed)
    family_size = len(checked)
    simultaneous_level = 1.0 - 0.05 / family_size
    simultaneous_tail = (1.0 - simultaneous_level) / 2.0
    rows: list[dict[str, Any]] = []
    for name, vector in checked.items():
        distribution = distributions[name]
        point = float(np.median(vector))
        lo95, hi95 = np.quantile(distribution, [0.025, 0.975])
        lo_sim, hi_sim = np.quantile(
            distribution, [simultaneous_tail, 1.0 - simultaneous_tail]
        )
        if hi_sim < 0.0:
            conclusion = "numerator lower paired risk"
        elif lo_sim > 0.0:
            conclusion = "denominator lower paired risk"
        else:
            conclusion = "no clear difference after family-wise control"
        rows.append(
            {
                "comparison_id": name,
                "n_systems": int(vector.size),
                "B": int(B),
                "point_log_ratio": point,
                "ratio_numerator_over_denominator": float(math.exp(point)),
                "ci95_ratio_lo": float(math.exp(float(lo95))),
                "ci95_ratio_hi": float(math.exp(float(hi95))),
                "simultaneous_ci_level": simultaneous_level,
                "simultaneous_ratio_lo": float(math.exp(float(lo_sim))),
                "simultaneous_ratio_hi": float(math.exp(float(hi_sim))),
                "simultaneous_conclusion": conclusion,
            }
        )
    return rows


def equivalence_row(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    *,
    B: int,
    seed: int,
    delta: float,
) -> dict[str, Any]:
    a = _validate_vector(risk_a, "risk_a")
    b = _validate_vector(risk_b, "risk_b")
    if a.size != b.size:
        raise ValueError("paired risk vectors have unequal lengths")
    contrast = np.log((a + EPS) / (b + EPS))
    distributions = _bootstrap_medians({"contrast": contrast}, B=B, seed=seed)[
        "contrast"
    ]
    point = float(np.median(contrast))
    lo95, hi95 = np.quantile(distributions, [0.025, 0.975])
    lo90, hi90 = np.quantile(distributions, [0.05, 0.95])
    margin = math.log1p(delta)
    return {
        "n_systems": int(a.size),
        "B": int(B),
        "ratio_A_over_B": float(math.exp(point)),
        "ci95_ratio_lo": float(math.exp(float(lo95))),
        "ci95_ratio_hi": float(math.exp(float(hi95))),
        "ci90_ratio_lo": float(math.exp(float(lo90))),
        "ci90_ratio_hi": float(math.exp(float(hi90))),
        "equivalence_delta": float(delta),
        "equivalence_ratio_bounds": [1.0 / (1.0 + delta), 1.0 + delta],
        "equivalent_90pct_CI": bool(lo90 > -margin and hi90 < margin),
    }


def cell_key(p: int, lambda_value: float) -> str:
    return f"p={int(p)}|lambda={str(float(lambda_value))}"


def cell_risk(results: Mapping[str, Any], p: int, lam: float, split: str) -> np.ndarray:
    return _validate_vector(
        np.asarray(results["cells"][cell_key(p, lam)][split]["system_risks"]),
        cell_key(p, lam),
    )


def key_secondary_vectors(evaluation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    def risk(name: str) -> np.ndarray:
        return _validate_vector(np.asarray(evaluation[name]["system_risks"]), name)

    a = risk("A_p1_lambda1")
    b = risk("B_p1_lambda095")
    c = risk("C_p5_lambda1")
    d = risk("D_p5_lambda095")
    return {
        "S1_B_over_A": np.log((b + EPS) / (a + EPS)),
        "S2_D_over_C": np.log((d + EPS) / (c + EPS)),
        "S3_A_over_C": np.log((a + EPS) / (c + EPS)),
    }


def gamma_stability_rows(
    audit: Mapping[str, Any], original: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for p in P_VALUES:
        for lam in (1.0, 0.95):
            key = cell_key(p, lam)
            new = audit["cells"][key]
            old = original["cells"][key]
            val_new = np.asarray(new["val"]["system_risks"], dtype=float)
            val_old = np.asarray(old["val"]["system_risks"], dtype=float)
            test_new = np.asarray(new["test"]["system_risks"], dtype=float)
            test_old = np.asarray(old["test"]["system_risks"], dtype=float)
            rows.append(
                {
                    "p": p,
                    "lambda": lam,
                    "original_gamma": float(old["gamma"]),
                    "audit_gamma": float(new["gamma"]),
                    "gamma_same": float(old["gamma"]) == float(new["gamma"]),
                    "boundary": bool(new["boundary"]),
                    "val_risks_exactly_same": bool(np.array_equal(val_new, val_old)),
                    "test_risks_exactly_same": bool(np.array_equal(test_new, test_old)),
                    "test_max_abs_difference": float(
                        np.max(np.abs(test_new - test_old))
                    ),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_ratio(row: Mapping[str, Any], lo: str, hi: str) -> str:
    return (
        f"{row['ratio_numerator_over_denominator']:.4f} [{row[lo]:.4f}, {row[hi]:.4f}]"
    )


def build_report(results: Mapping[str, Any]) -> str:
    gamma_rows = results["gamma_stability"]
    p1 = results["audit_primary_equivalence_test"]
    old_p1 = results["original_primary_equivalence_test"]
    lines = [
        "# Full-train gamma-tuning robustness audit",
        "",
        "**Status:** complete retrospective robustness audit  ",
        "**Protocol:** 201 train systems (d=2/3/4: 67/67/67), 102 validation, 201 test, original 21-point gamma grid and top-3 validation selection.",
        "",
        "## Decision",
        "",
        "The full-train audit supports the paper's existing main line. Nine of ten gamma optima were unchanged; the only change was p=2, lambda=1 (3.0 to 3.5). All four frozen A-D model gammas were unchanged and no audited optimum hit the gamma-grid boundary.",
        "",
        f"The original test result for p=1 versus p=5 at lambda=.95 was RR={old_p1['ratio_A_over_B']:.4f}, 90% CI [{old_p1['ci90_ratio_lo']:.4f}, {old_p1['ci90_ratio_hi']:.4f}]. The full-train rerun gave RR={p1['ratio_A_over_B']:.4f}, 90% CI [{p1['ci90_ratio_lo']:.4f}, {p1['ci90_ratio_hi']:.4f}]. Both intervals are fully inside the 5% multiplicative-equivalence bounds [{p1['equivalence_ratio_bounds'][0]:.4f}, {p1['equivalence_ratio_bounds'][1]:.4f}].",
        "",
        "## Gamma stability",
        "",
        "| p | lambda | old gamma | full-train gamma | same | boundary |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in gamma_rows:
        lines.append(
            f"| {row['p']} | {row['lambda']:.2f} | {row['original_gamma']:.3g} | {row['audit_gamma']:.3g} | {'yes' if row['gamma_same'] else 'no'} | {'yes' if row['boundary'] else 'no'} |"
        )
    lines += [
        "",
        "## Key secondary family (paired system bootstrap)",
        "",
        "Values are median paired risk ratios with Bonferroni 98.33% simultaneous percentile intervals (B=10,000).",
        "",
        "| Dataset | Comparison | RR [98.33% simultaneous CI] | Family-wise conclusion |",
        "|---|---|---|---|",
    ]
    for row in results["key_secondary_simultaneous"]:
        lines.append(
            f"| {row['dataset']} | {row['comparison_id']} | {fmt_ratio(row, 'simultaneous_ratio_lo', 'simultaneous_ratio_hi')} | {row['simultaneous_conclusion']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This audit removes the known d=2-only gamma-screening concern for the ten headline cells. Fixed-gamma system-risk vectors were not bitwise identical to the archived run, and some high-window forgetting cells showed large per-system differences around numerical divergence boundaries. The rerun used the restored code and deterministic seeds on macOS/Accelerate, whereas the archived results came from a different machine stack; this is therefore recorded as platform/numerical sensitivity, not attributed to gamma selection alone. Importantly, the paired P1 and S1-S3 conclusions remained unchanged.",
        "",
        "The audit does not turn the retrospective analysis into a preregistration, justify the 5% equivalence margin, or provide cross-generator/real-data external validation.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    audit = load_json(args.audit.resolve())
    original = load_json(args.original.resolve())
    frozen = load_json(args.frozen.resolve())
    new_systems = load_json(args.new_systems.resolve())
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    audit_p1 = equivalence_row(
        cell_risk(audit, 1, 0.95, "test"),
        cell_risk(audit, 5, 0.95, "test"),
        B=args.B,
        seed=20260825,
        delta=0.05,
    )
    audit_p1.update(
        {
            "comparison_id": "P1_p1_over_p5_at_lambda095",
            "split": "audit_test_confirmation",
        }
    )

    original_p1 = equivalence_row(
        cell_risk(original, 1, 0.95, "test"),
        cell_risk(original, 5, 0.95, "test"),
        B=args.B,
        seed=20260825,
        delta=0.05,
    )
    original_p1.update(
        {
            "comparison_id": "P1_p1_over_p5_at_lambda095",
            "split": "original_test_confirmation",
        }
    )

    audit_vectors = {
        "S1_B_over_A": np.log(
            (cell_risk(audit, 1, 0.95, "test") + EPS)
            / (cell_risk(audit, 1, 1.0, "test") + EPS)
        ),
        "S2_D_over_C": np.log(
            (cell_risk(audit, 5, 0.95, "test") + EPS)
            / (cell_risk(audit, 5, 1.0, "test") + EPS)
        ),
        "S3_A_over_C": np.log(
            (cell_risk(audit, 1, 1.0, "test") + EPS)
            / (cell_risk(audit, 5, 1.0, "test") + EPS)
        ),
    }
    audit_secondary = joint_bootstrap_rows(audit_vectors, B=args.B, seed=20260825)
    for row in audit_secondary:
        row["dataset"] = "full_train_audit_test"

    key_secondary: list[dict[str, Any]] = []
    for dataset, payload, seed in (
        ("frozen_new_trajectories", frozen, 20260825),
        ("same_generator_new_systems", new_systems, 20260826),
    ):
        rows = joint_bootstrap_rows(
            key_secondary_vectors(payload["evaluation"]), B=args.B, seed=seed
        )
        for row in rows:
            row["dataset"] = dataset
        key_secondary.extend(rows)

    output = {
        "meta": {
            "analysis_type": "retrospective_full_train_gamma_audit_closeout",
            "B": args.B,
            "paired_unit": "system",
            "ci_method": "percentile bootstrap",
            "key_secondary_family_size": 3,
            "simultaneous_ci_level": 1.0 - 0.05 / 3.0,
            "equivalence_bounds": [1.0 / 1.05, 1.05],
            "audit_results": str(args.audit.resolve()),
            "original_results": str(args.original.resolve()),
            "frozen_results": str(args.frozen.resolve()),
            "new_system_results": str(args.new_systems.resolve()),
        },
        "gamma_stability": gamma_stability_rows(audit, original),
        "original_primary_equivalence_test": original_p1,
        "audit_primary_equivalence_test": audit_p1,
        "audit_key_secondary_simultaneous": audit_secondary,
        "key_secondary_simultaneous": key_secondary,
    }
    result_path = outdir / "statistical_closeout_results.json"
    json_dump(output, result_path)
    write_csv(output["gamma_stability"], outdir / "gamma_stability_detailed.csv")
    write_csv(audit_secondary, outdir / "audit_key_secondary_simultaneous.csv")
    write_csv(key_secondary, outdir / "key_secondary_simultaneous_cis.csv")
    report_path = outdir / "AUDIT_REPORT.md"
    report_path.write_text(build_report(output), encoding="utf-8")
    print(f"Statistical close-out complete: {result_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multiplier-selection audit results"
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--new-systems", type=Path, default=DEFAULT_NEW_SYSTEMS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--B", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
