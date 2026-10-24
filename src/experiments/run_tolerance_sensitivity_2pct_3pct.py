from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_CLOSEOUT = (
    WORKSPACE_ROOT
    / "03_analysis"
    / "robustness"
    / "03_gamma_tuning_audit"
    / "balanced_statistical_closeout_results"
    / "p_lambda_statistical_closeout_results.json"
)
DEFAULT_NEW_TRAJECTORIES = (
    WORKSPACE_ROOT
    / "02_analysis"
    / "core"
    / "04_frozen_trajectory_confirmation"
    / "final_frozen_comparison_results"
    / "final_frozen_comparison_results.json"
)
DEFAULT_NEW_SYSTEMS = (
    WORKSPACE_ROOT
    / "02_analysis"
    / "core"
    / "05_new_system_holdout"
    / "new_system_holdout_results"
    / "new_system_holdout_results.json"
)
DEFAULT_OUTDIR = WORKSPACE_ROOT / "data" / "tolerance_results"
DEFAULT_TOLERANCES = (0.02, 0.03)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
    temporary.replace(path)


def equivalence_bounds(delta: float) -> tuple[float, float]:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"tolerance must be between 0 and 1; received {delta}")
    return 1.0 / (1.0 + delta), 1.0 + delta


def is_contained(ci_low: float, ci_high: float, delta: float) -> bool:
    lower, upper = equivalence_bounds(delta)
    return ci_low >= lower and ci_high <= upper


def validate_interval(row: Mapping[str, Any], source: str) -> None:
    required = ("ratio_A_over_B", "ci90_ratio_lo", "ci90_ratio_hi", "n_systems")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{source} is missing fields: {', '.join(missing)}")
    point = float(row["ratio_A_over_B"])
    low = float(row["ci90_ratio_lo"])
    high = float(row["ci90_ratio_hi"])
    if not all(math.isfinite(value) and value > 0.0 for value in (point, low, high)):
        raise ValueError(f"{source} contains a nonpositive or nonfinite risk ratio")
    if low > high or not low <= point <= high:
        raise ValueError(f"{source} has an invalid or point-excluding 90% interval")


def closeout_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    step = payload.get("step1_p_closeout")
    if not isinstance(step, Mapping):
        raise ValueError("closeout JSON lacks step1_p_closeout")

    rows: list[dict[str, Any]] = []
    for field, dataset in (
        ("validation_pairwise", "validation systems"),
        ("test_confirmation_pairwise", "held-out test systems"),
    ):
        records = step.get(field)
        if not isinstance(records, list):
            raise ValueError(f"closeout JSON lacks {field}")
        for index, record in enumerate(records):
            validate_interval(record, f"{field}[{index}]")
            p_a = int(record["p_A"])
            p_b = int(record["p_B"])
            rows.append(
                {
                    "dataset": dataset,
                    "analysis_role": (
                        "selection-stage sensitivity"
                        if field == "validation_pairwise"
                        else "independent confirmation"
                    ),
                    "comparison_id": f"p{p_a}_vs_p{p_b}_lambda095",
                    "comparison": f"p={p_a} versus p={p_b} at lambda=0.95",
                    "p_A": p_a,
                    "p_B": p_b,
                    "ratio_A_over_B": float(record["ratio_A_over_B"]),
                    "ci90_ratio_lo": float(record["ci90_ratio_lo"]),
                    "ci90_ratio_hi": float(record["ci90_ratio_hi"]),
                    "n_systems": int(record["n_systems"]),
                    "source_scope": "complete p-pair closeout",
                }
            )
    return rows


def frozen_primary_row(
    payload: Mapping[str, Any], *, dataset: str, role: str
) -> dict[str, Any]:
    records = payload.get("paired_comparisons")
    if not isinstance(records, list):
        raise ValueError(f"{dataset} JSON lacks paired_comparisons")
    matches = [
        record
        for record in records
        if record.get("comparison_id") == "p1_vs_p5_with_forgetting"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{dataset} must contain exactly one p1_vs_p5_with_forgetting record"
        )
    record = matches[0]
    validate_interval(record, dataset)
    return {
        "dataset": dataset,
        "analysis_role": role,
        "comparison_id": "p1_vs_p5_lambda095",
        "comparison": "p=1 versus p=5 at lambda=0.95",
        "p_A": 1,
        "p_B": 5,
        "ratio_A_over_B": float(record["ratio_A_over_B"]),
        "ci90_ratio_lo": float(record["ci90_ratio_lo"]),
        "ci90_ratio_hi": float(record["ci90_ratio_hi"]),
        "n_systems": int(record["n_systems"]),
        "source_scope": "primary frozen comparison",
    }


def classify(
    source_rows: Iterable[Mapping[str, Any]], tolerances: Iterable[float]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in source_rows:
        for delta in tolerances:
            lower, upper = equivalence_bounds(float(delta))
            contained = is_contained(
                float(row["ci90_ratio_lo"]),
                float(row["ci90_ratio_hi"]),
                float(delta),
            )
            results.append(
                {
                    **dict(row),
                    "tolerance": float(delta),
                    "tolerance_percent": 100.0 * float(delta),
                    "equivalence_bound_lo": lower,
                    "equivalence_bound_hi": upper,
                    "ci90_fully_contained": bool(contained),
                    "decision": (
                        "within tolerance"
                        if contained
                        else "not established within tolerance"
                    ),
                }
            )
    return results


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "analysis_role",
        "comparison_id",
        "comparison",
        "p_A",
        "p_B",
        "n_systems",
        "ratio_A_over_B",
        "ci90_ratio_lo",
        "ci90_ratio_hi",
        "tolerance_percent",
        "equivalence_bound_lo",
        "equivalence_bound_hi",
        "ci90_fully_contained",
        "decision",
        "source_scope",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def primary_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary_datasets = {
        "validation systems",
        "held-out test systems",
        "new trajectories on the frozen test systems",
        "new held-out systems",
    }
    primary = [
        row
        for row in rows
        if row["comparison_id"] == "p1_vs_p5_lambda095"
        and row["dataset"] in primary_datasets
    ]
    summaries = []
    for tolerance in sorted({float(row["tolerance"]) for row in primary}):
        subset = [row for row in primary if float(row["tolerance"]) == tolerance]
        summaries.append(
            {
                "tolerance": tolerance,
                "tolerance_percent": 100.0 * tolerance,
                "n_datasets": len(subset),
                "n_with_ci90_contained": sum(
                    bool(row["ci90_fully_contained"]) for row in subset
                ),
                "all_primary_datasets_contained": bool(subset)
                and all(bool(row["ci90_fully_contained"]) for row in subset),
            }
        )
    return summaries


def write_markdown(
    rows: list[Mapping[str, Any]], summaries: list[Mapping[str, Any]], path: Path
) -> None:
    primary = [row for row in rows if row["comparison_id"] == "p1_vs_p5_lambda095"]
    lines = [
        "# Tolerance sensitivity for the p=1 versus p=5 comparison",
        "",
        (
            "This report reclassifies the stored 90% confidence intervals using "
            "reciprocal risk-ratio bounds. No trajectories or confidence intervals "
            "were recomputed."
        ),
        "",
        "| Tolerance | Ratio bounds | Datasets passing / assessed | All pass |",
        "|---:|---:|---:|:---:|",
    ]
    for summary in summaries:
        lower, upper = equivalence_bounds(float(summary["tolerance"]))
        lines.append(
            f"| {summary['tolerance_percent']:.0f}% | "
            f"[{lower:.6f}, {upper:.6f}] | "
            f"{summary['n_with_ci90_contained']} / {summary['n_datasets']} | "
            f"{'yes' if summary['all_primary_datasets_contained'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-level results",
            "",
            "| Dataset | Tolerance | Ratio | 90% CI | Decision |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in primary:
        lines.append(
            f"| {row['dataset']} | {row['tolerance_percent']:.0f}% | "
            f"{row['ratio_A_over_B']:.6f} | "
            f"[{row['ci90_ratio_lo']:.6f}, {row['ci90_ratio_hi']:.6f}] | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            (
                "A contained interval supports compatibility with the stated "
                "tolerance for the paired median risk ratio. It does not show that "
                "the two window lengths are identical or interchangeable under every "
                "system-identification setting."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    paths = {
        "closeout": args.closeout.expanduser().resolve(),
        "new trajectories": args.new_trajectories.expanduser().resolve(),
        "new systems": args.new_systems.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} results not found: {path}")

    tolerances = sorted({float(value) for value in args.tolerances})
    if not tolerances:
        raise ValueError("at least one tolerance is required")
    for value in tolerances:
        equivalence_bounds(value)

    source_rows = closeout_rows(load_json(paths["closeout"]))
    source_rows.append(
        frozen_primary_row(
            load_json(paths["new trajectories"]),
            dataset="new trajectories on the frozen test systems",
            role="frozen-config trajectory replication",
        )
    )
    source_rows.append(
        frozen_primary_row(
            load_json(paths["new systems"]),
            dataset="new held-out systems",
            role="frozen-config system replication",
        )
    )

    rows = classify(source_rows, tolerances)
    summaries = primary_summary(rows)
    outdir = args.outdir.expanduser().resolve()
    csv_path = outdir / "tolerance_sensitivity_2pct_3pct.csv"
    json_path = outdir / "tolerance_sensitivity_2pct_3pct.json"
    markdown_path = outdir / "tolerance_sensitivity_2pct_3pct.md"
    write_csv(rows, csv_path)
    atomic_json_dump(
        {
            "meta": {
                "analysis_type": "equivalence_tolerance_sensitivity",
                "confidence_interval": "stored paired-system 90% CI",
                "ratio_bounds": "[1/(1+delta), 1+delta]",
                "tolerances": tolerances,
                "recomputed_trajectories": False,
                "recomputed_confidence_intervals": False,
                "source_files": {key: str(value) for key, value in paths.items()},
            },
            "primary_summary": summaries,
            "results": rows,
        },
        json_path,
    )
    write_markdown(rows, summaries, markdown_path)

    print("Tolerance sensitivity complete")
    for summary in summaries:
        print(
            f"  {summary['tolerance_percent']:.0f}%: "
            f"{summary['n_with_ci90_contained']}/{summary['n_datasets']} "
            "primary datasets have fully contained 90% CIs"
        )
    print(f"  CSV      : {csv_path}")
    print(f"  JSON     : {json_path}")
    print(f"  Markdown : {markdown_path}")
    return csv_path, json_path, markdown_path


def self_test() -> None:
    assert equivalence_bounds(0.02) == (1.0 / 1.02, 1.02)
    assert is_contained(0.9904, 1.019, 0.02)
    assert not is_contained(0.9800, 1.019, 0.02)
    sample = {
        "dataset": "validation systems",
        "analysis_role": "selection-stage sensitivity",
        "comparison_id": "p1_vs_p5_lambda095",
        "comparison": "p=1 versus p=5 at lambda=0.95",
        "p_A": 1,
        "p_B": 5,
        "ratio_A_over_B": 0.995,
        "ci90_ratio_lo": 0.990,
        "ci90_ratio_hi": 1.001,
        "n_systems": 102,
        "source_scope": "unit test",
    }
    classified = classify([sample], (0.02, 0.03))
    assert len(classified) == 2
    assert all(row["ci90_fully_contained"] for row in classified)
    print("SELF-TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate practical-similarity margins"
    )
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument(
        "--new-trajectories", type=Path, default=DEFAULT_NEW_TRAJECTORIES
    )
    parser.add_argument("--new-systems", type=Path, default=DEFAULT_NEW_SYSTEMS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="Relative tolerances as decimals (default: 0.02 0.03).",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
