from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

P_VALUES = [1, 2, 5, 8, 12]
LAMBDA_SELECTED = 0.95
LAMBDA_BASELINE = 1.0

FOCUSED_P_PAIRS = [
    (1, 5),
    (1, 2),
    (5, 8),
    (5, 12),
]

DEFAULT_BOOTSTRAP_B = 10000
DEFAULT_EQUIVALENCE_DELTA = 0.05
DEFAULT_SEED = 20260825
EPS = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )


def find_interaction_results(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()

        if not path.exists():
            raise SystemExit(f"Interaction results JSON not found: {path}")

        return path

    sibling_names = [
        "02_p_lambda_factorial",
        "Test- p-lambda interaction",
        "Test-p- lambda interaction",
        "Test-p-lambda interaction",
        "Test- p- lambda interaction",
    ]

    candidates: list[Path] = []

    for name in sibling_names:
        root = SCRIPT_DIR.parent / name

        candidates.extend(
            [
                root
                / "p_lambda_interaction_results"
                / "p_lambda_interaction_results.json",
                root / "p_lambda_interaction_results.json",
            ]
        )

    candidates.extend(
        [
            SCRIPT_DIR
            / "p_lambda_interaction_results"
            / "p_lambda_interaction_results.json",
            SCRIPT_DIR / "p_lambda_interaction_results.json",
        ]
    )

    for path in candidates:
        if path.exists():
            return path.resolve()

    discovered: list[Path] = []

    try:
        sibling_dirs = [item for item in SCRIPT_DIR.parent.iterdir() if item.is_dir()]
    except OSError:
        sibling_dirs = []

    for folder in sibling_dirs:
        for candidate in [
            folder
            / "p_lambda_interaction_results"
            / "p_lambda_interaction_results.json",
            folder / "p_lambda_interaction_results.json",
        ]:
            if candidate.exists():
                discovered.append(candidate.resolve())

    unique_discovered: list[Path] = []

    for path in discovered:
        if path not in unique_discovered:
            unique_discovered.append(path)

    if len(unique_discovered) == 1:
        return unique_discovered[0]

    if len(unique_discovered) > 1:
        raise SystemExit(
            "Multiple p_lambda_interaction_results.json files were found.\n"
            "Please specify the intended one explicitly with:\n"
            "  --interaction-results '/full/path/to/p_lambda_interaction_results.json'\n"
            "Found:\n  " + "\n  ".join(str(path) for path in unique_discovered)
        )

    raise SystemExit(
        "Could not find p_lambda_interaction_results.json automatically.\n"
        "Recommended sibling layout:\n"
        "  parent/\n"
        "  ├── 02_p_lambda_factorial/\n"
        "  │   └── p_lambda_interaction_results/\n"
        "  │       └── p_lambda_interaction_results.json\n"
        "  └── Test- p-lambda statistical closeout/\n"
        "      └── run_p_lambda_statistical_closeout_AUTO.py\n"
        "\nOr pass:\n"
        "  --interaction-results '/full/path/to/p_lambda_interaction_results.json'"
    )


def float_key(value: float) -> str:
    return str(float(value))


def cell_key(p: int, lambda_value: float) -> str:
    return f"p={int(p)}|lambda={float_key(lambda_value)}"


def validate_results_structure(results: dict[str, Any]) -> None:
    if "meta" not in results or "cells" not in results:
        raise SystemExit("Invalid interaction JSON: missing meta/cells.")

    meta = results["meta"]

    actual_p = [int(x) for x in meta.get("p_values", [])]
    actual_lambda = [float(x) for x in meta.get("lambda_values", [])]

    for p in P_VALUES:
        if p not in actual_p:
            raise SystemExit(f"Interaction JSON is missing p={p}.")

    for lam in [LAMBDA_SELECTED, LAMBDA_BASELINE]:
        if not any(
            math.isclose(
                x,
                lam,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for x in actual_lambda
        ):
            raise SystemExit(f"Interaction JSON is missing lambda={lam}.")

    if meta.get("selection_split") != "validation":
        raise SystemExit(
            "Expected selection_split='validation'. "
            "Refusing to reinterpret a different protocol."
        )

    for p in P_VALUES:
        for lam in [LAMBDA_SELECTED, LAMBDA_BASELINE]:
            key = cell_key(p, lam)

            if key not in results["cells"]:
                raise SystemExit(f"Interaction JSON is missing cell {key}.")

            record = results["cells"][key]

            for split in ["val", "test"]:
                if split not in record:
                    raise SystemExit(f"{key} is missing split '{split}'.")

                if "system_risks" not in record[split]:
                    raise SystemExit(f"{key}/{split} has no system_risks.")

    for split in ["val", "test"]:
        lengths = set()

        for p in P_VALUES:
            for lam in [LAMBDA_SELECTED, LAMBDA_BASELINE]:
                vector = np.asarray(
                    results["cells"][cell_key(p, lam)][split]["system_risks"],
                    dtype=float,
                )

                lengths.add(int(vector.size))

        if len(lengths) != 1:
            raise SystemExit(
                f"Paired analysis is invalid: {split} system-risk "
                f"vectors have different lengths: {sorted(lengths)}"
            )


def safe_risk_vector(
    results: dict[str, Any],
    *,
    p: int,
    lambda_value: float,
    split: str,
) -> np.ndarray:
    vector = np.asarray(
        results["cells"][cell_key(p, lambda_value)][split]["system_risks"],
        dtype=float,
    )

    if vector.ndim != 1:
        raise ValueError(f"Risk vector must be 1D; got shape {vector.shape}.")

    if vector.size == 0:
        raise ValueError("Risk vector is empty.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Risk vector for p={p}, lambda={lambda_value}, "
            f"split={split} contains NaN/Inf."
        )

    if np.any(vector < 0.0):
        raise ValueError("Risk values must be nonnegative.")

    return vector


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    B: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("bootstrap_median_ci requires a nonempty 1D vector.")

    if not np.all(np.isfinite(x)):
        raise ValueError("bootstrap_median_ci received NaN/Inf.")

    if B < 500:
        raise ValueError("Bootstrap B must be at least 500.")

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must lie in (0,1).")

    rng = np.random.default_rng(seed)
    n = x.size

    chunk_size = min(1000, B)
    bootstrap_parts: list[np.ndarray] = []

    completed = 0

    while completed < B:
        current = min(
            chunk_size,
            B - completed,
        )

        indices = rng.integers(
            0,
            n,
            size=(current, n),
        )

        estimates = np.median(
            x[indices],
            axis=1,
        )

        bootstrap_parts.append(estimates)

        completed += current

    distribution = np.concatenate(bootstrap_parts)

    alpha_tail = (1.0 - ci_level) / 2.0

    point = float(np.median(x))

    lo = float(
        np.quantile(
            distribution,
            alpha_tail,
        )
    )

    hi = float(
        np.quantile(
            distribution,
            1.0 - alpha_tail,
        )
    )

    return point, lo, hi


def paired_log_risk_ratio(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
) -> np.ndarray:
    if risk_a.size != risk_b.size:
        raise ValueError("Paired vectors have unequal length.")

    return np.log((risk_a + EPS) / (risk_b + EPS))


def compare_p_pair(
    results: dict[str, Any],
    *,
    p_a: int,
    p_b: int,
    split: str,
    B: int,
    delta: float,
    seed: int,
) -> dict[str, Any]:
    risk_a = safe_risk_vector(
        results,
        p=p_a,
        lambda_value=LAMBDA_SELECTED,
        split=split,
    )

    risk_b = safe_risk_vector(
        results,
        p=p_b,
        lambda_value=LAMBDA_SELECTED,
        split=split,
    )

    d = paired_log_risk_ratio(
        risk_a,
        risk_b,
    )

    point95, ci95_lo, ci95_hi = bootstrap_median_ci(
        d,
        B=B,
        ci_level=0.95,
        seed=seed,
    )

    _, ci90_lo, ci90_hi = bootstrap_median_ci(
        d,
        B=B,
        ci_level=0.90,
        seed=seed + 1,
    )

    margin = float(math.log1p(delta))

    equivalent = bool(ci90_lo > -margin and ci90_hi < margin)

    if ci95_hi < 0.0:
        superiority = f"p={p_a} lower paired risk"
    elif ci95_lo > 0.0:
        superiority = f"p={p_b} lower paired risk"
    else:
        superiority = "no clear superiority at 95% CI"

    return {
        "split": split,
        "lambda_fixed": LAMBDA_SELECTED,
        "p_A": int(p_a),
        "p_B": int(p_b),
        "n_systems": int(risk_a.size),
        "statistic": "median paired log risk ratio log(R_A/R_B)",
        "point_log_ratio": float(point95),
        "ratio_A_over_B": float(math.exp(point95)),
        "ci95_log_lo": float(ci95_lo),
        "ci95_log_hi": float(ci95_hi),
        "ci95_ratio_lo": float(math.exp(ci95_lo)),
        "ci95_ratio_hi": float(math.exp(ci95_hi)),
        "ci90_log_lo": float(ci90_lo),
        "ci90_log_hi": float(ci90_hi),
        "ci90_ratio_lo": float(math.exp(ci90_lo)),
        "ci90_ratio_hi": float(math.exp(ci90_hi)),
        "equivalence_delta": float(delta),
        "equivalence_ratio_bounds": [
            float(1.0 / (1.0 + delta)),
            float(1.0 + delta),
        ],
        "equivalent_90pct_CI": equivalent,
        "superiority_95pct_CI": superiority,
        "median_R_A": float(np.median(risk_a)),
        "median_R_B": float(np.median(risk_b)),
        "mean_R_A": float(np.mean(risk_a)),
        "mean_R_B": float(np.mean(risk_b)),
    }


def run_step1_pairwise(
    results: dict[str, Any],
    *,
    split: str,
    B: int,
    delta: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counter = 0

    for i, p_a in enumerate(P_VALUES):
        for p_b in P_VALUES[i + 1 :]:
            counter += 1

            row = compare_p_pair(
                results,
                p_a=p_a,
                p_b=p_b,
                split=split,
                B=B,
                delta=delta,
                seed=seed + 1000 * counter,
            )

            row["focused_pair"] = (p_a, p_b) in FOCUSED_P_PAIRS or (
                p_b,
                p_a,
            ) in FOCUSED_P_PAIRS

            rows.append(row)

    return rows


def step1_summary(
    results: dict[str, Any],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    median_by_p = {}

    for p in P_VALUES:
        risk = safe_risk_vector(
            results,
            p=p,
            lambda_value=LAMBDA_SELECTED,
            split="val",
        )

        median_by_p[str(p)] = float(np.median(risk))

    numerical_best_p = min(
        P_VALUES,
        key=lambda p: median_by_p[str(p)],
    )

    equivalent_to_best = [int(numerical_best_p)]

    for p in P_VALUES:
        if p == numerical_best_p:
            continue

        matching = [
            row
            for row in validation_rows
            if {
                int(row["p_A"]),
                int(row["p_B"]),
            }
            == {
                int(numerical_best_p),
                int(p),
            }
        ]

        if len(matching) == 1 and matching[0]["equivalent_90pct_CI"]:
            equivalent_to_best.append(int(p))

    equivalent_to_best = sorted(equivalent_to_best)

    return {
        "selection_split": "validation",
        "lambda_fixed": LAMBDA_SELECTED,
        "validation_median_R_by_p": median_by_p,
        "numerical_best_p_by_validation_median_R": int(numerical_best_p),
        "p_values_practically_equivalent_to_numerical_best": (equivalent_to_best),
        "equivalence_delta": None,
        "interpretation_rule": (
            "Use validation for p conclusions; test is confirmation only. "
            "Statistical superiority and practical equivalence are reported "
            "separately because both can be true for a very small effect."
        ),
    }


def forgetting_gain_vector(
    results: dict[str, Any],
    *,
    p: int,
    split: str,
) -> np.ndarray:
    selected = safe_risk_vector(
        results,
        p=p,
        lambda_value=LAMBDA_SELECTED,
        split=split,
    )

    baseline = safe_risk_vector(
        results,
        p=p,
        lambda_value=LAMBDA_BASELINE,
        split=split,
    )

    if selected.size != baseline.size:
        raise ValueError("Selected/baseline risk vectors have unequal length.")

    return np.log((selected + EPS) / (baseline + EPS))


def summarize_forgetting_gain(
    results: dict[str, Any],
    *,
    split: str,
    B: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []

    for index, p in enumerate(P_VALUES):
        gain = forgetting_gain_vector(
            results,
            p=p,
            split=split,
        )

        point, lo, hi = bootstrap_median_ci(
            gain,
            B=B,
            ci_level=0.95,
            seed=seed + 5000 + 100 * index,
        )

        ratio = float(math.exp(point))

        rows.append(
            {
                "split": split,
                "p": int(p),
                "n_systems": int(gain.size),
                "lambda_selected": LAMBDA_SELECTED,
                "lambda_baseline": LAMBDA_BASELINE,
                "statistic": ("median G_p = median log(R_lambda095/R_lambda1)"),
                "median_G_log_ratio": float(point),
                "risk_ratio_lambda095_over_lambda1": ratio,
                "ci95_log_lo": float(lo),
                "ci95_log_hi": float(hi),
                "ci95_ratio_lo": float(math.exp(lo)),
                "ci95_ratio_hi": float(math.exp(hi)),
                "relative_risk_reduction_point": float(1.0 - ratio),
                "forgetting_benefit_clear_95pct_CI": bool(hi < 0.0),
            }
        )

    return rows


def compare_interaction_pair(
    results: dict[str, Any],
    *,
    p_a: int,
    p_b: int,
    split: str,
    B: int,
    delta: float,
    seed: int,
) -> dict[str, Any]:
    gain_a = forgetting_gain_vector(
        results,
        p=p_a,
        split=split,
    )

    gain_b = forgetting_gain_vector(
        results,
        p=p_b,
        split=split,
    )

    if gain_a.size != gain_b.size:
        raise ValueError("Interaction contrast vectors have unequal length.")

    contrast = gain_a - gain_b

    point, ci95_lo, ci95_hi = bootstrap_median_ci(
        contrast,
        B=B,
        ci_level=0.95,
        seed=seed,
    )

    _, ci90_lo, ci90_hi = bootstrap_median_ci(
        contrast,
        B=B,
        ci_level=0.90,
        seed=seed + 1,
    )

    margin = float(math.log1p(delta))

    equivalent = bool(ci90_lo > -margin and ci90_hi < margin)

    if ci95_hi < 0.0:
        conclusion = f"forgetting gain is stronger at p={p_a}"
    elif ci95_lo > 0.0:
        conclusion = f"forgetting gain is stronger at p={p_b}"
    else:
        conclusion = "no clear gain-magnitude difference at 95% CI"

    return {
        "split": split,
        "p_A": int(p_a),
        "p_B": int(p_b),
        "n_systems": int(gain_a.size),
        "lambda_selected": LAMBDA_SELECTED,
        "lambda_baseline": LAMBDA_BASELINE,
        "contrast_definition": "D = G_pA - G_pB",
        "point_log_ratio_of_risk_ratios": float(point),
        "ratio_of_risk_ratios_A_over_B": float(math.exp(point)),
        "ci95_log_lo": float(ci95_lo),
        "ci95_log_hi": float(ci95_hi),
        "ci95_ratio_lo": float(math.exp(ci95_lo)),
        "ci95_ratio_hi": float(math.exp(ci95_hi)),
        "ci90_log_lo": float(ci90_lo),
        "ci90_log_hi": float(ci90_hi),
        "ci90_ratio_lo": float(math.exp(ci90_lo)),
        "ci90_ratio_hi": float(math.exp(ci90_hi)),
        "equivalence_delta": float(delta),
        "equivalence_ratio_bounds": [
            float(1.0 / (1.0 + delta)),
            float(1.0 + delta),
        ],
        "gain_magnitude_equivalent_90pct_CI": equivalent,
        "interaction_conclusion_95pct_CI": conclusion,
    }


def run_step2_interaction_pairs(
    results: dict[str, Any],
    *,
    split: str,
    B: int,
    delta: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    counter = 0

    for i, p_a in enumerate(P_VALUES):
        for p_b in P_VALUES[i + 1 :]:
            counter += 1

            rows.append(
                compare_interaction_pair(
                    results,
                    p_a=p_a,
                    p_b=p_b,
                    split=split,
                    B=B,
                    delta=delta,
                    seed=seed + 20000 + 1000 * counter,
                )
            )

    return rows


def step2_summary(
    gain_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_p_benefit = all(row["forgetting_benefit_clear_95pct_CI"] for row in gain_rows)

    no_clear_pairwise_interaction = all(
        row["interaction_conclusion_95pct_CI"]
        == "no clear gain-magnitude difference at 95% CI"
        for row in contrast_rows
    )

    all_gain_magnitudes_equivalent = all(
        row["gain_magnitude_equivalent_90pct_CI"] for row in contrast_rows
    )

    clear_interaction_pairs = [
        {
            "p_A": int(row["p_A"]),
            "p_B": int(row["p_B"]),
            "conclusion": row["interaction_conclusion_95pct_CI"],
            "ratio_of_risk_ratios_A_over_B": float(
                row["ratio_of_risk_ratios_A_over_B"]
            ),
            "ci95_ratio": [
                float(row["ci95_ratio_lo"]),
                float(row["ci95_ratio_hi"]),
            ],
        }
        for row in contrast_rows
        if row["interaction_conclusion_95pct_CI"]
        != "no clear gain-magnitude difference at 95% CI"
    ]

    return {
        "selection_split": "validation",
        "all_p_show_clear_forgetting_benefit_at_95pct_CI": bool(all_p_benefit),
        "all_pairwise_gain_contrasts_show_no_clear_difference": bool(
            no_clear_pairwise_interaction
        ),
        "all_pairwise_gain_magnitudes_practically_equivalent": bool(
            all_gain_magnitudes_equivalent
        ),
        "clear_interaction_pairs": clear_interaction_pairs,
        "interpretation_note": (
            "The preferred lambda can remain constant across p while the "
            "magnitude of the forgetting benefit still differs across p. "
            "Therefore this contrast is a stricter interaction diagnostic "
            "than simply plotting best lambda versus p."
        ),
    }


def write_csvs(
    *,
    outdir: Path,
    step1_validation: list[dict[str, Any]],
    step1_test: list[dict[str, Any]],
    step2_gain_validation: list[dict[str, Any]],
    step2_contrast_validation: list[dict[str, Any]],
    step2_gain_test: list[dict[str, Any]],
    step2_contrast_test: list[dict[str, Any]],
) -> None:
    import pandas as pd

    step1_validation_df = pd.DataFrame(step1_validation)

    step1_validation_df.to_csv(
        outdir / "step1_p_pairwise_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    focused_df = step1_validation_df[step1_validation_df["focused_pair"]].copy()

    focused_df.to_csv(
        outdir / "step1_p_focused_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(step1_test).to_csv(
        outdir / "step1_p_pairwise_test_confirmation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(step2_gain_validation).to_csv(
        outdir / "step2_forgetting_gain_by_p_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(step2_contrast_validation).to_csv(
        outdir / "step2_interaction_contrasts_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(step2_gain_test).to_csv(
        outdir / "step2_forgetting_gain_by_p_test_confirmation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(step2_contrast_test).to_csv(
        outdir / "step2_interaction_contrasts_test_confirmation.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_outputs(
    *,
    results: dict[str, Any],
    outdir: Path,
    step1_validation: list[dict[str, Any]],
    step2_gain_validation: list[dict[str, Any]],
    step2_contrast_validation: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    x = P_VALUES

    y = [
        float(
            np.median(
                safe_risk_vector(
                    results,
                    p=p,
                    lambda_value=LAMBDA_SELECTED,
                    split="val",
                )
            )
        )
        for p in P_VALUES
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.plot(
        x,
        y,
    )

    ax.set_xlabel("Innovation window p")

    ax.set_ylabel("Validation median risk")

    ax.set_title("p profile at fixed lambda=0.95")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "01_p_risk_at_lambda095_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    n_p = len(P_VALUES)

    matrix = np.full(
        (n_p, n_p),
        np.nan,
    )

    for i in range(n_p):
        matrix[i, i] = 1.0

    for row in step1_validation:
        p_a = int(row["p_A"])
        p_b = int(row["p_B"])

        i = P_VALUES.index(p_a)
        j = P_VALUES.index(p_b)

        value = 1.0 if row["equivalent_90pct_CI"] else 0.0

        matrix[i, j] = value
        matrix[j, i] = value

    fig, ax = plt.subplots(figsize=(6.6, 5.6))

    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        range(n_p),
        labels=[str(p) for p in P_VALUES],
    )

    ax.set_yticks(
        range(n_p),
        labels=[str(p) for p in P_VALUES],
    )

    ax.set_xlabel("p")

    ax.set_ylabel("p")

    ax.set_title("Practical equivalence of p at lambda=0.95")

    for i in range(n_p):
        for j in range(n_p):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    ("EQ" if matrix[i, j] == 1.0 else "NO"),
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        image,
        ax=ax,
        label="1 = equivalent",
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "02_p_equivalence_matrix_lambda095.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    gain_by_p = {int(row["p"]): row for row in step2_gain_validation}

    ratios = [
        float(gain_by_p[p]["risk_ratio_lambda095_over_lambda1"]) for p in P_VALUES
    ]

    ci_lo = [float(gain_by_p[p]["ci95_ratio_lo"]) for p in P_VALUES]

    ci_hi = [float(gain_by_p[p]["ci95_ratio_hi"]) for p in P_VALUES]

    lower_err = np.asarray(ratios) - np.asarray(ci_lo)

    upper_err = np.asarray(ci_hi) - np.asarray(ratios)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.plot(
        P_VALUES,
        ratios,
    )

    ax.errorbar(
        P_VALUES,
        ratios,
        yerr=np.vstack(
            [
                lower_err,
                upper_err,
            ]
        ),
    )

    ax.axhline(1.0)

    ax.set_xlabel("Innovation window p")

    ax.set_ylabel("Paired risk ratio: lambda=.95 / lambda=1")

    ax.set_title("Forgetting gain magnitude by p (validation)")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "03_forgetting_gain_by_p_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    contrast_matrix = np.ones(
        (n_p, n_p),
        dtype=float,
    )

    for row in step2_contrast_validation:
        p_a = int(row["p_A"])
        p_b = int(row["p_B"])

        i = P_VALUES.index(p_a)
        j = P_VALUES.index(p_b)

        ratio = float(row["ratio_of_risk_ratios_A_over_B"])

        contrast_matrix[
            i,
            j,
        ] = ratio

        contrast_matrix[
            j,
            i,
        ] = 1.0 / ratio if ratio > 0.0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.8, 5.8))

    image = ax.imshow(
        contrast_matrix,
    )

    ax.set_xticks(
        range(n_p),
        labels=[str(p) for p in P_VALUES],
    )

    ax.set_yticks(
        range(n_p),
        labels=[str(p) for p in P_VALUES],
    )

    ax.set_xlabel("p_B")

    ax.set_ylabel("p_A")

    ax.set_title("Interaction contrast: ratio of forgetting risk ratios")

    for i in range(n_p):
        for j in range(n_p):
            value = contrast_matrix[
                i,
                j,
            ]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        image,
        ax=ax,
        label="exp(G_pA - G_pB); 1 = equal gain",
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "04_interaction_contrast_matrix_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_fake_results() -> dict[str, Any]:
    rng = np.random.default_rng(12345)

    cells = {}

    n_val = 40
    n_test = 50

    for p in P_VALUES:
        for lam in [
            LAMBDA_SELECTED,
            LAMBDA_BASELINE,
        ]:
            base_val = (
                0.10
                if math.isclose(
                    lam,
                    LAMBDA_BASELINE,
                )
                else 0.01
            )

            base_test = (
                0.12
                if math.isclose(
                    lam,
                    LAMBDA_BASELINE,
                )
                else 0.02
            )

            p_factor = 1.0 + 0.001 * p

            val = np.maximum(
                1e-6,
                base_val
                * p_factor
                * np.exp(
                    rng.normal(
                        0.0,
                        0.1,
                        size=n_val,
                    )
                ),
            )

            test = np.maximum(
                1e-6,
                base_test
                * p_factor
                * np.exp(
                    rng.normal(
                        0.0,
                        0.1,
                        size=n_test,
                    )
                ),
            )

            cells[
                cell_key(
                    p,
                    lam,
                )
            ] = {
                "p": p,
                "lambda": lam,
                "val": {
                    "system_risks": val.tolist(),
                },
                "test": {
                    "system_risks": test.tolist(),
                },
            }

    return {
        "meta": {
            "p_values": P_VALUES,
            "lambda_values": [
                1.0,
                0.99,
                0.95,
                0.925,
                0.9,
                0.85,
            ],
            "selection_split": "validation",
            "test_usage": "confirmation only",
        },
        "cells": cells,
    }


def self_test() -> None:
    fake = make_fake_results()

    validate_results_structure(fake)

    x = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=float,
    )

    d = paired_log_risk_ratio(
        x,
        x,
    )

    assert np.allclose(
        d,
        0.0,
    )

    point, lo, hi = bootstrap_median_ci(
        d,
        B=500,
        ci_level=0.90,
        seed=10,
    )

    assert abs(point) < 1e-12 and abs(lo) < 1e-12 and abs(hi) < 1e-12

    step1 = run_step1_pairwise(
        fake,
        split="val",
        B=500,
        delta=0.05,
        seed=20,
    )

    assert len(step1) == 10

    gains = summarize_forgetting_gain(
        fake,
        split="val",
        B=500,
        seed=30,
    )

    contrasts = run_step2_interaction_pairs(
        fake,
        split="val",
        B=500,
        delta=0.05,
        seed=40,
    )

    assert len(gains) == 5
    assert len(contrasts) == 10

    assert all(row["risk_ratio_lambda095_over_lambda1"] < 1.0 for row in gains)

    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired factorial comparisons"
    )

    parser.add_argument(
        "--interaction-results",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=(SCRIPT_DIR / "p_lambda_statistical_closeout_results"),
    )

    parser.add_argument(
        "--bootstrap-B",
        type=int,
        default=DEFAULT_BOOTSTRAP_B,
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_EQUIVALENCE_DELTA,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstrap_B < 500:
        parser.error("--bootstrap-B must be >= 500.")

    if not (0.0 < args.delta < 0.5):
        parser.error("--delta must lie in (0, 0.5).")

    interaction_path = find_interaction_results(args.interaction_results)

    results = load_json(interaction_path)

    validate_results_structure(results)

    outdir = args.outdir.expanduser().resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    step1_validation = run_step1_pairwise(
        results,
        split="val",
        B=args.bootstrap_B,
        delta=args.delta,
        seed=args.seed,
    )

    step1_test = run_step1_pairwise(
        results,
        split="test",
        B=args.bootstrap_B,
        delta=args.delta,
        seed=args.seed + 100000,
    )

    step1_info = step1_summary(
        results,
        step1_validation,
    )

    step1_info["equivalence_delta"] = float(args.delta)

    step2_gain_validation = summarize_forgetting_gain(
        results,
        split="val",
        B=args.bootstrap_B,
        seed=args.seed,
    )

    step2_contrast_validation = run_step2_interaction_pairs(
        results,
        split="val",
        B=args.bootstrap_B,
        delta=args.delta,
        seed=args.seed,
    )

    step2_gain_test = summarize_forgetting_gain(
        results,
        split="test",
        B=args.bootstrap_B,
        seed=args.seed + 200000,
    )

    step2_contrast_test = run_step2_interaction_pairs(
        results,
        split="test",
        B=args.bootstrap_B,
        delta=args.delta,
        seed=args.seed + 300000,
    )

    step2_info = step2_summary(
        step2_gain_validation,
        step2_contrast_validation,
    )

    write_csvs(
        outdir=outdir,
        step1_validation=step1_validation,
        step1_test=step1_test,
        step2_gain_validation=step2_gain_validation,
        step2_contrast_validation=step2_contrast_validation,
        step2_gain_test=step2_gain_test,
        step2_contrast_test=step2_contrast_test,
    )

    plot_outputs(
        results=results,
        outdir=outdir,
        step1_validation=step1_validation,
        step2_gain_validation=step2_gain_validation,
        step2_contrast_validation=step2_contrast_validation,
    )

    final = {
        "meta": {
            "interaction_results": str(interaction_path),
            "p_values": P_VALUES,
            "lambda_selected": LAMBDA_SELECTED,
            "lambda_baseline": LAMBDA_BASELINE,
            "bootstrap_B": int(args.bootstrap_B),
            "equivalence_delta": float(args.delta),
            "selection_split": "validation",
            "test_usage": "confirmation only",
        },
        "step1_p_closeout": {
            "summary": step1_info,
            "validation_pairwise": step1_validation,
            "test_confirmation_pairwise": step1_test,
        },
        "step2_interaction_closeout": {
            "summary": step2_info,
            "validation_gain_by_p": step2_gain_validation,
            "validation_pairwise_contrasts": step2_contrast_validation,
            "test_confirmation_gain_by_p": step2_gain_test,
            "test_confirmation_pairwise_contrasts": step2_contrast_test,
        },
    }

    final_path = outdir / "p_lambda_statistical_closeout_results.json"

    json_dump(
        final,
        final_path,
    )

    print("=" * 78)

    print("p × lambda statistical close-out")

    print(f"interaction JSON : {interaction_path}")

    print(f"output           : {outdir}")

    print(f"bootstrap B      : {args.bootstrap_B}")

    print(f"equivalence delta: {args.delta:.1%}")

    print("=" * 78)

    print("\nSTEP 1 — p at lambda=0.95")

    print("validation median risk by p:")

    for p in P_VALUES:
        value = step1_info["validation_median_R_by_p"][str(p)]

        print(f"  p={p:<2d}  median_R={value:.8f}")

    print(f"numerical best p: {step1_info['numerical_best_p_by_validation_median_R']}")

    print(
        "practically equivalent to numerical best: "
        f"{step1_info['p_values_practically_equivalent_to_numerical_best']}"
    )

    print("\nFocused comparisons:")

    for row in step1_validation:
        if row["focused_pair"]:
            print(
                f"  p={row['p_A']} vs p={row['p_B']} | "
                f"ratio={row['ratio_A_over_B']:.6f} | "
                f"95%CI=[{row['ci95_ratio_lo']:.6f}, "
                f"{row['ci95_ratio_hi']:.6f}] | "
                f"equivalent={row['equivalent_90pct_CI']} | "
                f"{row['superiority_95pct_CI']}"
            )

    print("\nSTEP 2 — forgetting gain interaction")

    for row in step2_gain_validation:
        print(
            f"  p={row['p']:<2d} | "
            f"R(.95)/R(1)={row['risk_ratio_lambda095_over_lambda1']:.6f} | "
            f"95%CI=[{row['ci95_ratio_lo']:.6f}, "
            f"{row['ci95_ratio_hi']:.6f}] | "
            f"relative reduction={row['relative_risk_reduction_point']:.1%}"
        )

    print("\nClear pairwise gain-magnitude differences:")

    clear_pairs = step2_info["clear_interaction_pairs"]

    if clear_pairs:
        for item in clear_pairs:
            print(
                f"  p={item['p_A']} vs p={item['p_B']} | "
                f"{item['conclusion']} | "
                f"ratio-of-ratios="
                f"{item['ratio_of_risk_ratios_A_over_B']:.6f}"
            )
    else:
        print("  none at 95% CI")

    print("\nFinal JSON:")

    print(f"  {final_path}")

    print("=" * 78)


if __name__ == "__main__":
    main()
