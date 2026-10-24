from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

STATE_MODE = "observer"

DEFAULT_TRAJ_START = 100
DEFAULT_N_TRAJ = 5

BOOTSTRAP_B = 10000
EQUIVALENCE_DELTA = 0.05
BOOTSTRAP_SEED = 20260825

MODEL_SPECS = {
    "A_p1_lambda1": {
        "label": "p=1, lambda=1",
        "p": 1,
        "lambda": 1.0,
        "role": "single-innovation no-forgetting baseline",
    },
    "B_p1_lambda095": {
        "label": "p=1, lambda=0.95",
        "p": 1,
        "lambda": 0.95,
        "role": "preselected parsimonious final candidate",
    },
    "C_p5_lambda1": {
        "label": "p=5, lambda=1",
        "p": 5,
        "lambda": 1.0,
        "role": "representative multi-innovation no-forgetting baseline",
    },
    "D_p5_lambda095": {
        "label": "p=5, lambda=0.95",
        "p": 5,
        "lambda": 0.95,
        "role": "representative multi-innovation forgetting model",
    },
}


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


def float_key(value: float) -> str:
    return str(float(value))


def interaction_cell_key(p: int, lambda_value: float) -> str:
    return f"p={int(p)}|lambda={float_key(lambda_value)}"


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
        siblings = [item for item in SCRIPT_DIR.parent.iterdir() if item.is_dir()]
    except OSError:
        siblings = []

    for folder in siblings:
        for candidate in [
            folder
            / "p_lambda_interaction_results"
            / "p_lambda_interaction_results.json",
            folder / "p_lambda_interaction_results.json",
        ]:
            if candidate.exists():
                discovered.append(candidate.resolve())

    unique = []

    for path in discovered:
        if path not in unique:
            unique.append(path)

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        raise SystemExit(
            "Multiple interaction result JSON files were found.\n"
            "Specify the intended one with --interaction-results.\n"
            + "\n".join(str(path) for path in unique)
        )

    raise SystemExit(
        "Could not find p_lambda_interaction_results.json automatically.\n"
        "Use --interaction-results '/full/path/to/p_lambda_interaction_results.json'."
    )


def find_project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()

        if (root / "src" / "core").is_dir():
            return root

        raise SystemExit(
            f"--project is not a valid state-estimation project root: {root}"
        )

    candidates: list[Path] = []

    for base in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidates.append(base)
        candidates.append(base / "state-estimation")

    seen: set[Path] = set()

    for root in candidates:
        try:
            root = root.resolve()
        except OSError:
            continue

        if root in seen:
            continue

        seen.add(root)

        if (
            (root / "src" / "core" / "config.py").exists()
            and (root / "src" / "core" / "systems.py").exists()
            and (root / "src" / "core" / "estimators.py").exists()
            and (root / "src" / "core" / "metrics.py").exists()
        ):
            return root

    raise SystemExit(
        "Could not auto-discover the state-estimation project.\n"
        "Pass --project '/full/path/to/state-estimation'."
    )


def import_project(project: Path):
    src = project / "src" / "core"

    sys.path.insert(
        0,
        str(src),
    )

    from config import Config, traj_seed
    from estimators import misg_run
    from metrics import risk_R
    from systems import load_family, simulate

    return (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        load_family,
        simulate,
    )


def validate_interaction_results(results: dict[str, Any]) -> None:
    if "meta" not in results or "cells" not in results:
        raise SystemExit("Invalid interaction JSON: missing meta/cells.")

    meta = results["meta"]

    if meta.get("selection_split") != "validation":
        raise SystemExit("Interaction protocol is not validation-selected.")

    for spec in MODEL_SPECS.values():
        key = interaction_cell_key(
            spec["p"],
            spec["lambda"],
        )

        if key not in results["cells"]:
            raise SystemExit(f"Required frozen cell missing: {key}")

        record = results["cells"][key]

        if "gamma" not in record:
            raise SystemExit(f"Required gamma* missing for {key}")

        if bool(record.get("boundary", False)):
            raise SystemExit(
                f"Frozen gamma* is on search boundary for {key}. "
                "Do not proceed to final holdout."
            )


def extract_frozen_models(
    results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    frozen = {}

    for model_id, spec in MODEL_SPECS.items():
        key = interaction_cell_key(
            int(spec["p"]),
            float(spec["lambda"]),
        )

        record = results["cells"][key]

        frozen[model_id] = {
            **spec,
            "gamma": float(record["gamma"]),
            "selection_val_median_R": float(record["val"]["median_R"]),
            "selection_test_median_R_confirmation": float(record["test"]["median_R"]),
        }

    return frozen


def build_pairs(
    systems,
    *,
    j_start: int,
    n_traj: int,
    cfg,
    traj_seed,
    simulate,
):
    pairs = []

    for system_index, system_item in enumerate(systems):
        for j in range(j_start, j_start + n_traj):
            rng = np.random.default_rng(
                traj_seed(
                    cfg,
                    system_item["m"],
                    j,
                )
            )

            trajectory = simulate(
                system_item,
                rng,
                cfg.N,
            )

            pairs.append(
                (
                    system_index,
                    j,
                    system_item,
                    trajectory,
                )
            )

    return pairs


def capped_risk(
    result,
    system_item,
    cfg,
    risk_R,
) -> float:
    value = risk_R(
        result["rho_hat_traj"],
        system_item["rho_star"],
        cfg.N,
    )

    if not np.isfinite(value):
        return float(cfg.r_cap)

    return min(
        float(value),
        float(cfg.r_cap),
    )


def relative_error_series(
    rho_hat_traj: np.ndarray,
    rho_star: np.ndarray,
    N: int,
) -> np.ndarray:
    rho = np.asarray(
        rho_hat_traj,
        dtype=float,
    )

    truth = np.asarray(
        rho_star,
        dtype=float,
    )

    denominator = float(np.linalg.norm(truth))

    if denominator <= 0.0:
        return np.full(
            N,
            np.nan,
        )

    if rho.ndim != 2 or rho.shape[1] != truth.size:
        raise ValueError(f"Unexpected rho_hat_traj shape {rho.shape}.")

    if rho.shape[0] >= N + 1:
        sequence = rho[1 : N + 1]
    elif rho.shape[0] == N:
        sequence = rho
    else:
        raise ValueError(f"rho_hat_traj has {rho.shape[0]} rows for N={N}.")

    return (
        np.linalg.norm(
            sequence - truth[None, :],
            axis=1,
        )
        / denominator
    )


def first_hit_time(
    errors: np.ndarray,
    threshold: float,
) -> float:
    indices = np.flatnonzero(np.isfinite(errors) & (errors <= threshold))

    if indices.size == 0:
        return float("nan")

    return float(indices[0] + 1)


def evaluate_frozen_model(
    pairs,
    *,
    p: int,
    lambda_value: float,
    gamma: float,
    cfg,
    misg_run,
    risk_R,
) -> dict[str, Any]:
    weights = np.ones(
        int(p),
        dtype=float,
    )

    r_mode = (
        "cumulative"
        if math.isclose(
            lambda_value,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        else "forgetting"
    )

    risk_by_system: dict[int, list[float]] = {}
    hit30_by_system: dict[int, list[float]] = {}
    hit10_by_system: dict[int, list[float]] = {}

    trajectory_fail30 = 0
    trajectory_fail10 = 0
    divergences = 0
    n_trajectories = 0

    for (
        system_index,
        _trajectory_index,
        system_item,
        trajectory,
    ) in pairs:
        n_trajectories += 1

        result = misg_run(
            system_item,
            trajectory,
            weights,
            float(gamma),
            r_mode,
            STATE_MODE,
            cfg,
            lam_g=float(lambda_value),
        )

        diverged = result.get("status") == "diverged"

        if diverged:
            divergences += 1

        risk = capped_risk(
            result,
            system_item,
            cfg,
            risk_R,
        )

        risk_by_system.setdefault(
            system_index,
            [],
        ).append(risk)

        if diverged:
            hit30 = float("nan")
            hit10 = float("nan")
        else:
            errors = relative_error_series(
                result["rho_hat_traj"],
                system_item["rho_star"],
                cfg.N,
            )

            hit30 = first_hit_time(
                errors,
                0.30,
            )

            hit10 = first_hit_time(
                errors,
                0.10,
            )

        if not np.isfinite(hit30):
            trajectory_fail30 += 1

        if not np.isfinite(hit10):
            trajectory_fail10 += 1

        hit30_by_system.setdefault(
            system_index,
            [],
        ).append(hit30)

        hit10_by_system.setdefault(
            system_index,
            [],
        ).append(hit10)

    system_risks = np.asarray(
        [np.mean(values) for _, values in sorted(risk_by_system.items())],
        dtype=float,
    )

    system_hit30 = []
    system_hit10 = []

    for _, values in sorted(hit30_by_system.items()):
        arr = np.asarray(
            values,
            dtype=float,
        )

        finite = np.isfinite(arr)

        system_hit30.append(
            float(np.mean(arr[finite])) if np.any(finite) else float("nan")
        )

    for _, values in sorted(hit10_by_system.items()):
        arr = np.asarray(
            values,
            dtype=float,
        )

        finite = np.isfinite(arr)

        system_hit10.append(
            float(np.mean(arr[finite])) if np.any(finite) else float("nan")
        )

    system_hit30 = np.asarray(
        system_hit30,
        dtype=float,
    )

    system_hit10 = np.asarray(
        system_hit10,
        dtype=float,
    )

    finite30 = np.isfinite(system_hit30)

    finite10 = np.isfinite(system_hit10)

    return {
        "p": int(p),
        "lambda": float(lambda_value),
        "gamma": float(gamma),
        "median_R": float(np.median(system_risks)),
        "mean_R": float(np.mean(system_risks)),
        "system_risks": system_risks.tolist(),
        "n_systems": int(system_risks.size),
        "n_trajectories": int(n_trajectories),
        "divergences": int(divergences),
        "median_hit_30pct_success_only": (
            float(np.median(system_hit30[finite30]))
            if np.any(finite30)
            else float("nan")
        ),
        "median_hit_10pct_success_only": (
            float(np.median(system_hit10[finite10]))
            if np.any(finite10)
            else float("nan")
        ),
        "system_failure_rate_30pct": float(np.mean(~finite30)),
        "system_failure_rate_10pct": float(np.mean(~finite10)),
        "trajectory_failure_rate_30pct": float(trajectory_fail30 / n_trajectories),
        "trajectory_failure_rate_10pct": float(trajectory_fail10 / n_trajectories),
    }


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    B: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float]:
    x = np.asarray(
        values,
        dtype=float,
    )

    if x.ndim != 1 or x.size == 0:
        raise ValueError("bootstrap_median_ci requires a nonempty 1D vector.")

    if not np.all(np.isfinite(x)):
        raise ValueError("bootstrap input contains NaN/Inf.")

    rng = np.random.default_rng(seed)

    n = x.size

    chunk = min(
        1000,
        B,
    )

    parts = []

    done = 0

    while done < B:
        current = min(
            chunk,
            B - done,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                current,
                n,
            ),
        )

        parts.append(
            np.median(
                x[indices],
                axis=1,
            )
        )

        done += current

    distribution = np.concatenate(parts)

    tail = (1.0 - ci_level) / 2.0

    point = float(np.median(x))

    lo = float(
        np.quantile(
            distribution,
            tail,
        )
    )

    hi = float(
        np.quantile(
            distribution,
            1.0 - tail,
        )
    )

    return (
        point,
        lo,
        hi,
    )


def compare_model_pair(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    *,
    label_a: str,
    label_b: str,
    seed: int,
) -> dict[str, Any]:
    a = np.asarray(
        model_a["system_risks"],
        dtype=float,
    )

    b = np.asarray(
        model_b["system_risks"],
        dtype=float,
    )

    if a.size != b.size:
        raise ValueError("Paired model risk vectors have unequal length.")

    log_ratio = np.log((a + 1e-12) / (b + 1e-12))

    point95, ci95_lo, ci95_hi = bootstrap_median_ci(
        log_ratio,
        B=BOOTSTRAP_B,
        ci_level=0.95,
        seed=seed,
    )

    _, ci90_lo, ci90_hi = bootstrap_median_ci(
        log_ratio,
        B=BOOTSTRAP_B,
        ci_level=0.90,
        seed=seed + 1,
    )

    margin = float(math.log1p(EQUIVALENCE_DELTA))

    equivalent = bool(ci90_lo > -margin and ci90_hi < margin)

    if ci95_hi < 0.0:
        conclusion = f"{label_a} lower paired risk"
    elif ci95_lo > 0.0:
        conclusion = f"{label_b} lower paired risk"
    else:
        conclusion = "no clear superiority at 95% CI"

    return {
        "A": label_a,
        "B": label_b,
        "n_systems": int(a.size),
        "ratio_A_over_B": float(math.exp(point95)),
        "ci95_ratio_lo": float(math.exp(ci95_lo)),
        "ci95_ratio_hi": float(math.exp(ci95_hi)),
        "ci90_ratio_lo": float(math.exp(ci90_lo)),
        "ci90_ratio_hi": float(math.exp(ci90_hi)),
        "equivalence_delta": float(EQUIVALENCE_DELTA),
        "equivalent_90pct_CI": equivalent,
        "conclusion_95pct_CI": conclusion,
    }


def write_outputs(
    *,
    outdir: Path,
    frozen_models: dict[str, dict[str, Any]],
    evaluation: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    summary_rows = []

    for model_id, model in frozen_models.items():
        ev = evaluation[model_id]

        summary_rows.append(
            {
                "model_id": model_id,
                "label": model["label"],
                "role": model["role"],
                "p": model["p"],
                "lambda": model["lambda"],
                "gamma": model["gamma"],
                "median_R": ev["median_R"],
                "mean_R": ev["mean_R"],
                "system_failure_rate_30pct": ev["system_failure_rate_30pct"],
                "system_failure_rate_10pct": ev["system_failure_rate_10pct"],
                "trajectory_failure_rate_30pct": ev["trajectory_failure_rate_30pct"],
                "trajectory_failure_rate_10pct": ev["trajectory_failure_rate_10pct"],
                "divergences": ev["divergences"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        outdir / "final_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(comparisons).to_csv(
        outdir / "paired_model_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    labels = [frozen_models[model_id]["label"] for model_id in frozen_models]

    median_risks = [evaluation[model_id]["median_R"] for model_id in frozen_models]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.plot(
        range(len(labels)),
        median_risks,
    )

    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=20,
    )

    ax.set_ylabel("Median risk")

    ax.set_title("Frozen-model final trajectory holdout")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "01_final_median_risk.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fail30 = [
        evaluation[model_id]["trajectory_failure_rate_30pct"]
        for model_id in frozen_models
    ]

    fail10 = [
        evaluation[model_id]["trajectory_failure_rate_10pct"]
        for model_id in frozen_models
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x = np.arange(len(labels))

    ax.plot(
        x,
        fail30,
        label="30% threshold",
    )

    ax.plot(
        x,
        fail10,
        label="10% threshold",
    )

    ax.set_xticks(
        x,
        labels=labels,
        rotation=20,
    )

    ax.set_ylabel("Trajectory failure rate")

    ax.set_title("Frozen-model threshold failure rates")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "02_final_failure_rates.png",
        dpi=300,
        bbox_inches="tight",
    )

    p1_ratio = next(
        row["ratio_A_over_B"]
        for row in comparisons
        if row["comparison_id"] == "forgetting_effect_p1"
    )

    p5_ratio = next(
        row["ratio_A_over_B"]
        for row in comparisons
        if row["comparison_id"] == "forgetting_effect_p5"
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    ax.plot(
        [1, 5],
        [
            p1_ratio,
            p5_ratio,
        ],
    )

    ax.axhline(1.0)

    ax.set_xticks([1, 5])

    ax.set_xlabel("Innovation window p")

    ax.set_ylabel("Paired risk ratio: forgetting / no-forgetting")

    ax.set_title("Final holdout forgetting effect")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "03_forgetting_effect_by_p.png",
        dpi=300,
        bbox_inches="tight",
    )

    final_comp = next(
        row for row in comparisons if row["comparison_id"] == "p1_vs_p5_with_forgetting"
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    ratio = float(final_comp["ratio_A_over_B"])

    lo = float(final_comp["ci95_ratio_lo"])

    hi = float(final_comp["ci95_ratio_hi"])

    ax.errorbar(
        [0],
        [ratio],
        yerr=[
            [ratio - lo],
            [hi - ratio],
        ],
    )

    ax.axhline(1.0)

    ax.set_xticks(
        [0],
        labels=["p=1 λ=.95 / p=5 λ=.95"],
    )

    ax.set_ylabel("Paired risk ratio")

    ax.set_title("Parsimonious candidate vs representative MISG")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "04_p1_vs_p5_forgetting_ratio.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def self_test() -> None:
    fake = {
        "meta": {
            "selection_split": "validation",
        },
        "cells": {},
    }

    for model_id, spec in MODEL_SPECS.items():
        key = interaction_cell_key(
            spec["p"],
            spec["lambda"],
        )

        fake["cells"][key] = {
            "gamma": (
                1.0 if spec["lambda"] < 1.0 else (2.5 if spec["p"] == 1 else 3.5)
            ),
            "boundary": False,
            "val": {
                "median_R": 0.01,
            },
            "test": {
                "median_R": 0.02,
            },
        }

    validate_interaction_results(fake)

    frozen = extract_frozen_models(fake)

    assert math.isclose(
        frozen["B_p1_lambda095"]["gamma"],
        1.0,
    )

    a = {
        "system_risks": [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    }

    b = {
        "system_risks": [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    }

    comp = compare_model_pair(
        a,
        b,
        label_a="A",
        label_b="B",
        seed=123,
    )

    assert comp["equivalent_90pct_CI"] is True

    assert math.isclose(
        comp["ratio_A_over_B"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen new-trajectory confirmation"
    )

    parser.add_argument(
        "--interaction-results",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--family",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--traj-start",
        type=int,
        default=DEFAULT_TRAJ_START,
    )

    parser.add_argument(
        "--n-traj",
        type=int,
        default=DEFAULT_N_TRAJ,
    )

    parser.add_argument(
        "--n-systems",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=(SCRIPT_DIR / "final_frozen_comparison_results"),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.traj_start < 10:
        parser.error(
            "--traj-start should be well separated from earlier trajectory "
            "indices; use the default 100 unless you have a specific reason."
        )

    if args.n_traj < 1:
        parser.error("--n-traj must be >= 1.")

    interaction_path = find_interaction_results(args.interaction_results)

    interaction = load_json(interaction_path)

    validate_interaction_results(interaction)

    frozen_models = extract_frozen_models(interaction)

    meta = interaction["meta"]

    project = find_project_root(args.project)

    (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        load_family,
        simulate,
    ) = import_project(project)

    if args.family is not None:
        family_path = args.family.expanduser().resolve()
    else:
        stored = Path(str(meta["family"])).expanduser()

        if stored.exists():
            family_path = stored.resolve()
        else:
            family_path = project / "data" / "systems.npz"

    if not family_path.exists():
        raise SystemExit(f"systems.npz not found: {family_path}")

    N = int(meta["N"])

    cfg0 = Config()

    cfg = replace(
        cfg0,
        N=N,
    )

    family = load_family(family_path)

    systems = family[args.split]

    if args.n_systems is not None:
        systems = systems[: int(args.n_systems)]

    if len(systems) == 0:
        raise SystemExit("Selected system split is empty.")

    new_indices = list(
        range(
            args.traj_start,
            args.traj_start + args.n_traj,
        )
    )

    if any(
        index
        in {
            0,
            3,
            4,
        }
        for index in new_indices
    ):
        raise SystemExit(
            "New trajectory indices overlap earlier experiment trajectory indices."
        )

    outdir = args.outdir.expanduser().resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)

    print("Final frozen-model trajectory holdout")

    print(f"interaction JSON : {interaction_path}")

    print(f"family           : {family_path}")

    print(f"split            : {args.split}")

    print(f"N                : {N}")

    print(f"systems          : {len(systems)}")

    print(f"trajectory ids   : {new_indices}")

    print("Frozen models:")

    for model_id, model in frozen_models.items():
        print(
            f"  {model_id}: "
            f"p={model['p']}, "
            f"lambda={model['lambda']}, "
            f"gamma={model['gamma']}"
        )

    print("=" * 80)

    print("\nBuilding new holdout trajectories...")

    pairs = build_pairs(
        systems,
        j_start=args.traj_start,
        n_traj=args.n_traj,
        cfg=cfg,
        traj_seed=traj_seed,
        simulate=simulate,
    )

    evaluation = {}

    for model_id, model in frozen_models.items():
        print(f"\nRunning {model_id} ...")

        ev = evaluate_frozen_model(
            pairs,
            p=int(model["p"]),
            lambda_value=float(model["lambda"]),
            gamma=float(model["gamma"]),
            cfg=cfg,
            misg_run=misg_run,
            risk_R=risk_R,
        )

        evaluation[model_id] = ev

        print(
            f"  median_R={ev['median_R']:.8f}  "
            f"mean_R={ev['mean_R']:.8f}  "
            f"traj_fail30={ev['trajectory_failure_rate_30pct']:.3f}  "
            f"traj_fail10={ev['trajectory_failure_rate_10pct']:.3f}  "
            f"div={ev['divergences']}"
        )

    comparison_specs = [
        (
            "forgetting_effect_p1",
            "B_p1_lambda095",
            "A_p1_lambda1",
        ),
        (
            "forgetting_effect_p5",
            "D_p5_lambda095",
            "C_p5_lambda1",
        ),
        (
            "p_effect_no_forgetting",
            "A_p1_lambda1",
            "C_p5_lambda1",
        ),
        (
            "p1_vs_p5_with_forgetting",
            "B_p1_lambda095",
            "D_p5_lambda095",
        ),
    ]

    comparisons = []

    for index, (
        comparison_id,
        model_a,
        model_b,
    ) in enumerate(comparison_specs):
        comp = compare_model_pair(
            evaluation[model_a],
            evaluation[model_b],
            label_a=frozen_models[model_a]["label"],
            label_b=frozen_models[model_b]["label"],
            seed=(BOOTSTRAP_SEED + 1000 * index),
        )

        comp["comparison_id"] = comparison_id

        comparisons.append(comp)

    write_outputs(
        outdir=outdir,
        frozen_models=frozen_models,
        evaluation=evaluation,
        comparisons=comparisons,
    )

    final = {
        "meta": {
            "interaction_results": str(interaction_path),
            "project": str(project),
            "family": str(family_path),
            "split": args.split,
            "N": N,
            "n_systems": len(systems),
            "trajectory_indices": new_indices,
            "n_trajectories_per_system": args.n_traj,
            "preselected_final_candidate": "B_p1_lambda095",
            "representative_multi_innovation_candidate": "D_p5_lambda095",
            "selection_rule": (
                "No tuning in this script. This run is confirmation only."
            ),
        },
        "frozen_models": frozen_models,
        "evaluation": evaluation,
        "paired_comparisons": comparisons,
    }

    final_path = outdir / "final_frozen_comparison_results.json"

    json_dump(
        final,
        final_path,
    )

    print("\n" + "=" * 80)

    print("PAIRED FINAL COMPARISONS")

    for comp in comparisons:
        print(
            f"{comp['comparison_id']}: "
            f"ratio={comp['ratio_A_over_B']:.6f}  "
            f"95%CI=[{comp['ci95_ratio_lo']:.6f}, "
            f"{comp['ci95_ratio_hi']:.6f}]  "
            f"equivalent={comp['equivalent_90pct_CI']}  "
            f"{comp['conclusion_95pct_CI']}"
        )

    print(f"\nFinal JSON: {final_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
