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

DEFAULT_N_SYSTEMS = 201
DEFAULT_N_TRAJ = 5
DEFAULT_TRAJ_START = 200

NEW_SYSTEM_NAMESPACE = 99
NEW_SYSTEM_M_OFFSET = 1_000_000

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260826
EQUIVALENCE_DELTA = 0.05


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


def find_project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()

        if (
            (root / "src" / "core" / "config.py").exists()
            and (root / "src" / "core" / "systems.py").exists()
            and (root / "src" / "core" / "estimators.py").exists()
            and (root / "src" / "core" / "metrics.py").exists()
        ):
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
        "Could not find the state-estimation project automatically.\n"
        "Pass --project '/full/path/to/state-estimation'."
    )


def import_project(project: Path):
    src = project / "src" / "core"
    sys.path.insert(0, str(src))

    from config import Config, traj_seed
    from estimators import misg_run
    from metrics import risk_R
    from systems import (
        family_manifest,
        sample_system,
        save_family,
        simulate,
    )

    return (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        family_manifest,
        sample_system,
        save_family,
        simulate,
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
        "Test p-lambda interaction",
    ]

    candidates: list[Path] = []

    for folder_name in sibling_names:
        root = SCRIPT_DIR.parent / folder_name

        candidates.extend(
            [
                root
                / "p_lambda_interaction_results"
                / "p_lambda_interaction_results.json",
                root / "p_lambda_interaction_results.json",
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

    unique: list[Path] = []

    for path in discovered:
        if path not in unique:
            unique.append(path)

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        raise SystemExit(
            "Multiple interaction-result JSON files were found.\n"
            "Specify one with --interaction-results:\n  "
            + "\n  ".join(str(path) for path in unique)
        )

    raise SystemExit(
        "Could not find p_lambda_interaction_results.json automatically.\n"
        "Pass --interaction-results '/full/path/to/p_lambda_interaction_results.json'."
    )


def validate_interaction_results(results: dict[str, Any]) -> None:
    if "meta" not in results or "cells" not in results:
        raise SystemExit("Invalid interaction JSON: missing meta/cells.")

    if results["meta"].get("selection_split") != "validation":
        raise SystemExit("Expected a validation-selected interaction experiment.")

    for spec in MODEL_SPECS.values():
        key = interaction_cell_key(
            int(spec["p"]),
            float(spec["lambda"]),
        )

        if key not in results["cells"]:
            raise SystemExit(f"Required frozen cell is missing: {key}")

        record = results["cells"][key]

        if "gamma" not in record:
            raise SystemExit(f"Selected gamma* missing for {key}.")

        if bool(record.get("boundary", False)):
            raise SystemExit(
                f"gamma* is on the search boundary for {key}; "
                "do not use it in the final holdout."
            )


def extract_frozen_models(
    results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    frozen: dict[str, dict[str, Any]] = {}

    for model_id, spec in MODEL_SPECS.items():
        key = interaction_cell_key(
            int(spec["p"]),
            float(spec["lambda"]),
        )

        record = results["cells"][key]

        frozen[model_id] = {
            **spec,
            "gamma": float(record["gamma"]),
        }

    return frozen


def validate_new_system_count(
    n_systems: int,
    d_choices: tuple[int, ...] | list[int],
) -> None:
    if n_systems < 1:
        raise ValueError("n_systems must be positive.")

    if len(d_choices) == 0:
        raise ValueError("cfg.d_choices must not be empty.")

    if n_systems % len(d_choices) != 0:
        raise ValueError(
            f"n_systems={n_systems} must be divisible by "
            f"len(d_choices)={len(d_choices)} so the holdout is balanced."
        )


def generate_new_system_family(
    *,
    cfg,
    sample_system,
    n_systems: int,
) -> dict[str, list[dict[str, Any]]]:
    d_choices = tuple(int(d) for d in cfg.d_choices)

    validate_new_system_count(
        n_systems,
        d_choices,
    )

    per_delay = n_systems // len(d_choices)

    systems: list[dict[str, Any]] = []

    local_id = 0

    for d in d_choices:
        for _ in range(per_delay):
            seed = np.random.SeedSequence(
                [
                    int(cfg.seed_base),
                    int(NEW_SYSTEM_NAMESPACE),
                    int(local_id),
                ]
            )

            rng = np.random.default_rng(seed)

            system = sample_system(
                rng,
                cfg,
                d,
            )

            system["m"] = NEW_SYSTEM_M_OFFSET + local_id

            system["split"] = "final_new_system"

            system["new_system_local_id"] = local_id

            system["new_system_seed_entropy"] = [
                int(cfg.seed_base),
                int(NEW_SYSTEM_NAMESPACE),
                int(local_id),
            ]

            systems.append(system)

            local_id += 1

    if len(systems) != n_systems:
        raise RuntimeError("New-system generation produced the wrong count.")

    m_values = [int(system["m"]) for system in systems]

    if len(set(m_values)) != len(m_values):
        raise RuntimeError("New system m identifiers are not unique.")

    return {"final_new_system": systems}


def write_manifest_csv(
    *,
    family,
    family_manifest,
    path: Path,
) -> None:
    import pandas as pd

    rows, counts = family_manifest(family)

    df = pd.DataFrame(rows)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    json_dump(
        {"counts_by_delay": counts},
        path.with_suffix(".counts.json"),
    )


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

    for system_index, system in enumerate(systems):
        m = int(system["m"])

        for j in range(
            j_start,
            j_start + n_traj,
        ):
            rng = np.random.default_rng(
                traj_seed(
                    cfg,
                    m,
                    j,
                )
            )

            trajectory = simulate(
                system,
                rng,
                cfg.N,
            )

            pairs.append(
                (
                    system_index,
                    j,
                    system,
                    trajectory,
                )
            )

    return pairs


def capped_risk(
    result,
    system,
    cfg,
    risk_R,
) -> float:
    value = risk_R(
        result["rho_hat_traj"],
        system["rho_star"],
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
        raise ValueError(
            f"Unexpected rho_hat_traj shape {rho.shape}; truth dimension={truth.size}."
        )

    if rho.shape[0] >= N + 1:
        sequence = rho[1 : N + 1]
    elif rho.shape[0] == N:
        sequence = rho
    else:
        raise ValueError(f"rho_hat_traj has only {rho.shape[0]} rows for N={N}.")

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

    risk_by_system: dict[
        int,
        list[float],
    ] = {}

    hit30_by_system: dict[
        int,
        list[float],
    ] = {}

    hit10_by_system: dict[
        int,
        list[float],
    ] = {}

    divergences = 0
    trajectory_fail30 = 0
    trajectory_fail10 = 0
    n_trajectories = 0

    for (
        system_index,
        _j,
        system,
        trajectory,
    ) in pairs:
        n_trajectories += 1

        result = misg_run(
            system,
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
            system,
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
                system["rho_star"],
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

    if B < 500:
        raise ValueError("B must be >= 500.")

    rng = np.random.default_rng(seed)

    n = x.size
    chunk = min(
        1000,
        B,
    )

    parts: list[np.ndarray] = []

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
    evaluation_a: dict[str, Any],
    evaluation_b: dict[str, Any],
    *,
    label_a: str,
    label_b: str,
    comparison_id: str,
    seed: int,
) -> dict[str, Any]:
    a = np.asarray(
        evaluation_a["system_risks"],
        dtype=float,
    )

    b = np.asarray(
        evaluation_b["system_risks"],
        dtype=float,
    )

    if a.size != b.size:
        raise ValueError("Paired model risk vectors have unequal lengths.")

    log_ratio = np.log((a + 1e-12) / (b + 1e-12))

    point, ci95_lo, ci95_hi = bootstrap_median_ci(
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
        "comparison_id": comparison_id,
        "A": label_a,
        "B": label_b,
        "n_systems": int(a.size),
        "ratio_A_over_B": float(math.exp(point)),
        "ci95_ratio_lo": float(math.exp(ci95_lo)),
        "ci95_ratio_hi": float(math.exp(ci95_hi)),
        "ci90_ratio_lo": float(math.exp(ci90_lo)),
        "ci90_ratio_hi": float(math.exp(ci90_hi)),
        "equivalence_delta": EQUIVALENCE_DELTA,
        "equivalent_90pct_CI": equivalent,
        "conclusion_95pct_CI": conclusion,
    }


def write_result_tables(
    *,
    outdir: Path,
    frozen_models: dict[str, dict[str, Any]],
    evaluation: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    import pandas as pd

    rows = []

    for model_id, model in frozen_models.items():
        ev = evaluation[model_id]

        rows.append(
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

    pd.DataFrame(rows).to_csv(
        outdir / "new_system_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(comparisons).to_csv(
        outdir / "new_system_paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_outputs(
    *,
    outdir: Path,
    frozen_models: dict[str, dict[str, Any]],
    evaluation: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    model_ids = list(frozen_models)

    labels = [frozen_models[model_id]["label"] for model_id in model_ids]

    x = np.arange(len(model_ids))

    y = [evaluation[model_id]["median_R"] for model_id in model_ids]

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.0,
        )
    )

    ax.plot(
        x,
        y,
    )

    ax.set_xticks(
        x,
        labels=labels,
        rotation=20,
    )

    ax.set_ylabel("Median risk")

    ax.set_title("Independent new-system holdout")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "01_new_system_median_risk.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fail30 = [
        evaluation[model_id]["trajectory_failure_rate_30pct"] for model_id in model_ids
    ]

    fail10 = [
        evaluation[model_id]["trajectory_failure_rate_10pct"] for model_id in model_ids
    ]

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.0,
        )
    )

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

    ax.set_title("New-system threshold failure rates")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "02_new_system_failure_rates.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    comp_map = {row["comparison_id"]: row for row in comparisons}

    p1 = comp_map["forgetting_effect_p1"]

    p5 = comp_map["forgetting_effect_p5"]

    fig, ax = plt.subplots(
        figsize=(
            6.8,
            4.8,
        )
    )

    ax.plot(
        [1, 5],
        [
            p1["ratio_A_over_B"],
            p5["ratio_A_over_B"],
        ],
    )

    ax.axhline(1.0)

    ax.set_xticks([1, 5])

    ax.set_xlabel("Innovation window p")

    ax.set_ylabel("Paired risk ratio: forgetting / no-forgetting")

    ax.set_title("New-system forgetting effect")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "03_new_system_forgetting_effect.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    comp = comp_map["p1_vs_p5_with_forgetting"]

    ratio = float(comp["ratio_A_over_B"])

    lo = float(comp["ci95_ratio_lo"])

    hi = float(comp["ci95_ratio_hi"])

    fig, ax = plt.subplots(
        figsize=(
            6.8,
            4.8,
        )
    )

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

    ax.set_title("New-system parsimonious vs representative MISG")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        outdir / "04_new_system_p1_vs_p5.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


class _FakeCfg:
    seed_base = 1729
    d_choices = (2, 3, 4)
    N = 100
    r_cap = 10.0


def self_test() -> None:
    assert (
        interaction_cell_key(
            1,
            0.95,
        )
        == "p=1|lambda=0.95"
    )

    validate_new_system_count(
        201,
        (2, 3, 4),
    )

    try:
        validate_new_system_count(
            200,
            (2, 3, 4),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unbalanced new-system count was not rejected.")

    def fake_sample_system(
        rng,
        cfg,
        d,
    ):
        value = float(rng.normal())

        return {
            "A": np.array(
                [
                    [value, 0.0],
                    [0.0, value],
                ]
            ),
            "B": np.eye(2),
            "f": np.ones(2),
            "d": int(d),
            "n": 2,
            "snr_db": 10.0,
            "sigma_v": 0.1,
            "rho_star": np.array(
                [
                    value,
                    value,
                ]
            ),
            "rho_F": 0.5,
            "n_reject": 0,
        }

    family_a = generate_new_system_family(
        cfg=_FakeCfg(),
        sample_system=fake_sample_system,
        n_systems=6,
    )

    family_b = generate_new_system_family(
        cfg=_FakeCfg(),
        sample_system=fake_sample_system,
        n_systems=6,
    )

    systems_a = family_a["final_new_system"]

    systems_b = family_b["final_new_system"]

    assert len(systems_a) == 6

    assert np.array_equal(
        systems_a[0]["A"],
        systems_b[0]["A"],
    )

    assert not np.array_equal(
        systems_a[0]["A"],
        systems_a[1]["A"],
    )

    assert systems_a[0]["m"] == NEW_SYSTEM_M_OFFSET

    assert systems_a[-1]["m"] == NEW_SYSTEM_M_OFFSET + 5

    identical = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        dtype=float,
    )

    fake_eval = {"system_risks": identical.tolist()}

    comp = compare_model_pair(
        fake_eval,
        fake_eval,
        label_a="A",
        label_b="B",
        comparison_id="test",
        seed=1,
    )

    assert math.isclose(
        comp["ratio_A_over_B"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert comp["equivalent_90pct_CI"] is True

    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run newly sampled system confirmation"
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
        "--n-systems",
        type=int,
        default=DEFAULT_N_SYSTEMS,
    )

    parser.add_argument(
        "--n-traj",
        type=int,
        default=DEFAULT_N_TRAJ,
    )

    parser.add_argument(
        "--traj-start",
        type=int,
        default=DEFAULT_TRAJ_START,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=(SCRIPT_DIR / "new_system_holdout_results"),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.n_traj < 1:
        parser.error("--n-traj must be >= 1.")

    if args.traj_start < 20:
        parser.error(
            "--traj-start is too close to earlier experiments. "
            "Use the default 200 unless you have a specific reason."
        )

    interaction_path = find_interaction_results(args.interaction_results)

    interaction = load_json(interaction_path)

    validate_interaction_results(interaction)

    frozen_models = extract_frozen_models(interaction)

    project = find_project_root(args.project)

    (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        family_manifest,
        sample_system,
        save_family,
        simulate,
    ) = import_project(project)

    N = int(interaction["meta"]["N"])

    cfg0 = Config()

    cfg = replace(
        cfg0,
        N=N,
    )

    validate_new_system_count(
        int(args.n_systems),
        tuple(cfg.d_choices),
    )

    outdir = args.outdir.expanduser().resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_path = outdir / "new_systems_final.npz"

    manifest_path = outdir / "new_system_manifest.csv"

    if family_path.exists():
        raise SystemExit(
            f"Refusing to overwrite an existing final new-system family:\n"
            f"  {family_path}\n"
            "For a clean final holdout, either keep and analyze the existing run, "
            "or choose a new --outdir explicitly."
        )

    new_family = generate_new_system_family(
        cfg=cfg,
        sample_system=sample_system,
        n_systems=int(args.n_systems),
    )

    systems = new_family["final_new_system"]

    save_family(
        new_family,
        family_path,
    )

    write_manifest_csv(
        family=new_family,
        family_manifest=family_manifest,
        path=manifest_path,
    )

    trajectory_indices = list(
        range(
            int(args.traj_start),
            int(args.traj_start) + int(args.n_traj),
        )
    )

    print("=" * 80)

    print("INDEPENDENT NEW-SYSTEM HOLDOUT")

    print(f"interaction JSON : {interaction_path}")

    print(f"project          : {project}")

    print(f"new family       : {family_path}")

    print(f"N                : {N}")

    print(f"new systems      : {len(systems)}")

    print(f"d choices        : {tuple(cfg.d_choices)}")

    print(f"system namespace : {NEW_SYSTEM_NAMESPACE}")

    print(f"system m offset  : {NEW_SYSTEM_M_OFFSET}")

    print(f"trajectory ids   : {trajectory_indices}")

    print("Frozen models:")

    for model_id, model in frozen_models.items():
        print(
            f"  {model_id}: "
            f"p={model['p']}, "
            f"lambda={model['lambda']}, "
            f"gamma={model['gamma']}"
        )

    print("=" * 80)

    print("\nBuilding trajectories for the NEW systems...")

    pairs = build_pairs(
        systems,
        j_start=int(args.traj_start),
        n_traj=int(args.n_traj),
        cfg=cfg,
        traj_seed=traj_seed,
        simulate=simulate,
    )

    evaluation: dict[
        str,
        dict[str, Any],
    ] = {}

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
            comparison_id=comparison_id,
            seed=(BOOTSTRAP_SEED + 1000 * index),
        )

        comparisons.append(comp)

    write_result_tables(
        outdir=outdir,
        frozen_models=frozen_models,
        evaluation=evaluation,
        comparisons=comparisons,
    )

    plot_outputs(
        outdir=outdir,
        frozen_models=frozen_models,
        evaluation=evaluation,
        comparisons=comparisons,
    )

    final = {
        "meta": {
            "experiment": "independent_new_system_holdout",
            "interaction_results": str(interaction_path),
            "project": str(project),
            "N": N,
            "n_new_systems": len(systems),
            "d_choices": [int(d) for d in cfg.d_choices],
            "new_system_namespace": NEW_SYSTEM_NAMESPACE,
            "new_system_m_offset": NEW_SYSTEM_M_OFFSET,
            "trajectory_indices": trajectory_indices,
            "n_trajectories_per_system": int(args.n_traj),
            "new_family_npz": str(family_path),
            "selection_rule": (
                "No tuning. All p, lambda, and gamma values were frozen "
                "before this new-system holdout."
            ),
            "preselected_final_candidate": "B_p1_lambda095",
            "representative_multi_innovation_candidate": "D_p5_lambda095",
        },
        "frozen_models": frozen_models,
        "evaluation": evaluation,
        "paired_comparisons": comparisons,
    }

    final_json = outdir / "new_system_holdout_results.json"

    json_dump(
        final,
        final_json,
    )

    print("\n" + "=" * 80)

    print("PAIRED NEW-SYSTEM COMPARISONS")

    for comp in comparisons:
        print(
            f"{comp['comparison_id']}: "
            f"ratio={comp['ratio_A_over_B']:.6f}  "
            f"95%CI=[{comp['ci95_ratio_lo']:.6f}, "
            f"{comp['ci95_ratio_hi']:.6f}]  "
            f"equivalent={comp['equivalent_90pct_CI']}  "
            f"{comp['conclusion_95pct_CI']}"
        )

    print(f"\nFinal JSON: {final_json}")

    print("=" * 80)


if __name__ == "__main__":
    main()
