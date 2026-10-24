from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
P_FIXED = 5
STATE_MODE = "observer"


NEW_FORGETTING_LAMBDAS = [0.999, 0.995, 0.925]

DEFAULT_GAMMA_GRID = [
    0.02,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.80,
    1.00,
    1.20,
    1.50,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
    5.00,
    6.00,
]

TRAIN_TRAJ_START = 0
EVAL_TRAJ_START = 3


def find_project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if (root / "src" / "core").is_dir():
            return root
        raise SystemExit(
            f"--project does not look like the state-estimation project root: {root}"
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
        "Keep this script under the same 'State Estimation' folder, or pass:\n"
        "  --project '/full/path/to/state-estimation'"
    )


def import_project(project: Path):
    src = project / "src" / "core"
    sys.path.insert(0, str(src))

    from config import Config, traj_seed
    from estimators import misg_run
    from metrics import risk_R
    from systems import load_family, simulate

    return Config, traj_seed, misg_run, risk_R, load_family, simulate


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_base_results(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Base results JSON not found: {path}")
        return path

    candidates = [
        SCRIPT_DIR / "lambda_profile_p5_results.json",
        SCRIPT_DIR / "lambda_profile_p5_results" / "lambda_profile_p5_results.json",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise SystemExit(
        "Could not find lambda_profile_p5_results.json automatically.\n"
        "Put this script in the same 01_lambda_selection folder, or pass:\n"
        "  --base-results '/full/path/to/lambda_profile_p5_results.json'"
    )


def float_key(value: float) -> str:
    return str(float(value))


def safe_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional vector; got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf values")
    return arr


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
            rng = np.random.default_rng(traj_seed(cfg, system_item["m"], j))
            trajectory = simulate(system_item, rng, cfg.N)
            pairs.append((system_index, system_item, trajectory))

    return pairs


def capped_risk(result, system_item, cfg, risk_R) -> float:
    value = risk_R(
        result["rho_hat_traj"],
        system_item["rho_star"],
        cfg.N,
    )

    if not np.isfinite(value):
        return float(cfg.r_cap)

    return min(float(value), float(cfg.r_cap))


def relative_error_series(
    rho_hat_traj: np.ndarray,
    rho_star: np.ndarray,
    N: int,
) -> np.ndarray:
    rho = np.asarray(rho_hat_traj, dtype=float)
    truth = np.asarray(rho_star, dtype=float)

    denominator = float(np.linalg.norm(truth))
    if denominator <= 0.0:
        return np.full(N, np.nan)

    if rho.ndim != 2 or rho.shape[1] != truth.size:
        raise ValueError(
            f"Unexpected rho_hat_traj shape {rho.shape}; truth dimension={truth.size}"
        )

    if rho.shape[0] >= N + 1:
        sequence = rho[1 : N + 1]
    elif rho.shape[0] == N:
        sequence = rho
    else:
        raise ValueError(f"rho_hat_traj has only {rho.shape[0]} rows for N={N}")

    return np.linalg.norm(sequence - truth[None, :], axis=1) / denominator


def first_hit_time(errors: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(np.isfinite(errors) & (errors <= threshold))

    if indices.size == 0:
        return float("nan")

    return float(indices[0] + 1)


def mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)

    if not np.any(finite):
        return float("nan")

    return float(np.mean(arr[finite]))


def evaluate(
    pairs,
    *,
    gamma: float,
    lambda_value: float,
    r_mode: str,
    cfg,
    misg_run,
    risk_R,
) -> dict[str, Any]:
    if r_mode not in {"cumulative", "forgetting"}:
        raise ValueError(f"Unsupported r_mode: {r_mode!r}")

    weights = np.ones(P_FIXED, dtype=float)

    risk_by_system: dict[int, list[float]] = {}
    hit10_by_system: dict[int, list[float]] = {}
    hit30_by_system: dict[int, list[float]] = {}
    divergences = 0

    for system_index, system_item, trajectory in pairs:
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
        risk_by_system.setdefault(system_index, []).append(risk)

        if diverged:
            hit10 = float("nan")
            hit30 = float("nan")
        else:
            errors = relative_error_series(
                result["rho_hat_traj"],
                system_item["rho_star"],
                cfg.N,
            )
            hit10 = first_hit_time(errors, 0.10)
            hit30 = first_hit_time(errors, 0.30)

        hit10_by_system.setdefault(system_index, []).append(hit10)
        hit30_by_system.setdefault(system_index, []).append(hit30)

    system_risks = np.asarray(
        [np.mean(values) for _, values in sorted(risk_by_system.items())],
        dtype=float,
    )

    system_hit10 = np.asarray(
        [mean_or_nan(values) for _, values in sorted(hit10_by_system.items())],
        dtype=float,
    )

    system_hit30 = np.asarray(
        [mean_or_nan(values) for _, values in sorted(hit30_by_system.items())],
        dtype=float,
    )

    finite10 = np.isfinite(system_hit10)
    finite30 = np.isfinite(system_hit30)

    return {
        "median_R": float(np.median(system_risks)),
        "mean_R": float(np.mean(system_risks)),
        "divergences": int(divergences),
        "system_risks": system_risks.tolist(),
        "hit_times_10pct": system_hit10.tolist(),
        "hit_times_30pct": system_hit30.tolist(),
        "median_hit_10pct_success_only": (
            float(np.median(system_hit10[finite10]))
            if np.any(finite10)
            else float("nan")
        ),
        "median_hit_30pct_success_only": (
            float(np.median(system_hit30[finite30]))
            if np.any(finite30)
            else float("nan")
        ),
        "n_failed_10pct": int(np.sum(~finite10)),
        "n_failed_30pct": int(np.sum(~finite30)),
        "failure_rate_10pct": float(np.mean(~finite10)),
        "failure_rate_30pct": float(np.mean(~finite30)),
        "n_systems": int(system_risks.size),
    }


def retune_gamma(
    *,
    condition_name: str,
    lambda_value: float,
    r_mode: str,
    gamma_grid: list[float],
    top_k: int,
    train_pairs,
    val_pairs,
    test_pairs,
    cfg,
    misg_run,
    risk_R,
) -> dict[str, Any]:
    train_scan = []

    for gamma in gamma_grid:
        ev = evaluate(
            train_pairs,
            gamma=float(gamma),
            lambda_value=lambda_value,
            r_mode=r_mode,
            cfg=cfg,
            misg_run=misg_run,
            risk_R=risk_R,
        )

        train_scan.append(
            {
                "gamma": float(gamma),
                "mean_R": float(ev["mean_R"]),
                "median_R": float(ev["median_R"]),
                "divergences": int(ev["divergences"]),
            }
        )

    order = np.argsort([record["mean_R"] for record in train_scan])

    candidate_indices = [int(index) for index in order[: min(top_k, len(order))]]

    validation_candidates = []
    best_candidate = None

    for index in candidate_indices:
        gamma = float(train_scan[index]["gamma"])

        val_eval = evaluate(
            val_pairs,
            gamma=gamma,
            lambda_value=lambda_value,
            r_mode=r_mode,
            cfg=cfg,
            misg_run=misg_run,
            risk_R=risk_R,
        )

        candidate = {
            "gamma": gamma,
            "train_mean_R": float(train_scan[index]["mean_R"]),
            "val": val_eval,
        }
        validation_candidates.append(candidate)

        if (
            best_candidate is None
            or val_eval["median_R"] < best_candidate["val"]["median_R"]
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No gamma candidate was selected")

    gamma_star = float(best_candidate["gamma"])
    selected_index = gamma_grid.index(gamma_star)
    boundary = selected_index in (0, len(gamma_grid) - 1)

    test_eval = evaluate(
        test_pairs,
        gamma=gamma_star,
        lambda_value=lambda_value,
        r_mode=r_mode,
        cfg=cfg,
        misg_run=misg_run,
        risk_R=risk_R,
    )

    return {
        "condition": str(condition_name),
        "lambda": float(lambda_value),
        "p": P_FIXED,
        "r_mode": str(r_mode),
        "gamma": gamma_star,
        "boundary": bool(boundary),
        "train_mean_R": float(best_candidate["train_mean_R"]),
        "val_selection_risk": float(best_candidate["val"]["median_R"]),
        "train_scan": train_scan,
        "val_candidates": validation_candidates,
        "val": best_candidate["val"],
        "test": test_eval,
    }


def validate_base_results(base: dict[str, Any]) -> None:
    if "meta" not in base or "per_lambda" not in base:
        raise SystemExit(
            "Base JSON is not a valid lambda_profile_p5_results.json "
            "(missing meta/per_lambda)"
        )

    meta = base["meta"]

    if int(meta.get("p_fixed", -1)) != P_FIXED:
        raise SystemExit(
            f"Base results use p={meta.get('p_fixed')}; expected p={P_FIXED}"
        )

    for required in ["1.0", "0.99", "0.95", "0.9", "0.85", "0.8", "0.75", "0.7"]:
        if required not in base["per_lambda"]:
            raise SystemExit(f"Base results are incomplete: missing lambda={required}")

    one = base["per_lambda"]["1.0"]
    if int(one.get("p", -1)) != P_FIXED:
        raise SystemExit("Base lambda=1 result does not use p=5")
    if "train_scan" not in one or "val" not in one or "test" not in one:
        raise SystemExit("Base lambda=1 record is incomplete")


def _scan_map(record: dict[str, Any], field: str) -> dict[float, float]:
    return {float(item["gamma"]): float(item[field]) for item in record["train_scan"]}


def compare_branch_consistency(
    cumulative_record: dict[str, Any],
    forgetting_record: dict[str, Any],
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> dict[str, Any]:
    cum_mean = _scan_map(cumulative_record, "mean_R")
    fgt_mean = _scan_map(forgetting_record, "mean_R")
    common_gamma = sorted(set(cum_mean).intersection(fgt_mean))

    if not common_gamma:
        raise ValueError("No common gamma values for branch comparison")

    profile_abs_diff = [abs(cum_mean[g] - fgt_mean[g]) for g in common_gamma]

    same_gamma_star = math.isclose(
        float(cumulative_record["gamma"]),
        float(forgetting_record["gamma"]),
        rel_tol=rtol,
        abs_tol=atol,
    )

    selected_test_allclose = None
    max_selected_test_abs_diff = None

    if same_gamma_star:
        a = safe_array(
            cumulative_record["test"]["system_risks"],
            "cumulative test system_risks",
        )
        b = safe_array(
            forgetting_record["test"]["system_risks"],
            "forgetting test system_risks",
        )

        if a.size != b.size:
            raise ValueError(
                f"Branch test vectors have different sizes: {a.size} vs {b.size}"
            )

        selected_test_allclose = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        max_selected_test_abs_diff = float(np.max(np.abs(a - b)))

    profile_allclose = all(
        math.isclose(
            cum_mean[g],
            fgt_mean[g],
            rel_tol=rtol,
            abs_tol=atol,
        )
        for g in common_gamma
    )

    consistent = bool(
        profile_allclose and same_gamma_star and (selected_test_allclose is not False)
    )

    return {
        "consistent": consistent,
        "atol": float(atol),
        "rtol": float(rtol),
        "n_common_gamma": int(len(common_gamma)),
        "max_abs_train_mean_profile_diff": float(max(profile_abs_diff)),
        "cumulative_gamma_star": float(cumulative_record["gamma"]),
        "forgetting_gamma_star": float(forgetting_record["gamma"]),
        "same_gamma_star": bool(same_gamma_star),
        "selected_test_allclose": selected_test_allclose,
        "max_selected_test_abs_diff": max_selected_test_abs_diff,
        "cumulative_val_median_R": float(cumulative_record["val"]["median_R"]),
        "forgetting_val_median_R": float(forgetting_record["val"]["median_R"]),
        "cumulative_test_median_R": float(cumulative_record["test"]["median_R"]),
        "forgetting_test_median_R": float(forgetting_record["test"]["median_R"]),
    }


def paired_log_ratio_stat(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    if risk_a.size != risk_b.size:
        raise ValueError(
            f"Paired vectors must have equal length; got {risk_a.size} and {risk_b.size}"
        )

    if np.any(risk_a < 0.0) or np.any(risk_b < 0.0):
        raise ValueError("Risk values must be nonnegative")

    return np.log((risk_a + eps) / (risk_b + eps))


def bootstrap_median_ci(
    paired_values: np.ndarray,
    *,
    B: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    x = safe_array(paired_values, "paired_values")

    if B < 200:
        raise ValueError("Bootstrap B should be at least 200")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0,1)")

    point = float(np.median(x))
    n = x.size

    chunk = min(1000, B)
    estimates: list[np.ndarray] = []
    done = 0

    while done < B:
        current = min(chunk, B - done)
        indices = rng.integers(0, n, size=(current, n))
        boot = np.median(x[indices], axis=1)
        estimates.append(boot)
        done += current

    distribution = np.concatenate(estimates)

    alpha = 1.0 - ci_level
    lo = float(np.quantile(distribution, alpha / 2.0))
    hi = float(np.quantile(distribution, 1.0 - alpha / 2.0))

    return point, lo, hi


def paired_comparison(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    *,
    label_a: str,
    label_b: str,
    B: int,
    delta: float,
    seed: int,
) -> dict[str, Any]:
    a = safe_array(risk_a, f"{label_a} risks")
    b = safe_array(risk_b, f"{label_b} risks")

    d = paired_log_ratio_stat(a, b)

    rng95 = np.random.default_rng(seed)
    point, ci95_lo, ci95_hi = bootstrap_median_ci(
        d,
        B=B,
        ci_level=0.95,
        rng=rng95,
    )

    rng90 = np.random.default_rng(seed + 1)
    _, ci90_lo, ci90_hi = bootstrap_median_ci(
        d,
        B=B,
        ci_level=0.90,
        rng=rng90,
    )

    margin = float(math.log1p(delta))

    equivalent = bool(ci90_lo > -margin and ci90_hi < margin)

    if ci95_hi < 0.0:
        superiority = f"{label_a} lower risk"
    elif ci95_lo > 0.0:
        superiority = f"{label_b} lower risk"
    else:
        superiority = "no clear superiority at 95% CI"

    return {
        "A": label_a,
        "B": label_b,
        "n_systems": int(a.size),
        "statistic": "median paired log risk ratio log(A/B)",
        "point_log_ratio": float(point),
        "ratio_A_over_B": float(math.exp(point)),
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
        "median_R_A": float(np.median(a)),
        "median_R_B": float(np.median(b)),
        "mean_R_A": float(np.mean(a)),
        "mean_R_B": float(np.mean(b)),
    }


def merge_lambda_records(
    base: dict[str, Any],
    refinement: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for key, record in base["per_lambda"].items():
        merged[float_key(float(key))] = record

    for key, record in refinement.get("per_lambda", {}).items():
        merged[float_key(float(key))] = record

    return merged


def write_outputs(
    *,
    outdir: Path,
    base: dict[str, Any],
    refinement: dict[str, Any],
    branch: dict[str, Any],
    bootstrap_B: int,
    delta: float,
    seed: int,
) -> dict[str, Any]:
    import pandas as pd

    merged = merge_lambda_records(base, refinement)

    summary_rows = []

    for key in sorted(merged, key=float, reverse=True):
        record = merged[key]

        inferred_mode = (
            record.get("r_mode")
            if record.get("r_mode") is not None
            else ("cumulative" if abs(float(key) - 1.0) < 1e-12 else "forgetting")
        )

        test = record["test"]
        summary_rows.append(
            {
                "lambda": float(key),
                "mode": inferred_mode,
                "p": int(record.get("p", P_FIXED)),
                "gamma_star": float(record["gamma"]),
                "boundary": bool(record["boundary"]),
                "val_median_R": float(record["val"]["median_R"]),
                "test_median_R": float(test["median_R"]),
                "test_mean_R": float(test["mean_R"]),
                "test_divergences": int(test["divergences"]),
                "T30_success_only": float(test["median_hit_30pct_success_only"]),
                "failure_rate_30pct": float(test["failure_rate_30pct"]),
                "T10_success_only": float(test["median_hit_10pct_success_only"]),
                "failure_rate_10pct": float(test["failure_rate_10pct"]),
            }
        )

    fgt1 = refinement["branch_control"]["forgetting_lambda1"]
    t = fgt1["test"]
    summary_rows.append(
        {
            "lambda": 1.0,
            "mode": "forgetting",
            "p": P_FIXED,
            "gamma_star": float(fgt1["gamma"]),
            "boundary": bool(fgt1["boundary"]),
            "val_median_R": float(fgt1["val"]["median_R"]),
            "test_median_R": float(t["median_R"]),
            "test_mean_R": float(t["mean_R"]),
            "test_divergences": int(t["divergences"]),
            "T30_success_only": float(t["median_hit_30pct_success_only"]),
            "failure_rate_30pct": float(t["failure_rate_30pct"]),
            "T10_success_only": float(t["median_hit_10pct_success_only"]),
            "failure_rate_10pct": float(t["failure_rate_10pct"]),
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["lambda", "mode"],
        ascending=[False, True],
    )
    summary_df.to_csv(
        outdir / "step1_refinement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    branch_df = pd.DataFrame([branch])
    branch_df.to_csv(
        outdir / "branch_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plateau_values = [
        value
        for value in [0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95]
        if float_key(value) in merged
    ]

    validation_rows = []
    test_rows = []

    pair_counter = 0
    for i, a_value in enumerate(plateau_values):
        for b_value in plateau_values[i + 1 :]:
            a_key = float_key(a_value)
            b_key = float_key(b_value)

            val_a = np.asarray(
                merged[a_key]["val"]["system_risks"],
                dtype=float,
            )
            val_b = np.asarray(
                merged[b_key]["val"]["system_risks"],
                dtype=float,
            )

            pair_counter += 1
            val_cmp = paired_comparison(
                val_a,
                val_b,
                label_a=f"lambda={a_value:g}",
                label_b=f"lambda={b_value:g}",
                B=bootstrap_B,
                delta=delta,
                seed=seed + 100 * pair_counter,
            )
            val_cmp["split"] = "validation"
            validation_rows.append(val_cmp)

            test_a = np.asarray(
                merged[a_key]["test"]["system_risks"],
                dtype=float,
            )
            test_b = np.asarray(
                merged[b_key]["test"]["system_risks"],
                dtype=float,
            )

            test_cmp = paired_comparison(
                test_a,
                test_b,
                label_a=f"lambda={a_value:g}",
                label_b=f"lambda={b_value:g}",
                B=bootstrap_B,
                delta=delta,
                seed=seed + 100 * pair_counter + 50,
            )
            test_cmp["split"] = "test_confirmation_only"
            test_rows.append(test_cmp)

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)

    validation_df.to_csv(
        outdir / "pairwise_equivalence_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        outdir / "pairwise_confirmation_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    focused_rows = []

    for split_name in ["val", "test"]:
        a = np.asarray(
            merged[float_key(0.90)][split_name]["system_risks"],
            dtype=float,
        )
        b = np.asarray(
            merged[float_key(0.95)][split_name]["system_risks"],
            dtype=float,
        )

        cmp = paired_comparison(
            a,
            b,
            label_a="lambda=0.90",
            label_b="lambda=0.95",
            B=bootstrap_B,
            delta=delta,
            seed=seed + (900 if split_name == "val" else 950),
        )

        cmp["split"] = (
            "validation_selection" if split_name == "val" else "test_confirmation_only"
        )
        focused_rows.append(cmp)

    focused_df = pd.DataFrame(focused_rows)
    focused_df.to_csv(
        outdir / "lambda_090_vs_095_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_medians = {
        value: float(merged[float_key(value)]["val"]["median_R"])
        for value in plateau_values
    }

    best_validation_lambda = min(
        validation_medians,
        key=validation_medians.get,
    )
    best_validation_risk = validation_medians[best_validation_lambda]

    robust_candidates = []
    for value in plateau_values:
        ratio_to_best = (
            validation_medians[value] / best_validation_risk
            if best_validation_risk > 0.0
            else float("inf")
        )

        if value == best_validation_lambda:
            equivalent_to_best = True
        else:
            labels = {
                f"lambda={best_validation_lambda:g}",
                f"lambda={value:g}",
            }

            matching = [
                row for row in validation_rows if {row["A"], row["B"]} == labels
            ]

            equivalent_to_best = bool(matching and matching[0]["equivalent_90pct_CI"])

        if ratio_to_best <= (1.0 + delta) and equivalent_to_best:
            robust_candidates.append(float(value))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum = base["per_lambda"]["1.0"]
    fgt = refinement["branch_control"]["forgetting_lambda1"]

    cum_map = _scan_map(cum, "mean_R")
    fgt_map = _scan_map(fgt, "mean_R")
    gamma_common = sorted(set(cum_map).intersection(fgt_map))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(
        gamma_common,
        [cum_map[g] for g in gamma_common],
        marker="o",
        label="cumulative, lambda=1",
    )
    ax.plot(
        gamma_common,
        [fgt_map[g] for g in gamma_common],
        marker="x",
        label="forgetting mode, lambda=1",
    )
    ax.set_xlabel("gamma")
    ax.set_ylabel("Training mean risk")
    ax.set_title("Branch consistency control at lambda=1")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "01_branch_consistency_gamma_profile.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    local_values = [
        value
        for value in [0.90, 0.925, 0.95, 0.99, 0.995, 0.999, 1.0]
        if float_key(value) in merged
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(
        local_values,
        [merged[float_key(value)]["val"]["median_R"] for value in local_values],
        marker="o",
        label="validation median risk",
    )
    ax.plot(
        local_values,
        [merged[float_key(value)]["test"]["median_R"] for value in local_values],
        marker="o",
        label="test median risk",
    )
    ax.set_xlabel("Forgetting factor lambda")
    ax.set_ylabel("Median risk")
    ax.set_title("Local lambda refinement near the cliff and 0.90–0.95 region")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "02_lambda_cliff_refinement.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(
        local_values,
        [merged[float_key(value)]["gamma"] for value in local_values],
        marker="o",
    )
    ax.set_xlabel("Forgetting factor lambda")
    ax.set_ylabel("Selected gamma*")
    ax.set_title("Lambda-gamma coupling after local refinement")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        outdir / "03_lambda_refinement_gamma.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    labels = [f"{value:g}" for value in plateau_values]
    size = len(plateau_values)
    matrix = np.full((size, size), np.nan)

    for i in range(size):
        matrix[i, i] = 1.0

    for row in validation_rows:
        a = float(row["A"].split("=")[1])
        b = float(row["B"].split("=")[1])
        i = plateau_values.index(a)
        j = plateau_values.index(b)
        value = 1.0 if row["equivalent_90pct_CI"] else 0.0
        matrix[i, j] = value
        matrix[j, i] = value

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(size), labels=labels)
    ax.set_yticks(range(size), labels=labels)
    ax.set_xlabel("lambda")
    ax.set_ylabel("lambda")
    ax.set_title(f"Validation practical equivalence matrix (delta={delta:.0%})")

    for i in range(size):
        for j in range(size):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    "EQ" if matrix[i, j] == 1.0 else "NO",
                    ha="center",
                    va="center",
                )

    fig.colorbar(image, ax=ax, label="1 = equivalent")
    fig.tight_layout()
    fig.savefig(
        outdir / "04_validation_equivalence_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    statistical_summary = {
        "method": {
            "selection_split": "validation",
            "test_usage": "confirmation only; not used to select lambda",
            "paired_statistic": "median log risk ratio",
            "bootstrap_B": int(bootstrap_B),
            "superiority_CI": 0.95,
            "equivalence_CI": 0.90,
            "equivalence_delta": float(delta),
        },
        "best_validation_lambda_by_median_R": float(best_validation_lambda),
        "best_validation_median_R": float(best_validation_risk),
        "validation_median_R_by_lambda": {
            str(value): float(validation_medians[value]) for value in plateau_values
        },
        "robust_candidates_within_delta_and_equivalent_to_best": robust_candidates,
        "focused_090_vs_095": focused_rows,
    }

    all_results = {
        "branch_consistency": branch,
        "statistical_closeout": statistical_summary,
    }

    return all_results


def self_test() -> None:
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    b = a.copy()

    cmp_equal = paired_comparison(
        a,
        b,
        label_a="A",
        label_b="B",
        B=500,
        delta=0.05,
        seed=123,
    )

    assert cmp_equal["equivalent_90pct_CI"] is True
    assert abs(cmp_equal["point_log_ratio"]) < 1e-12

    a2 = np.array([0.05, 0.10, 0.15, 0.20, 0.25], dtype=float)
    b2 = np.array([0.10, 0.20, 0.30, 0.40, 0.50], dtype=float)

    cmp_better = paired_comparison(
        a2,
        b2,
        label_a="A",
        label_b="B",
        B=500,
        delta=0.05,
        seed=456,
    )

    assert cmp_better["ci95_log_hi"] < 0.0

    fake_record = {
        "gamma": 0.2,
        "train_scan": [
            {"gamma": 0.1, "mean_R": 0.3},
            {"gamma": 0.2, "mean_R": 0.2},
        ],
        "val": {"median_R": 0.1},
        "test": {
            "median_R": 0.1,
            "system_risks": [0.1, 0.2, 0.3],
        },
    }

    branch = compare_branch_consistency(
        fake_record,
        fake_record,
    )

    assert branch["consistent"] is True

    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete the forgetting-factor selection analysis"
    )

    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--family", type=Path, default=None)
    parser.add_argument("--base-results", type=Path, default=None)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "lambda_closeout_steps1_2",
    )

    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--n-train-traj", type=int, default=None)
    parser.add_argument("--n-eval-traj", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)

    parser.add_argument(
        "--gamma-grid",
        nargs="+",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--bootstrap-B",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Practical-equivalence multiplicative margin, default 0.05 = 5%%.",
    )

    parser.add_argument("--seed", type=int, default=20260825)

    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip new MISG runs and analyze an existing closeout checkpoint/results file.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal unit/smoke tests without importing the project.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstrap_B < 200:
        parser.error("--bootstrap-B must be at least 200")

    if not (0.0 < args.delta < 0.5):
        parser.error("--delta must be in (0, 0.5)")

    base_path = find_base_results(args.base_results)
    base = load_json(base_path)
    validate_base_results(base)

    meta = base["meta"]

    N = int(args.N if args.N is not None else meta["N"])
    n_train = int(args.n_train if args.n_train is not None else meta["n_train"])
    n_val = int(args.n_val if args.n_val is not None else meta["n_val"])
    n_test = int(args.n_test if args.n_test is not None else meta["n_test"])
    n_train_traj = int(
        args.n_train_traj if args.n_train_traj is not None else meta["n_train_traj"]
    )
    n_eval_traj = int(
        args.n_eval_traj if args.n_eval_traj is not None else meta["n_eval_traj"]
    )
    top_k = int(args.top_k if args.top_k is not None else meta["top_k"])

    gamma_grid = (
        [float(x) for x in args.gamma_grid]
        if args.gamma_grid is not None
        else [float(x) for x in meta.get("gamma_grid", DEFAULT_GAMMA_GRID)]
    )

    expected_protocol = {
        "N": int(meta["N"]),
        "n_train": int(meta["n_train"]),
        "n_val": int(meta["n_val"]),
        "n_test": int(meta["n_test"]),
        "n_train_traj": int(meta["n_train_traj"]),
        "n_eval_traj": int(meta["n_eval_traj"]),
        "top_k": int(meta["top_k"]),
    }
    actual_protocol = {
        "N": N,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_train_traj": n_train_traj,
        "n_eval_traj": n_eval_traj,
        "top_k": top_k,
    }

    if actual_protocol != expected_protocol:
        raise SystemExit(
            "Protocol mismatch with the completed lambda profile. "
            "For a valid close-out experiment, do not change N/split/trajectory/top-k settings.\n"
            f"Expected: {expected_protocol}\n"
            f"Actual  : {actual_protocol}"
        )

    expected_gamma_grid = [float(x) for x in meta.get("gamma_grid", DEFAULT_GAMMA_GRID)]
    if len(gamma_grid) != len(expected_gamma_grid) or not np.allclose(
        np.asarray(gamma_grid, dtype=float),
        np.asarray(expected_gamma_grid, dtype=float),
        atol=0.0,
        rtol=0.0,
    ):
        raise SystemExit(
            "Gamma-grid mismatch with the completed lambda profile. "
            "Use the same gamma grid for a fair close-out comparison."
        )

    if N < 100:
        parser.error("N is unrealistically small")
    if n_train < 1 or n_val < 1 or n_test < 1:
        parser.error("train/val/test sizes must be positive")
    if n_train_traj < 1 or n_eval_traj < 1:
        parser.error("trajectory counts must be positive")
    if top_k < 1:
        parser.error("top_k must be positive")
    if not gamma_grid or min(gamma_grid) <= 0.0:
        parser.error("gamma grid must contain positive values")

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = outdir / "lambda_closeout_checkpoint.json"
    final_path = outdir / "lambda_closeout_results.json"

    refinement: dict[str, Any] = {
        "meta": {
            "base_results": str(base_path),
            "N": N,
            "p_fixed": P_FIXED,
            "new_forgetting_lambdas": NEW_FORGETTING_LAMBDAS,
            "gamma_grid": gamma_grid,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_train_traj": n_train_traj,
            "n_eval_traj": n_eval_traj,
            "top_k": top_k,
        },
        "branch_control": {},
        "per_lambda": {},
    }

    if checkpoint_path.exists():
        saved = load_json(checkpoint_path)
        saved_meta = saved.get("meta", {})

        checkpoint_matches = (
            int(saved_meta.get("N", -1)) == N
            and int(saved_meta.get("p_fixed", -1)) == P_FIXED
            and int(saved_meta.get("n_train", -1)) == n_train
            and int(saved_meta.get("n_val", -1)) == n_val
            and int(saved_meta.get("n_test", -1)) == n_test
            and int(saved_meta.get("n_train_traj", -1)) == n_train_traj
            and int(saved_meta.get("n_eval_traj", -1)) == n_eval_traj
            and int(saved_meta.get("top_k", -1)) == top_k
            and saved_meta.get("gamma_grid") == gamma_grid
            and saved_meta.get("new_forgetting_lambdas") == NEW_FORGETTING_LAMBDAS
        )

        if checkpoint_matches:
            refinement = saved
            print(
                f"Checkpoint loaded: branch_control={list(refinement.get('branch_control', {}))}, "
                f"new lambdas={list(refinement.get('per_lambda', {}))}"
            )
        else:
            print("Existing checkpoint protocol does not match; starting fresh.")

    if not args.analysis_only:
        project = find_project_root(args.project)

        (
            Config,
            traj_seed,
            misg_run,
            risk_R,
            load_family,
            simulate,
        ) = import_project(project)

        if args.family is None:
            family_path = Path(meta["family"]).expanduser()

            if not family_path.exists():
                family_path = project / "data" / "systems.npz"
        else:
            family_path = args.family.expanduser().resolve()

        if not family_path.exists():
            raise SystemExit(f"Family NPZ not found: {family_path}")

        cfg0 = Config()
        cfg = replace(cfg0, N=N)

        family = load_family(family_path)
        train_systems = family["train"][:n_train]
        val_systems = family["val"][:n_val]
        test_systems = family["test"][:n_test]

        if len(train_systems) != n_train:
            raise SystemExit(
                f"Requested {n_train} train systems but family contains {len(train_systems)}"
            )
        if len(val_systems) != n_val:
            raise SystemExit(
                f"Requested {n_val} val systems but family contains {len(val_systems)}"
            )
        if len(test_systems) != n_test:
            raise SystemExit(
                f"Requested {n_test} test systems but family contains {len(test_systems)}"
            )

        print("=" * 76)
        print("Lambda close-out: steps 1 + 2")
        print(f"project      : {project}")
        print(f"family       : {family_path}")
        print(f"base results : {base_path}")
        print(f"output       : {outdir}")
        print(f"N            : {N}")
        print(f"fixed p      : {P_FIXED}")
        print(f"new lambdas  : {NEW_FORGETTING_LAMBDAS}")
        print(f"gamma grid   : {gamma_grid}")
        print(
            f"splits       : train={len(train_systems)}, "
            f"val={len(val_systems)}, test={len(test_systems)}"
        )
        print("=" * 76)

        print("\nBuilding shared trajectories...")
        train_pairs = build_pairs(
            train_systems,
            j_start=TRAIN_TRAJ_START,
            n_traj=n_train_traj,
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        val_pairs = build_pairs(
            val_systems,
            j_start=EVAL_TRAJ_START,
            n_traj=n_eval_traj,
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        test_pairs = build_pairs(
            test_systems,
            j_start=EVAL_TRAJ_START,
            n_traj=n_eval_traj,
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        if "forgetting_lambda1" not in refinement["branch_control"]:
            print("\n" + "#" * 76)
            print("STEP 1A: forgetting mode at lambda=1")
            print("#" * 76)

            start = time.perf_counter()

            fgt1 = retune_gamma(
                condition_name="forgetting_lambda1",
                lambda_value=1.0,
                r_mode="forgetting",
                gamma_grid=gamma_grid,
                top_k=top_k,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                test_pairs=test_pairs,
                cfg=cfg,
                misg_run=misg_run,
                risk_R=risk_R,
            )

            refinement["branch_control"]["forgetting_lambda1"] = fgt1
            json_dump(refinement, checkpoint_path)

            print(
                f"lambda=1 forgetting  gamma*={fgt1['gamma']:.3g}  "
                f"val_R={fgt1['val']['median_R']:.6f}  "
                f"test_R={fgt1['test']['median_R']:.6f}  "
                f"time={time.perf_counter() - start:.1f}s"
            )
        else:
            print("\nSTEP 1A already completed -> skip.")

        for lambda_value in NEW_FORGETTING_LAMBDAS:
            key = float_key(lambda_value)

            if key in refinement["per_lambda"]:
                print(f"\nlambda={lambda_value:g} already completed -> skip.")
                continue

            print("\n" + "#" * 76)
            print(f"STEP 1B: lambda={lambda_value:g}, forgetting mode, p={P_FIXED}")
            print("#" * 76)

            start = time.perf_counter()

            record = retune_gamma(
                condition_name=f"forgetting_lambda_{lambda_value:g}",
                lambda_value=lambda_value,
                r_mode="forgetting",
                gamma_grid=gamma_grid,
                top_k=top_k,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                test_pairs=test_pairs,
                cfg=cfg,
                misg_run=misg_run,
                risk_R=risk_R,
            )

            refinement["per_lambda"][key] = record
            json_dump(refinement, checkpoint_path)

            test = record["test"]
            print(
                f"lambda={lambda_value:g}  "
                f"gamma*={record['gamma']:.3g}  "
                f"val_R={record['val']['median_R']:.6f}  "
                f"test_R={test['median_R']:.6f}  "
                f"T30={test['median_hit_30pct_success_only']:.1f}  "
                f"fail30={test['failure_rate_30pct']:.3f}  "
                f"T10={test['median_hit_10pct_success_only']:.1f}  "
                f"fail10={test['failure_rate_10pct']:.3f}  "
                f"boundary={record['boundary']}  "
                f"time={time.perf_counter() - start:.1f}s"
            )

    if "forgetting_lambda1" not in refinement.get("branch_control", {}):
        raise SystemExit(
            "Missing forgetting-mode lambda=1 control. "
            "Run without --analysis-only first."
        )

    missing_new = [
        value
        for value in NEW_FORGETTING_LAMBDAS
        if float_key(value) not in refinement.get("per_lambda", {})
    ]

    if missing_new:
        raise SystemExit(
            "Missing refined lambda points: "
            + ", ".join(str(x) for x in missing_new)
            + ". Run without --analysis-only first."
        )

    branch = compare_branch_consistency(
        base["per_lambda"]["1.0"],
        refinement["branch_control"]["forgetting_lambda1"],
    )

    print("\n" + "=" * 76)
    print("STEP 1A BRANCH CONSISTENCY")
    print(f"consistent                     : {branch['consistent']}")
    print(
        "max train-profile abs diff    : "
        f"{branch['max_abs_train_mean_profile_diff']:.6g}"
    )
    print(
        "cumulative gamma* / forgetting gamma*: "
        f"{branch['cumulative_gamma_star']} / {branch['forgetting_gamma_star']}"
    )
    print(
        "test median R cumulative / forgetting: "
        f"{branch['cumulative_test_median_R']:.6f} / "
        f"{branch['forgetting_test_median_R']:.6f}"
    )
    print("=" * 76)

    statistical = write_outputs(
        outdir=outdir,
        base=base,
        refinement=refinement,
        branch=branch,
        bootstrap_B=args.bootstrap_B,
        delta=args.delta,
        seed=args.seed,
    )

    refinement["branch_consistency"] = branch
    refinement["statistical_closeout"] = statistical["statistical_closeout"]

    json_dump(refinement, final_path)

    stats = statistical["statistical_closeout"]
    focus = stats["focused_090_vs_095"][0]

    print("\n" + "=" * 76)
    print("STEP 2 STATISTICAL CLOSE-OUT (VALIDATION USED FOR SELECTION)")
    print(
        "best validation lambda by aggregate median R: "
        f"{stats['best_validation_lambda_by_median_R']}"
    )
    print(
        "robust candidates (within delta AND equivalent to best): "
        f"{stats['robust_candidates_within_delta_and_equivalent_to_best']}"
    )
    print("\nFocused lambda=0.90 vs lambda=0.95 (validation):")
    print(f"  median ratio R_0.90/R_0.95 = {focus['ratio_A_over_B']:.4f}")
    print(
        f"  95% CI ratio               = "
        f"[{focus['ci95_ratio_lo']:.4f}, {focus['ci95_ratio_hi']:.4f}]"
    )
    print(
        f"  90% equivalence CI ratio   = "
        f"[{focus['ci90_ratio_lo']:.4f}, {focus['ci90_ratio_hi']:.4f}]"
    )
    print(f"  practically equivalent?    = {focus['equivalent_90pct_CI']}")
    print(f"  superiority conclusion     = {focus['superiority_95pct_CI']}")

    print("\nOutputs:")
    for name in [
        "lambda_closeout_results.json",
        "step1_refinement_summary.csv",
        "branch_consistency_summary.csv",
        "pairwise_equivalence_validation.csv",
        "pairwise_confirmation_test.csv",
        "lambda_090_vs_095_summary.csv",
        "01_branch_consistency_gamma_profile.png",
        "02_lambda_cliff_refinement.png",
        "03_lambda_refinement_gamma.png",
        "04_validation_equivalence_matrix.png",
    ]:
        print(f"  {outdir / name}")

    print("=" * 76)


if __name__ == "__main__":
    main()
