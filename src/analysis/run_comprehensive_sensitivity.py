from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parents[1]
DEFAULT_PROJECT = WORK_ROOT
DEFAULT_FAMILY = WORK_ROOT / "data" / "systems.npz"
DEFAULT_OUT = WORK_ROOT / "data" / "sensitivity_results"

MODELS = {
    "A_p1_lambda1": (1, 1.0, 2.5),
    "B_p1_lambda095": (1, 0.95, 1.0),
    "C_p5_lambda1": (5, 1.0, 3.5),
    "D_p5_lambda095": (5, 0.95, 1.0),
}

MODEL_LABELS = {
    "A_p1_lambda1": r"p=1, lambda=1",
    "B_p1_lambda095": r"p=1, lambda=0.95",
    "C_p5_lambda1": r"p=5, lambda=1",
    "D_p5_lambda095": r"p=5, lambda=0.95",
}

CONDITIONS = {
    "baseline": {},
    "div_strict": {
        "div_rho_factor": 50.0,
        "div_spec_radius": 1.1,
        "div_spec_patience": 25,
    },
    "div_lenient": {
        "div_rho_factor": 200.0,
        "div_spec_radius": 1.3,
        "div_spec_patience": 100,
    },
    "p0_1e4": {"p0_init": 1e4},
    "p0_1e8": {"p0_init": 1e8},
    "r0_0p1": {"r0_value": 0.1},
    "r0_10": {"r0_value": 10.0},
}

HORIZONS = (6000, 9000, 12000, 16000)
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40)
CAPS = (2.0, 5.0, 10.0, 20.0)
EPS = 1e-12
BOOTSTRAP_SEED = 20260926

_FAMILY: dict[str, list[dict[str, Any]]] | None = None
_CONFIG = None
_TRAJ_SEED = None
_SIMULATE = None
_MISG_RUN = None


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict_json(data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _init_worker(project: str, family_path: str) -> None:
    global _FAMILY, _CONFIG, _TRAJ_SEED, _SIMULATE, _MISG_RUN
    project_src = str(Path(project) / "src" / "core")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from config import Config, traj_seed
    from systems import load_family, simulate
    from misg_sensitivity_estimator import misg_run

    _CONFIG = Config
    _TRAJ_SEED = traj_seed
    _SIMULATE = simulate
    _MISG_RUN = misg_run
    _FAMILY = load_family(family_path)


def _extend_original_trajectory(
    system: dict[str, Any], rng: np.random.Generator, base_n: int, final_n: int
) -> dict[str, Any]:
    assert _SIMULATE is not None
    trajectory = _SIMULATE(system, rng, base_n)
    if final_n == base_n:
        return trajectory
    if final_n < base_n:
        raise ValueError("final_n must not be shorter than base_n")

    extra = final_n - base_n
    u_extra = rng.standard_normal(extra)
    v_extra = rng.normal(0.0, float(system["sigma_v"]), extra)
    n = int(system["n"])
    d = int(system["d"])
    A = np.asarray(system["A"])
    B = np.asarray(system["B"])
    f = np.asarray(system["f"])

    u = np.zeros(final_n + 1)
    v = np.zeros(final_n + 1)
    y = np.zeros(final_n + 1)
    x = np.zeros((final_n + 2, n))
    u[: base_n + 1] = np.asarray(trajectory["u"])
    v[: base_n + 1] = np.asarray(trajectory["v"])
    y[: base_n + 1] = np.asarray(trajectory["y"])
    x[: base_n + 2] = np.asarray(trajectory["x"])
    u[base_n + 1 :] = u_extra
    v[base_n + 1 :] = v_extra

    for k in range(base_n + 1, final_n + 1):
        y[k] = x[k, 0] + v[k]
        delayed = np.zeros(n) if k - d <= 0 else x[k - d]
        x[k + 1] = A @ x[k] + B @ delayed + f * u[k]
    return {"u": u, "v": v, "y": y, "x": x, "N": final_n}


def _relative_errors(rho_hat: np.ndarray, truth: np.ndarray, n: int) -> np.ndarray:
    estimate = np.asarray(rho_hat[:n], dtype=float)
    denominator = float(np.linalg.norm(truth))
    return np.linalg.norm(estimate - truth[None, :], axis=1) / denominator


def _raw_risk(rho_hat: np.ndarray, truth: np.ndarray, n: int) -> float:
    estimate = np.asarray(rho_hat[:n], dtype=float)
    if estimate.shape[0] < n or not np.all(np.isfinite(estimate)):
        return float("nan")
    denominator = float(np.linalg.norm(truth) ** 2)
    errors = np.sum((estimate[n // 2 : n] - truth[None, :]) ** 2, axis=1)
    return float(np.mean(errors) / denominator)


def _first_hit(errors: np.ndarray, threshold: float) -> float | None:
    index = np.flatnonzero(np.isfinite(errors) & (errors <= threshold))
    return None if index.size == 0 else float(index[0] + 1)


def _task(task: tuple[str, int, int, str, str]) -> dict[str, Any]:
    dataset, system_index, trajectory_index, model_key, condition = task
    assert _FAMILY is not None
    assert _CONFIG is not None and _TRAJ_SEED is not None
    assert _SIMULATE is not None and _MISG_RUN is not None

    system = _FAMILY["test"][system_index]
    p, lambda_value, gamma = MODELS[model_key]
    condition_values = CONDITIONS.get(condition, {})
    cfg_values = {
        key: value for key, value in condition_values.items() if key != "r0_value"
    }
    final_n = 16000 if condition == "baseline" and dataset == "test" else 12000
    cfg = replace(_CONFIG(), N=final_n, **cfg_values)
    rng = np.random.default_rng(_TRAJ_SEED(cfg, int(system["m"]), trajectory_index))
    if condition == "baseline" and dataset == "test":
        trajectory = _extend_original_trajectory(system, rng, 12000, final_n)
    else:
        trajectory = _SIMULATE(system, rng, final_n)

    result = _MISG_RUN(
        system,
        trajectory,
        np.ones(p),
        gamma,
        "cumulative" if math.isclose(lambda_value, 1.0) else "forgetting",
        "observer",
        cfg,
        lam_g=lambda_value,
        r0_value=float(condition_values.get("r0_value", 1.0)),
    )
    rho_hat = np.asarray(result["rho_hat_traj"])
    truth = np.asarray(system["rho_star"])
    diverged_at = result.get("diverged_at")
    output: dict[str, Any] = {
        "dataset": dataset,
        "condition": condition,
        "system_index": system_index,
        "m": int(system["m"]),
        "d": int(system["d"]),
        "snr_db": float(system["snr_db"]),
        "rho_F": float(system["rho_F"]),
        "B_norm": float(np.linalg.norm(system["B"])),
        "trajectory_index": trajectory_index,
        "model": model_key,
        "status": str(result.get("status", "unknown")),
        "diverged_at": diverged_at,
        "div_reason": result.get("div_reason"),
    }
    horizons = HORIZONS if condition == "baseline" and dataset == "test" else (12000,)
    for horizon in horizons:
        diverged = diverged_at is not None and int(diverged_at) <= horizon
        raw = float("nan") if diverged else _raw_risk(rho_hat, truth, horizon)
        output[f"raw_R_{horizon}"] = raw
        output[f"diverged_{horizon}"] = bool(diverged or not math.isfinite(raw))
        errors = _relative_errors(rho_hat, truth, horizon)
        for threshold in THRESHOLDS:
            key = f"hit_{int(round(threshold * 100)):02d}_{horizon}"
            output[key] = None if diverged else _first_hit(errors, threshold)
    return output


def _paired_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.exp(np.median(np.log((a + EPS) / (b + EPS)))))


def _paired_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    confidence: float,
    B: int,
    seed: int,
) -> tuple[float, float, float]:
    log_ratio = np.log((np.asarray(a) + EPS) / (np.asarray(b) + EPS))
    point = float(np.exp(np.median(log_ratio)))
    rng = np.random.default_rng(seed)
    statistics = np.empty(B)
    n = log_ratio.size
    chunk = 1000
    start = 0
    while start < B:
        stop = min(B, start + chunk)
        indices = rng.integers(0, n, size=(stop - start, n))
        statistics[start:stop] = np.median(log_ratio[indices], axis=1)
        start = stop
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(statistics, [tail, 1.0 - tail])
    return point, float(np.exp(lo)), float(np.exp(hi))


def _system_arrays(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    condition: str,
    horizon: int,
    cap: float,
    trajectory_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset and row["condition"] == condition
    ]
    if trajectory_limit is not None:
        ids = sorted({int(row["trajectory_index"]) for row in selected})[
            :trajectory_limit
        ]
        selected = [row for row in selected if int(row["trajectory_index"]) in ids]
    systems = sorted({int(row["system_index"]) for row in selected})
    arrays: dict[str, np.ndarray] = {}
    for model in MODELS:
        values = []
        for system_index in systems:
            candidates = [
                row
                for row in selected
                if row["model"] == model and int(row["system_index"]) == system_index
            ]
            if not candidates:
                continue
            trajectory_values = []
            for row in candidates:
                raw = row.get(f"raw_R_{horizon}")
                if (
                    raw is None
                    or not math.isfinite(float(raw))
                    or row.get(f"diverged_{horizon}")
                ):
                    trajectory_values.append(cap)
                else:
                    trajectory_values.append(min(float(raw), cap))
            values.append(float(np.mean(trajectory_values)))
        if values:
            arrays[model] = np.asarray(values)
    metadata = []
    for system_index in systems:
        row = next(row for row in selected if int(row["system_index"]) == system_index)
        metadata.append(
            {
                "system_index": system_index,
                "m": int(row["m"]),
                "d": int(row["d"]),
                "snr_db": float(row["snr_db"]),
                "rho_F": float(row["rho_F"]),
                "B_norm": float(row["B_norm"]),
            }
        )
    return arrays, metadata


def _comparison_rows(
    arrays: dict[str, np.ndarray],
    context: dict[str, Any],
    *,
    B: int = 10000,
) -> list[dict[str, Any]]:
    comparisons = [
        ("primary_p1_over_p5_lambda095", "B_p1_lambda095", "D_p5_lambda095", 0.90),
        ("p1_over_p5_lambda1", "A_p1_lambda1", "C_p5_lambda1", 0.95),
        ("forgetting_effect_p1", "B_p1_lambda095", "A_p1_lambda1", 0.95),
        ("forgetting_effect_p5", "D_p5_lambda095", "C_p5_lambda1", 0.95),
    ]
    output = []
    for index, (name, numerator, denominator, confidence) in enumerate(comparisons):
        if numerator not in arrays or denominator not in arrays:
            continue
        point, lo, hi = _paired_ci(
            arrays[numerator],
            arrays[denominator],
            confidence=confidence,
            B=B,
            seed=BOOTSTRAP_SEED + index,
        )
        output.append(
            {
                **context,
                "comparison": name,
                "n_systems": int(arrays[numerator].size),
                "ratio": point,
                "confidence_level": confidence,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return output


def _threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        r for r in rows if r["dataset"] == "test" and r["condition"] == "baseline"
    ]
    output = []
    for threshold in THRESHOLDS:
        threshold_key = int(round(threshold * 100))
        for model in MODELS:
            model_rows = [r for r in selected if r["model"] == model]
            failures = 0
            systems = sorted({int(r["system_index"]) for r in model_rows})
            for system_index in systems:
                trajectories = [
                    r for r in model_rows if int(r["system_index"]) == system_index
                ]
                hits = [r.get(f"hit_{threshold_key:02d}_12000") for r in trajectories]
                if all(value is None for value in hits):
                    failures += 1
            output.append(
                {
                    "threshold": threshold,
                    "model": model,
                    "n_systems": len(systems),
                    "n_failed": failures,
                    "failure_rate": failures / len(systems),
                }
            )
    return output


def _bootstrap_convergence(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    a = arrays["B_p1_lambda095"]
    b = arrays["D_p5_lambda095"]
    log_ratio = np.log((a + EPS) / (b + EPS))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 100)
    max_b = 50000
    statistics = np.empty(max_b)
    for start in range(0, max_b, 1000):
        stop = min(max_b, start + 1000)
        indices = rng.integers(0, log_ratio.size, size=(stop - start, log_ratio.size))
        statistics[start:stop] = np.median(log_ratio[indices], axis=1)
    output = []
    for B in (1000, 5000, 10000, 50000):
        lo, hi = np.quantile(statistics[:B], [0.05, 0.95])
        output.append(
            {
                "bootstrap_resamples": B,
                "ratio": float(np.exp(np.median(log_ratio))),
                "ci90_lo": float(np.exp(lo)),
                "ci90_hi": float(np.exp(hi)),
                "ci_width": float(np.exp(hi) - np.exp(lo)),
            }
        )
    return output


def _sample_size_stability(
    arrays: dict[str, np.ndarray], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    log_ratio = np.log(
        (arrays["B_p1_lambda095"] + EPS) / (arrays["D_p5_lambda095"] + EPS)
    )
    delays = np.asarray([row["d"] for row in metadata])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 200)
    output = []
    for per_delay in (10, 20, 34, 50, 67):
        draws = 1 if per_delay == 67 else 2000
        estimates = np.empty(draws)
        for replicate in range(draws):
            indices = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(delays == delay), size=per_delay, replace=False
                    )
                    for delay in sorted(set(delays.tolist()))
                ]
            )
            estimates[replicate] = float(np.exp(np.median(log_ratio[indices])))
        lo, hi = (
            np.quantile(estimates, [0.05, 0.95])
            if draws > 1
            else (estimates[0], estimates[0])
        )
        within_two = np.mean((estimates >= 1 / 1.02) & (estimates <= 1.02))
        output.append(
            {
                "n_systems": per_delay * 3,
                "balanced_per_delay": per_delay,
                "subsamples": draws,
                "median_ratio_across_subsamples": float(np.median(estimates)),
                "p05_ratio": float(lo),
                "p95_ratio": float(hi),
                "proportion_point_estimate_within_2pct": float(within_two),
            }
        )
    return output


def _heterogeneity_rows(
    arrays: dict[str, np.ndarray], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    meta = {
        key: np.asarray([row[key] for row in metadata])
        for key in ("d", "snr_db", "rho_F", "B_norm")
    }
    groups: list[tuple[str, str, np.ndarray]] = []
    for delay in (2, 3, 4):
        groups.append(("delay", str(delay), meta["d"] == delay))
    snr_edges = (5.0, 35.0 / 3.0, 55.0 / 3.0, 25.000001)
    for index, label in enumerate(("low", "middle", "high")):
        groups.append(
            (
                "snr_db",
                label,
                (meta["snr_db"] >= snr_edges[index])
                & (meta["snr_db"] < snr_edges[index + 1]),
            )
        )
    for variable in ("rho_F", "B_norm"):
        q1, q2 = np.quantile(meta[variable], [1 / 3, 2 / 3])
        masks = (
            meta[variable] <= q1,
            (meta[variable] > q1) & (meta[variable] <= q2),
            meta[variable] > q2,
        )
        for label, mask in zip(("low", "middle", "high"), masks):
            groups.append((variable, label, mask))

    output = []
    for index, (variable, label, mask) in enumerate(groups):
        point, lo, hi = _paired_ci(
            arrays["B_p1_lambda095"][mask],
            arrays["D_p5_lambda095"][mask],
            confidence=0.90,
            B=10000,
            seed=BOOTSTRAP_SEED + 300 + index,
        )
        output.append(
            {
                "stratification": variable,
                "stratum": label,
                "n_systems": int(np.sum(mask)),
                "ratio": point,
                "ci90_lo": lo,
                "ci90_hi": hi,
                "contained_5pct": bool(lo >= 1 / 1.05 and hi <= 1.05),
                "contained_3pct": bool(lo >= 1 / 1.03 and hi <= 1.03),
                "contained_2pct": bool(lo >= 1 / 1.02 and hi <= 1.02),
            }
        )
    return output


def _make_tasks() -> list[tuple[str, int, int, str, str]]:
    tasks: list[tuple[str, int, int, str, str]] = []
    for system_index in range(201):
        for trajectory_index in (3, 4):
            for model in MODELS:
                for condition in ("baseline", "div_strict", "div_lenient"):
                    tasks.append(
                        ("test", system_index, trajectory_index, model, condition)
                    )
            for model in ("B_p1_lambda095", "D_p5_lambda095"):
                for condition in ("p0_1e4", "p0_1e8", "r0_0p1", "r0_10"):
                    tasks.append(
                        ("test", system_index, trajectory_index, model, condition)
                    )
        for trajectory_index in (100, 101, 102, 103, 104):
            for model in MODELS:
                tasks.append(
                    ("confirmation", system_index, trajectory_index, model, "baseline")
                )
    return tasks


def _run_tasks(args: argparse.Namespace, outdir: Path) -> list[dict[str, Any]]:
    checkpoint = outdir / "trajectory_level_results.jsonl"
    completed: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    if checkpoint.exists() and not args.restart:
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (
                row["dataset"],
                int(row["system_index"]),
                int(row["trajectory_index"]),
                row["model"],
                row["condition"],
            )
            completed[key] = row
    tasks = [task for task in _make_tasks() if task not in completed]
    print(f"Completed/resumed: {len(completed)}; remaining: {len(tasks)}", flush=True)
    if tasks:
        outdir.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as handle:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(str(args.project), str(args.family)),
            ) as executor:
                future_map = {executor.submit(_task, task): task for task in tasks}
                for count, future in enumerate(as_completed(future_map), start=1):
                    row = future.result()
                    key = future_map[future]
                    completed[key] = row
                    handle.write(
                        json.dumps(_strict_json(row), ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    if count % 100 == 0 or count == len(tasks):
                        print(
                            f"Completed {count}/{len(tasks)} new trajectory evaluations",
                            flush=True,
                        )
    return list(completed.values())


def _summarize(rows: list[dict[str, Any]], outdir: Path) -> dict[str, Any]:
    comparison_rows: list[dict[str, Any]] = []

    for cap in CAPS:
        arrays, metadata = _system_arrays(
            rows, dataset="test", condition="baseline", horizon=12000, cap=cap
        )
        comparison_rows.extend(
            _comparison_rows(arrays, {"analysis": "risk_cap", "setting": cap})
        )
    for horizon in HORIZONS:
        arrays, metadata = _system_arrays(
            rows, dataset="test", condition="baseline", horizon=horizon, cap=10.0
        )
        comparison_rows.extend(
            _comparison_rows(
                arrays, {"analysis": "trajectory_horizon", "setting": horizon}
            )
        )

    for condition in ("div_strict", "baseline", "div_lenient"):
        arrays, _ = _system_arrays(
            rows, dataset="test", condition=condition, horizon=12000, cap=10.0
        )
        comparison_rows.extend(
            _comparison_rows(
                arrays, {"analysis": "divergence_rule", "setting": condition}
            )
        )

    for condition in ("p0_1e4", "baseline", "p0_1e8", "r0_0p1", "r0_10"):
        arrays, _ = _system_arrays(
            rows, dataset="test", condition=condition, horizon=12000, cap=10.0
        )
        comparison_rows.extend(
            _comparison_rows(
                arrays, {"analysis": "initialization", "setting": condition}
            )
        )

    trajectory_count_rows = []
    for count in (1, 2, 3, 5):
        arrays, _ = _system_arrays(
            rows,
            dataset="confirmation",
            condition="baseline",
            horizon=12000,
            cap=10.0,
            trajectory_limit=count,
        )
        current = _comparison_rows(
            arrays, {"analysis": "confirmation_trajectory_count", "setting": count}
        )
        comparison_rows.extend(current)
        trajectory_count_rows.extend(current)

    divergence_rows = []
    for condition in ("div_strict", "baseline", "div_lenient"):
        for model in MODELS:
            selected = [
                row
                for row in rows
                if row["dataset"] == "test"
                and row["condition"] == condition
                and row["model"] == model
            ]
            count = sum(bool(row.get("diverged_12000")) for row in selected)
            reasons: dict[str, int] = {}
            for row in selected:
                if row.get("diverged_12000"):
                    reason = str(row.get("div_reason") or "nonfinite_risk")
                    reasons[reason] = reasons.get(reason, 0) + 1
            divergence_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "n_trajectories": len(selected),
                    "divergences": count,
                    "divergence_rate": count / len(selected),
                    "reasons": json.dumps(reasons, sort_keys=True),
                }
            )

    threshold_rows = _threshold_rows(rows)
    baseline_arrays, baseline_meta = _system_arrays(
        rows, dataset="test", condition="baseline", horizon=12000, cap=10.0
    )
    bootstrap_rows = _bootstrap_convergence(baseline_arrays)
    sample_rows = _sample_size_stability(baseline_arrays, baseline_meta)
    heterogeneity_rows = _heterogeneity_rows(baseline_arrays, baseline_meta)

    _write_csv(comparison_rows, outdir / "comparison_sensitivity.csv")
    _write_csv(divergence_rows, outdir / "divergence_sensitivity.csv")
    _write_csv(threshold_rows, outdir / "threshold_sensitivity.csv")
    _write_csv(bootstrap_rows, outdir / "bootstrap_resample_sensitivity.csv")
    _write_csv(sample_rows, outdir / "sample_size_sensitivity.csv")
    _write_csv(heterogeneity_rows, outdir / "heterogeneity_sensitivity.csv")

    summary = {
        "meta": {
            "analysis": "comprehensive_numeric_sensitivity",
            "independent_unit": "system",
            "n_test_systems": 201,
            "test_trajectory_indices": [3, 4],
            "confirmation_trajectory_indices": [100, 101, 102, 103, 104],
            "frozen_models": MODELS,
            "horizons": HORIZONS,
            "thresholds": THRESHOLDS,
            "risk_caps": CAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "comparison_sensitivity": comparison_rows,
        "divergence_sensitivity": divergence_rows,
        "threshold_sensitivity": threshold_rows,
        "bootstrap_resample_sensitivity": bootstrap_rows,
        "sample_size_sensitivity": sample_rows,
        "heterogeneity_sensitivity": heterogeneity_rows,
    }
    _write_json(summary, outdir / "comprehensive_sensitivity_results.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1))
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project = args.project.expanduser().resolve()
    args.family = args.family.expanduser().resolve()
    if args.self_test:
        _init_worker(str(args.project), str(args.family))
        first = _task(("test", 0, 3, "B_p1_lambda095", "baseline"))
        second = _task(("test", 0, 4, "B_p1_lambda095", "baseline"))
        observed = (float(first["raw_R_12000"]) + float(second["raw_R_12000"])) / 2.0
        expected = 0.04084023481917347
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-14):
            raise SystemExit(
                f"Self-test failed: observed {observed}, expected {expected}"
            )
        print(f"Self-test passed: first test-system R={observed:.17g}")
        return
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        checkpoint = outdir / "trajectory_level_results.jsonl"
        rows = [
            json.loads(line)
            for line in checkpoint.read_text(encoding="utf-8").splitlines()
        ]
    else:
        rows = _run_tasks(args, outdir)
    summary = _summarize(rows, outdir)
    print(json.dumps(_strict_json(summary["meta"]), indent=2), flush=True)
    print(f"Results written to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
