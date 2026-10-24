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
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "data" / "record_length_results"
SOURCE = PROJECT / "src" / "core"
FAMILY_PATH = PROJECT / "data" / "systems.npz"

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from config import Config, repro_seed, traj_seed
from estimators import misg_run
from systems import load_family, output_variance, simulate


N_MAIN = 12_000
N_LONG = 100_000
EPS = 1e-12
BOOTSTRAP_SEED = 20261006
P_VALUES = (1, 5)
LAMBDAS = (1.0, 0.99, 0.95)
SELECTED_GAMMA = {
    (1, 1.0): 2.5,
    (5, 1.0): 3.5,
    (1, 0.99): 2.5,
    (5, 0.99): 3.5,
    (1, 0.95): 1.0,
    (5, 0.95): 1.0,
}
GAMMA_GRID = (
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
)
CHECKPOINTS = tuple(
    sorted(
        set(
            [
                1,
                10,
                30,
                100,
                300,
                1_000,
                2_000,
                4_000,
                6_000,
                8_000,
                12_000,
                20_000,
                40_000,
                60_000,
                80_000,
                100_000,
            ]
        )
    )
)
PREFIX_HORIZONS = (
    2_000,
    4_000,
    6_000,
    8_000,
    12_000,
    20_000,
    40_000,
    60_000,
    80_000,
    100_000,
)

_FAMILY: dict[str, list[dict[str, Any]]] | None = None


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(strict_json(data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_risk(estimates: np.ndarray, truth: np.ndarray, start: int, stop: int) -> float:
    block = np.asarray(estimates[start:stop], dtype=float)
    if block.shape[0] != stop - start or not np.all(np.isfinite(block)):
        return float("nan")
    denominator = float(np.linalg.norm(truth) ** 2)
    return float(np.mean(np.sum((block - truth[None, :]) ** 2, axis=1)) / denominator)


def capped(value: float, cap: float = 10.0) -> float:
    return cap if not math.isfinite(float(value)) else min(float(value), cap)


def bootstrap_interval(
    values: np.ndarray, confidence: float = 0.95, B: int = 10_000
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    statistics = np.empty(B)
    for start in range(0, B, 1000):
        stop = min(B, start + 1000)
        idx = rng.integers(0, sample.size, size=(stop - start, sample.size))
        statistics[start:stop] = np.median(sample[idx], axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(statistics, [tail, 1.0 - tail])
    return float(lo), float(hi)


def bootstrap_log_ratio(
    a: np.ndarray, b: np.ndarray, confidence: float = 0.90, B: int = 10_000
) -> tuple[float, float, float]:
    log_ratio = np.log((np.asarray(a) + EPS) / (np.asarray(b) + EPS))
    point = float(np.exp(np.median(log_ratio)))
    lo, hi = bootstrap_interval(log_ratio, confidence=confidence, B=B)
    return point, float(np.exp(lo)), float(np.exp(hi))


def extend_original_trajectory(
    system: dict[str, Any], trajectory_index: int, final_n: int
) -> dict[str, Any]:
    cfg = replace(Config(), N=final_n)
    rng = np.random.default_rng(traj_seed(cfg, int(system["m"]), trajectory_index))
    trajectory = simulate(system, rng, N_MAIN)
    if final_n == N_MAIN:
        return trajectory
    extra = final_n - N_MAIN
    u_extra = rng.standard_normal(extra)
    v_extra = rng.normal(0.0, float(system["sigma_v"]), extra)
    n, d = int(system["n"]), int(system["d"])
    A, B, f = np.asarray(system["A"]), np.asarray(system["B"]), np.asarray(system["f"])
    u = np.zeros(final_n + 1)
    v = np.zeros(final_n + 1)
    y = np.zeros(final_n + 1)
    x = np.zeros((final_n + 2, n))
    u[: N_MAIN + 1], v[: N_MAIN + 1] = trajectory["u"], trajectory["v"]
    y[: N_MAIN + 1], x[: N_MAIN + 2] = trajectory["y"], trajectory["x"]
    u[N_MAIN + 1 :], v[N_MAIN + 1 :] = u_extra, v_extra
    for k in range(N_MAIN + 1, final_n + 1):
        y[k] = x[k, 0] + v[k]
        delayed = np.zeros(n) if k - d <= 0 else x[k - d]
        x[k + 1] = A @ x[k] + B @ delayed + f * u[k]
    return {"u": u, "v": v, "y": y, "x": x, "N": final_n}


def init_family() -> None:
    global _FAMILY
    _FAMILY = load_family(FAMILY_PATH)


def split_features(system: dict[str, Any]) -> dict[str, float | int]:
    A, B, f = np.asarray(system["A"]), np.asarray(system["B"]), np.asarray(system["f"])
    a1, a2 = float(A[1, 0]), float(A[1, 1])
    f1, f2 = float(f[0]), float(f[1])
    h = f1 / f2
    g = a2 + a1 * h
    L = float(B[1, 1] - B[0, 0] - g * B[0, 1] + h * B[1, 0])
    return {
        "m": int(system["m"]),
        "d": int(system["d"]),
        "snr_db": float(system["snr_db"]),
        "rho_F": float(system["rho_F"]),
        "B_norm": float(np.linalg.norm(B)),
        "rho_norm": float(np.linalg.norm(system["rho_star"])),
        "abs_L": abs(L),
        "sigma_v": float(system["sigma_v"]),
    }


def trajectory_swap_task(task: tuple[str, int, tuple[int, int]]) -> dict[str, Any]:
    split, index, trajectory_ids = task
    assert _FAMILY is not None
    system = _FAMILY[split][index]
    cfg = replace(Config(), N=N_MAIN)
    risks = []
    statuses = []
    for trajectory_index in trajectory_ids:
        rng = np.random.default_rng(traj_seed(cfg, int(system["m"]), trajectory_index))
        trajectory = simulate(system, rng, N_MAIN)
        result = misg_run(
            system,
            trajectory,
            np.ones(5),
            1.0,
            "forgetting",
            "observer",
            cfg,
            lam_g=0.95,
        )
        risk = raw_risk(
            np.asarray(result["rho_hat_traj"]),
            np.asarray(system["rho_star"]),
            N_MAIN // 2,
            N_MAIN,
        )
        risks.append(capped(risk))
        statuses.append(str(result["status"]))
    return {
        "split": split,
        "system_index": index,
        "m": int(system["m"]),
        "trajectory_ids": "-".join(str(x) for x in trajectory_ids),
        "system_risk": float(np.mean(risks)),
        "statuses": ";".join(statuses),
    }


def run_split_diagnostic(workers: int) -> dict[str, Any]:
    family = load_family(FAMILY_PATH)
    rows = []
    for split in ("val", "test"):
        for system in family[split]:
            rows.append({"split": split, **split_features(system)})
    write_csv(rows, RESULTS / "split_features.csv")
    tests = []
    for feature in ("snr_db", "rho_F", "B_norm", "rho_norm", "abs_L", "sigma_v"):
        val = np.asarray([r[feature] for r in rows if r["split"] == "val"], dtype=float)
        test = np.asarray(
            [r[feature] for r in rows if r["split"] == "test"], dtype=float
        )
        ks = stats.ks_2samp(val, test, alternative="two-sided", method="auto")
        tests.append(
            {
                "feature": feature,
                "val_median": float(np.median(val)),
                "test_median": float(np.median(test)),
                "median_ratio_test_over_val": float(np.median(test) / np.median(val)),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
            }
        )
    write_csv(tests, RESULTS / "split_distribution_tests.csv")

    task_groups = {
        "val_original": [("val", i, (1, 2)) for i in range(len(family["val"]))],
        "val_swapped": [("val", i, (3, 4)) for i in range(len(family["val"]))],
        "test_original": [("test", i, (3, 4)) for i in range(len(family["test"]))],
        "test_swapped": [("test", i, (1, 2)) for i in range(len(family["test"]))],
    }
    swap_rows: list[dict[str, Any]] = []
    all_tasks = [
        (label, task) for label, tasks in task_groups.items() for task in tasks
    ]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_family) as executor:
        future_map = {
            executor.submit(trajectory_swap_task, task): label
            for label, task in all_tasks
        }
        for count, future in enumerate(as_completed(future_map), start=1):
            row = future.result()
            row["condition"] = future_map[future]
            swap_rows.append(row)
            if count % 100 == 0 or count == len(all_tasks):
                print(f"split diagnostic {count}/{len(all_tasks)}", flush=True)
    write_csv(swap_rows, RESULTS / "trajectory_swap_system_risks.csv")
    summaries = []
    for label in task_groups:
        values = np.asarray(
            [r["system_risk"] for r in swap_rows if r["condition"] == label],
            dtype=float,
        )
        lo, hi = bootstrap_interval(values)
        summaries.append(
            {
                "condition": label,
                "n_systems": values.size,
                "median_R": float(np.median(values)),
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    write_csv(summaries, RESULTS / "trajectory_swap_summary.csv")
    output = {"distribution_tests": tests, "trajectory_swap": summaries}
    write_json(output, RESULTS / "split_diagnostic.json")
    return output


def example_system_2017() -> dict[str, Any]:
    A = np.array([[0.0, 1.0], [-0.01, -0.22]])
    B = np.array([[0.19, -0.08], [0.16, -0.12]])
    f = np.array([0.8, -0.8])
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": 2,
        "n": 2,
        "c": np.array([1.0, 0.0]),
        "sigma_v": 0.30,
        "rho_star": np.array([-0.01, -0.22, 0.19, -0.08, 0.16, -0.12, 0.8, -0.8]),
    }


def anchor_task(task: tuple[int, int, str, float]) -> dict[str, Any]:
    seed_index, p, regime, gamma = task
    system = example_system_2017()
    cfg = replace(Config(), N=3000)
    rng = np.random.default_rng(repro_seed(cfg, seed_index))
    trajectory = simulate(system, rng, 3000)
    if regime == "A_newest_lambda1":
        r_window, mode, lam = "newest", "cumulative", 1.0
    elif regime in {"B_full_lambda1", "C_full_lambda1_tuned"}:
        r_window, mode, lam = "full", "cumulative", 1.0
    elif regime == "D_full_lambda095_tuned":
        r_window, mode, lam = "full", "forgetting", 0.95
    else:
        raise ValueError(regime)
    result = misg_run(
        system,
        trajectory,
        np.ones(p),
        gamma,
        mode,
        "observer",
        cfg,
        lam_g=lam,
        r_window=r_window,
    )
    estimates = np.asarray(result["rho_hat_traj"])
    risk = raw_risk(estimates, np.asarray(system["rho_star"]), 1500, 3000)
    final_delta = (
        float(
            np.linalg.norm(estimates[-1] - system["rho_star"])
            / np.linalg.norm(system["rho_star"])
            * 100.0
        )
        if result["status"] == "ok"
        else float("nan")
    )
    return {
        "seed_index": seed_index,
        "p": p,
        "regime": regime,
        "gamma": gamma,
        "R": capped(risk),
        "delta_3000": final_delta,
        "status": result["status"],
    }


def parallel_rows(
    tasks: Iterable[tuple], function: Any, workers: int, label: str
) -> list[dict[str, Any]]:
    task_list = list(tasks)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, task) for task in task_list]
        for count, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if count % 200 == 0 or count == len(futures):
                print(f"{label} {count}/{len(futures)}", flush=True)
    return rows


def run_anchor(workers: int) -> dict[str, Any]:
    tune_tasks = [
        (seed, p, regime, gamma)
        for seed in range(50)
        for p in P_VALUES
        for regime in ("C_full_lambda1_tuned", "D_full_lambda095_tuned")
        for gamma in GAMMA_GRID
    ]
    tune_rows = parallel_rows(tune_tasks, anchor_task, workers, "anchor tuning")
    selected: dict[tuple[int, str], float] = {}
    tuning_summary = []
    for p in P_VALUES:
        for regime in ("C_full_lambda1_tuned", "D_full_lambda095_tuned"):
            candidates = []
            for gamma in GAMMA_GRID:
                values = [
                    r["R"]
                    for r in tune_rows
                    if r["p"] == p
                    and r["regime"] == regime
                    and math.isclose(r["gamma"], gamma)
                ]
                candidates.append((float(np.median(values)), gamma))
            median_R, gamma = min(candidates)
            selected[(p, regime)] = gamma
            tuning_summary.append(
                {
                    "p": p,
                    "regime": regime,
                    "selected_gamma": gamma,
                    "training_median_R": median_R,
                }
            )
    write_csv(tuning_summary, RESULTS / "anchor_gamma_selection.csv")

    evaluation_tasks = []
    for seed in range(50, 100):
        for p in P_VALUES:
            evaluation_tasks.append((seed, p, "A_newest_lambda1", 1.0))
            evaluation_tasks.append((seed, p, "B_full_lambda1", 1.0))
            for regime in ("C_full_lambda1_tuned", "D_full_lambda095_tuned"):
                evaluation_tasks.append((seed, p, regime, selected[(p, regime)]))
    evaluation = parallel_rows(
        evaluation_tasks, anchor_task, workers, "anchor evaluation"
    )
    write_csv(evaluation, RESULTS / "anchor_evaluation_rows.csv")
    summary = []
    for regime in (
        "A_newest_lambda1",
        "B_full_lambda1",
        "C_full_lambda1_tuned",
        "D_full_lambda095_tuned",
    ):
        for p in P_VALUES:
            rows = [r for r in evaluation if r["regime"] == regime and r["p"] == p]
            R = np.asarray([r["R"] for r in rows])
            delta = np.asarray(
                [
                    r["delta_3000"]
                    for r in rows
                    if r["delta_3000"] is not None
                    and math.isfinite(float(r["delta_3000"]))
                ]
            )
            summary.append(
                {
                    "regime": regime,
                    "p": p,
                    "gamma": rows[0]["gamma"],
                    "n_evaluation_trajectories": len(rows),
                    "median_R": float(np.median(R)),
                    "median_delta_3000_pct": float(np.median(delta)),
                    "q25_delta_3000_pct": float(np.quantile(delta, 0.25)),
                    "q75_delta_3000_pct": float(np.quantile(delta, 0.75)),
                    "n_diverged": sum(r["status"] != "ok" for r in rows),
                }
            )
    write_csv(summary, RESULTS / "anchor_summary.csv")
    output = {
        "output_variance": output_variance(
            example_system_2017()["A"],
            example_system_2017()["B"],
            example_system_2017()["f"],
            2,
        ),
        "tuning": tuning_summary,
        "evaluation": summary,
    }
    write_json(output, RESULTS / "anchor_results.json")
    return output


def long_task(task: tuple[int, int, int, float, float]) -> dict[str, Any]:
    system_index, trajectory_index, p, lam, gamma = task
    family = load_family(FAMILY_PATH)
    system = family["test"][system_index]
    cfg = replace(Config(), N=N_LONG)
    trajectory = extend_original_trajectory(system, trajectory_index, N_LONG)
    mode = "cumulative" if math.isclose(lam, 1.0) else "forgetting"
    result = misg_run(
        system,
        trajectory,
        np.ones(p),
        gamma,
        mode,
        "observer",
        cfg,
        lam_g=lam,
        r_window="full",
    )
    estimates = np.asarray(result["rho_hat_traj"])
    truth = np.asarray(system["rho_star"])
    denom = float(np.linalg.norm(truth))
    row: dict[str, Any] = {
        "system_index": system_index,
        "m": int(system["m"]),
        "d": int(system["d"]),
        "trajectory_index": trajectory_index,
        "p": p,
        "lambda": lam,
        "gamma": gamma,
        "status": result["status"],
        "diverged_at": result["diverged_at"],
    }
    for k in CHECKPOINTS:
        estimate = estimates[k - 1]
        row[f"delta_{k}"] = (
            float(np.linalg.norm(estimate - truth) / denom * 100.0)
            if np.all(np.isfinite(estimate))
            else None
        )
    for horizon in PREFIX_HORIZONS:
        row[f"R_prefix_{horizon}"] = capped(
            raw_risk(estimates, truth, horizon // 2, horizon)
        )
    row["R_first_2000"] = capped(raw_risk(estimates, truth, 0, 2000))
    row["R_1001_2000"] = capped(raw_risk(estimates, truth, 1000, 2000))
    return row


def run_resumable(
    tasks: list[tuple], function: Any, checkpoint: Path, workers: int, label: str
) -> list[dict[str, Any]]:
    completed: dict[tuple, dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (
                int(row["system_index"]),
                int(row["trajectory_index"]),
                int(row["p"]),
                float(row["lambda"]),
                float(row["gamma"]),
            )
            completed[key] = row
    remaining = [task for task in tasks if task not in completed]
    print(f"{label}: resumed {len(completed)}, remaining {len(remaining)}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if remaining:
        with checkpoint.open("a", encoding="utf-8") as handle:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(function, task): task for task in remaining
                }
                for count, future in enumerate(as_completed(future_map), start=1):
                    row = future.result()
                    completed[future_map[future]] = row
                    handle.write(
                        json.dumps(strict_json(row), ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    if count % 50 == 0 or count == len(remaining):
                        print(f"{label} {count}/{len(remaining)} new runs", flush=True)
    return list(completed.values())


def run_long_horizon(workers: int) -> dict[str, Any]:
    tasks = [
        (system_index, trajectory_index, p, lam, SELECTED_GAMMA[(p, lam)])
        for system_index in range(201)
        for trajectory_index in (3, 4)
        for p in P_VALUES
        for lam in LAMBDAS
    ]
    rows = run_resumable(
        tasks, long_task, RESULTS / "long_horizon_rows.jsonl", workers, "long horizon"
    )
    system_rows = []
    for system_index in range(201):
        for p in P_VALUES:
            for lam in LAMBDAS:
                selected = [
                    r
                    for r in rows
                    if int(r["system_index"]) == system_index
                    and int(r["p"]) == p
                    and math.isclose(float(r["lambda"]), lam)
                ]
                row: dict[str, Any] = {
                    "system_index": system_index,
                    "p": p,
                    "lambda": lam,
                    "gamma": SELECTED_GAMMA[(p, lam)],
                }
                for k in CHECKPOINTS:
                    values = [r.get(f"delta_{k}") for r in selected]

                    penalty_delta = math.sqrt(10.0) * 100.0
                    row[f"delta_{k}"] = float(
                        np.mean(
                            [penalty_delta if v is None else float(v) for v in values]
                        )
                    )
                for h in PREFIX_HORIZONS:
                    row[f"R_prefix_{h}"] = float(
                        np.mean([float(r[f"R_prefix_{h}"]) for r in selected])
                    )
                row["R_first_2000"] = float(
                    np.mean([float(r["R_first_2000"]) for r in selected])
                )
                row["R_1001_2000"] = float(
                    np.mean([float(r["R_1001_2000"]) for r in selected])
                )
                system_rows.append(row)
    write_csv(system_rows, RESULTS / "long_horizon_system_summary.csv")

    curves = []
    for p in P_VALUES:
        for lam in LAMBDAS:
            selected = [
                r for r in system_rows if r["p"] == p and math.isclose(r["lambda"], lam)
            ]
            for k in CHECKPOINTS:
                values = np.asarray(
                    [r[f"delta_{k}"] for r in selected if r[f"delta_{k}"] is not None],
                    dtype=float,
                )
                curves.append(
                    {
                        "p": p,
                        "lambda": lam,
                        "gamma": SELECTED_GAMMA[(p, lam)],
                        "k": k,
                        "n_systems": values.size,
                        "median_delta_pct": float(np.median(values)),
                        "q25_delta_pct": float(np.quantile(values, 0.25)),
                        "q75_delta_pct": float(np.quantile(values, 0.75)),
                    }
                )
    write_csv(curves, RESULTS / "long_horizon_curves.csv")

    comparisons = []
    for lam in LAMBDAS:
        p1 = [r for r in system_rows if r["p"] == 1 and math.isclose(r["lambda"], lam)]
        p5 = [r for r in system_rows if r["p"] == 5 and math.isclose(r["lambda"], lam)]
        for metric in ("R_first_2000", "R_1001_2000"):
            point, lo, hi = bootstrap_log_ratio(
                np.asarray([r[metric] for r in p1]), np.asarray([r[metric] for r in p5])
            )
            comparisons.append(
                {
                    "lambda": lam,
                    "metric": metric,
                    "horizon": 2000,
                    "ratio_p1_over_p5": point,
                    "ci90_lo": lo,
                    "ci90_hi": hi,
                }
            )
        for h in PREFIX_HORIZONS:
            key = f"R_prefix_{h}"
            point, lo, hi = bootstrap_log_ratio(
                np.asarray([r[key] for r in p1]), np.asarray([r[key] for r in p5])
            )
            comparisons.append(
                {
                    "lambda": lam,
                    "metric": "second_half_prefix_R",
                    "horizon": h,
                    "ratio_p1_over_p5": point,
                    "ci90_lo": lo,
                    "ci90_hi": hi,
                }
            )
    write_csv(comparisons, RESULTS / "horizon_metric_comparisons.csv")
    status_summary = []
    for p in P_VALUES:
        for lam in LAMBDAS:
            selected = [
                r
                for r in rows
                if int(r["p"]) == p and math.isclose(float(r["lambda"]), lam)
            ]
            status_summary.append(
                {
                    "p": p,
                    "lambda": lam,
                    "gamma": SELECTED_GAMMA[(p, lam)],
                    "n_trajectories": len(selected),
                    "n_diverged": sum(str(r["status"]) != "ok" for r in selected),
                    "n_systems_with_any_divergence": len(
                        {
                            int(r["system_index"])
                            for r in selected
                            if str(r["status"]) != "ok"
                        }
                    ),
                }
            )
    write_csv(status_summary, RESULTS / "long_horizon_status.csv")
    output = {"curves": curves, "comparisons": comparisons, "status": status_summary}
    write_json(output, RESULTS / "long_horizon_results.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("all", "diagnostic", "anchor", "long"), default="all"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(10, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"workers": args.workers, "phase": args.phase}
    if args.phase in {"all", "diagnostic"}:
        manifest["diagnostic"] = run_split_diagnostic(args.workers)
    if args.phase in {"all", "anchor"}:
        manifest["anchor"] = run_anchor(args.workers)
    if args.phase in {"all", "long"}:
        manifest["long"] = run_long_horizon(args.workers)
    write_json(manifest, RESULTS / f"run_manifest_{args.phase}.json")


if __name__ == "__main__":
    main()
