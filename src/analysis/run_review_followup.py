from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
VERSION_ROOT = HERE.parents[1]
OUT = VERSION_ROOT / "data" / "review_followup_results"
SOURCE_ROOT = VERSION_ROOT / "src" / "core"
FAMILY_PATH = VERSION_ROOT / "data" / "systems.npz"
FACTORIAL_PATH = VERSION_ROOT / "data" / "balanced_full_factorial_results.json"
LONG_ROWS = VERSION_ROOT / "data" / "long_horizon_rows.jsonl"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config import Config, traj_seed


N = 12_000
CAP = 10.0
SEED = 20261007
EPS = 1e-12


def load_family(path: Path) -> dict[str, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as archive:
        split_names = [str(value) for value in archive["split_names"]]
        item_splits = [str(value) for value in archive["item_splits"]]
        family: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
        for index, split in enumerate(item_splits):
            suffix = f"{index:06d}"
            family[split].append(
                {
                    "A": np.array(archive[f"A_{suffix}"], copy=True),
                    "B": np.array(archive[f"B_{suffix}"], copy=True),
                    "f": np.array(archive[f"f_{suffix}"], copy=True),
                    "rho_star": np.array(archive[f"rho_star_{suffix}"], copy=True),
                    "d": int(archive["d"][index]),
                    "n": int(archive["n"][index]),
                    "snr_db": float(archive["snr_db"][index]),
                    "sigma_v": float(archive["sigma_v"][index]),
                    "rho_F": float(archive["rho_F"][index]),
                    "n_reject": int(archive["n_reject"][index]),
                    "m": int(archive["m"][index]),
                    "split": split,
                }
            )
    return family


def build_F(A: np.ndarray, B: np.ndarray, d: int) -> np.ndarray:
    n = A.shape[0]
    F = np.zeros((n * (d + 1), n * (d + 1)))
    F[:n, :n] = A
    F[:n, n * d : n * (d + 1)] = B
    for block in range(1, d + 1):
        F[block * n : (block + 1) * n, (block - 1) * n : block * n] = np.eye(n)
    return F


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def unpack_rho(
    rho: np.ndarray, n: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d < n:
        raise ValueError("The frozen family requires d >= n")
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)
    A[-1] = rho[:n]
    B = rho[n : n + n * n].reshape(n, n).copy()
    f = rho[n + n * n :].copy()
    return A, B, f


def get_u(trajectory: dict[str, Any], k: int) -> float:
    return 0.0 if k <= 0 else float(trajectory["u"][k])


def get_y(trajectory: dict[str, Any], k: int) -> float:
    return 0.0 if k <= 0 else float(trajectory["y"][k])


def simulate(
    system: dict[str, Any], rng: np.random.Generator, N: int
) -> dict[str, Any]:
    n, d = int(system["n"]), int(system["d"])
    A, B, f = np.asarray(system["A"]), np.asarray(system["B"]), np.asarray(system["f"])
    u = np.zeros(N + 1)
    v = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x = np.zeros((N + 2, n))
    u[1:] = rng.standard_normal(N)
    v[1:] = rng.normal(0.0, float(system["sigma_v"]), N)
    for k in range(1, N + 1):
        y[k] = x[k, 0] + v[k]
        delayed = np.zeros(n) if k - d <= 0 else x[k - d]
        x[k + 1] = A @ x[k] + B @ delayed + f * u[k]
    return {"u": u, "v": v, "y": y, "x": x, "N": N}


def _initial_history(n: int, d: int, scale: float) -> dict[int, np.ndarray]:
    value = np.ones(n) / scale
    return {time_index: value.copy() for time_index in range(1 - d - n, 2)}


def _estimated_phi(
    time_index: int,
    n: int,
    d: int,
    history: dict[int, np.ndarray],
    trajectory: dict[str, Any],
) -> np.ndarray:
    if time_index <= 0:
        return np.zeros(n * n + 2 * n)
    x_lags = [n, *range(d + 1, d + n + 1)]
    u_lags = list(range(1, n + 1))
    blocks: list[np.ndarray] = [history[time_index - lag] for lag in x_lags]
    blocks.extend(np.asarray([get_u(trajectory, time_index - lag)]) for lag in u_lags)
    return np.concatenate(blocks)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def percentile_interval(
    values: np.ndarray, confidence: float, seed: int = SEED
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(10_000)
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        idx = rng.integers(0, sample.size, size=(stop - start, sample.size))
        stats[start:stop] = np.median(sample[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_ratio(
    a: np.ndarray, b: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    logs = np.log((np.asarray(a) + EPS) / (np.asarray(b) + EPS))
    lo, hi = percentile_interval(logs, confidence)
    return float(np.exp(np.median(logs))), float(np.exp(lo)), float(np.exp(hi))


def random_partition_diagnostic() -> dict[str, Any]:
    path = VERSION_ROOT / "review_1_4_analysis/results/trajectory_swap_system_risks.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    validation = np.asarray(
        [float(r["system_risk"]) for r in rows if r["condition"] == "val_original"],
        dtype=float,
    )
    test = np.asarray(
        [float(r["system_risk"]) for r in rows if r["condition"] == "test_original"],
        dtype=float,
    )
    pooled = np.concatenate([validation, test])
    observed = float(np.median(test) / np.median(validation))
    rng = np.random.default_rng(SEED)
    ratios = np.empty(10_000)
    for index in range(ratios.size):
        order = rng.permutation(pooled.size)
        val_draw = pooled[order[: validation.size]]
        test_draw = pooled[order[validation.size :]]
        ratios[index] = np.median(test_draw) / np.median(val_draw)
    centered = np.abs(np.log(ratios))
    observed_distance = abs(math.log(observed))
    result = {
        "seed": SEED,
        "n_permutations": int(ratios.size),
        "n_validation": int(validation.size),
        "n_test": int(test.size),
        "validation_median": float(np.median(validation)),
        "test_median": float(np.median(test)),
        "observed_test_over_validation_ratio": observed,
        "permutation_ratio_interval_90": [
            float(x) for x in np.quantile(ratios, [0.05, 0.95])
        ],
        "permutation_ratio_interval_95": [
            float(x) for x in np.quantile(ratios, [0.025, 0.975])
        ],
        "two_sided_log_ratio_p": float(
            (1 + np.sum(centered >= observed_distance)) / (ratios.size + 1)
        ),
        "one_sided_ratio_p": float(
            (1 + np.sum(ratios >= observed)) / (ratios.size + 1)
        ),
    }
    write_json(result, OUT / "partition_permutation.json")
    return result


def late_record_summary() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in LONG_ROWS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    system_rows: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["system_index"]), int(row["p"]), float(row["lambda"]))
        system_rows.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for lam in (1.0, 0.99, 0.95):
        for p in (1, 5):
            values_60_80 = []
            values_80_100 = []
            for system_index in range(201):
                pair = system_rows[(system_index, p, lam)]

                values_60_80.append(np.mean([float(r["R_prefix_80000"]) for r in pair]))
                values_80_100.append(
                    np.mean([float(r["R_prefix_100000"]) for r in pair])
                )
            a = np.asarray(values_60_80)
            b = np.asarray(values_80_100)
            summaries.append(
                {
                    "lambda": lam,
                    "p": p,
                    "median_R_prefix_80000": float(np.median(a)),
                    "median_R_prefix_100000": float(np.median(b)),
                    "median_within_system_ratio_100k_over_80k": float(
                        np.exp(np.median(np.log((b + EPS) / (a + EPS))))
                    ),
                }
            )
    write_csv(summaries, OUT / "late_record_summary.csv")
    return summaries


def rls_run(
    system: dict[str, Any],
    trajectory: dict[str, Any],
    cfg: Config,
    covariance_scale: float,
) -> dict[str, Any]:
    n, d = int(system["n"]), int(system["d"])
    m = int(np.asarray(system["rho_star"]).size)
    rho = np.ones(m) / cfg.p0_init
    covariance = covariance_scale * np.eye(m)
    history = _initial_history(n, d, cfg.p0_init)
    estimates = np.full((N, m), np.nan)
    spec_counter = 0
    status = "ok"
    diverged_at: int | None = None

    for k in range(1, N + 1):
        current_state = history[k]
        phi = _estimated_phi(k, n, d, history, trajectory)
        covariance_phi = covariance @ phi
        denominator = 1.0 + float(phi @ covariance_phi)
        gain = covariance_phi / denominator
        innovation = get_y(trajectory, k) - float(phi @ rho)
        rho = rho + gain * innovation
        covariance = covariance - np.outer(gain, covariance_phi)
        covariance = (covariance + covariance.T) / 2.0
        estimates[k - 1] = rho

        if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(current_state)):
            status, diverged_at = "diverged", k
            break
        if np.linalg.norm(rho) > cfg.div_rho_factor * np.linalg.norm(
            system["rho_star"]
        ):
            status, diverged_at = "diverged", k
            break
        A_hat, B_hat, f_hat = unpack_rho(rho, n, d)
        F_hat = build_F(A_hat, B_hat, d)
        if (
            float(np.abs(F_hat).sum(axis=1).max()) > cfg.div_spec_radius
            and spectral_radius(F_hat) > cfg.div_spec_radius
        ):
            spec_counter += 1
        else:
            spec_counter = 0
        if spec_counter >= cfg.div_spec_patience:
            status, diverged_at = "diverged", k
            break
        if k < N:
            history[k + 1] = (
                A_hat @ current_state
                + B_hat @ history[k - d]
                + f_hat * get_u(trajectory, k)
            )

    truth = np.asarray(system["rho_star"])
    if status == "ok" and np.all(np.isfinite(estimates[N // 2 :])):
        risk = float(
            np.mean(np.sum((estimates[N // 2 :] - truth) ** 2, axis=1))
            / np.linalg.norm(truth) ** 2
        )
        risk = min(risk, CAP)
        final_error = float(
            np.linalg.norm(estimates[-1] - truth) / np.linalg.norm(truth) * 100.0
        )
    else:
        risk = CAP
        final_error = math.sqrt(CAP) * 100.0
    return {
        "risk": risk,
        "final_error_pct": final_error,
        "status": status,
        "diverged_at": diverged_at,
    }


def rls_task(task: tuple[str, int, int, float]) -> dict[str, Any]:
    split, system_index, trajectory_index, covariance_scale = task
    cfg = replace(Config(), N=N)
    family = load_family(FAMILY_PATH)
    system = family[split][system_index]
    rng = np.random.default_rng(traj_seed(cfg, int(system["m"]), trajectory_index))
    trajectory = simulate(system, rng, N)
    start = time.perf_counter()
    result = rls_run(system, trajectory, cfg, covariance_scale)
    result.update(
        {
            "system_index": system_index,
            "split": split,
            "trajectory_index": trajectory_index,
            "covariance_scale": covariance_scale,
            "elapsed_seconds": time.perf_counter() - start,
        }
    )
    return result


def rls_reference(workers: int = 8) -> dict[str, Any]:
    covariance_grid = (0.01, 0.1, 1.0, 100.0, 10_000.0, 1_000_000.0)
    validation_tasks = [
        ("val", i, j, scale)
        for scale in covariance_grid
        for i in range(102)
        for j in (3, 4)
    ]
    validation_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(rls_task, task) for task in validation_tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            validation_rows.append(future.result())
            if count % 100 == 0 or count == len(futures):
                print(f"RLS validation {count}/{len(futures)}", flush=True)
    write_csv(validation_rows, OUT / "rls_validation_trajectory_results.csv")

    selection_rows = []
    for scale in covariance_grid:
        system_risks = []
        system_divergence = []
        for i in range(102):
            pair = [
                row
                for row in validation_rows
                if int(row["system_index"]) == i
                and math.isclose(float(row["covariance_scale"]), scale)
            ]
            system_risks.append(np.mean([float(row["risk"]) for row in pair]))
            system_divergence.append(any(row["status"] != "ok" for row in pair))
        selection_rows.append(
            {
                "covariance_scale": scale,
                "validation_median_R": float(np.median(system_risks)),
                "systems_with_any_divergence": int(sum(system_divergence)),
            }
        )
    selected_scale = min(
        selection_rows,
        key=lambda row: (row["validation_median_R"], row["covariance_scale"]),
    )["covariance_scale"]
    for row in selection_rows:
        row["selected"] = int(
            math.isclose(float(row["covariance_scale"]), float(selected_scale))
        )
    write_csv(selection_rows, OUT / "rls_covariance_selection.csv")

    tasks = [("test", i, j, float(selected_scale)) for i in range(201) for j in (3, 4)]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(rls_task, task): task for task in tasks}
        for count, future in enumerate(as_completed(future_map), start=1):
            rows.append(future.result())
            if count % 50 == 0 or count == len(tasks):
                print(f"RLS test {count}/{len(tasks)}", flush=True)
    rows.sort(key=lambda r: (int(r["system_index"]), int(r["trajectory_index"])))
    write_csv(rows, OUT / "rls_trajectory_results.csv")

    system_values = []
    for i in range(201):
        pair = [r for r in rows if int(r["system_index"]) == i]
        system_values.append(
            {
                "system_index": i,
                "R": float(np.mean([float(r["risk"]) for r in pair])),
                "final_error_pct": float(
                    np.mean([float(r["final_error_pct"]) for r in pair])
                ),
                "any_divergence": int(any(r["status"] != "ok" for r in pair)),
                "mean_runtime_seconds": float(
                    np.mean([float(r["elapsed_seconds"]) for r in pair])
                ),
            }
        )
    write_csv(system_values, OUT / "rls_system_results.csv")
    rls_r = np.asarray([r["R"] for r in system_values], dtype=float)
    lo, hi = percentile_interval(rls_r, 0.95)

    factorial = json.loads(FACTORIAL_PATH.read_text(encoding="utf-8"))
    comparisons = []
    for p, lam in ((1, 1.0), (5, 1.0), (1, 0.95), (5, 0.95)):
        cell = factorial["cells"][f"p={p}|lambda={lam}"]
        misg = np.asarray(cell["test"]["system_risks"], dtype=float)
        point, ratio_lo, ratio_hi = paired_ratio(misg, rls_r)
        comparisons.append(
            {
                "method": f"MISG p={p}, lambda={lam:g}",
                "gamma": float(cell["gamma"]),
                "median_R": float(np.median(misg)),
                "paired_ratio_MISG_over_RLS": point,
                "ratio_ci95_lo": ratio_lo,
                "ratio_ci95_hi": ratio_hi,
            }
        )
    write_csv(comparisons, OUT / "rls_misg_comparisons.csv")
    summary = {
        "design": {
            "n_systems": 201,
            "trajectories_per_system": 2,
            "trajectory_indices": [3, 4],
            "N": N,
            "initial_parameter_scale": cfg.p0_init
            if (cfg := replace(Config(), N=N))
            else None,
            "candidate_covariance_scales": list(covariance_grid),
            "selected_covariance_scale": selected_scale,
        },
        "rls": {
            "median_R": float(np.median(rls_r)),
            "ci95": [lo, hi],
            "median_final_error_pct": float(
                np.median([r["final_error_pct"] for r in system_values])
            ),
            "systems_with_any_divergence": int(
                sum(r["any_divergence"] for r in system_values)
            ),
            "median_runtime_seconds": float(
                np.median([r["mean_runtime_seconds"] for r in system_values])
            ),
        },
        "comparisons": comparisons,
        "validation_selection": selection_rows,
    }
    write_json(summary, OUT / "rls_summary.json")
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "partition": random_partition_diagnostic(),
        "late_record": late_record_summary(),
        "rls": rls_reference(),
    }
    write_json(result, OUT / "followup_manifest.json")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Run the stored-review follow-up analyses, including the RLS reference."
    ).parse_args()
    main()
