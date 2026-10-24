from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / "src" / "core"
sys.path.insert(0, str(SOURCE))

from config import Config, traj_seed
from estimators import misg_run
from systems import load_family, simulate


N = 12_000
LAMBDA = 0.95
GAMMA = 1.0
P_VALUES = (1, 5)
OUT = PROJECT / "data" / "prop3_results"


def rms_scalar(values: np.ndarray) -> float:
    work = np.asarray(values, dtype=np.longdouble)
    scale = np.max(np.abs(work))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean(np.square(work / scale))))


def rms_vector(values: np.ndarray) -> float:
    work = np.asarray(values, dtype=np.longdouble)
    scale = np.max(np.abs(work))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean(np.sum(np.square(work / scale), axis=1))))


def relative_scalar_gap(left: np.ndarray, right: np.ndarray) -> float:
    denominator = 0.5 * (rms_scalar(left) + rms_scalar(right))
    return rms_scalar(left - right) / max(denominator, np.finfo(float).tiny)


def relative_vector_gap(left: np.ndarray, right: np.ndarray) -> float:
    denominator = 0.5 * (rms_vector(left) + rms_vector(right))
    return rms_vector(left - right) / max(denominator, np.finfo(float).tiny)


def run_configuration(system, trajectory, p: int, cfg: Config):
    result = misg_run(
        system,
        trajectory,
        np.ones(p),
        GAMMA,
        "forgetting",
        "observer",
        cfg,
        lam_g=LAMBDA,
        r_window="full",
        return_diagnostics=True,
    )
    if result["status"] != "ok":
        raise RuntimeError(
            f"Diagnostic configuration p={p} stopped at "
            f"{result['diverged_at']} ({result['div_reason']})"
        )
    return result


def trajectory_diagnostics(
    system, trajectory_index: int, cfg: Config
) -> dict[str, object]:
    trajectory = simulate(
        system,
        np.random.default_rng(traj_seed(cfg, int(system["m"]), trajectory_index)),
        N,
    )
    runs = {p: run_configuration(system, trajectory, p, cfg) for p in P_VALUES}
    one, five = runs[1], runs[5]

    early = slice(N // 4, N // 2)
    late = slice(3 * N // 4, N)
    second_half = slice(N // 2, N)

    row: dict[str, object] = {
        "system_id": int(system["m"]),
        "trajectory": trajectory_index,
    }
    diagnostic_arrays = [
        run[key]
        for run in (one, five)
        for key in ("a_traj", "q_traj", "gbar_traj", "correction_traj")
    ]
    safe_norm_limit = np.sqrt(np.finfo(float).max / 16.0)
    if not all(
        np.all(np.isfinite(array)) and np.max(np.abs(array)) < safe_norm_limit
        for array in diagnostic_arrays
    ):
        row["diagnostic_finite"] = 0
        return row
    row["diagnostic_finite"] = 1
    for label, window in (("early", early), ("late", late)):
        row[f"Da_{label}"] = relative_scalar_gap(
            one["a_traj"][window], five["a_traj"][window]
        )
        row[f"Dq_{label}"] = relative_scalar_gap(
            one["q_traj"][window], five["q_traj"][window]
        )
        row[f"Dg_{label}"] = relative_vector_gap(
            one["gbar_traj"][window], five["gbar_traj"][window]
        )
        row[f"Ddelta_{label}"] = relative_vector_gap(
            one["correction_traj"][window], five["correction_traj"][window]
        )

    a1 = one["a_traj"][second_half]
    a5 = five["a_traj"][second_half]
    q1 = one["q_traj"][second_half]
    q5 = five["q_traj"][second_half]
    g1 = one["gbar_traj"][second_half]
    g5 = five["gbar_traj"][second_half]
    delta1 = one["correction_traj"][second_half]
    delta5 = five["correction_traj"][second_half]

    q_min = float(min(np.min(q1), np.min(q5)))
    G = float(np.max(np.linalg.norm(g5, axis=1)))
    eps_a = float(np.max(np.abs(a1 - a5)))
    eps_g = float(np.max(np.linalg.norm(g1 - g5, axis=1)))
    d0 = float(q1[0] - q5[0])
    powers = np.power(LAMBDA, np.arange(q1.size, dtype=float))
    bound = GAMMA * eps_g / q_min + GAMMA * G / (q_min**2) * (
        powers * abs(d0) + (1.0 - powers) * eps_a / (1.0 - LAMBDA)
    )
    actual = np.linalg.norm(delta1 - delta5, axis=1)
    tolerance = 1e-12 * np.maximum(1.0, bound)

    row.update(
        q_min=q_min,
        G=G,
        eps_a=eps_a,
        eps_g=eps_g,
        max_actual_correction_gap=float(np.max(actual)),
        max_proposition_bound=float(np.max(bound)),
        bound_violations=int(np.count_nonzero(actual > bound + tolerance)),
    )
    return row


def task(index: int) -> list[dict[str, object]]:
    system = load_family(PROJECT / "data" / "systems.npz")["test"][index]
    cfg = replace(Config(), N=N)
    return [
        trajectory_diagnostics(system, trajectory_index, cfg)
        for trajectory_index in (3, 4)
    ]


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def summarize(rows: list[dict[str, object]]) -> None:
    metrics = [
        "Da_early",
        "Da_late",
        "Dq_early",
        "Dq_late",
        "Dg_early",
        "Dg_late",
        "Ddelta_early",
        "Ddelta_late",
        "q_min",
        "G",
        "eps_a",
        "eps_g",
        "max_actual_correction_gap",
        "max_proposition_bound",
    ]
    finite_rows = [row for row in rows if int(row["diagnostic_finite"]) == 1]
    finite_counts = {
        system_id: sum(int(row["system_id"]) == system_id for row in finite_rows)
        for system_id in {int(row["system_id"]) for row in rows}
    }
    system_ids = sorted(
        system_id for system_id, count in finite_counts.items() if count == 2
    )
    finite_rows = [row for row in finite_rows if int(row["system_id"]) in system_ids]
    system_values: dict[str, np.ndarray] = {}
    for metric in metrics:
        system_values[metric] = np.asarray(
            [
                np.mean(
                    [
                        float(row[metric])
                        for row in finite_rows
                        if int(row["system_id"]) == system_id
                    ]
                )
                for system_id in system_ids
            ]
        )

    summary = {
        "protocol": {
            "systems_with_two_finite_diagnostics": len(system_ids),
            "requested_systems": len({int(row["system_id"]) for row in rows}),
            "trajectories_per_system": 2,
            "N": N,
            "configurations": [[1, LAMBDA, GAMMA], [5, LAMBDA, GAMMA]],
            "early_window": [N // 4 + 1, N // 2],
            "late_window": [3 * N // 4 + 1, N],
            "finite_record_witness_window": [N // 2 + 1, N],
            "unit_of_summary": "system; two trajectory values averaged within system",
        },
        "metrics": {
            metric: interval(values) for metric, values in system_values.items()
        },
        "late_to_early_median_ratios": {
            prefix: float(
                np.median(system_values[f"{prefix}_late"])
                / np.median(system_values[f"{prefix}_early"])
            )
            for prefix in ("Da", "Dq", "Dg", "Ddelta")
        },
        "positive_q_min_trajectories": int(
            sum(float(row["q_min"]) > 0 for row in finite_rows)
        ),
        "finite_diagnostic_trajectories": len(finite_rows),
        "total_trajectories": len(rows),
        "bound_violations": int(
            sum(int(row["bound_violations"]) for row in finite_rows)
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trajectory_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=201)
    args = parser.parse_args()

    collected: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, index): index for index in range(args.limit)}
        for completed, future in enumerate(as_completed(futures), start=1):
            collected.extend(future.result())
            if completed % 10 == 0 or completed == args.limit:
                print(f"{completed}/{args.limit} systems completed", flush=True)
    collected.sort(key=lambda row: (int(row["system_id"]), int(row["trajectory"])))
    summarize(collected)


if __name__ == "__main__":
    main()
