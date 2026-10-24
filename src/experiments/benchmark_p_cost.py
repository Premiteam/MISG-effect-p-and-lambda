from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


P_VALUES = (1, 5)
LAMBDA = 0.95
GAMMA_BY_P = {1: 1.0, 5: 1.0}
STATE_MODE = "observer"
R_MODE = "forgetting"
R_WINDOW = "full"
N_DEFAULT = 12000
N_PER_DELAY_DEFAULT = 3
REPEATS_DEFAULT = 10
WARMUP_DEFAULT = 2
BOOTSTRAP_B_DEFAULT = 10000
BOOTSTRAP_SEED = 20260828
TRAJECTORY_INDEX = 4

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROJECT = WORKSPACE_ROOT
DEFAULT_FAMILY = WORKSPACE_ROOT / "data" / "systems.npz"
DEFAULT_OUTDIR = WORKSPACE_ROOT / "data" / "cost_results"


def protocol_meta(
    *,
    N: int = N_DEFAULT,
    n_per_delay: int = N_PER_DELAY_DEFAULT,
    repeats: int = REPEATS_DEFAULT,
    warmup: int = WARMUP_DEFAULT,
    bootstrap_B: int = BOOTSTRAP_B_DEFAULT,
) -> dict[str, Any]:
    if min(N, n_per_delay, repeats, bootstrap_B) < 1 or warmup < 0:
        raise ValueError("benchmark scales must be positive; warmup may be zero")
    return {
        "schema_version": 1,
        "analysis": "p1_vs_p5_computational_cost",
        "p_values": list(P_VALUES),
        "lambda": LAMBDA,
        "gamma_by_p": {str(p): GAMMA_BY_P[p] for p in P_VALUES},
        "state_mode": STATE_MODE,
        "r_mode": R_MODE,
        "r_window": R_WINDOW,
        "N": int(N),
        "split": "val",
        "test_split_used": False,
        "delays": [2, 3, 4],
        "n_systems_per_delay": int(n_per_delay),
        "n_systems": 3 * int(n_per_delay),
        "trajectory_index": TRAJECTORY_INDEX,
        "warmup_runs_per_configuration_and_system": int(warmup),
        "technical_repetitions_per_configuration_and_system": int(repeats),
        "bootstrap_B": int(bootstrap_B),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "timing_order": "alternating p=1/p=5 and p=5/p=1 by system and repetition",
        "inference_unit": "system",
    }


def misg_update_flops(m: int, p: int) -> int:
    if m < 1 or p < 1:
        raise ValueError("m and p must be positive")
    return (4 * p + 2) * m + p


def phi_buffer_bytes(m: int, p: int, *, itemsize: int = 8) -> int:
    if m < 1 or p < 1 or itemsize < 1:
        raise ValueError("m, p and itemsize must be positive")
    return int(m * p * itemsize)


def run_order(*, system_index: int, repeat_index: int) -> tuple[int, int]:
    if system_index < 0 or repeat_index < 0:
        raise ValueError("indices must be non-negative")
    return (1, 5) if (system_index + repeat_index) % 2 == 0 else (5, 1)


def select_stratified(
    systems: Iterable[Mapping[str, Any]],
    *,
    delays: Sequence[int],
    n_per_delay: int,
    rng: np.random.Generator,
) -> list[Mapping[str, Any]]:
    if n_per_delay < 1:
        raise ValueError("n_per_delay must be positive")
    systems_list = list(systems)
    if any(str(item.get("split")) != "val" for item in systems_list):
        raise ValueError("timing benchmark accepts validation systems only")
    selected: list[Mapping[str, Any]] = []
    for delay in delays:
        candidates = [item for item in systems_list if int(item["d"]) == int(delay)]
        if len(candidates) < n_per_delay:
            raise ValueError(
                f"delay {delay} has {len(candidates)} systems; need {n_per_delay}"
            )
        indices = np.sort(rng.choice(len(candidates), size=n_per_delay, replace=False))
        selected.extend(candidates[int(index)] for index in indices)
    counts = Counter(int(item["d"]) for item in selected)
    expected = {int(delay): int(n_per_delay) for delay in delays}
    if dict(counts) != expected:
        raise RuntimeError(
            f"stratification failed: expected {expected}, found {counts}"
        )
    return selected


def _bootstrap_median(
    values: np.ndarray,
    *,
    B: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(B, values.size))
    draws = np.median(values[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_timings(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_B: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    delays: dict[int, int] = {}
    repetition_counts: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in rows:
        system_id = int(row["system_id"])
        p = int(row["p"])
        elapsed = float(row["elapsed_sec"])
        if p not in P_VALUES or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("timing rows must contain positive finite p=1/p=5 times")
        grouped[(system_id, p)].append(elapsed)
        repetition_counts[(system_id, p)].add(int(row["repetition"]))
        delays[system_id] = int(row["d"])

    system_ids = sorted(delays)
    if not system_ids:
        raise ValueError("no timing rows supplied")
    system_rows: list[dict[str, Any]] = []
    repeat_values: set[int] = set()
    for system_id in system_ids:
        if any((system_id, p) not in grouped for p in P_VALUES):
            raise ValueError(f"system {system_id} lacks one comparison configuration")
        p1 = float(np.median(grouped[(system_id, 1)]))
        p5 = float(np.median(grouped[(system_id, 5)]))
        repeats_p1 = len(repetition_counts[(system_id, 1)])
        repeats_p5 = len(repetition_counts[(system_id, 5)])
        if repeats_p1 != repeats_p5:
            raise ValueError(f"system {system_id} has unbalanced repetitions")
        repeat_values.add(repeats_p1)
        system_rows.append(
            {
                "system_id": system_id,
                "d": delays[system_id],
                "median_sec_p1": p1,
                "median_sec_p5": p5,
                "ratio_p5_over_p1": p5 / p1,
                "log_ratio_p5_over_p1": math.log(p5 / p1),
                "technical_repetitions": repeats_p1,
            }
        )
    if len(repeat_values) != 1:
        raise ValueError("technical repetition count differs across systems")

    p1_values = np.asarray([row["median_sec_p1"] for row in system_rows], dtype=float)
    p5_values = np.asarray([row["median_sec_p5"] for row in system_rows], dtype=float)
    log_ratios = np.asarray(
        [row["log_ratio_p5_over_p1"] for row in system_rows], dtype=float
    )
    rng = np.random.default_rng(seed)
    p1_low, p1_high = _bootstrap_median(p1_values, B=bootstrap_B, rng=rng)
    p5_low, p5_high = _bootstrap_median(p5_values, B=bootstrap_B, rng=rng)
    log_low, log_high = _bootstrap_median(log_ratios, B=bootstrap_B, rng=rng)
    summary = {
        "independent_unit": "system",
        "n_systems": len(system_rows),
        "technical_repetitions_per_configuration": repeat_values.pop(),
        "median_sec_p1": float(np.median(p1_values)),
        "median_sec_p1_ci95_low": p1_low,
        "median_sec_p1_ci95_high": p1_high,
        "median_sec_p5": float(np.median(p5_values)),
        "median_sec_p5_ci95_low": p5_low,
        "median_sec_p5_ci95_high": p5_high,
        "runtime_ratio_p5_over_p1": float(math.exp(np.median(log_ratios))),
        "runtime_ratio_ci95_low": float(math.exp(log_low)),
        "runtime_ratio_ci95_high": float(math.exp(log_high)),
        "runtime_increase_percent": float(
            100.0 * (math.exp(np.median(log_ratios)) - 1.0)
        ),
        "bootstrap_B": int(bootstrap_B),
        "bootstrap_seed": int(seed),
    }
    return system_rows, summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _load_computational_modules(project: Path):
    src = project / "src" / "core"
    if not src.exists():
        raise FileNotFoundError(f"source directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from config import Config, timing_select_seed, traj_seed
    from estimators import misg_run
    from systems import load_family, rho_dim, simulate

    return (
        Config,
        timing_select_seed,
        traj_seed,
        misg_run,
        load_family,
        rho_dim,
        simulate,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project.expanduser().resolve()
    family_path = args.family.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    if not family_path.exists():
        raise FileNotFoundError(f"system family not found: {family_path}")
    (
        Config,
        timing_select_seed,
        traj_seed,
        misg_run,
        load_family,
        rho_dim,
        simulate,
    ) = _load_computational_modules(project)

    meta = protocol_meta(
        N=args.N,
        n_per_delay=args.n_per_delay,
        repeats=args.repeats,
        warmup=args.warmup,
        bootstrap_B=args.bootstrap_B,
    )
    cfg = replace(Config(), N=args.N)
    family = load_family(family_path)
    selected = select_stratified(
        family["val"],
        delays=cfg.d_choices,
        n_per_delay=args.n_per_delay,
        rng=np.random.default_rng(timing_select_seed(cfg)),
    )
    trajectories: dict[int, Mapping[str, Any]] = {}
    for system in selected:
        system_id = int(system["m"])
        trajectories[system_id] = simulate(
            system,
            np.random.default_rng(traj_seed(cfg, system_id, TRAJECTORY_INDEX)),
            N=args.N,
        )

    raw_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    for system_index, system in enumerate(selected):
        if str(system["split"]) != "val":
            raise RuntimeError("test or training system entered timing benchmark")
        system_id = int(system["m"])
        trajectory = trajectories[system_id]
        for p in P_VALUES:
            for _ in range(args.warmup):
                result = misg_run(
                    system,
                    trajectory,
                    np.ones(p),
                    GAMMA_BY_P[p],
                    R_MODE,
                    STATE_MODE,
                    cfg,
                    lam_g=LAMBDA,
                    r_window=R_WINDOW,
                    return_diagnostics=False,
                )
                if result["status"] != "ok":
                    raise RuntimeError(
                        f"warmup diverged for system={system_id}, p={p}: {result}"
                    )
        for repetition in range(args.repeats):
            order = run_order(system_index=system_index, repeat_index=repetition)
            for position, p in enumerate(order, start=1):
                started = time.perf_counter()
                result = misg_run(
                    system,
                    trajectory,
                    np.ones(p),
                    GAMMA_BY_P[p],
                    R_MODE,
                    STATE_MODE,
                    cfg,
                    lam_g=LAMBDA,
                    r_window=R_WINDOW,
                    return_diagnostics=False,
                )
                elapsed = float(time.perf_counter() - started)
                if result["status"] != "ok":
                    raise RuntimeError(
                        f"timed run diverged for system={system_id}, p={p}: {result}"
                    )
                raw_rows.append(
                    {
                        "system_id": system_id,
                        "d": int(system["d"]),
                        "trajectory_index": TRAJECTORY_INDEX,
                        "repetition": repetition,
                        "run_position": position,
                        "p": p,
                        "lambda": LAMBDA,
                        "gamma": GAMMA_BY_P[p],
                        "N": args.N,
                        "elapsed_sec": elapsed,
                    }
                )

    system_rows, timing_summary = summarize_timings(
        raw_rows,
        bootstrap_B=args.bootstrap_B,
        seed=BOOTSTRAP_SEED,
    )
    m_values = {rho_dim(int(item["n"]), int(item["d"])) for item in selected}
    if len(m_values) != 1:
        raise RuntimeError(
            f"selected systems have different rho dimensions: {m_values}"
        )
    m = int(m_values.pop())
    p1_flops = misg_update_flops(m, 1)
    p5_flops = misg_update_flops(m, 5)
    p1_buffer = phi_buffer_bytes(m, 1)
    p5_buffer = phi_buffer_bytes(m, 5)
    cost_summary = {
        **timing_summary,
        "parameter_dimension_m": m,
        "per_step_update_flops_p1": p1_flops,
        "per_step_update_flops_p5": p5_flops,
        "total_update_flops_p1": int(p1_flops * args.N),
        "total_update_flops_p5": int(p5_flops * args.N),
        "update_flop_ratio_p5_over_p1": float(p5_flops / p1_flops),
        "phi_buffer_bytes_p1": p1_buffer,
        "phi_buffer_bytes_p5": p5_buffer,
        "phi_buffer_ratio_p5_over_p1": float(p5_buffer / p1_buffer),
        "phi_buffer_increment_bytes_p5_minus_p1": int(p5_buffer - p1_buffer),
    }
    elapsed_total = float(time.perf_counter() - benchmark_started)
    machine = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "note": "Wall-clock values are machine- and load-specific; ratios use paired identical trajectories.",
    }
    payload = {
        "meta": {
            **meta,
            "family": str(family_path),
            "project_snapshot": str(project),
            "selected_system_ids": [int(item["m"]) for item in selected],
            "selected_delay_counts": {
                str(delay): count
                for delay, count in sorted(
                    Counter(int(item["d"]) for item in selected).items()
                )
            },
            "benchmark_elapsed_seconds": elapsed_total,
        },
        "machine": machine,
        "summary": cost_summary,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "raw_timing.csv", raw_rows)
    _write_csv(outdir / "system_timing_summary.csv", system_rows)
    _write_csv(outdir / "computational_cost_summary.csv", [cost_summary])
    _write_json(outdir / "computational_cost_results.json", payload)
    report = f"""# Computational cost of `p=1` and `p=5`

## Protocol

The frozen moderate-forgetting configurations (`lambda=0.95`, `gamma=1`) were compared using observer state estimation, full-window normalization and trajectories of `N={args.N:,}` samples. The benchmark used {len(selected)} validation systems, balanced across delays ({args.n_per_delay} systems for each of `d=2,3,4`), and one identical trajectory per system. Each configuration was warmed up {args.warmup} times and timed {args.repeats} times per system. Run order alternated within system and repetition. Technical timing repetitions were reduced to a median within system before the paired `p=5/p=1` log-runtime ratio was summarized across systems; the 95% percentile-bootstrap interval used {args.bootstrap_B:,} system-level resamples.

## Result

For the registered parameter dimension `m={m}`, the MISG update count was {p1_flops} FLOPs per step for `p=1` and {p5_flops} FLOPs per step for `p=5`, a {p5_flops / p1_flops:.2f}-fold difference in the window-dependent update kernel. Across `N={args.N:,}` steps, these counts correspond to {p1_flops * args.N:,} and {p5_flops * args.N:,} FLOPs, respectively. The persistent regressor buffer increased from {p1_buffer} to {p5_buffer} bytes ({p5_buffer / p1_buffer:.0f}-fold; +{p5_buffer - p1_buffer} bytes).

On this machine, the median trajectory runtime was {timing_summary["median_sec_p1"]:.4f} s for `p=1` (95% CI, {timing_summary["median_sec_p1_ci95_low"]:.4f}–{timing_summary["median_sec_p1_ci95_high"]:.4f}) and {timing_summary["median_sec_p5"]:.4f} s for `p=5` (95% CI, {timing_summary["median_sec_p5_ci95_low"]:.4f}–{timing_summary["median_sec_p5_ci95_high"]:.4f}). The paired runtime ratio was {timing_summary["runtime_ratio_p5_over_p1"]:.3f} (95% CI, {timing_summary["runtime_ratio_ci95_low"]:.3f}–{timing_summary["runtime_ratio_ci95_high"]:.3f}), equivalent to a {timing_summary["runtime_increase_percent"]:.1f}% longer runtime for `p=5` under this implementation and environment.

## Interpretation boundary

The FLOP and buffer counts describe the registered MISG update and its persistent regressor window. They exclude common observer propagation, stability checks, trajectory generation and output storage. The wall-clock ratio includes these common costs and is therefore smaller than the update-kernel FLOP ratio. Absolute times are machine- and load-specific; the paired ratio is the more relevant implementation-level comparison. This benchmark supports a computational rationale for preferring `p=1` when its risk is within the author-approved 5% multiplicative equivalence region, but it does not establish a universal runtime ratio for other implementations or hardware.
"""
    (outdir / "COMPUTATIONAL_COST_REPORT.md").write_text(report, encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the computational cost of innovation lengths"
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--n-per-delay", type=int, default=N_PER_DELAY_DEFAULT)
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT)
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--bootstrap-B", type=int, default=BOOTSTRAP_B_DEFAULT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_benchmark(args)
    summary = result["summary"]
    print(
        "p=5/p=1 runtime ratio "
        f"{summary['runtime_ratio_p5_over_p1']:.3f} "
        f"(95% CI {summary['runtime_ratio_ci95_low']:.3f}–"
        f"{summary['runtime_ratio_ci95_high']:.3f})"
    )
    print(f"Results written to {args.outdir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
