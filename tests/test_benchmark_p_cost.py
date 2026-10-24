from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1] / "src" / "experiments" / "benchmark_p_cost.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_p_cost", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load benchmark module: {SCRIPT}")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_protocol_is_the_frozen_main_equivalence_comparison() -> None:
    meta = benchmark.protocol_meta()
    assert meta["p_values"] == [1, 5]
    assert meta["lambda"] == 0.95
    assert meta["gamma_by_p"] == {"1": 1.0, "5": 1.0}
    assert meta["state_mode"] == "observer"
    assert meta["r_mode"] == "forgetting"
    assert meta["r_window"] == "full"
    assert meta["N"] == 12000
    assert meta["split"] == "val"
    assert meta["test_split_used"] is False
    assert meta["n_systems_per_delay"] == 3
    assert meta["n_systems"] == 9


def test_registered_update_cost_matches_misg_formula() -> None:
    assert benchmark.misg_update_flops(8, 1) == 49
    assert benchmark.misg_update_flops(8, 5) == 181
    assert np.isclose(benchmark.misg_update_flops(8, 5) / 49, 181 / 49)
    assert benchmark.phi_buffer_bytes(8, 1) == 64
    assert benchmark.phi_buffer_bytes(8, 5) == 320


def test_alternating_order_balances_first_position() -> None:
    assert benchmark.run_order(system_index=0, repeat_index=0) == (1, 5)
    assert benchmark.run_order(system_index=0, repeat_index=1) == (5, 1)
    assert benchmark.run_order(system_index=1, repeat_index=0) == (5, 1)
    first_positions = [
        benchmark.run_order(system_index=i, repeat_index=r)[0]
        for i in range(9)
        for r in range(10)
    ]
    assert first_positions.count(1) == first_positions.count(5) == 45


def test_system_level_summary_aggregates_repetitions_before_inference() -> None:
    rows = []
    for system_id, ratio in [(10, 2.0), (11, 3.0), (12, 4.0)]:
        for repetition in range(4):
            rows.extend(
                [
                    {
                        "system_id": system_id,
                        "d": 2,
                        "repetition": repetition,
                        "p": 1,
                        "elapsed_sec": 1.0,
                    },
                    {
                        "system_id": system_id,
                        "d": 2,
                        "repetition": repetition,
                        "p": 5,
                        "elapsed_sec": ratio,
                    },
                ]
            )
    systems, summary = benchmark.summarize_timings(rows, bootstrap_B=2000, seed=7)
    assert len(systems) == 3
    assert np.isclose(summary["runtime_ratio_p5_over_p1"], 3.0)
    assert summary["independent_unit"] == "system"
    assert summary["n_systems"] == 3
    assert summary["technical_repetitions_per_configuration"] == 4
    assert summary["runtime_ratio_ci95_low"] <= 3.0
    assert summary["runtime_ratio_ci95_high"] >= 3.0


def test_stratified_selection_never_uses_test_split() -> None:
    systems = [
        {"m": 100 * d + index, "d": d, "split": "val"}
        for d in (2, 3, 4)
        for index in range(5)
    ]
    chosen = benchmark.select_stratified(
        systems,
        delays=(2, 3, 4),
        n_per_delay=3,
        rng=np.random.default_rng(123),
    )
    assert len(chosen) == 9
    assert all(item["split"] == "val" for item in chosen)
    assert {d: sum(int(item["d"]) == d for item in chosen) for d in (2, 3, 4)} == {
        2: 3,
        3: 3,
        4: 3,
    }
