from __future__ import annotations

import math
import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import cma
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

import rng
import search


MANIFEST_HASH = "a" * 64
DIM_CONFIG = {1: 2, 2: 4, 6: 15}
ROW_KIND_TWO_FIELDS = frozenset(
    {
        "selection_kind",
        "J_val",
        "J_CMA",
        "J_inc",
        "source",
        "winner_restart_id",
        "n_tied",
        "tie_rule",
        "final_ref",
    }
)


def _identity(config_code: int, *, system_id: int = -1) -> dict[str, int]:
    return {
        "config_code": config_code,
        "scenario_code": 1,
        "condition_code": 0,
        "split_code": 0,
        "system_id": system_id,
    }


def _unit(
    n_dim: int, *, oracle: bool = False, scenario_code: int = 1
) -> search.SearchUnit:
    if oracle:
        identity = _identity(16, system_id=303)
        identity["scenario_code"] = scenario_code
        return search.SearchUnit(
            identity,
            6,
            MANIFEST_HASH,
            "oracle",
            seed_factory=partial(rng.bench_cma_key, 1, 15, scenario_code),
        )
    identity = _identity(DIM_CONFIG[n_dim])
    identity["scenario_code"] = scenario_code
    return search.SearchUnit(identity, n_dim, MANIFEST_HASH, "bench")


def _objective(candidate: search.Candidate, *, offset: float = 0.0) -> float:
    return offset + float(np.sum((candidate.x_box - 0.37) ** 2))


def _tell_quadratic(unit: search.SearchUnit, *, offset: float = 0.0) -> None:
    candidates = unit.ask()
    unit.tell(
        {
            candidate.candidate_id: search.EvalResult(
                candidate.candidate_id,
                _objective(candidate, offset=offset),
                0,
            )
            for candidate in candidates
        }
    )


def _finish_current_restart(
    unit: search.SearchUnit,
    *,
    offset: float = 0.0,
) -> dict:
    while unit.terminal is None:
        _tell_quadratic(unit, offset=offset)
    return unit.checkpoint_dict()


def _assert_records_equal(left: list[dict], right: list[dict]) -> None:
    assert len(left) == len(right)
    for lhs, rhs in zip(left, right, strict=True):
        assert set(lhs) == set(rhs) == search.GEN_RECORD_KEYS
        for key in search.GEN_RECORD_KEYS:
            if key in {"x_raw", "x_box"}:
                assert np.array_equal(lhs[key], rhs[key])
            else:
                assert lhs[key] == rhs[key]


def _global_state_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@pytest.fixture(scope="module")
def completed_w3() -> tuple[dict, dict]:
    unit = _unit(6)
    unit.start_restart()
    checkpoint = _finish_current_restart(unit)
    return checkpoint, unit.best_candidate_record()


@pytest.fixture(scope="module")
def completed_oracle_cma() -> dict:
    unit = _unit(6, oracle=True)
    unit.start_restart()
    return _finish_current_restart(unit, offset=10.0)


@pytest.mark.parametrize(
    ("value", "expected", "raises"),
    [
        (np.nan, 10.0, False),
        (np.inf, 10.0, False),
        (-np.inf, None, True),
        (-1e-12, None, True),
        (20.0, 10.0, False),
        (5.0, 5.0, False),
        (0.0, 0.0, False),
    ],
)
def test_phi_contract(value: float, expected: float | None, raises: bool) -> None:
    logged: list[float] = []
    if raises:
        with pytest.raises(search.SearchFailed, match="negative_R"):
            search.phi(value, 10.0, logged.append)
        assert len(logged) == 1
    else:
        assert search.phi(value, 10.0, logged.append) == expected
        assert len(logged) == int(not np.isfinite(value))


def test_objective_caps_each_trajectory_before_averaging() -> None:
    risks = [(0, 2, 0.01), (0, 0, 100.0), (0, 1, 0.01)]
    result = search.j_train(risks, 10.0, lambda _: None)
    assert np.isclose(result, 3.34, rtol=0.0, atol=1e-12)
    assert result != 10.0
    assert search.j_val({0: [100.0, 0.01, 0.01]}, 10.0, lambda _: None) == result


@pytest.mark.parametrize(
    ("n_dim", "expected_logical", "expected_discarded"),
    [(1, 300, 0), (2, 300, 0), (6, 297, 3)],
)
def test_fixed_budgets_and_two_level_conservation(
    n_dim: int,
    expected_logical: int,
    expected_discarded: int,
) -> None:
    unit = _unit(n_dim)
    checkpoints: dict[int, dict] = {}
    for restart_id in range(1, unit.n_restart + 1):
        unit.start_restart(restart_id)
        checkpoint = _finish_current_restart(unit)
        budget = checkpoint["budget"]
        assert (
            budget["logical_eval_count"] + budget["discarded_alloc"]
            == budget["allocation"]
        )
        checkpoints[restart_id] = checkpoint
    coordinator = search.rebuild_coordinator(
        checkpoints,
        frozenset(checkpoints),
        frozenset(checkpoints),
        require_complete=True,
    )
    assert coordinator == {
        "logical_eval_count": expected_logical,
        "discarded_alloc": expected_discarded,
        "allocation": 300,
    }


def test_w3_oracle_budget_and_incumbent_record(
    completed_w3: tuple[dict, dict],
    completed_oracle_cma: dict,
) -> None:
    w3_checkpoint, w3_best = completed_w3
    selection = search.make_selection(
        MANIFEST_HASH,
        _identity(15),
        [
            {
                "restart_id": 1,
                "J_val": 1.0,
                "final_ref": (f"r1g{w3_best['gen_id']}c{w3_best['candidate_id']}"),
            }
        ],
    )
    located = search.locate_w3_x_box(selection, {1: w3_checkpoint})
    assert (
        located
        is w3_checkpoint["gen_records"][
            next(
                index
                for index, record in enumerate(w3_checkpoint["gen_records"])
                if record["gen_id"] == w3_best["gen_id"]
                and record["candidate_id"] == w3_best["candidate_id"]
            )
        ]["x_box"]
    )

    oracle = search.SearchUnit.restore(completed_oracle_cma, MANIFEST_HASH)
    cma_value = oracle.best_candidate_record()["J"]
    oracle.complete_incumbent(located, cma_value + 1.0, 0, J_w3=cma_value + 1.0)
    checkpoint = oracle.checkpoint_dict()
    assert checkpoint["budget"] == {
        "logical_eval_count": 298,
        "discarded_alloc": 2,
        "allocation": 300,
    }
    incumbent = checkpoint["gen_records"][-1]
    candidate = checkpoint["gen_records"][0]
    assert (
        incumbent["restart_id"],
        incumbent["gen_id"],
        incumbent["candidate_id"],
    ) == (1, -1, -1)
    assert incumbent["x_raw"] is None and incumbent["stop_flags"] is None
    assert np.array_equal(incumbent["x_box"], located)
    assert incumbent["logical_eval_count_at_commit"] == 298
    assert incumbent["discarded_alloc_at_commit"] == 2
    assert incumbent["cum_scope"] == "restart"
    assert set(incumbent) == set(candidate) == search.GEN_RECORD_KEYS
    assert len(incumbent) == 12
    assert not (set(incumbent) & ROW_KIND_TWO_FIELDS)
    rebuilt = pickle.loads(pickle.dumps(checkpoint, protocol=5))
    _assert_records_equal(checkpoint["gen_records"], rebuilt["gen_records"])
    coordinator = search.rebuild_coordinator(
        {1: checkpoint}, frozenset({1}), frozenset({1}), require_complete=True
    )
    assert coordinator == {
        "logical_eval_count": 298,
        "discarded_alloc": 2,
        "allocation": 300,
    }


def test_rebuild_coordinator_complete_midrun_and_extra_restart() -> None:
    unit = _unit(1)
    checkpoints: dict[int, dict] = {}
    for restart_id in range(1, 4):
        unit.start_restart(restart_id)
        checkpoints[restart_id] = _finish_current_restart(unit)

    missing_middle = {1: checkpoints[1], 3: checkpoints[3]}
    with pytest.raises(search.IntegrityError, match="missing restarts"):
        search.rebuild_coordinator(
            missing_middle,
            frozenset({1, 2, 3}),
            frozenset({1, 2, 3}),
            require_complete=True,
        )

    missing_largest = {1: checkpoints[1], 2: checkpoints[2]}
    with pytest.raises(search.IntegrityError, match="missing restarts"):
        search.rebuild_coordinator(
            missing_largest,
            frozenset({1, 2, 3}),
            frozenset({1, 2}),
            require_complete=True,
        )

    midrun = search.rebuild_coordinator(
        {1: checkpoints[1]},
        frozenset({1, 2, 3}),
        frozenset({1}),
        require_complete=False,
    )
    assert midrun["logical_eval_count"] == 100
    with pytest.raises(search.IntegrityError, match="started restarts"):
        search.rebuild_coordinator(
            {1: checkpoints[1]},
            frozenset({1, 2, 3}),
            frozenset({1, 2}),
            require_complete=False,
        )

    extra = {**checkpoints, 4: checkpoints[3]}
    for require_complete in (False, True):
        with pytest.raises(search.IntegrityError, match="unexpected restart"):
            search.rebuild_coordinator(
                extra,
                frozenset({1, 2, 3}),
                frozenset({1, 2, 3}),
                require_complete=require_complete,
            )


def test_locator_rejects_nonunique_or_inconsistent_evidence(
    completed_w3: tuple[dict, dict],
) -> None:
    checkpoint, best = completed_w3
    final_ref = f"r1g{best['gen_id']}c{best['candidate_id']}"
    base = {
        "winner_restart_id": 1,
        "candidates": [{"restart_id": 1, "J_val": 1.0, "final_ref": final_ref}],
    }
    duplicate = {
        **base,
        "candidates": [*base["candidates"], dict(base["candidates"][0])],
    }
    with pytest.raises(search.IntegrityError, match="matched 2 candidates"):
        search.locate_w3_x_box(duplicate, {1: checkpoint})
    with pytest.raises(search.IntegrityError, match="missing checkpoint"):
        search.locate_w3_x_box(base, {})
    wrong_identity = pickle.loads(pickle.dumps(checkpoint, protocol=5))
    wrong_identity["l2_identity"]["restart_id"] = 2
    with pytest.raises(search.IntegrityError, match="identity mismatch"):
        search.locate_w3_x_box(base, {1: wrong_identity})
    duplicate_record = pickle.loads(pickle.dumps(checkpoint, protocol=5))
    matching = next(
        record
        for record in duplicate_record["gen_records"]
        if record["gen_id"] == best["gen_id"]
        and record["candidate_id"] == best["candidate_id"]
    )
    duplicate_record["gen_records"].append(pickle.loads(pickle.dumps(matching)))
    with pytest.raises(search.IntegrityError, match="matched 2 gen_records"):
        search.locate_w3_x_box(base, {1: duplicate_record})


def test_ask_tell_uses_raw_points_and_exactly_one_repair() -> None:
    unit = _unit(1)
    unit.start_restart()
    candidates = unit.ask()
    before = [
        (candidate.x_raw.copy(), candidate.x_box.copy()) for candidate in candidates
    ]
    unit.tell(
        {candidate.candidate_id: _objective(candidate) for candidate in candidates}
    )
    checkpoint = unit.checkpoint_dict()
    for record, (raw, boxed) in zip(checkpoint["gen_records"], before, strict=True):
        assert np.array_equal(record["x_raw"], raw)
        assert np.array_equal(record["x_box"], boxed)
        assert np.array_equal(cma.BoundTransform([0, 1]).repair(raw.copy()), boxed)
    once = np.asarray(cma.BoundTransform([0, 1]).repair(np.array([0.0])))
    twice = np.asarray(cma.BoundTransform([0, 1]).repair(once.copy()))
    assert float(once[0]) != 0.0
    assert not np.array_equal(once, twice)
    assert math.isclose(float(once[0]), 0.0125, rel_tol=1e-12)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate"])
def test_generation_integrity_failure_does_not_advance_state(case: str) -> None:
    unit = _unit(1)
    reference = _unit(1)
    unit.start_restart()
    reference.start_restart()
    candidates = unit.ask()
    reference_candidates = reference.ask()
    assert all(
        np.array_equal(left.x_raw, right.x_raw)
        for left, right in zip(candidates, reference_candidates, strict=True)
    )
    correct = [
        search.EvalResult(candidate.candidate_id, _objective(candidate), 0)
        for candidate in candidates
    ]
    if case == "missing":
        invalid: object = correct[:-1]
    elif case == "extra":
        invalid = {**{result.candidate_id: result for result in correct}, 4: 1.0}
    else:
        invalid = [*correct, correct[0]]
    with pytest.raises(search.GenerationIntegrityError):
        unit.tell(invalid)
    unit.tell(correct)
    reference.tell(
        {
            candidate.candidate_id: _objective(candidate)
            for candidate in reference_candidates
        }
    )
    next_candidates = unit.ask()
    next_reference = reference.ask()
    assert all(
        np.array_equal(left.x_raw, right.x_raw)
        for left, right in zip(next_candidates, next_reference, strict=True)
    )


def test_checkpoint_restore_replays_next_ask_and_finishes_identically() -> None:
    global_before = np.random.get_state()
    continuous = _unit(2)
    interrupted = _unit(2)
    continuous.start_restart()
    interrupted.start_restart()
    for _ in range(4):
        _tell_quadratic(continuous)
    for _ in range(2):
        _tell_quadratic(interrupted)

    checkpoint = pickle.loads(pickle.dumps(interrupted.checkpoint_dict(), protocol=5))
    crash_candidates = interrupted.ask()
    restored = search.SearchUnit.restore(checkpoint, MANIFEST_HASH)
    replayed = restored.ask()
    assert all(
        np.array_equal(left.x_raw, right.x_raw)
        and np.array_equal(left.x_box, right.x_box)
        for left, right in zip(crash_candidates, replayed, strict=True)
    )
    restored.tell(
        {candidate.candidate_id: _objective(candidate) for candidate in replayed}
    )
    _tell_quadratic(restored)
    _assert_records_equal(
        continuous.checkpoint_dict()["gen_records"],
        restored.checkpoint_dict()["gen_records"],
    )
    assert _global_state_equal(global_before, np.random.get_state())


def test_rng_pollution_and_interleaved_es_do_not_change_candidates() -> None:
    target = _unit(1)
    reference = _unit(1)
    other = _unit(1, scenario_code=2)
    for unit in (target, reference, other):
        unit.start_restart()

    for _ in range(3):
        expected = reference.ask()
        actual = target.ask()
        assert all(
            np.array_equal(left.x_raw, right.x_raw)
            for left, right in zip(actual, expected, strict=True)
        )
        np.random.default_rng(0).standard_normal(100)
        _tell_quadratic(other)
        target.tell(
            {candidate.candidate_id: _objective(candidate) for candidate in actual}
        )
        reference.tell(
            {candidate.candidate_id: _objective(candidate) for candidate in expected}
        )


def test_sobol_points_are_one_shot_deterministic_and_not_centre() -> None:
    first = _unit(1)
    second = _unit(1)
    first.start_restart()
    second.start_restart()
    first_state = first.checkpoint_dict()["sobol_state"]
    second_state = second.checkpoint_dict()["sobol_state"]
    assert first_state["scramble"] is True and first_state["skip"] == 0
    assert first_state["points"].shape == (2, 1)
    assert first_state["n_drawn"] == 2
    assert np.array_equal(first_state["points"], second_state["points"])
    for point in first_state["points"]:
        assert np.linalg.norm(point - np.array([0.5])) > 1e-9
    _finish_current_restart(first)
    first.start_restart(2)
    assert np.array_equal(
        first.checkpoint_dict()["sobol_state"]["points"],
        first_state["points"],
    )


def test_selection_tie_uses_smallest_restart_identity_and_exact_schema() -> None:
    selection = search.make_selection(
        MANIFEST_HASH,
        _identity(2),
        [
            {"restart_id": 3, "J_val": 2.0, "final_ref": "r3g25c0"},
            {"restart_id": 2, "J_val": 1.0 + 5e-13, "final_ref": "r2g25c1"},
            {"restart_id": 1, "J_val": 1.0, "final_ref": "r1g25c2"},
        ],
    )
    assert set(selection) == {
        "schema_version",
        "run_manifest_hash",
        "selection_kind",
        "search_key",
        "candidates",
        "winner_restart_id",
        "tie_rule",
        "n_tied",
        "J_val_winner",
    }
    assert selection["winner_restart_id"] == 1
    assert selection["n_tied"] == 2
    assert selection["tie_rule"] == search.TIE_RULE
    assert [candidate["restart_id"] for candidate in selection["candidates"]] == [
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    ("delta", "expected_source"),
    [(-1.0, "incumbent"), (0.0, "incumbent"), (1.0, "cma")],
)
def test_oracle_selection_and_conservative_tie(
    completed_oracle_cma: dict,
    delta: float,
    expected_source: str,
) -> None:
    unit = search.SearchUnit.restore(completed_oracle_cma, MANIFEST_HASH)
    cma_value = unit.best_candidate_record()["J"]
    incumbent_value = cma_value + delta
    oracle = unit.complete_incumbent(
        np.full(6, 0.4),
        incumbent_value,
        0,
        J_w3=incumbent_value,
    )
    assert oracle["source"] == expected_source
    assert unit.budget["logical_eval_count"] == 298


def test_oracle_bound_violation_is_hard_failure(completed_oracle_cma: dict) -> None:
    unit = search.SearchUnit.restore(completed_oracle_cma, MANIFEST_HASH)
    cma_value = unit.best_candidate_record()["J"]
    with pytest.raises(search.OracleImplError, match="worse than the W3"):
        unit.complete_incumbent(
            np.full(6, 0.4),
            cma_value + 1.0,
            0,
            J_w3=cma_value - 1.0,
        )
    assert unit.budget == {
        "logical_eval_count": 297,
        "discarded_alloc": 3,
        "allocation": 300,
    }


def test_all_inf_generation_discards_remainder_and_escalates() -> None:
    unit = _unit(1)
    checkpoints: list[dict] = []
    for restart_id in (1, 2):
        unit.start_restart(restart_id)
        candidates = unit.ask()
        with pytest.raises(search.RestartFailed, match="generation_all_inf"):
            unit.tell({candidate.candidate_id: np.inf for candidate in candidates})
        checkpoint = unit.checkpoint_dict()
        checkpoints.append(checkpoint)
        assert checkpoint["terminal"] == {
            "state": "RESTART FAILED",
            "reason": "generation_all_inf",
        }
        assert checkpoint["budget"] == {
            "logical_eval_count": 4,
            "discarded_alloc": 96,
            "allocation": 100,
        }
    unit.start_restart(3)
    candidates = unit.ask()
    with pytest.raises(search.SearchFailed, match="all_restarts_failed"):
        unit.tell({candidate.candidate_id: np.inf for candidate in candidates})
    assert unit.checkpoint_dict()["terminal"] == {
        "state": "SEARCH FAILED",
        "reason": "all_restarts_failed",
    }


def test_terminal_blocks_cover_success_restart_and_search_failure(
    completed_w3: tuple[dict, dict],
) -> None:
    success, _ = completed_w3
    assert success["terminal"] == {
        "state": "success",
        "reason": "fixed_generations_complete",
    }


def test_checkpoint_schema_identity_and_manifest_are_strict() -> None:
    unit = _unit(1)
    emitted: list[dict] = []
    unit.checkpoint_callback = emitted.append
    unit.start_restart()
    checkpoint = emitted[0]
    assert set(checkpoint) == search.CHECKPOINT_KEYS
    assert checkpoint["cursor"] == {
        "restart_id": 1,
        "gen_id": 0,
        "committed": True,
    }
    with pytest.raises(search.IntegrityError, match="manifest_hash mismatch"):
        search.SearchUnit.restore(checkpoint, "b" * 64)
    inconsistent = pickle.loads(pickle.dumps(checkpoint, protocol=5))
    inconsistent["cursor"]["gen_id"] = 1
    with pytest.raises(search.IntegrityError, match="gen_id differ"):
        search.SearchUnit.restore(inconsistent, MANIFEST_HASH)


def test_dimension_tiers_match_rng_restart_domains() -> None:
    assert search.DIM_TIERS[1]["n_restart"] == 3
    assert rng.ONE_DIM_RESTART_IDS == frozenset({1, 2, 3})
    assert search.DIM_TIERS[2]["n_restart"] == 2
    assert rng.TWO_DIM_RESTART_IDS == frozenset({1, 2})
    assert search.DIM_TIERS[6]["n_restart"] == 1
    assert rng.SIX_DIM_RESTART_IDS == frozenset({1})

    for n_dim, config_code in DIM_CONFIG.items():
        n_restart = int(search.DIM_TIERS[n_dim]["n_restart"])
        rng.cma_global_key(config_code, 1, n_restart)
        with pytest.raises(rng.SeedKeyError):
            rng.cma_global_key(config_code, 1, n_restart + 1)

    with pytest.raises(rng.SeedKeyError):
        rng.cma_global_key(16, 1, 1)
    rng.cma_oracle_key(1, 1, 303)
    w3 = _unit(6)
    w3_oracle = _unit(6, oracle=True)
    assert w3.n_restart == w3_oracle.n_restart == 1
    assert w3.tier["quota"] == w3_oracle.tier["quota"] == 300
    assert w3.tier == w3_oracle.tier
