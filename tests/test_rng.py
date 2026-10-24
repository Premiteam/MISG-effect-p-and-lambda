from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

from config import Config
import rng
from systems import sample_family


EXPECTED_CONFIG_CODE = {
    "W0": 1,
    "W0-gamma": 2,
    "R1-faithful": 3,
    "R1-matched": 4,
    "W1-free": 5,
    "WP1-faithful": 6,
    "WP1-matched": 7,
    "WP2-weight-faithful": 8,
    "WP2-matched": 9,
    "WP3-faithful": 10,
    "WP3-matched": 11,
    "WP4-faithful": 12,
    "WP4-matched": 13,
    "W2b": 14,
    "W3": 15,
    "W3-o": 16,
    "AP": 17,
}
EXPECTED_SCENARIO_CODE = {
    "S1": 1,
    "S2": 2,
    "S3-jump": 3,
    "S3-drift": 4,
    "S4": 5,
}
EXPECTED_SPLIT_CODE = {"train": 0, "val": 1, "test": 2, "pilot": 3}
EXPECTED_COMPONENT_COUNTS = {
    0: 504,
    1: 2520,
    2: 5,
    3: 51,
    4: 255,
    5: 2000,
    6: 1,
    7: 1,
    8: 201,
    9: 1005,
    10: 1292,
    11: 10584,
    12: 5,
}


def _entropy(seed: np.random.SeedSequence) -> tuple[int, ...]:
    return tuple(int(value) for value in seed.entropy)


def test_enumerated_keys_have_frozen_counts_and_no_derived_collisions() -> None:
    keys = list(rng.enumerate_legal_keys())
    assert len(keys) == 18_424
    assert Counter(key[1] for key in keys) == EXPECTED_COMPONENT_COUNTS
    assert len(set(keys)) == len(keys)

    derived: dict[bytes, tuple[int, ...]] = {}
    for key in keys:
        state = np.random.SeedSequence(key).generate_state(4).tobytes()
        assert state not in derived, f"derived collision: {derived.get(state)} vs {key}"
        derived[state] = key
    assert len(derived) == 18_424


def test_seedsequence_trailing_zero_fact() -> None:
    short = np.random.SeedSequence([rng.SEED_BASE, 11, 7]).generate_state(4)
    short_zero = np.random.SeedSequence([rng.SEED_BASE, 11, 7, 0]).generate_state(4)
    long = np.random.SeedSequence([rng.SEED_BASE, 10, 3, 2, 1]).generate_state(4)
    long_zero = np.random.SeedSequence([rng.SEED_BASE, 10, 3, 2, 1, 0]).generate_state(
        4
    )
    assert np.array_equal(short, short_zero)
    assert not np.array_equal(long, long_zero)


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (rng.sobol_key, (15, 1)),
        (rng.cma_global_key, (2, 1, 4)),
        (rng.cma_global_key, (2, 1, 0)),
        (rng.cma_oracle_key, (3, 1, 250)),
        (rng.resample_matrix_key, (0, 0)),
        (rng.tost_power_key, (1, 0)),
        (rng.bench_system_key, (2, 0)),
        (rng.bench_s3_second_key, (2, 0, 0)),
        (rng.bench_s3_second_key, (0, 0, 21)),
    ],
)
def test_invalid_keys_raise(call, args) -> None:
    with pytest.raises(rng.SeedKeyError):
        call(*args)


def test_frozen_code_tables_are_exact() -> None:
    assert rng.CONFIG_CODE == EXPECTED_CONFIG_CODE
    assert rng.SCENARIO_CODE == EXPECTED_SCENARIO_CODE
    assert rng.SPLIT_CODE == EXPECTED_SPLIT_CODE


def test_split_code_matches_sample_family_output_order() -> None:
    assert list(rng.SPLIT_CODE) == ["train", "val", "test", "pilot"]
    family = sample_family(Config())
    assert list(family) == list(rng.SPLIT_CODE)


def test_randn_adapter_pickles_and_preserves_stream_state() -> None:
    seed = np.random.SeedSequence([rng.SEED_BASE, 13, 1, 10, 2, 1, 1])
    original = rng.RandnFromGenerator(np.random.default_rng(seed))
    restored = pickle.loads(pickle.dumps(original))
    assert np.array_equal(original(3, 4), restored(3, 4))

    advanced = rng.RandnFromGenerator(np.random.default_rng(seed))
    advanced(2, 5)
    resumed = pickle.loads(pickle.dumps(advanced))
    assert np.array_equal(advanced(4, 2), resumed(4, 2))


def test_seed_base_matches_config() -> None:
    assert rng.SEED_BASE == Config().seed_base


def test_component_isolation_rejects_both_wrong_directions() -> None:
    production = _entropy(rng.sobol_key(2, 1))
    bench = _entropy(rng.bench_system_key(0, 0))
    rng.assert_component_isolation(production, namespace="production")
    rng.assert_component_isolation(bench, namespace="bench")
    with pytest.raises(rng.SeedKeyError):
        rng.assert_component_isolation(bench, namespace="production")
    with pytest.raises(rng.SeedKeyError):
        rng.assert_component_isolation(production, namespace="bench")


def test_all_public_key_functions_return_exact_registered_entropy() -> None:
    assert _entropy(rng.sobol_key(2, 5)) == (rng.SEED_BASE, 10, 2, 5)
    assert _entropy(rng.cma_global_key(15, 5, 1)) == (rng.SEED_BASE, 10, 15, 5, 1)
    assert _entropy(rng.cma_oracle_key(2, 1, 201)) == (rng.SEED_BASE, 10, 16, 2, 1, 201)
    assert _entropy(rng.s3_second_key(503, 20)) == (rng.SEED_BASE, 11, 503, 20)
    assert _entropy(rng.resample_matrix_key(2, 1)) == (rng.SEED_BASE, 12, 0, 2, 1)
    assert _entropy(rng.tost_power_key(2, 0)) == (rng.SEED_BASE, 12, 1, 2, 0)
    assert _entropy(rng.bench_system_key(1, 4)) == (rng.SEED_BASE, 13, 1, 0, 4)
    assert _entropy(rng.bench_traj_key(1, 4, 3)) == (rng.SEED_BASE, 13, 1, 1, 4, 3)
    assert _entropy(rng.bench_cma_key(0, 2, 1, 3)) == (
        rng.SEED_BASE,
        13,
        0,
        10,
        2,
        1,
        3,
    )
    assert _entropy(rng.bench_s3_second_key(0, 4, 20)) == (
        rng.SEED_BASE,
        13,
        0,
        11,
        4,
        20,
    )
    assert _entropy(rng.bench_resample_matrix_key(1, 2, 1)) == (
        rng.SEED_BASE,
        13,
        1,
        12,
        0,
        2,
        1,
    )
    assert _entropy(rng.bench_tost_power_key(1, 2, 0)) == (
        rng.SEED_BASE,
        13,
        1,
        12,
        1,
        2,
        0,
    )


def test_bench_resample_uses_caller_declared_mini_domain() -> None:
    with pytest.raises(rng.SeedKeyError):
        rng.bench_resample_matrix_key(0, 0, 7)
    seed = rng.bench_resample_matrix_key(
        0,
        0,
        7,
        allowed_split_codes={0},
        allowed_systemset_codes={7},
    )
    assert _entropy(seed) == (rng.SEED_BASE, 13, 0, 12, 0, 0, 7)


def test_bench_s3_second_is_isolated_from_the_production_key() -> None:
    production = _entropy(rng.s3_second_key(4, 0))
    bench = _entropy(rng.bench_s3_second_key(0, 4, 0))
    assert production == (rng.SEED_BASE, 11, 4, 0)
    assert bench == (rng.SEED_BASE, 13, 0, 11, 4, 0)
    assert production != bench
    rng.assert_component_isolation(production, namespace="production")
    rng.assert_component_isolation(bench, namespace="bench")


@pytest.mark.parametrize(
    ("config_code", "scenario_code", "condition_code"),
    [
        (1, 1, 0),
        (17, 5, 0),
        (15, 4, 1),
        (15, 4, 2),
        (15, 5, 3),
        (15, 5, 4),
        (15, 5, 5),
        (1, 3, 6),
        (1, 3, 7),
        (1, 3, 8),
        (8, 3, 8),
    ],
)
def test_legal_condition_code_combinations(
    config_code, scenario_code, condition_code
) -> None:
    assert rng.condition_tuple_is_legal(config_code, scenario_code, condition_code)
    rng.validate_condition_tuple(config_code, scenario_code, condition_code)


@pytest.mark.parametrize(
    ("config_code", "scenario_code", "condition_code"),
    [(2, 2, 1), (2, 4, 3), (2, 1, 6), (8, 1, 7), (2, 1, 8), (1, 1, 9)],
)
def test_illegal_condition_code_combinations_raise(
    config_code, scenario_code, condition_code
) -> None:
    assert not rng.condition_tuple_is_legal(config_code, scenario_code, condition_code)
    with pytest.raises(rng.SeedKeyError):
        rng.validate_condition_tuple(config_code, scenario_code, condition_code)


def test_registry_covers_components_and_fixed_shapes() -> None:
    rng.validate_registry()
    assert {component for component, _ in rng.SEED_REGISTRY} == set(range(14))
    assert tuple(
        field.name for field in rng.SEED_REGISTRY[(10, "sobol")].exact_shape
    ) == (
        "seed_base",
        "component",
        "config_code",
        "scenario_code",
    )
    assert tuple(
        field.name for field in rng.SEED_REGISTRY[(10, "cma_global")].exact_shape
    ) == (
        "seed_base",
        "component",
        "config_code",
        "scenario_code",
        "restart_id",
    )
    assert tuple(
        field.name for field in rng.SEED_REGISTRY[(10, "cma_oracle")].exact_shape
    ) == (
        "seed_base",
        "component",
        "config_code",
        "scenario_code",
        "restart_id",
        "m",
    )
    assert tuple(
        field.name for field in rng.SEED_REGISTRY[(13, "bench_s3_second")].exact_shape
    ) == (
        "seed_base",
        "component",
        "purpose_code",
        "component_kind",
        "m",
        "retry",
    )


def test_power_rep_count_matches_run_power_source_without_importing_it() -> None:
    run_power = Path(__file__).resolve().parents[1] / "run_power.py"
    tree = ast.parse(run_power.read_text(encoding="utf-8"))
    values = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "N_REP"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Constant):
                values.append(node.value.value)
    assert values == [rng.POWER_N_REP] == [2000]


def test_rng_source_has_no_hash_or_legacy_numpy_random_calls() -> None:
    source = (SRC / "rng.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_legacy = {"seed", "rand", "randn", "choice", "shuffle", "permutation"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "hash"
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "random"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "np"
        ):
            assert node.attr not in forbidden_legacy
