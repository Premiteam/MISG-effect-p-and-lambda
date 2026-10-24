from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from config import Config


SEED_BASE = 20260729
POWER_N_REP = 2000

MIN_NONNEGATIVE = 0
UNIT_STEP = 1
KEY_SEED_BASE_INDEX = 0
KEY_COMPONENT_INDEX = 1
KEY_PREFIX_LENGTH = 2
MAIN_CONDITION_CODE = 0
PRIMARY_RESTART_ID = 1

CONFIG_CODE = {
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
SCENARIO_CODE = {"S1": 1, "S2": 2, "S3-jump": 3, "S3-drift": 4, "S4": 5}
SPLIT_CODE = {"train": 0, "val": 1, "test": 2, "pilot": 3}

COMPONENT_MAIN_SYSTEM = 0
COMPONENT_MAIN_TRAJECTORY = 1
COMPONENT_REPRODUCTION = 2
COMPONENT_PILOT_SYSTEM = 3
COMPONENT_PILOT_TRAJECTORY = 4
COMPONENT_POWER = 5
COMPONENT_Q_SELECTION = 6
COMPONENT_TIMING_SELECTION = 7
COMPONENT_E2_SYSTEM = 8
COMPONENT_E2_TRAJECTORY = 9
COMPONENT_OPTIMIZER = 10
COMPONENT_S3_SECOND = 11
COMPONENT_RESAMPLING = 12
COMPONENT_BENCH = 13

RESAMPLE_MATRIX_KIND = 0
TOST_POWER_KIND = 1
BENCH_SYSTEM_KIND = COMPONENT_MAIN_SYSTEM
BENCH_TRAJECTORY_KIND = COMPONENT_MAIN_TRAJECTORY
BENCH_CMA_KIND = COMPONENT_OPTIMIZER
BENCH_S3_SECOND_KIND = COMPONENT_S3_SECOND
BENCH_RESAMPLING_KIND = COMPONENT_RESAMPLING

PURPOSE_CODES = frozenset({0, 1})
CONFIG_CODES = frozenset(CONFIG_CODE.values())
SCENARIO_CODES = frozenset(SCENARIO_CODE.values())
CONDITION_CODES = frozenset(range(9))
S3_DRIFT_CONDITION_CODES = frozenset({1, 2})
S4_CONDITION_CODES = frozenset({3, 4, 5})
W0_ONLY_CONDITION_CODES = frozenset({6, 7})
P8_CONDITION_CODE = 8
ONE_DIM_CONFIG_CODES = frozenset({2, 3, 10})
TWO_DIM_CONFIG_CODES = frozenset({4, 5, 7, 9, 11, 13, 14, 17})
SIX_DIM_GLOBAL_CONFIG_CODES = frozenset({15})
ORACLE_CONFIG_CODE = 16
SOBOL_CONFIG_CODES = ONE_DIM_CONFIG_CODES | TWO_DIM_CONFIG_CODES
GLOBAL_CMA_CONFIG_CODES = (
    ONE_DIM_CONFIG_CODES | TWO_DIM_CONFIG_CODES | SIX_DIM_GLOBAL_CONFIG_CODES
)

MAIN_SYSTEM_IDS = range(504)
TRAJECTORY_IDS = range(5)
REPRODUCTION_IDS = range(5)
PILOT_SYSTEM_IDS = range(51)
POWER_REPLICATE_IDS = range(POWER_N_REP)
E2_SYSTEM_IDS = range(201)
S3_RETRY_IDS = range(21)
ONE_DIM_RESTART_IDS = frozenset({1, 2, 3})
TWO_DIM_RESTART_IDS = frozenset({1, 2})
SIX_DIM_RESTART_IDS = frozenset({PRIMARY_RESTART_ID})
VAL_SYSTEM_IDS = range(201, 303)
TEST_SYSTEM_IDS = range(303, 504)
INFERENCE_SPLIT_CODES = frozenset({SPLIT_CODE["val"], SPLIT_CODE["test"]})
SYSTEMSET_CODES = frozenset({0, 1})

PRODUCTION_COMPONENTS = frozenset(range(COMPONENT_BENCH))
BENCH_COMPONENTS = frozenset({COMPONENT_BENCH})
EXPECTED_REGISTRY_COMPONENTS = frozenset(range(COMPONENT_BENCH + 1))


class SeedKeyError(ValueError):
    pass


@dataclass(frozen=True)
class IntDomain:
    minimum: int = 0
    maximum: int | None = None
    allowed: frozenset[int] | None = None
    description: str = "non-negative integer"

    def contains(self, value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, Integral):
            return False
        integer = int(value)
        if self.allowed is not None:
            return integer in self.allowed
        if integer < self.minimum:
            return False
        return self.maximum is None or integer <= self.maximum


@dataclass(frozen=True)
class SeedField:
    name: str
    domain: IntDomain


SeedKey = tuple[int, ...]
LegalTuple = Callable[[SeedKey], bool]


@dataclass(frozen=True)
class SeedSpec:
    component: int
    seed_kind: str
    exact_shape: tuple[SeedField, ...]
    legal_tuple: LegalTuple


def _fixed(value: int, description: str) -> IntDomain:
    return IntDomain(allowed=frozenset({value}), description=description)


def _bounded(values: range | frozenset[int], description: str) -> IntDomain:
    if isinstance(values, range):
        if values.step != UNIT_STEP or values.stop <= values.start:
            raise RuntimeError(
                f"invalid registered range for {description}: {values!r}"
            )
        return IntDomain(
            minimum=values.start,
            maximum=values.stop - UNIT_STEP,
            description=description,
        )
    return IntDomain(allowed=values, description=description)


NONNEGATIVE = IntDomain(description="non-negative integer declared by the mini spec")


def _shape(
    component: int,
    *fields: tuple[str, IntDomain],
) -> tuple[SeedField, ...]:
    return (
        SeedField("seed_base", _fixed(SEED_BASE, f"exactly {SEED_BASE}")),
        SeedField("component", _fixed(component, f"exactly {component}")),
        *(SeedField(name, domain) for name, domain in fields),
    )


def _always_legal(key: SeedKey) -> bool:
    return bool(key)


def _legal_sobol(key: SeedKey) -> bool:
    _, _, config_code, scenario_code = key
    return config_code in SOBOL_CONFIG_CODES and scenario_code in SCENARIO_CODES


def _legal_cma_global(key: SeedKey) -> bool:
    _, _, config_code, scenario_code, restart_id = key
    if scenario_code not in SCENARIO_CODES:
        return False
    if config_code in ONE_DIM_CONFIG_CODES:
        return restart_id in ONE_DIM_RESTART_IDS
    if config_code in TWO_DIM_CONFIG_CODES:
        return restart_id in TWO_DIM_RESTART_IDS
    return (
        config_code in SIX_DIM_GLOBAL_CONFIG_CODES and restart_id in SIX_DIM_RESTART_IDS
    )


def _legal_cma_oracle(key: SeedKey) -> bool:
    _, _, config_code, scenario_code, restart_id, system_id = key
    if config_code != ORACLE_CONFIG_CODE or restart_id != PRIMARY_RESTART_ID:
        return False
    if system_id in TEST_SYSTEM_IDS:
        return scenario_code in SCENARIO_CODES
    return system_id in VAL_SYSTEM_IDS and scenario_code == SCENARIO_CODE["S2"]


def _legal_s3_second(key: SeedKey) -> bool:
    _, _, system_id, retry = key
    return system_id in MAIN_SYSTEM_IDS and retry in S3_RETRY_IDS


def _legal_resample_matrix(key: SeedKey) -> bool:
    _, _, discriminator, split_code, systemset_code = key
    return (
        discriminator == RESAMPLE_MATRIX_KIND
        and split_code in INFERENCE_SPLIT_CODES
        and systemset_code in SYSTEMSET_CODES
    )


def _legal_tost_power(key: SeedKey) -> bool:
    _, _, discriminator, scenario_code, condition_code = key
    return (
        discriminator == TOST_POWER_KIND
        and scenario_code == SCENARIO_CODE["S2"]
        and condition_code == MAIN_CONDITION_CODE
    )


def _legal_bench_system(key: SeedKey) -> bool:
    _, _, purpose_code, discriminator, system_id = key
    return (
        purpose_code in PURPOSE_CODES
        and discriminator == BENCH_SYSTEM_KIND
        and system_id >= MIN_NONNEGATIVE
    )


def _legal_bench_trajectory(key: SeedKey) -> bool:
    _, _, purpose_code, discriminator, system_id, trajectory_id = key
    return (
        purpose_code in PURPOSE_CODES
        and discriminator == BENCH_TRAJECTORY_KIND
        and system_id >= MIN_NONNEGATIVE
        and trajectory_id >= MIN_NONNEGATIVE
    )


def _legal_bench_s3_second(key: SeedKey) -> bool:
    _, _, purpose_code, discriminator, system_id, retry = key
    return (
        purpose_code in PURPOSE_CODES
        and discriminator == BENCH_S3_SECOND_KIND
        and system_id >= MIN_NONNEGATIVE
        and retry in S3_RETRY_IDS
    )


def _legal_bench_cma(key: SeedKey) -> bool:
    (
        _,
        _,
        purpose_code,
        discriminator,
        config_code,
        scenario_code,
        restart_id,
    ) = key
    production_key = (
        SEED_BASE,
        COMPONENT_OPTIMIZER,
        config_code,
        scenario_code,
        restart_id,
    )
    return (
        purpose_code in PURPOSE_CODES
        and discriminator == BENCH_CMA_KIND
        and _legal_cma_global(production_key)
    )


def _legal_bench_resample_matrix(key: SeedKey) -> bool:
    (
        _,
        _,
        purpose_code,
        discriminator,
        seed_kind,
        split_code,
        systemset_code,
    ) = key
    return (
        purpose_code in PURPOSE_CODES
        and discriminator == BENCH_RESAMPLING_KIND
        and seed_kind == RESAMPLE_MATRIX_KIND
        and split_code >= MIN_NONNEGATIVE
        and systemset_code >= MIN_NONNEGATIVE
    )


def _legal_bench_tost_power(key: SeedKey) -> bool:
    (
        _,
        _,
        purpose_code,
        discriminator,
        seed_kind,
        scenario_code,
        condition_code,
    ) = key
    production_key = (
        SEED_BASE,
        COMPONENT_RESAMPLING,
        seed_kind,
        scenario_code,
        condition_code,
    )
    return (
        purpose_code in PURPOSE_CODES
        and discriminator == BENCH_RESAMPLING_KIND
        and _legal_tost_power(production_key)
    )


_MAIN_M = _bounded(MAIN_SYSTEM_IDS, "main system id 0..503")
_TRAJ_J = _bounded(TRAJECTORY_IDS, "trajectory id 0..4")
_REPRO_J = _bounded(REPRODUCTION_IDS, "reproduction id 0..4")
_PILOT_M = _bounded(PILOT_SYSTEM_IDS, "pilot system id 0..50")
_POWER_REP = _bounded(POWER_REPLICATE_IDS, "power replicate 0..1999")
_E2_M = _bounded(E2_SYSTEM_IDS, "E2 system id 0..200")
_CONFIG = _bounded(CONFIG_CODES, "config code 1..17")
_SCENARIO = _bounded(SCENARIO_CODES, "scenario code 1..5")
_CONDITION = _bounded(CONDITION_CODES, "condition code 0..8")
_RESTART = _bounded(range(1, 4), "restart id 1..3")
_RETRY = _bounded(S3_RETRY_IDS, "retry id 0..20")
_PURPOSE = _bounded(PURPOSE_CODES, "purpose code 0..1")
_INFERENCE_SPLIT = _bounded(INFERENCE_SPLIT_CODES, "split code 1..2")
_SYSTEMSET = _bounded(SYSTEMSET_CODES, "system-set code 0..1")


_REGISTRY_SPECS = (
    SeedSpec(
        COMPONENT_MAIN_SYSTEM,
        "main_system",
        _shape(COMPONENT_MAIN_SYSTEM, ("m", _MAIN_M)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_MAIN_TRAJECTORY,
        "main_trajectory",
        _shape(COMPONENT_MAIN_TRAJECTORY, ("m", _MAIN_M), ("j", _TRAJ_J)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_REPRODUCTION,
        "reproduction",
        _shape(COMPONENT_REPRODUCTION, ("j", _REPRO_J)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_PILOT_SYSTEM,
        "pilot_system",
        _shape(COMPONENT_PILOT_SYSTEM, ("m", _PILOT_M)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_PILOT_TRAJECTORY,
        "pilot_trajectory",
        _shape(COMPONENT_PILOT_TRAJECTORY, ("m", _PILOT_M), ("j", _TRAJ_J)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_POWER,
        "power",
        _shape(COMPONENT_POWER, ("rep", _POWER_REP)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_Q_SELECTION,
        "q_selection",
        _shape(COMPONENT_Q_SELECTION),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_TIMING_SELECTION,
        "timing_selection",
        _shape(COMPONENT_TIMING_SELECTION),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_E2_SYSTEM,
        "e2_system",
        _shape(COMPONENT_E2_SYSTEM, ("m", _E2_M)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_E2_TRAJECTORY,
        "e2_trajectory",
        _shape(COMPONENT_E2_TRAJECTORY, ("m", _E2_M), ("j", _TRAJ_J)),
        _always_legal,
    ),
    SeedSpec(
        COMPONENT_OPTIMIZER,
        "sobol",
        _shape(
            COMPONENT_OPTIMIZER, ("config_code", _CONFIG), ("scenario_code", _SCENARIO)
        ),
        _legal_sobol,
    ),
    SeedSpec(
        COMPONENT_OPTIMIZER,
        "cma_global",
        _shape(
            COMPONENT_OPTIMIZER,
            ("config_code", _CONFIG),
            ("scenario_code", _SCENARIO),
            ("restart_id", _RESTART),
        ),
        _legal_cma_global,
    ),
    SeedSpec(
        COMPONENT_OPTIMIZER,
        "cma_oracle",
        _shape(
            COMPONENT_OPTIMIZER,
            ("config_code", _fixed(ORACLE_CONFIG_CODE, "exactly 16")),
            ("scenario_code", _SCENARIO),
            ("restart_id", _fixed(1, "exactly 1")),
            ("m", _MAIN_M),
        ),
        _legal_cma_oracle,
    ),
    SeedSpec(
        COMPONENT_S3_SECOND,
        "s3_second",
        _shape(COMPONENT_S3_SECOND, ("m", _MAIN_M), ("retry", _RETRY)),
        _legal_s3_second,
    ),
    SeedSpec(
        COMPONENT_RESAMPLING,
        "resample_matrix",
        _shape(
            COMPONENT_RESAMPLING,
            ("seed_kind", _fixed(RESAMPLE_MATRIX_KIND, "matrix discriminator 0")),
            ("split_code", _INFERENCE_SPLIT),
            ("systemset_code", _SYSTEMSET),
        ),
        _legal_resample_matrix,
    ),
    SeedSpec(
        COMPONENT_RESAMPLING,
        "tost_power",
        _shape(
            COMPONENT_RESAMPLING,
            ("seed_kind", _fixed(TOST_POWER_KIND, "TOST discriminator 1")),
            ("scenario_code", _SCENARIO),
            ("condition_code", _CONDITION),
        ),
        _legal_tost_power,
    ),
    SeedSpec(
        COMPONENT_BENCH,
        "bench_system",
        _shape(
            COMPONENT_BENCH,
            ("purpose_code", _PURPOSE),
            ("component_kind", _fixed(BENCH_SYSTEM_KIND, "system discriminator 0")),
            ("m", NONNEGATIVE),
        ),
        _legal_bench_system,
    ),
    SeedSpec(
        COMPONENT_BENCH,
        "bench_trajectory",
        _shape(
            COMPONENT_BENCH,
            ("purpose_code", _PURPOSE),
            (
                "component_kind",
                _fixed(BENCH_TRAJECTORY_KIND, "trajectory discriminator 1"),
            ),
            ("m", NONNEGATIVE),
            ("j", NONNEGATIVE),
        ),
        _legal_bench_trajectory,
    ),
    SeedSpec(
        COMPONENT_BENCH,
        "bench_cma",
        _shape(
            COMPONENT_BENCH,
            ("purpose_code", _PURPOSE),
            ("component_kind", _fixed(BENCH_CMA_KIND, "optimizer discriminator 10")),
            ("config_code", _CONFIG),
            ("scenario_code", _SCENARIO),
            ("restart_id", _RESTART),
        ),
        _legal_bench_cma,
    ),
    SeedSpec(
        COMPONENT_BENCH,
        "bench_s3_second",
        _shape(
            COMPONENT_BENCH,
            ("purpose_code", _PURPOSE),
            (
                "component_kind",
                _fixed(BENCH_S3_SECOND_KIND, "S3 second-system discriminator 11"),
            ),
            ("m", NONNEGATIVE),
            ("retry", _RETRY),
        ),
        _legal_bench_s3_second,
    ),
    SeedSpec(
        COMPONENT_BENCH,
        "bench_resample_matrix",
        _shape(
            COMPONENT_BENCH,
            ("purpose_code", _PURPOSE),
            (
                "component_kind",
                _fixed(BENCH_RESAMPLING_KIND, "resampling discriminator 12"),
            ),
            ("seed_kind", _fixed(RESAMPLE_MATRIX_KIND, "matrix discriminator 0")),
            ("split_code", NONNEGATIVE),
            ("systemset_code", NONNEGATIVE),
        ),
        _legal_bench_resample_matrix,
    ),
    SeedSpec(
        COMPONENT_BENCH,
        "bench_tost_power",
        _shape(
            COMPONENT_BENCH,
            ("purpose_code", _PURPOSE),
            (
                "component_kind",
                _fixed(BENCH_RESAMPLING_KIND, "resampling discriminator 12"),
            ),
            ("seed_kind", _fixed(TOST_POWER_KIND, "TOST discriminator 1")),
            ("scenario_code", _SCENARIO),
            ("condition_code", _CONDITION),
        ),
        _legal_bench_tost_power,
    ),
)

_REGISTRY_KEYS = tuple((spec.component, spec.seed_kind) for spec in _REGISTRY_SPECS)
if len(_REGISTRY_KEYS) != len(set(_REGISTRY_KEYS)):
    raise RuntimeError("duplicate (component, seed_kind) in seed registry")

SEED_REGISTRY = {(spec.component, spec.seed_kind): spec for spec in _REGISTRY_SPECS}


def validate_registry() -> None:
    components = {component for component, _ in SEED_REGISTRY}
    if components != EXPECTED_REGISTRY_COMPONENTS:
        raise RuntimeError(
            f"registry components {sorted(components)} != 0..{COMPONENT_BENCH}"
        )
    for registry_key, spec in SEED_REGISTRY.items():
        if registry_key != (spec.component, spec.seed_kind):
            raise RuntimeError(f"registry key/value mismatch: {registry_key!r}")
        names = tuple(field.name for field in spec.exact_shape)
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate field name in {registry_key!r}: {names!r}")
        if names[:KEY_PREFIX_LENGTH] != ("seed_base", "component"):
            raise RuntimeError(f"invalid leading shape in {registry_key!r}: {names!r}")


def _validate_key(component: int, seed_kind: str, key: SeedKey) -> SeedKey:
    registry_key = (component, seed_kind)
    if registry_key not in SEED_REGISTRY:
        raise SeedKeyError(
            f"unregistered seed kind: component={component}, kind={seed_kind!r}"
        )
    spec = SEED_REGISTRY[registry_key]
    if len(key) != len(spec.exact_shape):
        raise SeedKeyError(
            f"{seed_kind}: key length {len(key)} != {len(spec.exact_shape)}"
        )
    for value, field in zip(key, spec.exact_shape, strict=True):
        if not field.domain.contains(value):
            raise SeedKeyError(
                f"{seed_kind}: {field.name}={value!r} outside {field.domain.description}"
            )
    if not spec.legal_tuple(key):
        raise SeedKeyError(f"{seed_kind}: illegal joint key {key!r}")
    return key


def _seed_sequence(
    component: int, seed_kind: str, key: SeedKey
) -> np.random.SeedSequence:
    return np.random.SeedSequence(_validate_key(component, seed_kind, key))


def _registered_key(component: int, seed_kind: str, key: SeedKey) -> SeedKey:
    return _validate_key(component, seed_kind, key)


def condition_tuple_is_legal(
    config_code: int,
    scenario_code: int,
    condition_code: int,
) -> bool:
    if config_code not in CONFIG_CODES or scenario_code not in SCENARIO_CODES:
        return False
    if condition_code == MAIN_CONDITION_CODE:
        return True
    if condition_code in S3_DRIFT_CONDITION_CODES:
        return scenario_code == SCENARIO_CODE["S3-drift"]
    if condition_code in S4_CONDITION_CODES:
        return scenario_code == SCENARIO_CODE["S4"]
    if condition_code in W0_ONLY_CONDITION_CODES:
        return config_code == CONFIG_CODE["W0"]
    if condition_code == P8_CONDITION_CODE:
        return config_code in {CONFIG_CODE["W0"], CONFIG_CODE["WP2-weight-faithful"]}
    return False


def validate_condition_tuple(
    config_code: int,
    scenario_code: int,
    condition_code: int,
) -> None:
    if not condition_tuple_is_legal(config_code, scenario_code, condition_code):
        raise SeedKeyError(
            "illegal (config_code, scenario_code, condition_code): "
            f"{(config_code, scenario_code, condition_code)!r}"
        )


class RandnFromGenerator:
    __slots__ = ("rng",)

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, m: int, n: int) -> np.ndarray:
        return self.rng.standard_normal((m, n))


def sobol_key(config_code: int, scenario_code: int) -> np.random.SeedSequence:
    key = (SEED_BASE, COMPONENT_OPTIMIZER, config_code, scenario_code)
    return _seed_sequence(COMPONENT_OPTIMIZER, "sobol", key)


def cma_global_key(
    config_code: int,
    scenario_code: int,
    restart_id: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_OPTIMIZER,
        config_code,
        scenario_code,
        restart_id,
    )
    return _seed_sequence(COMPONENT_OPTIMIZER, "cma_global", key)


def cma_oracle_key(
    scenario_code: int,
    restart_id: int,
    m: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_OPTIMIZER,
        ORACLE_CONFIG_CODE,
        scenario_code,
        restart_id,
        m,
    )
    return _seed_sequence(COMPONENT_OPTIMIZER, "cma_oracle", key)


def s3_second_key(m: int, retry: int) -> np.random.SeedSequence:
    key = (SEED_BASE, COMPONENT_S3_SECOND, m, retry)
    return _seed_sequence(COMPONENT_S3_SECOND, "s3_second", key)


def resample_matrix_key(
    split_code: int,
    systemset_code: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_RESAMPLING,
        RESAMPLE_MATRIX_KIND,
        split_code,
        systemset_code,
    )
    return _seed_sequence(COMPONENT_RESAMPLING, "resample_matrix", key)


def tost_power_key(
    scenario_code: int,
    condition_code: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_RESAMPLING,
        TOST_POWER_KIND,
        scenario_code,
        condition_code,
    )
    return _seed_sequence(COMPONENT_RESAMPLING, "tost_power", key)


def bench_system_key(purpose_code: int, m: int) -> np.random.SeedSequence:
    key = (SEED_BASE, COMPONENT_BENCH, purpose_code, BENCH_SYSTEM_KIND, m)
    return _seed_sequence(COMPONENT_BENCH, "bench_system", key)


def bench_traj_key(
    purpose_code: int,
    m: int,
    j: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_BENCH,
        purpose_code,
        BENCH_TRAJECTORY_KIND,
        m,
        j,
    )
    return _seed_sequence(COMPONENT_BENCH, "bench_trajectory", key)


def bench_s3_second_key(
    purpose_code: int,
    m: int,
    retry: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_BENCH,
        purpose_code,
        BENCH_S3_SECOND_KIND,
        m,
        retry,
    )
    return _seed_sequence(COMPONENT_BENCH, "bench_s3_second", key)


def bench_cma_key(
    purpose_code: int,
    config_code: int,
    scenario_code: int,
    restart_id: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_BENCH,
        purpose_code,
        BENCH_CMA_KIND,
        config_code,
        scenario_code,
        restart_id,
    )
    return _seed_sequence(COMPONENT_BENCH, "bench_cma", key)


def _declared_domain(values: Collection[int], name: str) -> frozenset[int]:
    domain = frozenset(values)
    if not domain:
        raise SeedKeyError(f"{name} must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < MIN_NONNEGATIVE
        for value in domain
    ):
        raise SeedKeyError(f"{name} must contain only non-negative integers")
    return domain


def bench_resample_matrix_key(
    purpose_code: int,
    split_code: int,
    systemset_code: int,
    *,
    allowed_split_codes: Collection[int] = INFERENCE_SPLIT_CODES,
    allowed_systemset_codes: Collection[int] = SYSTEMSET_CODES,
) -> np.random.SeedSequence:
    split_domain = _declared_domain(allowed_split_codes, "allowed_split_codes")
    systemset_domain = _declared_domain(
        allowed_systemset_codes,
        "allowed_systemset_codes",
    )
    if split_code not in split_domain or systemset_code not in systemset_domain:
        raise SeedKeyError(
            "bench resampling key is outside the mini spec's declared domain: "
            f"split={split_code}, systemset={systemset_code}"
        )
    key = (
        SEED_BASE,
        COMPONENT_BENCH,
        purpose_code,
        BENCH_RESAMPLING_KIND,
        RESAMPLE_MATRIX_KIND,
        split_code,
        systemset_code,
    )
    return _seed_sequence(COMPONENT_BENCH, "bench_resample_matrix", key)


def bench_tost_power_key(
    purpose_code: int,
    scenario_code: int,
    condition_code: int,
) -> np.random.SeedSequence:
    key = (
        SEED_BASE,
        COMPONENT_BENCH,
        purpose_code,
        BENCH_RESAMPLING_KIND,
        TOST_POWER_KIND,
        scenario_code,
        condition_code,
    )
    return _seed_sequence(COMPONENT_BENCH, "bench_tost_power", key)


def assert_component_isolation(
    key: Collection[int],
    *,
    namespace: Literal["production", "bench"],
) -> None:
    entropy = tuple(key)
    if len(entropy) < KEY_PREFIX_LENGTH:
        raise SeedKeyError(f"seed key is too short: {entropy!r}")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in entropy
    ):
        raise SeedKeyError(f"seed key must contain only integers: {entropy!r}")
    if entropy[KEY_SEED_BASE_INDEX] != SEED_BASE:
        raise SeedKeyError(f"unexpected seed base: {entropy[KEY_SEED_BASE_INDEX]!r}")
    component = int(entropy[KEY_COMPONENT_INDEX])
    if namespace == "production":
        if component not in PRODUCTION_COMPONENTS:
            raise SeedKeyError(f"component {component} is not a production component")
        return
    if namespace == "bench":
        if component not in BENCH_COMPONENTS:
            raise SeedKeyError(f"component {component} is not the bench component")
        return
    raise SeedKeyError(f"unknown namespace {namespace!r}")


def enumerate_legal_keys() -> Iterator[SeedKey]:
    for m in MAIN_SYSTEM_IDS:
        yield _registered_key(
            COMPONENT_MAIN_SYSTEM, "main_system", (SEED_BASE, COMPONENT_MAIN_SYSTEM, m)
        )
        for j in TRAJECTORY_IDS:
            yield _registered_key(
                COMPONENT_MAIN_TRAJECTORY,
                "main_trajectory",
                (SEED_BASE, COMPONENT_MAIN_TRAJECTORY, m, j),
            )
    for j in REPRODUCTION_IDS:
        yield _registered_key(
            COMPONENT_REPRODUCTION,
            "reproduction",
            (SEED_BASE, COMPONENT_REPRODUCTION, j),
        )
    for m in PILOT_SYSTEM_IDS:
        yield _registered_key(
            COMPONENT_PILOT_SYSTEM,
            "pilot_system",
            (SEED_BASE, COMPONENT_PILOT_SYSTEM, m),
        )
        for j in TRAJECTORY_IDS:
            yield _registered_key(
                COMPONENT_PILOT_TRAJECTORY,
                "pilot_trajectory",
                (SEED_BASE, COMPONENT_PILOT_TRAJECTORY, m, j),
            )
    for rep in POWER_REPLICATE_IDS:
        yield _registered_key(
            COMPONENT_POWER, "power", (SEED_BASE, COMPONENT_POWER, rep)
        )
    yield _registered_key(
        COMPONENT_Q_SELECTION, "q_selection", (SEED_BASE, COMPONENT_Q_SELECTION)
    )
    yield _registered_key(
        COMPONENT_TIMING_SELECTION,
        "timing_selection",
        (SEED_BASE, COMPONENT_TIMING_SELECTION),
    )
    for m in E2_SYSTEM_IDS:
        yield _registered_key(
            COMPONENT_E2_SYSTEM, "e2_system", (SEED_BASE, COMPONENT_E2_SYSTEM, m)
        )
        for j in TRAJECTORY_IDS:
            yield _registered_key(
                COMPONENT_E2_TRAJECTORY,
                "e2_trajectory",
                (SEED_BASE, COMPONENT_E2_TRAJECTORY, m, j),
            )
    for config_code in sorted(SOBOL_CONFIG_CODES):
        for scenario_code in sorted(SCENARIO_CODES):
            yield _registered_key(
                COMPONENT_OPTIMIZER,
                "sobol",
                (SEED_BASE, COMPONENT_OPTIMIZER, config_code, scenario_code),
            )
    for config_code in sorted(GLOBAL_CMA_CONFIG_CODES):
        if config_code in ONE_DIM_CONFIG_CODES:
            restart_ids = ONE_DIM_RESTART_IDS
        elif config_code in TWO_DIM_CONFIG_CODES:
            restart_ids = TWO_DIM_RESTART_IDS
        else:
            restart_ids = SIX_DIM_RESTART_IDS
        for scenario_code in sorted(SCENARIO_CODES):
            for restart_id in restart_ids:
                yield _registered_key(
                    COMPONENT_OPTIMIZER,
                    "cma_global",
                    (
                        SEED_BASE,
                        COMPONENT_OPTIMIZER,
                        config_code,
                        scenario_code,
                        restart_id,
                    ),
                )
    for m in VAL_SYSTEM_IDS:
        yield _registered_key(
            COMPONENT_OPTIMIZER,
            "cma_oracle",
            (
                SEED_BASE,
                COMPONENT_OPTIMIZER,
                ORACLE_CONFIG_CODE,
                SCENARIO_CODE["S2"],
                PRIMARY_RESTART_ID,
                m,
            ),
        )
    for m in TEST_SYSTEM_IDS:
        for scenario_code in sorted(SCENARIO_CODES):
            yield _registered_key(
                COMPONENT_OPTIMIZER,
                "cma_oracle",
                (
                    SEED_BASE,
                    COMPONENT_OPTIMIZER,
                    ORACLE_CONFIG_CODE,
                    scenario_code,
                    PRIMARY_RESTART_ID,
                    m,
                ),
            )
    for m in MAIN_SYSTEM_IDS:
        for retry in S3_RETRY_IDS:
            yield _registered_key(
                COMPONENT_S3_SECOND,
                "s3_second",
                (SEED_BASE, COMPONENT_S3_SECOND, m, retry),
            )
    for split_code in sorted(INFERENCE_SPLIT_CODES):
        for systemset_code in sorted(SYSTEMSET_CODES):
            yield _registered_key(
                COMPONENT_RESAMPLING,
                "resample_matrix",
                (
                    SEED_BASE,
                    COMPONENT_RESAMPLING,
                    RESAMPLE_MATRIX_KIND,
                    split_code,
                    systemset_code,
                ),
            )
    yield _registered_key(
        COMPONENT_RESAMPLING,
        "tost_power",
        (
            SEED_BASE,
            COMPONENT_RESAMPLING,
            TOST_POWER_KIND,
            SCENARIO_CODE["S2"],
            MAIN_CONDITION_CODE,
        ),
    )


validate_registry()
if SEED_BASE != Config().seed_base:
    raise RuntimeError(
        f"SEED_BASE={SEED_BASE} != Config().seed_base={Config().seed_base}"
    )
