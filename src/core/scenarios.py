from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Callable, Mapping
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from systems import (
    sample_system,
    simulate,
    pack_rho,
    unpack_rho,
    build_F,
    spectral_radius,
    rho_dim,
)
from rng import (
    SCENARIO_CODE,
    s3_second_key,
    bench_system_key,
    bench_s3_second_key,
    assert_component_isolation,
)
from config import Config, traj_seed


SCENARIO_CONDITION_DOMAIN: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 0),
        (5, 3),
        (5, 4),
        (5, 5),
    }
)

_DRIFT_SEVERITY = {0: 0.5, 1: 0.25, 2: 1.0}
_S4_EPSILON = {0: 0.05, 3: 0.0, 4: 0.01, 5: 0.10}
_S4_KAPPA = 10.0
_S3_JUMP_INDEX = 1500
_OUTER_RETRY_PENDING = "PENDING: merge the registered s3_outer_retry_cost measurement"


class ScenarioError(Exception):
    pass


class ScenarioInfeasible(ScenarioError):
    pass


class ScenarioGateUnspecified(ScenarioError):
    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class DriftPlanEntry:
    m: int
    second_system: Mapping[str, Any]
    n_retry: int
    source_rho_sha256: str


def _packed(sys: Mapping[str, Any]) -> np.ndarray:
    return pack_rho(
        np.asarray(sys["A"]),
        np.asarray(sys["B"]),
        np.asarray(sys["f"]),
        int(sys["d"]),
    )


def _rho_sha256(sys: Mapping[str, Any]) -> str:
    buf = np.ascontiguousarray(_packed(sys), dtype="<f8").tobytes()
    return hashlib.sha256(buf).hexdigest()


def _canonical_list_sha256(values: list[int]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_condition(scenario_code: int, condition_code: int) -> None:
    pair = (int(scenario_code), int(condition_code))
    if pair in SCENARIO_CONDITION_DOMAIN:
        return
    if condition_code in {6, 7, 8}:
        raise ScenarioError(
            "condition_code 6/7 is W0-only and 8 is the D12 p sweep; these are "
            "method-layer conditions and do not alter data generation; pass "
            "condition_code=0 to obtain data"
        )
    raise ScenarioError(f"illegal scenario/condition pair: {pair!r}")


def _second_system(m: int, retry: int, cfg: Config, d: int) -> Mapping[str, Any]:
    key = s3_second_key(m, retry)
    assert_component_isolation(key.entropy, namespace="production")
    return sample_system(np.random.default_rng(key), cfg, d)


def _rho_path_is_stable(
    rho_1: np.ndarray, rho_2: np.ndarray, cfg: Config, d: int
) -> bool:
    for severity in (0.25, 0.5, 1.0):
        for k in range(1, cfg.N + 1):
            alpha = severity * k / cfg.N
            rho_k = (1.0 - alpha) * rho_1 + alpha * rho_2
            A_k, B_k, _ = unpack_rho(rho_k, cfg.n, d)
            if spectral_radius(build_F(A_k, B_k, d)) >= cfg.rho_max:
                return False
    return True


def _simulate_rho_path(
    base: Mapping[str, Any], rho_path: np.ndarray, cfg: Config, d: int
) -> dict[str, Any]:
    u = np.asarray(base["u"]).copy()
    v = np.asarray(base["v"]).copy()
    x = np.zeros((cfg.N + 2, cfg.n), dtype=float)
    y = np.zeros(cfg.N + 1, dtype=float)
    for k in range(1, cfg.N + 1):
        y[k] = x[k, 0] + v[k]
        A_k, B_k, f_k = unpack_rho(rho_path[k], cfg.n, d)
        delayed = x[k - d] if k - d > 0 else np.zeros(cfg.n, dtype=float)
        x[k + 1] = A_k @ x[k] + B_k @ delayed + f_k * u[k]
    return {"u": u, "v": v, "y": y, "x": x, "N": cfg.N}


def _jump_rho_path(
    rho_1: np.ndarray,
    rho_2: np.ndarray,
    cfg: Config,
    *,
    enforce_registered_length: bool,
) -> np.ndarray:
    if enforce_registered_length and cfg.N != 3000:
        raise ScenarioError("S3-jump is pre-registered at N=3000 with k_jump=1500")
    jump = _S3_JUMP_INDEX if enforce_registered_length else cfg.N // 2
    shape = (cfg.N + 1, rho_1.size)
    rho_path = np.broadcast_to(rho_1, shape).copy()
    rho_path[jump + 1 :] = rho_2
    return rho_path


def _drift_rho_path(
    rho_1: np.ndarray,
    rho_2: np.ndarray,
    cfg: Config,
    severity: float,
) -> np.ndarray:
    alpha = severity * np.arange(cfg.N + 1, dtype=float) / cfg.N
    return (1.0 - alpha[:, None]) * rho_1 + alpha[:, None] * rho_2


def _apply_s4_transform(
    base: Mapping[str, Any],
    generator: np.random.Generator,
    cfg: Config,
    epsilon: float,
) -> tuple[dict[str, Any], np.ndarray]:
    branch_uniform = generator.random(cfg.N)
    outlier_index = np.flatnonzero(branch_uniform < epsilon) + 1
    multiplier = np.where(branch_uniform < epsilon, _S4_KAPPA, 1.0)
    v_s4 = np.asarray(base["v"]).copy()
    v_s4[1:] = v_s4[1:] * multiplier / np.sqrt(1.0 + epsilon * (_S4_KAPPA**2 - 1.0))
    y_s4 = np.asarray(base["y"]).copy()
    y_s4[1:] = np.asarray(base["x"])[1 : cfg.N + 1, 0] + v_s4[1:]
    return {**base, "v": v_s4, "y": y_s4}, outlier_index


def _unit_trajectory_for_scenario(
    sampled: Mapping[str, Any],
    generator: np.random.Generator,
    scenario_code: int,
    cfg: Config,
) -> dict[str, Any]:
    base = simulate(sampled, generator, cfg.N)
    rho_1 = _packed(sampled)
    d = int(sampled["d"])
    if scenario_code == SCENARIO_CODE["S3-jump"]:
        rho_path = _jump_rho_path(
            rho_1,
            rho_1,
            cfg,
            enforce_registered_length=False,
        )
        return _simulate_rho_path(base, rho_path, cfg, d)
    if scenario_code == SCENARIO_CODE["S3-drift"]:
        rho_path = _drift_rho_path(
            rho_1,
            rho_1,
            cfg,
            _DRIFT_SEVERITY[0],
        )
        return _simulate_rho_path(base, rho_path, cfg, d)
    if scenario_code == SCENARIO_CODE["S4"]:
        transformed, _ = _apply_s4_transform(
            base,
            generator,
            cfg,
            _S4_EPSILON[0],
        )
        return transformed
    return base


def scenario_trajectory(
    sys: Mapping[str, Any],
    m: int,
    j: int,
    scenario_code: int,
    condition_code: int,
    cfg: Config,
    drift_plan: Mapping[int, DriftPlanEntry] | None = None,
) -> dict[str, Any]:
    _validate_condition(scenario_code, condition_code)
    d = int(sys["d"])
    rho_1 = _packed(sys)

    if scenario_code == SCENARIO_CODE["S3-jump"] and cfg.N != 3000:
        raise ScenarioError("S3-jump is pre-registered at N=3000 with k_jump=1500")

    drift_entry: DriftPlanEntry | None = None
    if scenario_code == SCENARIO_CODE["S3-drift"]:
        if drift_plan is None:
            raise ScenarioError("S3-drift requires the prebuilt drift_plan")
        if m not in drift_plan:
            raise ScenarioError(f"S3-drift plan has no entry for global m={m}")
        drift_entry = drift_plan[m]
        if int(sys["m"]) != m or int(drift_entry.m) != m:
            raise ScenarioError("S3-drift plan/system m identity mismatch")
        if d != int(drift_entry.second_system["d"]):
            raise ScenarioError("S3-drift plan/system delay stratum mismatch")
        if _rho_sha256(sys) != drift_entry.source_rho_sha256:
            raise ScenarioError("S3-drift source system SHA-256 mismatch")

    key = traj_seed(cfg, m, j)
    assert_component_isolation(key.entropy, namespace="production")
    generator = np.random.default_rng(key)
    base = simulate(sys, generator, cfg.N)

    shape = (cfg.N + 1, rho_dim(cfg.n, d))
    rho_path = np.broadcast_to(rho_1, shape).copy()
    n_retry = 0
    jump_magnitude: float | None = None
    severity: float | None = None
    epsilon: float | None = None
    outlier_index: np.ndarray | None = None

    if scenario_code == SCENARIO_CODE["S3-jump"]:
        second = _second_system(m, 0, cfg, d)
        rho_2 = _packed(second)
        rho_path = _jump_rho_path(
            rho_1,
            rho_2,
            cfg,
            enforce_registered_length=True,
        )
        jump_magnitude = float(np.linalg.norm(rho_2 - rho_1) / np.linalg.norm(rho_1))
        base = _simulate_rho_path(base, rho_path, cfg, d)
    elif scenario_code == SCENARIO_CODE["S3-drift"]:
        if drift_entry is None:
            raise ScenarioError("S3-drift plan validation did not produce an entry")
        rho_2 = _packed(drift_entry.second_system)
        severity = _DRIFT_SEVERITY[condition_code]
        rho_path = _drift_rho_path(rho_1, rho_2, cfg, severity)
        n_retry = int(drift_entry.n_retry)
        base = _simulate_rho_path(base, rho_path, cfg, d)
    elif scenario_code == SCENARIO_CODE["S4"]:
        epsilon = _S4_EPSILON[condition_code]
        base, outlier_index = _apply_s4_transform(base, generator, cfg, epsilon)

    provenance = {
        "seed_key": key.entropy,
        "n_retry": n_retry,
        "jump_magnitude": jump_magnitude,
        "s": severity,
        "epsilon": epsilon,
        "outlier_index": outlier_index,
        "scenario_code": int(scenario_code),
        "condition_code": int(condition_code),
    }
    return {
        **base,
        "rho_star_traj": rho_path,
        "scenario_code": int(scenario_code),
        "condition_code": int(condition_code),
        "provenance": provenance,
    }


def build_s3_drift_family(
    family: Mapping[str, list[Mapping[str, Any]]], cfg: Config
) -> dict[str, Any]:
    required_splits = {"train", "val", "test", "pilot"}
    if set(family) != required_splits:
        raise ScenarioError(
            f"family splits must be exactly {sorted(required_splits)!r}"
        )

    plan: dict[int, DriftPlanEntry] = {}
    audit: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        surviving: list[int] = []
        d_labels: list[int] = []
        excluded: list[int] = []
        retries: dict[int, int] = {}
        jumps: dict[int, float] = {}
        for sys in family[split]:
            m = int(sys["m"])
            d = int(sys["d"])
            rho_1 = _packed(sys)
            accepted: Mapping[str, Any] | None = None
            accepted_retry = 20
            for retry in range(21):
                candidate = _second_system(m, retry, cfg, d)
                rho_2 = _packed(candidate)
                if _rho_path_is_stable(rho_1, rho_2, cfg, d):
                    accepted = candidate
                    accepted_retry = retry
                    break
            retries[m] = int(accepted_retry)
            if accepted is None:
                excluded.append(m)
                continue
            entry = DriftPlanEntry(m, accepted, accepted_retry, _rho_sha256(sys))
            plan[m] = entry
            surviving.append(m)
            d_labels.append(d)
            rho_2 = _packed(accepted)
            jumps[m] = float(np.linalg.norm(rho_2 - rho_1) / np.linalg.norm(rho_1))

        per_d_remaining = {
            int(d): sum(label == d for label in d_labels) for d in cfg.d_choices
        }
        audit[split] = {
            "surviving_system_ids": surviving,
            "d_labels": d_labels,
            "sha256_ids": _canonical_list_sha256(surviving),
            "sha256_labels": _canonical_list_sha256(d_labels),
            "n_retry_by_system": retries,
            "excluded_system_ids": excluded,
            "exclusion_ratio": len(excluded) / len(family[split]),
            "per_d_remaining": per_d_remaining,
            "jump_magnitude_by_system": jumps,
        }

    audit["pilot"] = {
        "status": "not applicable",
        "reason": "D9 never states whether S3-drift is built on pilot",
    }

    unspecified = [
        split for split in ("train", "val") if audit[split]["excluded_system_ids"]
    ]
    if unspecified:
        names = ", ".join(unspecified)
        raise ScenarioGateUnspecified(
            "D9 only defines the S3-drift feasibility gate for test; "
            f"{names} contains exclusions and requires human adjudication",
            audit=audit,
        )
    test_audit = audit["test"]
    if len(test_audit["excluded_system_ids"]) > 10 or any(
        test_audit["per_d_remaining"][int(d)] < 60 for d in cfg.d_choices
    ):
        raise ScenarioInfeasible(
            "S3-drift test split is SCENARIO INFEASIBLE under the D9 gate"
        )
    return {"drift_plan": plan, "audit_by_split": audit}


StreamFactory = Callable[[int, int, str, int], np.random.Generator]
SystemIdFactory = Callable[[int, int, int], int]


def scenario_unit_cost(
    cfg: Config,
    n_repeat: int,
    *,
    stream_factory: StreamFactory,
) -> dict[str, dict[str, dict[str, float | None | str]]]:
    if isinstance(n_repeat, bool) or not isinstance(n_repeat, int) or n_repeat < 1:
        raise ValueError("n_repeat must be a positive integer")
    result: dict[str, dict[str, dict[str, float | None | str]]] = {}
    for scenario_code in sorted(SCENARIO_CODE.values()):
        by_delay: dict[str, dict[str, float | None | str]] = {}
        for d in cfg.d_choices:
            sys_times: list[float] = []
            traj_times: list[float] = []
            rejects: list[float] = []
            for index in (-1, *range(n_repeat)):
                sys_rng = stream_factory(scenario_code, int(d), "system", index)
                traj_rng = stream_factory(scenario_code, int(d), "trajectory", index)
                t0 = perf_counter()
                sampled = sample_system(sys_rng, cfg, int(d))
                t1 = perf_counter()
                t2 = perf_counter()
                _unit_trajectory_for_scenario(
                    sampled,
                    traj_rng,
                    scenario_code,
                    cfg,
                )
                t3 = perf_counter()
                if index >= 0:
                    sys_times.append(float(t1 - t0))
                    traj_times.append(float(t3 - t2))
                    rejects.append(float(sampled["n_reject"]))
            t_sys = float(np.median(np.asarray(sys_times, dtype=float)))
            t_traj = float(np.median(np.asarray(traj_times, dtype=float)))
            n_reject = float(np.median(np.asarray(rejects, dtype=float)))
            if scenario_code in {
                SCENARIO_CODE["S3-jump"],
                SCENARIO_CODE["S3-drift"],
            }:
                t_outer: float | None = None
                n_outer: float | None = None
                status = _OUTER_RETRY_PENDING
            else:
                t_outer = t_sys
                n_outer = 0.0
                status = "no outer retry by construction"
            by_delay[str(d)] = {
                "t_sys_call_s": t_sys,
                "t_sys_with_outer_retry_s": t_outer,
                "t_traj_s": t_traj,
                "n_reject_mean": n_reject,
                "n_outer_retry_mean": n_outer,
                "outer_retry_status": status,
            }
        result[str(scenario_code)] = by_delay
    return result


def s3_outer_retry_cost(
    cfg: Config,
    n_repeat: int,
    *,
    purpose_code: int,
    system_id_factory: SystemIdFactory,
) -> dict[str, dict[str, dict[str, float | str]]]:
    if isinstance(n_repeat, bool) or not isinstance(n_repeat, int) or n_repeat < 1:
        raise ValueError("n_repeat must be a positive integer")

    sample_ids: dict[tuple[int, int, int], int] = {}
    seen_ids: set[int] = set()
    s3_codes = (
        SCENARIO_CODE["S3-jump"],
        SCENARIO_CODE["S3-drift"],
    )
    for scenario_code in s3_codes:
        for d in cfg.d_choices:
            for index in (-1, *range(n_repeat)):
                m = system_id_factory(scenario_code, int(d), index)
                if isinstance(m, bool) or not isinstance(m, Integral) or m < 0:
                    raise ValueError(
                        "system_id_factory must return nonnegative integers"
                    )
                m = int(m)
                if m in seen_ids:
                    raise ValueError(
                        "system_id_factory must return a distinct m for every S3 cell/sample"
                    )
                seen_ids.add(m)
                sample_ids[(scenario_code, int(d), index)] = m

    result: dict[str, dict[str, dict[str, float | str]]] = {}
    for scenario_code in s3_codes:
        by_delay: dict[str, dict[str, float | str]] = {}
        for d in cfg.d_choices:
            total_times: list[float] = []
            outer_retries: list[float] = []
            exhausted_samples = 0
            for index in (-1, *range(n_repeat)):
                m = sample_ids[(scenario_code, int(d), index)]
                first_key = bench_system_key(purpose_code, m)
                assert_component_isolation(first_key.entropy, namespace="bench")
                first_rng = np.random.default_rng(first_key)

                t0 = perf_counter()
                first = sample_system(first_rng, cfg, int(d))
                rho_1 = _packed(first)
                retry_count = 0
                exhausted = False
                for retry in range(21):
                    second_key = bench_s3_second_key(purpose_code, m, retry)
                    assert_component_isolation(second_key.entropy, namespace="bench")
                    second_rng = np.random.default_rng(second_key)
                    second = sample_system(second_rng, cfg, int(d))
                    if scenario_code == SCENARIO_CODE["S3-jump"] or _rho_path_is_stable(
                        rho_1,
                        _packed(second),
                        cfg,
                        int(d),
                    ):
                        retry_count = retry
                        break
                    if retry == 20:
                        retry_count = 20
                        exhausted = True
                t1 = perf_counter()

                if index >= 0:
                    total_times.append(float(t1 - t0))
                    outer_retries.append(float(retry_count))
                    exhausted_samples += int(exhausted)

            by_delay[str(d)] = {
                "t_sys_with_outer_retry_s": float(
                    np.median(np.asarray(total_times, dtype=float))
                ),
                "n_outer_retry_mean": float(
                    np.median(np.asarray(outer_retries, dtype=float))
                ),
                "outer_retry_status": (
                    "measured: component-13 bench_s3_second; "
                    f"exhausted_samples={exhausted_samples}"
                ),
            }
        result[str(scenario_code)] = by_delay
    return result


__all__ = [
    "SCENARIO_CONDITION_DOMAIN",
    "ScenarioError",
    "ScenarioInfeasible",
    "ScenarioGateUnspecified",
    "DriftPlanEntry",
    "scenario_trajectory",
    "build_s3_drift_family",
    "scenario_unit_cost",
    "s3_outer_retry_cost",
]
