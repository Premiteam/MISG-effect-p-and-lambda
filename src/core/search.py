from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import cma
import numpy as np
from scipy.stats import qmc

from rng import (
    RandnFromGenerator,
    bench_cma_key,
    cma_global_key,
    cma_oracle_key,
    sobol_key,
)


CHECKPOINT_SCHEMA_VERSION = 1
SEARCH_ALLOCATION = 300
SIGMA0 = 0.3
BOX = (0.0, 1.0)
INITIAL_MEAN = 0.5
CENTRE_TOL = 1e-9
TIE_TOL = 1e-12
TIE_RULE = "J_val diff <= 1e-12 -> smallest restart_id"
ORACLE_CONFIG_CODE = 16
TRAJECTORIES_PER_SYSTEM = 3
INCUMBENT_RESTART_ID = 1
INCUMBENT_GEN_ID = -1
INCUMBENT_CANDIDATE_ID = -1
TERMINAL_BUDGET_STATES = frozenset({"success", "RESTART FAILED"})
GEN_RECORD_KEYS = frozenset(
    {
        "row_kind",
        "restart_id",
        "gen_id",
        "candidate_id",
        "x_raw",
        "x_box",
        "J",
        "n_div",
        "stop_flags",
        "logical_eval_count_at_commit",
        "discarded_alloc_at_commit",
        "cum_scope",
    }
)
CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "run_manifest_hash",
        "l2_identity",
        "cma_state",
        "sobol_state",
        "cursor",
        "budget",
        "gen_records",
        "oracle",
        "terminal",
    }
)
SEARCH_KEY_FIELDS = (
    "config_code",
    "scenario_code",
    "condition_code",
    "split_code",
    "system_id",
)

DIM_TIERS = MappingProxyType(
    {
        1: MappingProxyType({"popsize": 4, "n_restart": 3, "quota": 100, "gens": 25}),
        2: MappingProxyType({"popsize": 6, "n_restart": 2, "quota": 150, "gens": 25}),
        6: MappingProxyType({"popsize": 9, "n_restart": 1, "quota": 300, "gens": 33}),
    }
)

_FINAL_REF_RE = re.compile(r"r([1-9][0-9]*)g([1-9][0-9]*)c(0|[1-9][0-9]*)")


class SearchError(RuntimeError):
    pass


class IntegrityError(SearchError):
    pass


class GenerationIntegrityError(IntegrityError):
    pass


class RestartFailed(SearchError):
    pass


class SearchFailed(SearchError):
    pass


class OracleImplError(SearchError):
    pass


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    x_raw: np.ndarray
    x_box: np.ndarray


@dataclass(frozen=True)
class EvalResult:
    candidate_id: int
    J: float
    n_div: int = 0


def _is_plain_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _require_plain_int(value: object, name: str) -> int:
    if not _is_plain_int(value):
        raise IntegrityError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _validate_identity(identity: Mapping[str, object]) -> dict[str, int]:
    if set(identity) != set(SEARCH_KEY_FIELDS):
        raise IntegrityError(
            f"identity keys must be {SEARCH_KEY_FIELDS!r}, got {tuple(identity)!r}"
        )
    return {
        field: _require_plain_int(identity[field], f"identity[{field!r}]")
        for field in SEARCH_KEY_FIELDS
    }


def _validate_manifest_hash(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError("run_manifest_hash must be a 64-character string")
    if any(char not in "0123456789abcdef" for char in value):
        raise IntegrityError("run_manifest_hash must contain lowercase hexadecimal")
    return value


def phi(R: float, r_cap: float, log_nonfinite: Callable[[float], None]) -> float:
    value = float(R)
    cap = float(r_cap)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError("r_cap must be finite and non-negative")
    if np.isnan(value) or value == np.inf:
        log_nonfinite(value)
        return cap
    if value < 0.0:
        log_nonfinite(value)
        raise SearchFailed("negative_R")
    return min(value, cap)


def _ordered_risks(
    risks: Mapping[int, Sequence[float]] | Sequence[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    if isinstance(risks, Mapping):
        rows: list[tuple[int, int, float]] = []
        for system_id in sorted(risks):
            m = _require_plain_int(system_id, "system_id")
            values = risks[system_id]
            if len(values) != TRAJECTORIES_PER_SYSTEM:
                raise IntegrityError(
                    f"system_id={m} must have exactly three trajectories"
                )
            rows.extend((m, j, float(value)) for j, value in enumerate(values))
        return rows

    rows = [
        (
            _require_plain_int(row[0], "system_id"),
            _require_plain_int(row[1], "trajectory_id"),
            float(row[2]),
        )
        for row in risks
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def aggregate_objective(
    risks: Mapping[int, Sequence[float]] | Sequence[tuple[int, int, float]],
    r_cap: float,
    log_nonfinite: Callable[[float], None],
) -> float:
    rows = _ordered_risks(risks)
    if not rows:
        raise IntegrityError("risk collection must contain at least one system")

    systems: dict[int, list[int]] = {}
    total = 0.0
    for system_id, trajectory_id, value in rows:
        systems.setdefault(system_id, []).append(trajectory_id)
        total += phi(value, r_cap, log_nonfinite)
    for system_id, trajectory_ids in systems.items():
        if trajectory_ids != list(range(TRAJECTORIES_PER_SYSTEM)):
            raise IntegrityError(
                f"system_id={system_id} trajectory ids must be [0, 1, 2], "
                f"got {trajectory_ids!r}"
            )
    objective = total / (TRAJECTORIES_PER_SYSTEM * len(systems))
    if not np.isfinite(objective):
        raise SearchFailed("aggregate_nonfinite")
    return float(objective)


def j_train(
    risks: Mapping[int, Sequence[float]] | Sequence[tuple[int, int, float]],
    r_cap: float,
    log_nonfinite: Callable[[float], None],
) -> float:
    return aggregate_objective(risks, r_cap, log_nonfinite)


def j_val(
    risks: Mapping[int, Sequence[float]] | Sequence[tuple[int, int, float]],
    r_cap: float,
    log_nonfinite: Callable[[float], None],
) -> float:
    return aggregate_objective(risks, r_cap, log_nonfinite)


def parse_final_ref(final_ref: str) -> tuple[int, int, int]:
    if not isinstance(final_ref, str):
        raise IntegrityError("final_ref must be a string")
    match = _FINAL_REF_RE.fullmatch(final_ref)
    if match is None:
        raise IntegrityError(f"invalid final_ref={final_ref!r}")
    return tuple(int(part) for part in match.groups())


def locate_w3_x_box(
    selection: Mapping[str, Any],
    checkpoints_by_restart: Mapping[int, Mapping[str, Any]],
) -> np.ndarray:
    winner_id = _require_plain_int(
        selection.get("winner_restart_id"), "winner_restart_id"
    )
    candidates = selection.get("candidates")
    if not isinstance(candidates, Sequence):
        raise IntegrityError("selection candidates must be a sequence")
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("restart_id") == winner_id
    ]
    if len(matches) != 1:
        raise IntegrityError(
            f"winner_restart_id={winner_id} matched {len(matches)} candidates"
        )
    final_ref = matches[0].get("final_ref")
    r, g, c = parse_final_ref(final_ref)
    if r not in checkpoints_by_restart:
        raise IntegrityError(f"missing checkpoint for restart_id={r}")
    checkpoint = checkpoints_by_restart[r]
    checkpoint_restart = checkpoint.get("l2_identity", {}).get("restart_id")
    if checkpoint_restart != r:
        raise IntegrityError(f"checkpoint identity mismatch: expected restart_id={r}")
    records = checkpoint.get("gen_records")
    if not isinstance(records, Sequence):
        raise IntegrityError("checkpoint gen_records must be a sequence")
    matching_records = [
        record
        for record in records
        if record.get("gen_id") == g and record.get("candidate_id") == c
    ]
    if len(matching_records) != 1:
        raise IntegrityError(
            f"final_ref={final_ref!r} matched {len(matching_records)} gen_records"
        )
    x_box = matching_records[0].get("x_box")
    if not isinstance(x_box, np.ndarray):
        raise IntegrityError("selected gen_record x_box must be an ndarray")
    return x_box


def rebuild_coordinator(
    checkpoints_by_restart: Mapping[int, Mapping[str, Any]],
    expected_restart_ids: frozenset[int],
    started_evidence: frozenset[int],
    require_complete: bool,
) -> dict[str, int]:
    expected = frozenset(
        _require_plain_int(restart_id, "expected restart_id")
        for restart_id in expected_restart_ids
    )
    evidence = frozenset(
        _require_plain_int(restart_id, "evidence restart_id")
        for restart_id in started_evidence
    )
    present = frozenset(
        _require_plain_int(restart_id, "checkpoint restart_id")
        for restart_id in checkpoints_by_restart
    )
    if not expected:
        raise IntegrityError("expected_restart_ids must not be empty")
    extra = present - expected
    if extra:
        raise IntegrityError(f"unexpected restart checkpoints: {sorted(extra)!r}")
    if require_complete:
        missing = expected - present
        if missing:
            raise IntegrityError(
                f"terminal search is missing restarts: {sorted(missing)!r}"
            )
    else:
        missing_with_evidence = (expected - present) & evidence
        if missing_with_evidence:
            raise IntegrityError(
                "started restarts are missing checkpoints: "
                f"{sorted(missing_with_evidence)!r}"
            )

    logical_total = 0
    discarded_total = 0
    allocation_total = 0
    manifest_hash: str | None = None
    search_identity: dict[str, int] | None = None
    for restart_id in sorted(present):
        checkpoint = checkpoints_by_restart[restart_id]
        _validate_checkpoint_shape(checkpoint)
        identity = checkpoint["l2_identity"]
        if identity["restart_id"] != restart_id:
            raise IntegrityError(
                f"checkpoint map key {restart_id} != identity restart_id "
                f"{identity['restart_id']}"
            )
        candidate_identity = {field: identity[field] for field in SEARCH_KEY_FIELDS}
        if search_identity is None:
            search_identity = candidate_identity
        elif candidate_identity != search_identity:
            raise IntegrityError("checkpoints belong to different search keys")
        current_hash = checkpoint["run_manifest_hash"]
        if manifest_hash is None:
            manifest_hash = current_hash
        elif current_hash != manifest_hash:
            raise IntegrityError("checkpoints have different run_manifest_hash values")

        budget = checkpoint["budget"]
        logical = _require_plain_int(
            budget.get("logical_eval_count"), "logical_eval_count"
        )
        discarded = _require_plain_int(budget.get("discarded_alloc"), "discarded_alloc")
        allocation = _require_plain_int(budget.get("allocation"), "allocation")
        if min(logical, discarded, allocation) < 0:
            raise IntegrityError("budget values must be non-negative")
        terminal = checkpoint["terminal"]
        if require_complete and terminal is None:
            raise IntegrityError(
                f"terminal search contains non-terminal restart_id={restart_id}"
            )
        terminal_state = (
            terminal.get("state") if isinstance(terminal, Mapping) else None
        )
        if (
            terminal_state in TERMINAL_BUDGET_STATES
            and logical + discarded != allocation
        ):
            raise IntegrityError("terminal restart budget does not conserve allocation")
        logical_total += logical
        discarded_total += discarded
        allocation_total += allocation

    if require_complete and allocation_total != SEARCH_ALLOCATION:
        raise IntegrityError(
            f"terminal restart allocations total {allocation_total}, expected {SEARCH_ALLOCATION}"
        )
    return {
        "logical_eval_count": logical_total,
        "discarded_alloc": discarded_total,
        "allocation": SEARCH_ALLOCATION,
    }


def make_selection(
    run_manifest_hash: str,
    search_key: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    manifest_hash = _validate_manifest_hash(run_manifest_hash)
    key = _validate_identity(search_key)
    if not candidates:
        raise SearchFailed("selection has no candidates")
    normalized: list[dict[str, object]] = []
    seen_restarts: set[int] = set()
    for candidate in candidates:
        if set(candidate) != {"restart_id", "J_val", "final_ref"}:
            raise IntegrityError("selection candidate keys are not exact")
        restart_id = _require_plain_int(candidate["restart_id"], "restart_id")
        if restart_id in seen_restarts:
            raise IntegrityError(f"duplicate restart_id={restart_id} in selection")
        seen_restarts.add(restart_id)
        value = float(candidate["J_val"])
        if not np.isfinite(value):
            raise SearchFailed("selection_nonfinite")
        final_restart_id, _, _ = parse_final_ref(candidate["final_ref"])
        if final_restart_id != restart_id:
            raise IntegrityError(
                f"restart_id={restart_id} disagrees with final_ref={candidate['final_ref']!r}"
            )
        normalized.append(
            {
                "restart_id": restart_id,
                "J_val": value,
                "final_ref": candidate["final_ref"],
            }
        )
    normalized.sort(key=lambda candidate: int(candidate["restart_id"]))
    best_value = min(float(candidate["J_val"]) for candidate in normalized)
    tied = [
        candidate
        for candidate in normalized
        if abs(float(candidate["J_val"]) - best_value) <= TIE_TOL
    ]
    winner = min(tied, key=lambda candidate: int(candidate["restart_id"]))
    return {
        "schema_version": 1,
        "run_manifest_hash": manifest_hash,
        "selection_kind": 0,
        "search_key": key,
        "candidates": normalized,
        "winner_restart_id": int(winner["restart_id"]),
        "tie_rule": TIE_RULE,
        "n_tied": len(tied),
        "J_val_winner": float(winner["J_val"]),
    }


def _validate_checkpoint_shape(checkpoint: Mapping[str, Any]) -> None:
    if set(checkpoint) != CHECKPOINT_KEYS:
        raise IntegrityError("checkpoint top-level keys do not match schema version 1")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise IntegrityError(
            f"unknown checkpoint schema {checkpoint['schema_version']!r}"
        )
    _validate_manifest_hash(checkpoint["run_manifest_hash"])
    identity = checkpoint["l2_identity"]
    expected_identity_keys = set(SEARCH_KEY_FIELDS) | {"restart_id", "gen_id"}
    if not isinstance(identity, Mapping) or set(identity) != expected_identity_keys:
        raise IntegrityError("checkpoint l2_identity keys are not exact")
    for key, value in identity.items():
        _require_plain_int(value, f"l2_identity[{key!r}]")
    cursor = checkpoint["cursor"]
    if not isinstance(cursor, Mapping) or set(cursor) != {
        "restart_id",
        "gen_id",
        "committed",
    }:
        raise IntegrityError("checkpoint cursor keys are not exact")
    if identity["restart_id"] != cursor["restart_id"]:
        raise IntegrityError("l2_identity and cursor restart_id differ")
    if identity["gen_id"] != cursor["gen_id"]:
        raise IntegrityError("l2_identity and cursor gen_id differ")
    if cursor["committed"] is not True:
        raise IntegrityError("persisted checkpoint cursor must be committed")
    budget = checkpoint["budget"]
    if not isinstance(budget, Mapping) or set(budget) != {
        "logical_eval_count",
        "discarded_alloc",
        "allocation",
    }:
        raise IntegrityError("checkpoint budget keys are not exact")
    records = checkpoint["gen_records"]
    if not isinstance(records, list):
        raise IntegrityError("checkpoint gen_records must be a list")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != GEN_RECORD_KEYS:
            raise IntegrityError("gen_record keys are not exact")


class SearchUnit:
    def __init__(
        self,
        identity: Mapping[str, object],
        n_dim: int,
        run_manifest_hash: str,
        seed_kind: Literal["global", "oracle", "bench"],
        *,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
        bench_purpose: int = 1,
        seed_factory: Callable[[int], np.random.SeedSequence] | None = None,
    ) -> None:
        dimension = _require_plain_int(n_dim, "n_dim")
        if dimension not in DIM_TIERS:
            raise ValueError(f"unsupported search dimension {dimension}")
        if seed_kind not in {"global", "oracle", "bench"}:
            raise ValueError(f"unknown seed_kind={seed_kind!r}")
        self.identity = _validate_identity(identity)
        self.n_dim = dimension
        self.run_manifest_hash = _validate_manifest_hash(run_manifest_hash)
        self.seed_kind = seed_kind
        self.bench_purpose = _require_plain_int(bench_purpose, "bench_purpose")
        self.checkpoint_callback = checkpoint_callback
        self._seed_factory = seed_factory
        self.tier = DIM_TIERS[dimension]
        self.n_restart = int(self.tier["n_restart"])
        self._sobol_state = self._make_sobol_state()
        self._current_restart_id = 0
        self._gen_id = 0
        self._es: Any | None = None
        self._bt: Any | None = None
        self._pending: tuple[Candidate, ...] | None = None
        self._logical_eval_count = 0
        self._discarded_alloc = 0
        self._gen_records: list[dict[str, Any]] = []
        self._oracle: dict[str, Any] | None = None
        self._terminal: dict[str, str] | None = None
        self._failed_restarts: set[int] = set()

    @property
    def current_restart_id(self) -> int:
        return self._current_restart_id

    @property
    def generation_id(self) -> int:
        return self._gen_id

    @property
    def budget(self) -> dict[str, int]:
        return {
            "logical_eval_count": self._logical_eval_count,
            "discarded_alloc": self._discarded_alloc,
            "allocation": int(self.tier["quota"]),
        }

    @property
    def terminal(self) -> dict[str, str] | None:
        return copy.deepcopy(self._terminal)

    def _seed_for_restart(self, restart_id: int) -> np.random.SeedSequence:
        if self._seed_factory is not None:
            seed = self._seed_factory(restart_id)
            if not isinstance(seed, np.random.SeedSequence):
                raise TypeError("seed_factory must return numpy.random.SeedSequence")
            return seed
        config_code = self.identity["config_code"]
        scenario_code = self.identity["scenario_code"]
        if self.seed_kind == "global":
            return cma_global_key(config_code, scenario_code, restart_id)
        if self.seed_kind == "oracle":
            return cma_oracle_key(
                scenario_code,
                restart_id,
                self.identity["system_id"],
            )
        return bench_cma_key(
            self.bench_purpose,
            config_code,
            scenario_code,
            restart_id,
        )

    def _sobol_seed(self) -> np.random.SeedSequence:
        if self.seed_kind == "bench":
            return bench_cma_key(
                self.bench_purpose,
                self.identity["config_code"],
                self.identity["scenario_code"],
                1,
            )
        return sobol_key(self.identity["config_code"], self.identity["scenario_code"])

    def _make_sobol_state(self) -> dict[str, Any] | None:
        if self.n_restart == 1:
            return None
        engine = qmc.Sobol(
            d=self.n_dim,
            scramble=True,
            seed=np.random.default_rng(self._sobol_seed()),
        )
        required = self.n_restart - 1
        initial_points = np.asarray(engine.random(required), dtype=float)
        n_drawn = required
        centre = np.full(self.n_dim, INITIAL_MEAN)
        points = [
            point
            for point in initial_points
            if np.linalg.norm(point - centre) > CENTRE_TOL
        ]
        while len(points) < required:
            point = np.asarray(engine.random(1)[0], dtype=float)
            n_drawn += 1
            if np.linalg.norm(point - centre) > CENTRE_TOL:
                points.append(point)
        return {
            "scramble": True,
            "skip": 0,
            "points": np.asarray(points, dtype=float),
            "n_drawn": n_drawn,
        }

    def start_restart(
        self,
        restart_id: int | None = None,
        *,
        box_override: tuple[float, float] | None = None,
        initial_mean_override: Sequence[float] | None = None,
    ) -> None:
        requested = self._current_restart_id + 1 if restart_id is None else restart_id
        restart = _require_plain_int(requested, "restart_id")
        if restart != self._current_restart_id + 1:
            raise IntegrityError(
                "restarts must start once in strictly increasing order"
            )
        if restart < 1 or restart > self.n_restart:
            raise IntegrityError(f"restart_id={restart} is outside 1..{self.n_restart}")
        if self._current_restart_id and self._terminal is None:
            raise IntegrityError(
                "cannot start a new restart before the current one terminates"
            )
        if self._pending is not None:
            raise IntegrityError("cannot start a restart with a pending generation")

        box = (
            BOX
            if box_override is None
            else tuple(float(value) for value in box_override)
        )
        if len(box) != 2 or not box[0] < box[1]:
            raise ValueError("box_override must be an increasing pair")
        if initial_mean_override is not None:
            mean = np.asarray(initial_mean_override, dtype=float)
        elif restart == 1:
            mean = np.full(self.n_dim, INITIAL_MEAN)
        else:
            if self._sobol_state is None:
                raise IntegrityError("multi-restart search is missing Sobol state")
            mean = np.array(self._sobol_state["points"][restart - 2], copy=True)
        if mean.shape != (self.n_dim,) or not np.all(np.isfinite(mean)):
            raise ValueError(
                f"initial mean must have shape ({self.n_dim},) and be finite"
            )

        generator = np.random.default_rng(self._seed_for_restart(restart))
        options = {
            "popsize": int(self.tier["popsize"]),
            "verbose": -9,
            "seed": np.nan,
            "randn": RandnFromGenerator(generator),
        }
        self._es = cma.CMAEvolutionStrategy(mean.tolist(), SIGMA0, options)
        self._bt = cma.BoundTransform(list(box))
        self._current_restart_id = restart
        self._gen_id = 0
        self._pending = None
        self._logical_eval_count = 0
        self._discarded_alloc = int(self.tier["quota"]) - (
            int(self.tier["gens"]) * int(self.tier["popsize"])
        )
        self._gen_records = []
        self._oracle = None
        self._terminal = None
        self._emit_checkpoint()

    def ask(self) -> list[Candidate]:
        if self._es is None or self._bt is None:
            raise IntegrityError("start_restart must be called before ask")
        if self._terminal is not None:
            raise IntegrityError("cannot ask after restart termination")
        if self._pending is not None:
            raise IntegrityError("the previous generation has not been told")
        raw_points = self._es.ask()
        pending: list[Candidate] = []
        for candidate_id, point in enumerate(raw_points):
            x_raw = np.asarray(point, dtype=float).copy()
            x_box = np.asarray(self._bt.repair(x_raw.copy()), dtype=float)
            pending.append(Candidate(candidate_id, x_raw, x_box))
        if len(pending) != int(self.tier["popsize"]):
            raise IntegrityError("pycma returned an unexpected population size")
        self._pending = tuple(pending)
        return [
            Candidate(item.candidate_id, item.x_raw.copy(), item.x_box.copy())
            for item in pending
        ]

    def _normalize_results(
        self,
        results: Mapping[int, EvalResult | Real] | Sequence[EvalResult],
    ) -> list[EvalResult]:
        if isinstance(results, Mapping):
            normalized: list[EvalResult] = []
            for candidate_id, result in results.items():
                cid = _require_plain_int(candidate_id, "candidate_id")
                if isinstance(result, EvalResult):
                    if result.candidate_id != cid:
                        raise GenerationIntegrityError(
                            f"mapping key {cid} != EvalResult candidate_id {result.candidate_id}"
                        )
                    normalized.append(result)
                elif isinstance(result, Real) and not isinstance(
                    result, (bool, np.bool_)
                ):
                    normalized.append(EvalResult(cid, float(result), 0))
                else:
                    raise GenerationIntegrityError("unsupported result value")
            return normalized
        if not isinstance(results, Sequence):
            raise GenerationIntegrityError("results must be a mapping or sequence")
        if any(not isinstance(result, EvalResult) for result in results):
            raise GenerationIntegrityError(
                "result sequences must contain EvalResult values"
            )
        return list(results)

    def tell(
        self,
        results: Mapping[int, EvalResult | Real] | Sequence[EvalResult],
    ) -> None:
        if self._es is None or self._pending is None:
            raise IntegrityError("ask must produce a pending generation before tell")
        normalized = self._normalize_results(results)
        ids = [result.candidate_id for result in normalized]
        expected = list(range(int(self.tier["popsize"])))
        if len(ids) != len(set(ids)):
            raise GenerationIntegrityError("duplicate candidate_id")
        if sorted(ids) != expected:
            raise GenerationIntegrityError(
                f"candidate ids must be {expected!r}, got {sorted(ids)!r}"
            )
        ordered = sorted(normalized, key=lambda result: result.candidate_id)
        for result in ordered:
            if not _is_plain_int(result.n_div) or result.n_div < 0:
                raise GenerationIntegrityError("n_div must be a non-negative integer")
            if np.isnan(result.J) or result.J == -np.inf:
                self._pending = None
                self._terminal = {
                    "state": "SEARCH FAILED",
                    "reason": "aggregate_nonfinite",
                }
                self._emit_checkpoint()
                raise SearchFailed("aggregate_nonfinite")

        next_generation = self._gen_id + 1
        logical_after = self._logical_eval_count + len(ordered)
        if all(result.J == np.inf for result in ordered):
            self._logical_eval_count = logical_after
            self._discarded_alloc = int(self.tier["quota"]) - logical_after
            self._gen_id = next_generation
            self._append_records(ordered, stop_flags=())
            self._pending = None
            self._failed_restarts.add(self._current_restart_id)
            all_failed = (
                self._current_restart_id == self.n_restart
                and self._failed_restarts == set(range(1, self.n_restart + 1))
            )
            if all_failed:
                self._terminal = {
                    "state": "SEARCH FAILED",
                    "reason": "all_restarts_failed",
                }
                self._emit_checkpoint()
                raise SearchFailed("all_restarts_failed")
            self._terminal = {"state": "RESTART FAILED", "reason": "generation_all_inf"}
            self._emit_checkpoint()
            raise RestartFailed("generation_all_inf")

        raw_points = [self._pending[result.candidate_id].x_raw for result in ordered]
        objective_values = [float(result.J) for result in ordered]
        self._es.tell(raw_points, objective_values)
        stop_flags = tuple(sorted(str(key) for key in self._es.stop()))
        self._logical_eval_count = logical_after
        self._gen_id = next_generation
        if self._gen_id == int(self.tier["gens"]):
            self._discarded_alloc = int(self.tier["quota"]) - self._logical_eval_count
            self._terminal = {
                "state": "success",
                "reason": "fixed_generations_complete",
            }
        self._append_records(ordered, stop_flags=stop_flags)
        self._pending = None
        self._emit_checkpoint()

    def _append_records(
        self,
        ordered: Sequence[EvalResult],
        *,
        stop_flags: Sequence[str],
    ) -> None:
        if self._pending is None:
            raise IntegrityError("cannot record candidates without pending points")
        for result in ordered:
            candidate = self._pending[result.candidate_id]
            record = {
                "row_kind": 0,
                "restart_id": self._current_restart_id,
                "gen_id": self._gen_id,
                "candidate_id": result.candidate_id,
                "x_raw": candidate.x_raw.copy(),
                "x_box": candidate.x_box.copy(),
                "J": float(result.J),
                "n_div": int(result.n_div),
                "stop_flags": tuple(stop_flags),
                "logical_eval_count_at_commit": self._logical_eval_count,
                "discarded_alloc_at_commit": self._discarded_alloc,
                "cum_scope": "restart",
            }
            if set(record) != GEN_RECORD_KEYS:
                raise IntegrityError("constructed gen_record keys are not exact")
            self._gen_records.append(record)

    def best_candidate_record(self) -> dict[str, Any]:
        candidates = [
            record
            for record in self._gen_records
            if record["row_kind"] == 0 and np.isfinite(record["J"])
        ]
        if not candidates:
            raise SearchFailed("restart has no finite candidate")
        best = min(
            candidates,
            key=lambda record: (record["J"], record["gen_id"], record["candidate_id"]),
        )
        return copy.deepcopy(best)

    def complete_incumbent(
        self,
        x_box: np.ndarray,
        J_inc: float,
        n_div: int,
        *,
        J_w3: float,
    ) -> dict[str, Any]:
        if (
            self.seed_kind != "oracle"
            or self.identity["config_code"] != ORACLE_CONFIG_CODE
        ):
            raise IntegrityError("incumbent completion is only valid for W3-o")
        if self._pending is not None or self._terminal != {
            "state": "success",
            "reason": "fixed_generations_complete",
        }:
            raise IntegrityError("CMA generations must complete before the incumbent")
        cma_logical = int(self.tier["gens"]) * int(self.tier["popsize"])
        cma_discarded = int(self.tier["quota"]) - cma_logical
        if (
            self._logical_eval_count != cma_logical
            or self._discarded_alloc != cma_discarded
        ):
            raise IntegrityError("W3-o CMA budget must be (297, 3) before incumbent")
        incumbent_x = np.asarray(x_box, dtype=float)
        if incumbent_x.shape != (self.n_dim,) or not np.all(np.isfinite(incumbent_x)):
            raise IntegrityError("incumbent x_box has invalid shape or values")
        incumbent_value = float(J_inc)
        w3_value = float(J_w3)
        if not np.isfinite(incumbent_value) or not np.isfinite(w3_value):
            raise OracleImplError("oracle objectives must be finite")
        n_division = _require_plain_int(n_div, "n_div")
        if n_division < 0:
            raise IntegrityError("n_div must be non-negative")

        cma_record = self.best_candidate_record()
        cma_value = float(cma_record["J"])
        if cma_value < incumbent_value:
            source = "cma"
            final_params = np.array(cma_record["x_box"], copy=True)
            final_ref = (
                f"r{cma_record['restart_id']}g{cma_record['gen_id']}"
                f"c{cma_record['candidate_id']}"
            )
            selected_value = cma_value
        else:
            source = "incumbent"
            final_params = incumbent_x.copy()
            final_ref = "incumbent"
            selected_value = incumbent_value
        if selected_value > w3_value:
            raise OracleImplError(
                "oracle selection is worse than the W3 fitted objective"
            )

        self._logical_eval_count += 1
        self._discarded_alloc -= 1
        self._gen_id = INCUMBENT_GEN_ID
        incumbent_record = {
            "row_kind": 1,
            "restart_id": INCUMBENT_RESTART_ID,
            "gen_id": INCUMBENT_GEN_ID,
            "candidate_id": INCUMBENT_CANDIDATE_ID,
            "x_raw": None,
            "x_box": incumbent_x.copy(),
            "J": incumbent_value,
            "n_div": n_division,
            "stop_flags": None,
            "logical_eval_count_at_commit": self._logical_eval_count,
            "discarded_alloc_at_commit": self._discarded_alloc,
            "cum_scope": "restart",
        }
        if set(incumbent_record) != GEN_RECORD_KEYS:
            raise IntegrityError("constructed incumbent record keys are not exact")
        self._gen_records.append(incumbent_record)
        self._oracle = {
            "J_CMA": cma_value,
            "J_inc": incumbent_value,
            "source": source,
            "final_params": final_params,
            "final_ref": final_ref,
        }
        self._terminal = {"state": "success", "reason": "oracle_complete"}
        self._emit_checkpoint()
        return copy.deepcopy(self._oracle)

    def checkpoint_dict(self) -> dict[str, Any]:
        if self._es is None or self._current_restart_id == 0:
            raise IntegrityError("no restart has been started")
        if self._pending is not None:
            raise IntegrityError("checkpoint is only valid before ask or after tell")
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_manifest_hash": self.run_manifest_hash,
            "l2_identity": {
                **self.identity,
                "restart_id": self._current_restart_id,
                "gen_id": self._gen_id,
            },
            "cma_state": copy.deepcopy(self._es),
            "sobol_state": copy.deepcopy(self._sobol_state),
            "cursor": {
                "restart_id": self._current_restart_id,
                "gen_id": self._gen_id,
                "committed": True,
            },
            "budget": self.budget,
            "gen_records": copy.deepcopy(self._gen_records),
            "oracle": copy.deepcopy(self._oracle),
            "terminal": copy.deepcopy(self._terminal),
        }
        _validate_checkpoint_shape(checkpoint)
        return checkpoint

    def _emit_checkpoint(self) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(self.checkpoint_dict())

    @classmethod
    def restore(
        cls,
        checkpoint: Mapping[str, Any],
        run_manifest_hash: str,
        *,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> "SearchUnit":
        _validate_checkpoint_shape(checkpoint)
        expected_hash = _validate_manifest_hash(run_manifest_hash)
        if checkpoint["run_manifest_hash"] != expected_hash:
            raise IntegrityError("run_manifest_hash mismatch")
        es = copy.deepcopy(checkpoint["cma_state"])
        n_dim = _require_plain_int(es.N, "cma_state.N")
        if n_dim not in DIM_TIERS:
            raise IntegrityError(f"checkpoint has unsupported dimension {n_dim}")
        if int(es.popsize) != int(DIM_TIERS[n_dim]["popsize"]):
            raise IntegrityError("checkpoint CMA popsize does not match DIM_TIERS")

        restored = cls.__new__(cls)
        identity = checkpoint["l2_identity"]
        restored.identity = {field: int(identity[field]) for field in SEARCH_KEY_FIELDS}
        restored.n_dim = n_dim
        restored.run_manifest_hash = expected_hash
        restored.seed_kind = (
            "oracle"
            if restored.identity["config_code"] == ORACLE_CONFIG_CODE
            else "global"
        )
        restored.bench_purpose = 1
        restored.checkpoint_callback = checkpoint_callback
        restored._seed_factory = None
        restored.tier = DIM_TIERS[n_dim]
        restored.n_restart = int(restored.tier["n_restart"])
        restored._sobol_state = copy.deepcopy(checkpoint["sobol_state"])
        restored._current_restart_id = int(identity["restart_id"])
        restored._gen_id = int(identity["gen_id"])
        restored._es = es
        restored._bt = cma.BoundTransform(list(BOX))
        restored._pending = None
        budget = checkpoint["budget"]
        restored._logical_eval_count = int(budget["logical_eval_count"])
        restored._discarded_alloc = int(budget["discarded_alloc"])
        restored._gen_records = copy.deepcopy(checkpoint["gen_records"])
        restored._oracle = copy.deepcopy(checkpoint["oracle"])
        restored._terminal = copy.deepcopy(checkpoint["terminal"])
        restored._failed_restarts = set()
        return restored


__all__ = [
    "Candidate",
    "CHECKPOINT_KEYS",
    "DIM_TIERS",
    "EvalResult",
    "GEN_RECORD_KEYS",
    "GenerationIntegrityError",
    "IntegrityError",
    "OracleImplError",
    "RestartFailed",
    "SearchFailed",
    "SearchUnit",
    "aggregate_objective",
    "j_train",
    "j_val",
    "locate_w3_x_box",
    "make_selection",
    "parse_final_ref",
    "phi",
    "rebuild_coordinator",
]
