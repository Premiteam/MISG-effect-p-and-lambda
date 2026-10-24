import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import queue as queue_module
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
import threadpoolctl


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]
SRC_DIR = CODE_DIR.parent / "core"
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TOOLS_DIR))

import rng
import search
import snapshot_env


PRODUCTION_ENABLED: bool = False
DEFAULT_WORKERS = 12
PRODUCTION_DISABLED = 10
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
L1_SEARCH_FIELDS = (
    "config_code",
    "scenario_code",
    "condition_code",
    "split_code",
    "eval_kind",
    "restart_id",
    "gen_id",
    "candidate_id",
    "m",
    "R0",
    "R1",
    "R2",
    "st0",
    "st1",
    "st2",
    "h",
)
L1_EVAL_FIELDS = (
    "config_code",
    "scenario_code",
    "condition_code",
    "split_code",
    "eval_kind",
    "restart_id",
    "gen_id",
    "candidate_id",
    "m",
    "R3",
    "R4",
    "st3",
    "st4",
    "h",
)
RESUME_EVENT_FIELDS = (
    "config_code",
    "scenario_code",
    "condition_code",
    "split_code",
    "system_id",
    "restart_id",
    "gen_id",
    "lambda_pop",
    "run_manifest_hash",
    "h",
)
MINI_SEARCH_IDENTITY = {
    "config_code": 2,
    "scenario_code": 1,
    "condition_code": 0,
    "split_code": 0,
    "system_id": -1,
}
HONEST_A1_LIMITATION = (
    "本机制无法观测「握手完成之前发生、且 START 未送达」的初始化失败"
    "（实测：live == audited，握手判 OK）。\n"
    "A1 判绿的准确表述是「执行实验的那批 worker 全部通过线程审计」，"
    "不是「从未发生过初始化失败」。"
)
READ_BLOCK_BYTES = 1024 * 1024


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise search.IntegrityError(f"expected directory, got {path}")
        return
    path.mkdir(parents=True)


def _atomic_json(path: Path, value: Any, *, read_only: bool = False) -> None:
    _ensure_directory(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if read_only:
        os.chmod(path, stat.S_IREAD)


def _atomic_pickle(path: Path, value: Any) -> None:
    _ensure_directory(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def serialize_seed_registry() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (component, seed_kind), spec in sorted(rng.SEED_REGISTRY.items()):
        fields: list[dict[str, Any]] = []
        for field in spec.exact_shape:
            domain = field.domain
            fields.append(
                {
                    "name": field.name,
                    "domain": {
                        "minimum": domain.minimum,
                        "maximum": domain.maximum,
                        "allowed": (
                            None if domain.allowed is None else sorted(domain.allowed)
                        ),
                        "description": domain.description,
                    },
                }
            )
        output.append(
            {
                "component": component,
                "seed_kind": seed_kind,
                "exact_shape": fields,
            }
        )
    return output


def _library_projection(pool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "internal_api": pool.get("internal_api"),
        "version": pool.get("version"),
        "threading_layer": pool.get("threading_layer"),
        "architecture": pool.get("architecture"),
        "num_threads": pool.get("num_threads"),
    }


def build_threadpool_projection(
    profiles: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    groups: Counter[bytes] = Counter()
    decoded: dict[bytes, list[dict[str, Any]]] = {}
    for profile in profiles.values():
        pools = profile.get("pools")
        if not isinstance(pools, list):
            raise search.IntegrityError("worker pools must be a list")
        libs = sorted(
            (_library_projection(item) for item in pools),
            key=lambda item: canonical_bytes(item),
        )
        encoded = canonical_bytes(libs)
        groups[encoded] += 1
        decoded[encoded] = libs
    projection = {
        "n_workers": len(profiles),
        "distinct_worker_profiles": [
            {"count": groups[key], "libs": decoded[key]} for key in sorted(groups)
        ],
    }
    validate_threadpool_projection(projection)
    return projection


def validate_threadpool_projection(projection: Mapping[str, Any]) -> bool:
    expected = projection.get("n_workers")
    groups = projection.get("distinct_worker_profiles")
    if not isinstance(expected, int) or expected < 1 or not isinstance(groups, list):
        raise search.IntegrityError("invalid threadpool projection schema")
    if len(groups) != 1 or groups[0].get("count") != expected:
        raise search.IntegrityError("worker threadpool profiles are not identical")
    libs = groups[0].get("libs")
    if not isinstance(libs, list) or not libs:
        raise search.IntegrityError("worker library projection is empty")
    if any(lib.get("num_threads") != 1 for lib in libs):
        raise search.IntegrityError("worker library thread count is not one")
    return True


def _assert_thread_env() -> None:
    wrong = {
        name: os.environ.get(name)
        for name in THREAD_ENV_VARS
        if os.environ.get(name) != "1"
    }
    if wrong:
        raise RuntimeError(f"thread environment not pinned: {wrong!r}")


def _init(
    queue: Any,
    fail_once: Any = None,
    fail_lock: Any = None,
    fail_mode: str | None = None,
) -> None:
    if fail_mode == "hard_before_start" and fail_once is not None:
        hard = False
        with fail_lock:
            if fail_once.value:
                fail_once.value = 0
                hard = True
        if hard:
            os._exit(70)
    pid = os.getpid()
    queue.put(("START", pid))
    try:
        if fail_once is not None:
            hard = False
            with fail_lock:
                if fail_once.value:
                    fail_once.value = 0
                    if fail_mode == "hard":
                        hard = True
                    else:
                        raise RuntimeError("E1_TEST: injected init failure")
            if hard:
                time.sleep(0.05)
                os._exit(70)
        _assert_thread_env()
        profile = {
            "pid": pid,
            "env_vars": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
            "pools": threadpoolctl.threadpool_info(),
        }
        queue.put(("READY", pid, profile))
    except Exception as exc:
        queue.put(("ERROR", pid, type(exc).__name__, str(exc)))
        raise


def create_pool(
    n_workers: int, test_hooks: tuple[Any, Any, str] | None = None
) -> tuple[Any, Any, Any]:
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    args = (queue,) if test_hooks is None else (queue, *test_hooks)
    pool_factory = multiprocessing.get_context("spawn").Pool
    pool = pool_factory(processes=n_workers, initializer=_init, initargs=args)
    return ctx, queue, pool


def _apply_handshake_message(
    states: dict[int, str],
    profiles: dict[int, Mapping[str, Any]],
    message: Sequence[Any],
    n_workers: int,
) -> None:
    if len(message) < 2:
        raise search.IntegrityError("malformed handshake message")
    kind, pid = message[0], int(message[1])
    if kind == "START":
        if pid in states:
            raise search.IntegrityError(f"duplicate START for PID {pid}")
        states[pid] = "STARTED"
        if len(states) > n_workers:
            raise search.IntegrityError("more START PIDs than n_workers")
        return
    if pid not in states:
        raise search.IntegrityError(f"{kind} arrived before START for PID {pid}")
    if kind == "READY":
        if states[pid] == "READY":
            raise search.IntegrityError(f"duplicate READY for PID {pid}")
        if states[pid] != "STARTED":
            raise search.IntegrityError(f"invalid READY state for PID {pid}")
        states[pid] = "READY"
        profiles[pid] = message[2]
        return
    if kind == "ERROR":
        states[pid] = "ERROR"
        detail = ": ".join(str(part) for part in message[2:])
        raise search.IntegrityError(f"worker initializer ERROR PID {pid}: {detail}")
    raise search.IntegrityError(f"unknown handshake message {kind!r}")


def _state_counts(states: Mapping[int, str], n_workers: int) -> str:
    counts = Counter(states.values())
    return (
        f"UNSEEN={max(0, n_workers - len(states))}, "
        f"STARTED={counts['STARTED']}, READY={counts['READY']}, ERROR={counts['ERROR']}"
    )


def collect_handshake_from_messages(
    messages: Iterable[Sequence[Any]], n_workers: int, *, complete: bool
) -> dict[int, Any]:
    states: dict[int, str] = {}
    profiles: dict[int, Mapping[str, Any]] = {}
    for message in messages:
        _apply_handshake_message(states, profiles, message, n_workers)
    if complete:
        if len(states) != n_workers or any(
            state != "READY" for state in states.values()
        ):
            raise search.IntegrityError(
                f"handshake deadline: {_state_counts(states, n_workers)}"
            )
        return profiles
    return states


def collect_handshake(
    queue: Any, n_workers: int, *, deadline_seconds: float = 120.0
) -> dict[int, Mapping[str, Any]]:
    states: dict[int, str] = {}
    profiles: dict[int, Mapping[str, Any]] = {}
    deadline = time.monotonic() + deadline_seconds
    while len(profiles) < n_workers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise search.IntegrityError(
                f"handshake deadline: {_state_counts(states, n_workers)}"
            )
        try:
            message = queue.get(timeout=remaining)
        except queue_module.Empty as exc:
            raise search.IntegrityError(
                f"handshake deadline: {_state_counts(states, n_workers)}"
            ) from exc
        _apply_handshake_message(states, profiles, message, n_workers)
    if set(states) != set(profiles) or any(
        state != "READY" for state in states.values()
    ):
        raise search.IntegrityError(
            f"handshake incomplete: {_state_counts(states, n_workers)}"
        )
    return profiles


def live_pool_pids(pool: Any) -> set[int]:
    return {worker.pid for worker in pool._pool if worker.is_alive()}


def _csv_prefix(fields: Sequence[str], row: Mapping[str, Any]) -> str:
    values: list[str] = []
    for field in fields[:-1]:
        value = row[field]
        if isinstance(value, (float, np.floating)):
            values.append(repr(float(value)))
        else:
            values.append(str(value))
    return ",".join(values)


def checked_row_hash(row: Mapping[str, Any]) -> str:
    fields = tuple(row)
    if not fields or fields[-1] != "h":
        raise search.IntegrityError("checked row must end in h")
    return hashlib.sha256(_csv_prefix(fields, row).encode("utf-8")).hexdigest()[:16]


def initialize_checked_csv(
    path: Path, run_manifest_hash: str, fields: Sequence[str]
) -> None:
    _ensure_directory(path.parent)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# run_manifest_hash={run_manifest_hash}\n")
        handle.write(",".join(fields) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_checked_row(
    path: Path,
    fields: Sequence[str],
    row: Mapping[str, Any],
    *,
    kill_half: bool = False,
) -> None:
    if tuple(fields)[-1] != "h" or set(row) != set(fields[:-1]):
        raise search.IntegrityError("checked row keys do not match schema")
    prefix = _csv_prefix(fields, row)
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]
    encoded = f"{prefix},{digest}\n".encode("utf-8")
    with path.open("ab") as handle:
        if (
            kill_half
            and os.environ.get("E1_TEST_MODE") == "1"
            and os.environ.get("E1_TEST_KILLPOINT") == "3"
        ):
            handle.write(encoded[: max(1, len(encoded) // 2)])
            handle.flush()
            os.fsync(handle.fileno())
            _killpoint_ready("3")
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_checked_line(line: bytes, fields: Sequence[str]) -> dict[str, str]:
    try:
        decoded = line.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise search.IntegrityError("checked CSV row is not UTF-8") from exc
    values = decoded.split(",")
    if len(values) != len(fields):
        raise search.IntegrityError("checked CSV row has wrong field count")
    row = dict(zip(fields, values, strict=True))
    if checked_row_hash(row) != row["h"]:
        raise search.IntegrityError("checked CSV row hash mismatch")
    return row


def _logical_identity(row: Mapping[str, str], fields: Sequence[str]) -> tuple[str, ...]:
    if tuple(fields) in (L1_SEARCH_FIELDS, L1_EVAL_FIELDS):
        return tuple(row[field] for field in fields[:9])
    if tuple(fields) == RESUME_EVENT_FIELDS:
        return tuple(row[field] for field in fields[:7])
    return tuple(row[field] for field in fields[:-1])


def scan_checked_csv(
    path: Path, fields: Sequence[str], *, truncate_tail: bool
) -> list[dict[str, str]]:
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) < 2:
        raise search.IntegrityError("checked CSV header is incomplete")
    if raw_lines[1].decode("utf-8").rstrip("\r\n") != ",".join(fields):
        raise search.IntegrityError("checked CSV header mismatch")
    rows: list[dict[str, str]] = []
    identities: set[tuple[str, ...]] = set()
    offset = len(raw_lines[0]) + len(raw_lines[1])
    last_good = offset
    for index, line in enumerate(raw_lines[2:]):
        try:
            row = _parse_checked_line(line, fields)
        except search.IntegrityError as exc:
            is_last = index == len(raw_lines[2:]) - 1
            if not is_last:
                raise search.IntegrityError(
                    "middle checked CSV row is corrupt"
                ) from exc
            if not truncate_tail:
                raise
            with path.open("r+b") as handle:
                handle.truncate(last_good)
                handle.flush()
                os.fsync(handle.fileno())
            break
        identity = _logical_identity(row, fields)
        if tuple(fields) in (L1_SEARCH_FIELDS, L1_EVAL_FIELDS):
            if identity in identities:
                raise search.IntegrityError(
                    f"duplicate complete logical key: {identity!r}"
                )
            identities.add(identity)
        rows.append(row)
        offset += len(line)
        last_good = offset
    return rows


def test_search_row(candidate_id: int = 0) -> dict[str, Any]:
    return {
        "config_code": 2,
        "scenario_code": 1,
        "condition_code": 0,
        "split_code": 0,
        "eval_kind": 0,
        "restart_id": 1,
        "gen_id": 1,
        "candidate_id": candidate_id,
        "m": 0,
        "R0": 0.0,
        "R1": 0.0,
        "R2": 0.0,
        "st0": 0,
        "st1": 0,
        "st2": 0,
    }


def count_n_div(rows: Sequence[Mapping[str, Any]], eval_kind: int) -> int:
    if eval_kind in (0, 1):
        fields = ("st0", "st1", "st2")
    elif eval_kind == 2:
        fields = ("st3", "st4")
    else:
        raise ValueError(f"unknown eval_kind={eval_kind}")
    divergent = {
        int(row["m"])
        for row in rows
        if int(row["eval_kind"]) == eval_kind
        and any(int(row[field]) != 0 for field in fields)
    }
    return len(divergent)


def validate_generation_buffer(
    results: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: set[int],
    system_ids: set[int],
) -> None:
    actual = [(int(item["candidate_id"]), int(item["m"])) for item in results]
    expected = {(candidate_id, m) for candidate_id in candidate_ids for m in system_ids}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise search.IntegrityError(
            "generation result buffer is incomplete or duplicated"
        )


def _seed_key(seed: np.random.SeedSequence) -> list[int]:
    entropy = seed.entropy
    if isinstance(entropy, Sequence) and not isinstance(entropy, (str, bytes)):
        return [int(value) for value in entropy]
    return [int(entropy)]


def _toy_target(m: int, n_dim: int) -> np.ndarray:
    generator = np.random.default_rng(rng.bench_system_key(1, m))
    return generator.uniform(0.15, 0.85, size=n_dim)


def _toy_risks(
    x_box: Sequence[float], m: int, eval_kind: int
) -> tuple[list[float], list[int]]:
    vector = np.asarray(x_box, dtype=float)
    target = _toy_target(m, vector.size)
    trajectory_ids = range(3) if eval_kind in (0, 1) else range(3, 5)
    risks: list[float] = []
    statuses: list[int] = []
    for trajectory_id in trajectory_ids:
        generator = np.random.default_rng(rng.bench_traj_key(1, m, trajectory_id))
        perturbation = float(generator.normal(0.0, 0.002))
        risk = float(np.dot(vector - target, vector - target) + abs(perturbation))
        risks.append(risk)
        statuses.append(0)
    return risks, statuses


def _worker_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if task.get("kill_worker"):
        os._exit(1)
    risks, statuses = _toy_risks(task["x_box"], int(task["m"]), int(task["eval_kind"]))
    return {
        "task_key": tuple(task["task_key"]),
        "candidate_id": int(task["candidate_id"]),
        "m": int(task["m"]),
        "risks": risks,
        "statuses": statuses,
        "pid": os.getpid(),
    }


def collect_results(
    pool: Any,
    tasks: Sequence[Mapping[str, Any]],
    audited_pids: set[int],
    *,
    poll_interval: float = 0.5,
) -> list[dict[str, Any]]:
    iterator = pool.imap_unordered(_worker_task, tasks, chunksize=1)
    results: list[dict[str, Any]] = []
    while len(results) < len(tasks):
        try:
            result = iterator.next(timeout=poll_interval)
        except multiprocessing.TimeoutError:
            if live_pool_pids(pool) != audited_pids:
                raise search.IntegrityError("worker PID set changed during computation")
            continue
        if int(result["pid"]) not in audited_pids:
            raise search.IntegrityError("result returned by an unaudited worker PID")
        results.append(result)
    return results


def _git_state(repo: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status_text = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(status_text), status_text


def _receipt_payload(
    path: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("verification_digest"), str):
        raise search.IntegrityError("input receipt lacks verification_digest")
    record = {
        "path": "input_receipt.json",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "expected_sha256": _sha256(path),
        "match": True,
    }
    return payload, record


def _production_preflight(
    repo: Path,
    receipt_path: Path | None,
    verify_result: Mapping[str, Any],
    git_status: str,
) -> tuple[int, str]:
    if git_status:
        return 11, "git worktree is not clean"
    if int(verify_result.get("returncode", -1)) != 0:
        return 12, f"input verification returned {verify_result.get('returncode')}"
    if receipt_path is None:
        return 13, "--input-receipt is required"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return 13, f"input receipt unreadable: {exc}"
    expected = receipt.get("verification_digest")
    actual = verify_result.get("verification_digest")
    if expected != actual:
        return (
            13,
            f"verification_digest differs: generated={actual!r}, receipt={expected!r}",
        )
    return 0, "ok"


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_mini_spec(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise search.IntegrityError(f"mini spec is unreadable: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"n_dim", "n_systems", "r_cap"}:
        raise search.IntegrityError("mini spec keys must be n_dim, n_systems, r_cap")
    n_dim = int(value["n_dim"])
    n_systems = int(value["n_systems"])
    r_cap = float(value["r_cap"])
    if (
        n_dim not in search.DIM_TIERS
        or n_systems < 1
        or not math.isfinite(r_cap)
        or r_cap <= 0
    ):
        raise search.IntegrityError("mini spec values are outside their frozen domains")
    return {"n_dim": n_dim, "n_systems": n_systems, "r_cap": r_cap}


def _build_config(
    spec: Mapping[str, Any],
    n_workers: int,
    receipt: Mapping[str, Any] | None,
    receipt_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "mini",
        "mini": dict(spec),
        "inference": {
            "B": 2000,
            "method": "percentile",
            "alpha": 0.05,
            "batch_means": None,
        },
        "family": {
            "kind": "isolated_toy_quadratic",
            "purpose_code": 1,
            "n_systems": int(spec["n_systems"]),
        },
        "seeds": {"base": rng.SEED_BASE, "registry": serialize_seed_registry()},
        "search": {
            "K": 300,
            "lambda_pop_by_n": {
                str(key): int(value["popsize"])
                for key, value in search.DIM_TIERS.items()
            },
            "n_restart_by_n": {
                str(key): int(value["n_restart"])
                for key, value in search.DIM_TIERS.items()
            },
            "objective": {"traj_aggregation": "per_trajectory_phi_then_mean"},
        },
        "n_workers": n_workers,
        "input_receipt": (
            None
            if receipt is None or receipt_record is None
            else {
                "path": receipt_record["path"],
                "sha256": receipt_record["sha256"],
                "verification_digest": receipt["verification_digest"],
            }
        ),
        "unavailable": [
            "C-a",
            "C-b",
            "C-c",
            "C-d",
            "C-f",
            "C-j",
            "C-k",
            "C-l",
            "C-n",
            "C-o",
        ],
    }


def _codebooks() -> dict[str, Any]:
    return {
        "config": {name: int(value) for name, value in rng.CONFIG_CODE.items()},
        "scenario": {name: int(value) for name, value in rng.SCENARIO_CODE.items()},
        "split": {name: int(value) for name, value in rng.SPLIT_CODE.items()},
    }


def _stable_environment(
    projection: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "python": snapshot["python"]["version"],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cma": search.cma.__version__,
        "platform": {
            "implementation": snapshot["python"]["implementation"],
            "sysconfig_platform": snapshot["python"]["sysconfig_platform"],
            "machine": snapshot["platform"]["machine"],
        },
        "cpu_count": snapshot["platform"]["cpu_count_logical"],
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "threadpool_projection": projection,
    }


def _input_records(
    spec_path: Path, receipt_path: Path | None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    digest = _sha256(spec_path)
    records = [
        {
            "path": "mini_spec.json",
            "bytes": spec_path.stat().st_size,
            "sha256": digest,
            "expected_sha256": digest,
            "match": True,
        }
    ]
    receipt, record = _receipt_payload(receipt_path)
    if record is not None:
        records.append(record)
    return records, receipt


def _build_manifest(
    config: Mapping[str, Any],
    projection: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    commit: str,
    dirty: bool,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "config": dict(config),
        "codebooks": _codebooks(),
        "environment": _stable_environment(projection, snapshot),
        "code": {"git_commit": commit, "git_dirty": dirty},
        "inputs": list(inputs),
    }
    if any(not record["match"] for record in inputs):
        raise search.IntegrityError("manifest contains a mismatched input")
    return manifest


def _raw_env(
    snapshot: Mapping[str, Any],
    profiles: Mapping[int, Mapping[str, Any]],
    run_manifest_hash: str,
    commit: str,
    n_workers: int,
) -> dict[str, Any]:
    return {
        "python": snapshot["python"],
        "numpy": snapshot["numpy"],
        "scipy": snapshot["scipy"],
        "platform": snapshot["platform"],
        "cpu_count": snapshot["platform"]["cpu_count_logical"],
        "numpy_config": np.show_config(mode="dicts"),
        "worker_threadpool_info": [profiles[pid] for pid in sorted(profiles)],
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "git_commit": commit,
        "n_workers": n_workers,
        "run_manifest_hash": run_manifest_hash,
        "executed_worker_pids": sorted(profiles),
        "a1_honest_limitation": HONEST_A1_LIMITATION,
    }


def _run_hash(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def _checkpoint_name(identity: Mapping[str, int], restart_id: int) -> str:
    base = (
        f"c{identity['config_code']}_s{identity['scenario_code']}_"
        f"d{identity['condition_code']}_p{identity['split_code']}_r{restart_id}"
    )
    if identity.get("system_id", -1) != -1:
        base += f"_m{identity['system_id']}"
    return base + ".pkl"


def _selection_stem(identity: Mapping[str, int]) -> str:
    base = (
        f"c{identity['config_code']}_s{identity['scenario_code']}_"
        f"d{identity['condition_code']}_p{identity['split_code']}"
    )
    if identity.get("system_id", -1) != -1:
        base += f"_m{identity['system_id']}"
    return base


def _load_checkpoint(
    path: Path, run_manifest_hash: str | None = None
) -> dict[str, Any]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if (
        run_manifest_hash is not None
        and checkpoint.get("run_manifest_hash") != run_manifest_hash
    ):
        raise search.IntegrityError(
            "checkpoint run_manifest_hash differs: "
            f"stored={checkpoint.get('run_manifest_hash')!r}, expected={run_manifest_hash!r}"
        )
    return checkpoint


def _matching_l1_rows(
    run_dir: Path, identity: Mapping[str, int]
) -> list[dict[str, str]]:
    path = run_dir / "logs" / "l1_search.csv"
    if not path.exists():
        return []
    rows = scan_checked_csv(path, L1_SEARCH_FIELDS, truncate_tail=False)
    return [
        row
        for row in rows
        if all(
            int(row[field]) == int(identity[field])
            for field in (
                "config_code",
                "scenario_code",
                "condition_code",
                "split_code",
            )
        )
    ]


def scan_coordinator(
    run_dir: Path,
    identity: Mapping[str, int],
    n_dim: int,
    *,
    require_complete: bool,
) -> dict[str, int]:
    checkpoint_dir = run_dir / "logs" / "checkpoints"
    checkpoints: dict[int, dict[str, Any]] = {}
    prefix = (
        f"c{identity['config_code']}_s{identity['scenario_code']}_"
        f"d{identity['condition_code']}_p{identity['split_code']}_r"
    )
    system_suffix = (
        "" if identity.get("system_id", -1) == -1 else f"_m{identity['system_id']}"
    )
    pattern = re.compile(
        rf"{re.escape(prefix)}([1-9][0-9]*){re.escape(system_suffix)}\.pkl"
    )
    for path in checkpoint_dir.glob(prefix + "*.pkl"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        restart_id = int(match.group(1))
        if restart_id in checkpoints:
            raise search.IntegrityError(f"duplicate checkpoint restart_id={restart_id}")
        checkpoints[restart_id] = _load_checkpoint(path)
    expected = frozenset(range(1, int(search.DIM_TIERS[n_dim]["n_restart"]) + 1))
    evidence = {int(row["restart_id"]) for row in _matching_l1_rows(run_dir, identity)}
    selection_path = (
        run_dir / "logs" / "selection" / (_selection_stem(identity) + ".json")
    )
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        evidence.update(int(item["restart_id"]) for item in selection["candidates"])
    present = set(checkpoints)
    for restart_id in expected:
        if any(higher > restart_id for higher in present):
            evidence.add(restart_id)
    return search.rebuild_coordinator(
        checkpoints, expected, frozenset(evidence), require_complete
    )


def trace_fields(n_dim: int) -> list[str]:
    return [
        "row_kind",
        "selection_kind",
        "restart_id",
        "gen_id",
        "candidate_id",
        *(f"x_raw_{index}" for index in range(n_dim)),
        *(f"x_box_{index}" for index in range(n_dim)),
        "J",
        "n_div",
        "J_val",
        "J_CMA",
        "J_inc",
        "source",
        "winner_restart_id",
        "n_tied",
        "tie_rule",
        "final_ref",
        "cum_scope",
        "logical_eval_count",
        "discarded_alloc",
        "h",
    ]


def _trace_rows(
    checkpoints: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
    n_dim: int,
    coordinator: Mapping[str, int],
) -> list[dict[str, Any]]:
    fields = trace_fields(n_dim)
    blank = {field: "" for field in fields[:-1]}
    rows: list[dict[str, Any]] = []
    for restart_id in sorted(checkpoints):
        for record in checkpoints[restart_id]["gen_records"]:
            row = dict(blank)
            row.update(
                {
                    "row_kind": record["row_kind"],
                    "restart_id": record["restart_id"],
                    "gen_id": record["gen_id"],
                    "candidate_id": record["candidate_id"],
                    "J": record["J"],
                    "n_div": record["n_div"],
                    "cum_scope": "restart",
                    "logical_eval_count": record["logical_eval_count_at_commit"],
                    "discarded_alloc": record["discarded_alloc_at_commit"],
                }
            )
            if record["x_raw"] is not None:
                row.update(
                    {f"x_raw_{index}": record["x_raw"][index] for index in range(n_dim)}
                )
            if record["x_box"] is not None:
                row.update(
                    {f"x_box_{index}": record["x_box"][index] for index in range(n_dim)}
                )
            rows.append(row)
    final = dict(blank)
    winner = int(selection["winner_restart_id"])
    candidate = next(
        item for item in selection["candidates"] if int(item["restart_id"]) == winner
    )
    final.update(
        {
            "row_kind": 2,
            "selection_kind": 0,
            "J_val": selection["J_val_winner"],
            "winner_restart_id": winner,
            "n_tied": selection["n_tied"],
            "tie_rule": selection["tie_rule"],
            "final_ref": candidate["final_ref"],
            "cum_scope": "search",
            "logical_eval_count": coordinator["logical_eval_count"],
            "discarded_alloc": coordinator["discarded_alloc"],
        }
    )
    rows.append(final)
    return rows


def _write_trace(
    path: Path,
    run_manifest_hash: str,
    checkpoints: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any],
    n_dim: int,
    coordinator: Mapping[str, int],
) -> None:
    fields = trace_fields(n_dim)
    if path.exists():
        path.unlink()
    initialize_checked_csv(path, run_manifest_hash, fields)
    for row in _trace_rows(checkpoints, selection, n_dim, coordinator):
        append_checked_row(path, fields, row)


def validate_trace(
    path: Path, checkpoint_dir: Path, selection_dir: Path, n_dim: int
) -> bool:
    rows = scan_checked_csv(path, trace_fields(n_dim), truncate_tail=False)
    selection_path = next(selection_dir.glob("*.json"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    search_key = selection["search_key"]
    checkpoints: dict[int, dict[str, Any]] = {}
    for item in sorted(checkpoint_dir.glob("*.pkl")):
        checkpoint = _load_checkpoint(item)
        identity = checkpoint["l2_identity"]
        if any(int(identity[field]) != int(search_key[field]) for field in search_key):
            continue
        restart_id = int(identity["restart_id"])
        if restart_id in checkpoints:
            raise search.IntegrityError(f"duplicate checkpoint restart_id={restart_id}")
        checkpoints[restart_id] = checkpoint
    coordinator = search.rebuild_coordinator(
        checkpoints,
        frozenset(range(1, int(search.DIM_TIERS[n_dim]["n_restart"]) + 1)),
        frozenset(checkpoints),
        True,
    )
    expected = _trace_rows(checkpoints, selection, n_dim, coordinator)
    if len(rows) != len(expected):
        raise search.IntegrityError("search trace row count differs from authorities")
    for actual, wanted in zip(rows, expected, strict=True):
        normalized = {
            field: str(wanted[field]) if wanted[field] != "" else ""
            for field in trace_fields(n_dim)[:-1]
        }
        for field in trace_fields(n_dim)[:-1]:
            if actual[field] != normalized[field]:
                raise search.IntegrityError(f"search trace differs at {field}")
    return True


def copy_coordinator_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    shutil.copytree(
        source / "logs" / "checkpoints", destination / "logs" / "checkpoints"
    )
    shutil.copytree(source / "logs" / "selection", destination / "logs" / "selection")
    shutil.copy2(
        source / "logs" / "l1_search.csv", destination / "logs" / "l1_search.csv"
    )


def remove_restart_evidence(run_dir: Path, restart_id: int) -> None:
    l1 = run_dir / "logs" / "l1_search.csv"
    rows = scan_checked_csv(l1, L1_SEARCH_FIELDS, truncate_tail=False)
    run_hash = l1.read_text(encoding="utf-8").splitlines()[0].split("=", 1)[1]
    l1.unlink()
    initialize_checked_csv(l1, run_hash, L1_SEARCH_FIELDS)
    for row in rows:
        if int(row["restart_id"]) != restart_id:
            append_checked_row(
                l1,
                L1_SEARCH_FIELDS,
                {field: row[field] for field in L1_SEARCH_FIELDS[:-1]},
            )
    selection_path = next((run_dir / "logs" / "selection").glob("*.json"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidates"] = [
        item
        for item in selection["candidates"]
        if int(item["restart_id"]) != restart_id
    ]
    _atomic_json(selection_path, selection)


def _killpoint_ready(point_id: str) -> None:
    if os.environ.get("E1_TEST_MODE") != "1":
        return
    if os.environ.get("E1_TEST_KILLPOINT") != point_id:
        return
    sys.stderr.write(f"E1-READY {point_id}\n")
    sys.stderr.flush()
    while True:
        time.sleep(1.0)


def _seed_records(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for m in range(3 * int(spec["n_systems"])):
        system_seed = rng.bench_system_key(1, m)
        records.append(
            {
                "seed_kind": "bench_system",
                "key": _seed_key(system_seed),
                "entropy": _seed_key(system_seed),
            }
        )
        for trajectory_id in range(5):
            trajectory_seed = rng.bench_traj_key(1, m, trajectory_id)
            records.append(
                {
                    "seed_kind": "bench_trajectory",
                    "key": _seed_key(trajectory_seed),
                    "entropy": _seed_key(trajectory_seed),
                }
            )
    n_restart = int(search.DIM_TIERS[int(spec["n_dim"])]["n_restart"])
    for restart_id in range(1, n_restart + 1):
        cma_seed = rng.bench_cma_key(
            1,
            MINI_SEARCH_IDENTITY["config_code"],
            MINI_SEARCH_IDENTITY["scenario_code"],
            restart_id,
        )
        records.append(
            {
                "seed_kind": "bench_cma",
                "key": _seed_key(cma_seed),
                "entropy": _seed_key(cma_seed),
            }
        )
    return records


def _task_key(
    run_manifest_hash: str,
    eval_kind: int,
    restart_id: int,
    gen_id: int,
    candidate_id: int,
    m: int,
) -> tuple[Any, ...]:
    identity = MINI_SEARCH_IDENTITY
    return (
        run_manifest_hash,
        identity["config_code"],
        identity["scenario_code"],
        identity["condition_code"],
        identity["split_code"],
        eval_kind,
        restart_id,
        gen_id,
        candidate_id,
        m,
    )


def _row_key(row: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(row[field])
        for field in (
            "config_code",
            "scenario_code",
            "condition_code",
            "split_code",
            "eval_kind",
            "restart_id",
            "gen_id",
            "candidate_id",
            "m",
        )
    )


def _existing_row_map(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, ...], Mapping[str, str]]:
    return {_row_key(row): row for row in rows}


def _make_l1_row(result: Mapping[str, Any], eval_kind: int) -> dict[str, Any]:
    key = tuple(result["task_key"])
    base = {
        "config_code": key[1],
        "scenario_code": key[2],
        "condition_code": key[3],
        "split_code": key[4],
        "eval_kind": key[5],
        "restart_id": key[6],
        "gen_id": key[7],
        "candidate_id": key[8],
        "m": key[9],
    }
    if eval_kind in (0, 1):
        base.update({f"R{index}": result["risks"][index] for index in range(3)})
        base.update({f"st{index}": result["statuses"][index] for index in range(3)})
    else:
        base.update({f"R{index + 3}": result["risks"][index] for index in range(2)})
        base.update({f"st{index + 3}": result["statuses"][index] for index in range(2)})
    return base


def _result_from_l1(row: Mapping[str, str], run_manifest_hash: str) -> dict[str, Any]:
    eval_kind = int(row["eval_kind"])
    if eval_kind in (0, 1):
        risks = [float(row[f"R{index}"]) for index in range(3)]
        statuses = [int(row[f"st{index}"]) for index in range(3)]
    else:
        risks = [float(row[f"R{index}"]) for index in range(3, 5)]
        statuses = [int(row[f"st{index}"]) for index in range(3, 5)]
    key_values = (
        tuple(int(row[field]) for field in L1_SEARCH_FIELDS[:9])
        if eval_kind in (0, 1)
        else tuple(int(row[field]) for field in L1_EVAL_FIELDS[:9])
    )
    return {
        "task_key": (run_manifest_hash, *key_values),
        "candidate_id": int(row["candidate_id"]),
        "m": int(row["m"]),
        "risks": risks,
        "statuses": statuses,
        "pid": None,
    }


def _evaluate_candidate_batch(
    pool: Any,
    audited_pids: set[int],
    run_manifest_hash: str,
    candidates: Sequence[tuple[int, Sequence[float]]],
    system_ids: set[int],
    eval_kind: int,
    restart_id: int,
    gen_id: int,
    existing: Mapping[tuple[int, ...], Mapping[str, str]],
    l1_path: Path,
    fields: Sequence[str],
    executed_pids: set[int],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for candidate_id, x_box in candidates:
        for m in sorted(system_ids):
            task_key = _task_key(
                run_manifest_hash, eval_kind, restart_id, gen_id, candidate_id, m
            )
            short_key = tuple(int(value) for value in task_key[1:])
            if short_key in existing:
                results.append(_result_from_l1(existing[short_key], run_manifest_hash))
            else:
                tasks.append(
                    {
                        "task_key": task_key,
                        "candidate_id": candidate_id,
                        "m": m,
                        "eval_kind": eval_kind,
                        "x_box": list(map(float, x_box)),
                    }
                )
    if tasks:
        if live_pool_pids(pool) != audited_pids:
            raise search.IntegrityError("worker PID set changed before task dispatch")
        fresh = collect_results(pool, tasks, audited_pids)
        executed_pids.update(int(item["pid"]) for item in fresh)
        results.extend(fresh)
        for index, result in enumerate(
            sorted(fresh, key=lambda item: (int(item["candidate_id"]), int(item["m"])))
        ):
            append_checked_row(
                l1_path,
                fields,
                _make_l1_row(result, eval_kind),
                kill_half=index == 0,
            )
    validate_generation_buffer(
        results,
        candidate_ids={candidate_id for candidate_id, _ in candidates},
        system_ids=system_ids,
    )
    return sorted(results, key=lambda item: (int(item["candidate_id"]), int(item["m"])))


def _aggregate_results(
    results: Sequence[Mapping[str, Any]],
    candidate_id: int,
    eval_kind: int,
    r_cap: float,
) -> tuple[float, int]:
    selected = [item for item in results if int(item["candidate_id"]) == candidate_id]
    risks = {int(item["m"]): list(map(float, item["risks"])) for item in selected}
    if eval_kind not in (0, 1):
        raise search.IntegrityError("final_eval does not use j_train/j_val")
    objective_fn = search.j_train if eval_kind == 0 else search.j_val
    value = objective_fn(risks, r_cap, lambda _value: None)
    rows = [_make_l1_row(item, eval_kind) for item in selected]
    return value, count_n_div(rows, eval_kind)


def _load_all_checkpoints(
    run_dir: Path, n_dim: int, run_manifest_hash: str
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for restart_id in range(1, int(search.DIM_TIERS[n_dim]["n_restart"]) + 1):
        path = (
            run_dir
            / "logs"
            / "checkpoints"
            / _checkpoint_name(MINI_SEARCH_IDENTITY, restart_id)
        )
        if path.exists():
            output[restart_id] = _load_checkpoint(path, run_manifest_hash)
    return output


def _append_resume_event(
    run_dir: Path,
    run_manifest_hash: str,
    checkpoint: Mapping[str, Any],
    popsize: int,
) -> None:
    path = run_dir / "logs" / "resume_events.csv"
    if not path.exists():
        initialize_checked_csv(path, run_manifest_hash, RESUME_EVENT_FIELDS)
    identity = checkpoint["l2_identity"]
    append_checked_row(
        path,
        RESUME_EVENT_FIELDS,
        {
            "config_code": identity["config_code"],
            "scenario_code": identity["scenario_code"],
            "condition_code": identity["condition_code"],
            "split_code": identity["split_code"],
            "system_id": identity["system_id"],
            "restart_id": identity["restart_id"],
            "gen_id": identity["gen_id"],
            "lambda_pop": popsize,
            "run_manifest_hash": run_manifest_hash,
        },
    )


def _run_search(
    run_dir: Path,
    pool: Any,
    audited_pids: set[int],
    run_manifest_hash: str,
    spec: Mapping[str, Any],
    *,
    resume: bool,
) -> set[int]:
    logs = run_dir / "logs"
    l1_search = logs / "l1_search.csv"
    l1_eval = logs / "l1_eval.csv"
    resume_events = logs / "resume_events.csv"
    if not l1_search.exists():
        initialize_checked_csv(l1_search, run_manifest_hash, L1_SEARCH_FIELDS)
    if not l1_eval.exists():
        initialize_checked_csv(l1_eval, run_manifest_hash, L1_EVAL_FIELDS)
    if not resume_events.exists():
        initialize_checked_csv(resume_events, run_manifest_hash, RESUME_EVENT_FIELDS)
    search_rows = scan_checked_csv(l1_search, L1_SEARCH_FIELDS, truncate_tail=resume)
    eval_rows = scan_checked_csv(l1_eval, L1_EVAL_FIELDS, truncate_tail=resume)
    scan_checked_csv(resume_events, RESUME_EVENT_FIELDS, truncate_tail=resume)
    search_existing = _existing_row_map(search_rows)
    eval_existing = _existing_row_map(eval_rows)
    n_dim = int(spec["n_dim"])
    r_cap = float(spec["r_cap"])
    n_systems = int(spec["n_systems"])
    train_systems = set(range(n_systems))
    val_systems = set(range(n_systems, 2 * n_systems))
    final_systems = set(range(2 * n_systems, 3 * n_systems))
    checkpoints = _load_all_checkpoints(run_dir, n_dim, run_manifest_hash)
    selection_path = (
        logs / "selection" / (_selection_stem(MINI_SEARCH_IDENTITY) + ".json")
    )
    scan_coordinator(
        run_dir,
        MINI_SEARCH_IDENTITY,
        n_dim,
        require_complete=selection_path.exists(),
    )
    if resume and not selection_path.exists() and checkpoints:
        latest = checkpoints[max(checkpoints)]
        _append_resume_event(
            run_dir,
            run_manifest_hash,
            latest,
            int(search.DIM_TIERS[n_dim]["popsize"]),
        )
    executed_pids: set[int] = set()
    pending_checkpoint: list[dict[str, Any]] = []

    def remember(checkpoint: dict[str, Any]) -> None:
        pending_checkpoint[:] = [checkpoint]

    unit: search.SearchUnit | None = None
    n_restart = int(search.DIM_TIERS[n_dim]["n_restart"])
    for restart_id in range(1, n_restart + 1):
        checkpoint_path = (
            logs / "checkpoints" / _checkpoint_name(MINI_SEARCH_IDENTITY, restart_id)
        )
        if restart_id in checkpoints:
            unit = search.SearchUnit.restore(
                checkpoints[restart_id], run_manifest_hash, checkpoint_callback=remember
            )
            unit.seed_kind = "bench"
            unit.bench_purpose = 1
        else:
            if unit is None:
                unit = search.SearchUnit(
                    MINI_SEARCH_IDENTITY,
                    n_dim,
                    run_manifest_hash,
                    "bench",
                    checkpoint_callback=remember,
                    bench_purpose=1,
                )
            else:
                unit.checkpoint_callback = remember
            unit.start_restart(restart_id)
            _atomic_pickle(checkpoint_path, pending_checkpoint[-1])
        while unit.terminal is None:
            candidates = unit.ask()
            generation_id = unit.generation_id + 1
            _killpoint_ready("1")
            batch = _evaluate_candidate_batch(
                pool,
                audited_pids,
                run_manifest_hash,
                [(item.candidate_id, item.x_box) for item in candidates],
                train_systems,
                0,
                restart_id,
                generation_id,
                search_existing,
                l1_search,
                L1_SEARCH_FIELDS,
                executed_pids,
            )
            eval_results = [
                search.EvalResult(
                    candidate_id, *_aggregate_results(batch, candidate_id, 0, r_cap)
                )
                for candidate_id in range(int(search.DIM_TIERS[n_dim]["popsize"]))
            ]
            unit.tell(eval_results)
            _killpoint_ready("2")
            _atomic_pickle(checkpoint_path, pending_checkpoint[-1])
            checkpoints[restart_id] = pending_checkpoint[-1]
        checkpoints[restart_id] = unit.checkpoint_dict()

    best_records = {
        restart_id: checkpoints[restart_id]["gen_records"] for restart_id in checkpoints
    }
    selection_candidates: list[dict[str, Any]] = []
    for restart_id in sorted(checkpoints):
        records = best_records[restart_id]
        finite = [
            record
            for record in records
            if record["row_kind"] == 0 and np.isfinite(record["J"])
        ]
        best = min(
            finite,
            key=lambda record: (record["J"], record["gen_id"], record["candidate_id"]),
        )
        batch = _evaluate_candidate_batch(
            pool,
            audited_pids,
            run_manifest_hash,
            [(int(best["candidate_id"]), best["x_box"])],
            val_systems,
            1,
            restart_id,
            int(best["gen_id"]),
            search_existing,
            l1_search,
            L1_SEARCH_FIELDS,
            executed_pids,
        )
        value, _ = _aggregate_results(batch, int(best["candidate_id"]), 1, r_cap)
        selection_candidates.append(
            {
                "restart_id": restart_id,
                "J_val": value,
                "final_ref": f"r{restart_id}g{best['gen_id']}c{best['candidate_id']}",
            }
        )
    selection = search.make_selection(
        run_manifest_hash, MINI_SEARCH_IDENTITY, selection_candidates
    )
    if selection_path.exists():
        existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if canonical_bytes(existing_selection) != canonical_bytes(selection):
            raise search.IntegrityError("selection differs from rebuilt authorities")
    else:
        _atomic_json(selection_path, selection)

    winner_restart = int(selection["winner_restart_id"])
    winner_ref = next(
        item["final_ref"]
        for item in selection["candidates"]
        if int(item["restart_id"]) == winner_restart
    )
    _, winner_generation, winner_candidate = search.parse_final_ref(winner_ref)
    winner_record = next(
        record
        for record in checkpoints[winner_restart]["gen_records"]
        if int(record["gen_id"]) == winner_generation
        and int(record["candidate_id"]) == winner_candidate
    )
    _evaluate_candidate_batch(
        pool,
        audited_pids,
        run_manifest_hash,
        [(-1, winner_record["x_box"])],
        final_systems,
        2,
        -1,
        -1,
        eval_existing,
        l1_eval,
        L1_EVAL_FIELDS,
        executed_pids,
    )
    coordinator = scan_coordinator(
        run_dir, MINI_SEARCH_IDENTITY, n_dim, require_complete=True
    )
    trace_path = (
        run_dir / "search_trace" / (_selection_stem(MINI_SEARCH_IDENTITY) + ".csv")
    )
    if trace_path.exists():
        validate_trace(trace_path, logs / "checkpoints", logs / "selection", n_dim)
    else:
        _write_trace(
            trace_path,
            run_manifest_hash,
            checkpoints,
            selection,
            n_dim,
            coordinator,
        )
    return executed_pids


def _next_resume_number(run_dir: Path) -> int:
    root = run_dir / "logs" / "resume_env"
    if not root.exists():
        return 1
    numbers = [
        int(match.group(1))
        for path in root.glob("event_*.json")
        if (match := __import__("re").fullmatch(r"event_(\d{4})\.json", path.name))
    ]
    return max(numbers, default=0) + 1


def _record_verification_event(
    run_dir: Path,
    *,
    resume_number: int | None,
    spec_path: Path,
    receipt_path: Path | None,
) -> None:
    if resume_number is None:
        directory = run_dir / "logs" / "input_verification" / "fresh"
    else:
        directory = (
            run_dir / "logs" / "input_verification" / f"resume_{resume_number:04d}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    event = {
        "mode": "mini",
        "mini_spec_sha256": _sha256(spec_path),
        "input_receipt_sha256": None if receipt_path is None else _sha256(receipt_path),
        "note": "mini mode skips production B15 while still recomputing every declared input",
    }
    _atomic_json(directory / "inputs.json", event)


def _prepare_run(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], Any, dict[int, Mapping[str, Any]], str, bool]:
    if args.output_root is None:
        output_root = REPO_ROOT / "results"
    else:
        output_root = args.output_root
    run_dir = output_root / args.run_id
    if args.resume:
        if not run_dir.is_dir():
            raise search.IntegrityError("resume target directory does not exist")
        core = (
            run_dir / "logs" / "config.json",
            run_dir / "run_manifest.json",
            run_dir / "logs" / "env.json",
        )
        if not all(path.is_file() and os.access(path, os.R_OK) for path in core):
            raise search.IntegrityError(
                "incomplete startup directory cannot be resumed"
            )
    else:
        _ensure_directory(output_root)
        run_dir.mkdir(exist_ok=False)
    spec = _load_mini_spec(args.mini_spec)
    inputs, receipt = _input_records(args.mini_spec, args.input_receipt)
    receipt_record = next(
        (record for record in inputs if record["path"] == "input_receipt.json"), None
    )
    config = _build_config(spec, args.workers, receipt, receipt_record)
    commit, dirty, _ = _git_state(REPO_ROOT)
    ctx, queue, pool = create_pool(args.workers)
    try:
        profiles = collect_handshake(queue, args.workers)
        if live_pool_pids(pool) != set(profiles):
            raise search.IntegrityError(
                "worker PID set differs immediately after handshake"
            )
        projection = build_threadpool_projection(profiles)
        snapshot = snapshot_env.collect()
        manifest = _build_manifest(config, projection, snapshot, inputs, commit, dirty)
        run_manifest_hash = _run_hash(manifest)
        if args.resume:
            stored_config_path = run_dir / "logs" / "config.json"
            stored_manifest_path = run_dir / "run_manifest.json"
            if stored_config_path.read_bytes() != _pretty_json_bytes(config):
                raise search.IntegrityError(
                    "stored config bytes differ from recomputed config"
                )
            stored_manifest = json.loads(
                stored_manifest_path.read_text(encoding="utf-8")
            )
            stored_hash = _run_hash(stored_manifest)
            if stored_hash != run_manifest_hash:
                raise search.IntegrityError(
                    f"candidate manifest differs: stored={stored_hash}, current={run_manifest_hash}"
                )
            resume_number = _next_resume_number(run_dir)
            _record_verification_event(
                run_dir,
                resume_number=resume_number,
                spec_path=args.mini_spec,
                receipt_path=args.input_receipt,
            )
            _atomic_json(
                run_dir / "logs" / "resume_env" / f"event_{resume_number:04d}.json",
                {
                    "run_manifest_hash": run_manifest_hash,
                    "worker_threadpool_info": [
                        profiles[pid] for pid in sorted(profiles)
                    ],
                    "threadpool_projection": projection,
                },
            )
        else:
            _record_verification_event(
                run_dir,
                resume_number=None,
                spec_path=args.mini_spec,
                receipt_path=args.input_receipt,
            )
            _atomic_json(run_dir / "logs" / "config.json", config, read_only=True)
            if live_pool_pids(pool) != set(profiles):
                raise search.IntegrityError(
                    "worker PID set changed before manifest write"
                )
            _atomic_json(run_dir / "run_manifest.json", manifest, read_only=True)
            _atomic_json(
                run_dir / "logs" / "env.json",
                _raw_env(snapshot, profiles, run_manifest_hash, commit, args.workers),
            )
            _atomic_json(run_dir / "logs" / "seeds.json", _seed_records(spec))
        core = (
            run_dir / "logs" / "config.json",
            run_dir / "run_manifest.json",
            run_dir / "logs" / "env.json",
        )
        if not all(path.is_file() and os.access(path, os.R_OK) for path in core):
            raise search.IntegrityError("startup artifacts are not all readable")
        if live_pool_pids(pool) != set(profiles):
            raise search.IntegrityError("worker PID set changed before first task")
        return run_dir, spec, pool, profiles, run_manifest_hash, args.resume
    except BaseException:
        pool.terminate()
        pool.join()
        raise


def _write_sha256sums(run_dir: Path) -> None:
    files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(run_dir).as_posix()}" for path in files
    ]
    path = run_dir / "SHA256SUMS.txt"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the E1 simulation workflow")
    parser.add_argument("--mode", choices=("mini", "production"), required=True)
    parser.add_argument("--mini-spec", type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("E1_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--armed", action="store_true")
    return parser


def _run_cli(args: argparse.Namespace) -> int:
    if args.mode == "production":
        print("开跑前检查单总闸未开（PRODUCTION_ENABLED=False）")
        return PRODUCTION_DISABLED
    if args.mini_spec is None:
        raise search.IntegrityError("--mini-spec is required in mini mode")
    run_dir, spec, pool, profiles, run_manifest_hash, resume = _prepare_run(args)
    try:
        executed = _run_search(
            run_dir,
            pool,
            set(profiles),
            run_manifest_hash,
            spec,
            resume=resume,
        )
        if not resume and executed != set(profiles):
            raise search.IntegrityError(
                "executing worker PID set does not equal the audited worker PID set"
            )
        if not resume:
            _write_sha256sums(run_dir)
    finally:
        pool.close()
        pool.join()
    print(f"E1 mini completed: {run_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_streams()
    args = _parser().parse_args(argv)
    if args.mode == "production":
        return _run_cli(args)
    try:
        return _run_cli(args)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        pickle.PickleError,
        search.SearchError,
    ) as exc:
        print(f"E1 aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
