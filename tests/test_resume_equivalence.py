from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pytest


CODE = Path(__file__).resolve().parents[1] / "src" / "experiments"
REPO = CODE.parent
SRC = CODE.parent / "core"
sys.path.insert(0, str(SRC))

import search


MINI_SPEC = {"n_dim": 1, "n_systems": 2, "r_cap": 10.0}
READY_TIMEOUT_SECONDS = 120.0
PROCESS_TIMEOUT_SECONDS = 180.0
L1_KEY_FIELDS = (
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
L1_SEARCH_FIELDS = (
    *L1_KEY_FIELDS,
    "R0",
    "R1",
    "R2",
    "st0",
    "st1",
    "st2",
    "h",
)
L1_EVAL_FIELDS = (
    *L1_KEY_FIELDS,
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
RECORD_SCALARS = (
    "row_kind",
    "restart_id",
    "gen_id",
    "candidate_id",
    "J",
    "n_div",
    "stop_flags",
    "logical_eval_count_at_commit",
    "discarded_alloc_at_commit",
    "cum_scope",
)


@dataclass(frozen=True)
class ReferenceRun:
    spec: Path
    run_dir: Path


@dataclass(frozen=True)
class KillEvidence:
    restart_id: int
    generation_id: int
    bad_tail_offset: int | None = None
    bad_tail: bytes | None = None
    bad_size: int | None = None


def _write_spec(path: Path) -> Path:
    path.write_text(json.dumps(MINI_SPEC, sort_keys=True), encoding="utf-8")
    return path


def _command(root: Path, run_id: str, spec: Path, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        str(CODE / "run_e1.py"),
        "--mode",
        "mini",
        "--mini-spec",
        str(spec),
        "--run-id",
        run_id,
        "--output-root",
        str(root),
        "--workers",
        "2",
    ]
    if resume:
        command.append("--resume")
    return command


def _child_env(point_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("E1_TEST_MODE", None)
    env.pop("E1_TEST_KILLPOINT", None)
    if point_id is not None:
        env["E1_TEST_MODE"] = "1"
        env["E1_TEST_KILLPOINT"] = str(point_id)
    return env


def _run_complete(root: Path, run_id: str, spec: Path, *, resume: bool = False) -> Path:
    result = subprocess.run(
        _command(root, run_id, spec, resume=resume),
        cwd=REPO,
        env=_child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=PROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return root / run_id


def _stderr_reader(
    stream: Any, messages: queue.Queue[str | None], captured: list[str]
) -> None:
    try:
        for line in stream:
            captured.append(line)
            messages.put(line)
    finally:
        messages.put(None)


def _wait_for_ready(
    proc: subprocess.Popen[str], point_id: int
) -> tuple[threading.Thread, list[str]]:
    assert proc.stderr is not None
    messages: queue.Queue[str | None] = queue.Queue()
    captured: list[str] = []
    reader = threading.Thread(
        target=_stderr_reader,
        args=(proc.stderr, messages, captured),
        name=f"e1-ready-reader-{point_id}",
        daemon=True,
    )
    reader.start()
    expected = f"E1-READY {point_id}"
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            item = messages.get(timeout=remaining)
            if item is None:
                raise AssertionError(
                    f"child exited before {expected}: returncode={proc.poll()}\n"
                    + "".join(captured)
                )
            if item.rstrip("\r\n") == expected:
                return reader, captured
    except queue.Empty as exc:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        reader.join(timeout=30)
        raise AssertionError(
            f"timed out waiting for {expected}\n" + "".join(captured)
        ) from exc


def _kill_at(
    root: Path,
    run_id: str,
    spec: Path,
    point_id: int,
    *,
    resume: bool = False,
) -> KillEvidence:
    proc = subprocess.Popen(
        _command(root, run_id, spec, resume=resume),
        cwd=REPO,
        env=_child_env(point_id),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    reader, captured = _wait_for_ready(proc, point_id)
    proc.kill()
    proc.wait(timeout=30)
    reader.join(timeout=30)
    assert not reader.is_alive(), "stderr reader did not finish after child termination"
    stdout = ""
    if proc.stdout is not None:
        stdout = proc.stdout.read()
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()
    assert proc.returncode != 0, stdout + "".join(captured)
    return _inspect_kill_site(root / run_id, point_id)


def _row_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(",".join(values).encode("utf-8")).hexdigest()[:16]


def _parse_checked_line(line: bytes, fields: Sequence[str]) -> dict[str, str]:
    decoded = line.decode("utf-8").rstrip("\r\n")
    values = next(csv.reader([decoded]))
    if len(values) != len(fields):
        raise ValueError("checked row has the wrong field count")
    if _row_digest(values[:-1]) != values[-1]:
        raise ValueError("checked row has the wrong digest")
    return dict(zip(fields, values, strict=True))


def _read_checked(
    path: Path, expected_fields: Sequence[str] | None = None
) -> list[dict[str, str]]:
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 2
    assert lines[0].startswith(b"# run_manifest_hash=")
    fields = tuple(next(csv.reader([lines[1].decode("utf-8").rstrip("\r\n")])))
    if expected_fields is not None:
        assert fields == tuple(expected_fields)
    return [_parse_checked_line(line, fields) for line in lines[2:]]


def _bad_tail(path: Path, fields: Sequence[str]) -> tuple[int, bytes, int]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert len(lines) >= 3
    offset = len(lines[0]) + len(lines[1])
    for index, line in enumerate(lines[2:]):
        try:
            _parse_checked_line(line, fields)
        except (UnicodeDecodeError, ValueError):
            assert index == len(lines[2:]) - 1, (
                "corruption was not confined to the tail"
            )
            return offset, line, len(raw)
        offset += len(line)
    raise AssertionError("killpoint 3 did not leave a corrupt tail row")


def _load_checkpoints(run_dir: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "logs" / "checkpoints").glob("*.pkl")):
        checkpoint = pickle.loads(path.read_bytes())
        restart_id = int(checkpoint["l2_identity"]["restart_id"])
        assert restart_id not in output
        output[restart_id] = checkpoint
    return output


def _latest_checkpoint(
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    assert checkpoints
    return checkpoints[max(checkpoints)]


def _inspect_kill_site(run_dir: Path, point_id: int) -> KillEvidence:
    checkpoints = _load_checkpoints(run_dir)
    checkpoint = _latest_checkpoint(checkpoints)
    restart_id = int(checkpoint["l2_identity"]["restart_id"])
    committed_gen = int(checkpoint["l2_identity"]["gen_id"])
    assert checkpoint["terminal"] is None
    l1_path = run_dir / "logs" / "l1_search.csv"

    if point_id == 1:
        rows = _read_checked(l1_path, L1_SEARCH_FIELDS)
        generation_id = committed_gen + 1
        assert committed_gen == 0
        assert not any(
            int(row["restart_id"]) == restart_id and int(row["gen_id"]) == generation_id
            for row in rows
        )
        return KillEvidence(restart_id, generation_id)

    if point_id == 2:
        rows = _read_checked(l1_path, L1_SEARCH_FIELDS)
        generations = {
            int(row["gen_id"]) for row in rows if int(row["restart_id"]) == restart_id
        }
        assert generations
        generation_id = max(generations)
        assert committed_gen < generation_id
        return KillEvidence(restart_id, generation_id)

    assert point_id == 3
    offset, tail, size = _bad_tail(l1_path, L1_SEARCH_FIELDS)
    return KillEvidence(restart_id, committed_gen + 1, offset, tail, size)


def _l1_arrays(
    run_dir: Path,
    name: str,
    fields: Sequence[str],
    risk_fields: Sequence[str],
    status_fields: Sequence[str],
) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
    rows = _read_checked(run_dir / "logs" / name, fields)
    keyed: dict[tuple[int, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(int(row[field]) for field in L1_KEY_FIELDS)
        assert key not in keyed, f"duplicate complete L1 key: {key!r}"
        keyed[key] = row
    keys = sorted(keyed)
    risks = np.asarray(
        [[float(keyed[key][field]) for field in risk_fields] for key in keys],
        dtype=np.float64,
    )
    statuses = np.asarray(
        [[int(keyed[key][field]) for field in status_fields] for key in keys],
        dtype=np.int64,
    )
    return keys, risks, statuses


def _assert_l1_equal(reference: Path, resumed: Path) -> None:
    cases = (
        ("l1_search.csv", L1_SEARCH_FIELDS, ("R0", "R1", "R2"), ("st0", "st1", "st2")),
        ("l1_eval.csv", L1_EVAL_FIELDS, ("R3", "R4"), ("st3", "st4")),
    )
    for name, fields, risks, statuses in cases:
        left = _l1_arrays(reference, name, fields, risks, statuses)
        right = _l1_arrays(resumed, name, fields, risks, statuses)
        assert left[0] == right[0]
        assert np.array_equal(left[1], right[1])
        assert np.array_equal(left[2], right[2])


def _trace_projection(run_dir: Path) -> tuple[list[tuple[str, ...]], np.ndarray]:
    paths = list((run_dir / "search_trace").glob("*.csv"))
    assert len(paths) == 1
    rows = _read_checked(paths[0])
    assert rows
    assert all(row["source"] == "" for row in rows), (
        "ordinary mini expects an empty source column; this does not exercise "
        "the W3-o oracle path"
    )
    coordinate_fields = sorted(
        (field for field in rows[0] if field.startswith(("x_raw_", "x_box_"))),
        key=lambda field: (field.split("_")[0:2], int(field.rsplit("_", 1)[1])),
    )
    value_fields = (*coordinate_fields, "J", "source")
    keyed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(
            row[field] for field in ("row_kind", "restart_id", "gen_id", "candidate_id")
        )
        assert key not in keyed
        keyed[key] = row
    keys = sorted(keyed)
    values = np.asarray(
        [[keyed[key][field] for field in value_fields] for key in keys], dtype=str
    )
    return keys, values


def _assert_record_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    assert set(left) == set(right)
    for field in RECORD_SCALARS:
        assert left[field] == right[field]
    for field in ("x_raw", "x_box"):
        if left[field] is None or right[field] is None:
            assert left[field] is None and right[field] is None
        else:
            assert np.array_equal(left[field], right[field])


def _selection(run_dir: Path) -> dict[str, Any]:
    paths = list((run_dir / "logs" / "selection").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _assert_checkpoints_and_selection_equal(reference: Path, resumed: Path) -> None:
    left_cp = _load_checkpoints(reference)
    right_cp = _load_checkpoints(resumed)
    assert left_cp.keys() == right_cp.keys()
    for restart_id in left_cp:
        left = left_cp[restart_id]
        right = right_cp[restart_id]
        assert left["oracle"] is None and right["oracle"] is None
        assert left["budget"] == right["budget"]
        assert left["terminal"] == right["terminal"]
        assert len(left["gen_records"]) == len(right["gen_records"])
        for left_record, right_record in zip(
            left["gen_records"], right["gen_records"], strict=True
        ):
            _assert_record_equal(left_record, right_record)

    left_selection = _selection(reference)
    right_selection = _selection(resumed)
    for field in ("winner_restart_id", "J_val_winner", "n_tied", "tie_rule"):
        assert left_selection[field] == right_selection[field]
    left_candidates = {
        int(item["restart_id"]): (item["J_val"], item["final_ref"])
        for item in left_selection["candidates"]
    }
    right_candidates = {
        int(item["restart_id"]): (item["J_val"], item["final_ref"])
        for item in right_selection["candidates"]
    }
    assert left_candidates == right_candidates

    left_final = search.locate_w3_x_box(left_selection, left_cp)
    right_final = search.locate_w3_x_box(right_selection, right_cp)
    assert np.array_equal(left_final, right_final)


def _assert_scientific_outputs_equal(reference: Path, resumed: Path) -> None:
    _assert_l1_equal(reference, resumed)
    left_trace = _trace_projection(reference)
    right_trace = _trace_projection(resumed)
    assert left_trace[0] == right_trace[0]
    assert np.array_equal(left_trace[1], right_trace[1])
    _assert_checkpoints_and_selection_equal(reference, resumed)


def _resume_events(run_dir: Path) -> list[dict[str, str]]:
    return _read_checked(run_dir / "logs" / "resume_events.csv", RESUME_EVENT_FIELDS)


def _assert_resume_events(run_dir: Path, expected_count: int) -> list[dict[str, str]]:
    rows = _resume_events(run_dir)
    assert len(rows) == expected_count
    authorized_reissue_slots = sum(int(row["lambda_pop"]) for row in rows)
    assert authorized_reissue_slots == expected_count * 4
    return rows


def _assert_bad_tail_replaced(run_dir: Path, evidence: KillEvidence) -> None:
    assert evidence.bad_tail_offset is not None
    assert evidence.bad_tail is not None
    assert evidence.bad_size is not None
    path = run_dir / "logs" / "l1_search.csv"
    raw = path.read_bytes()
    assert len(raw) > evidence.bad_size
    assert evidence.bad_tail_offset < len(raw)
    newline = raw.find(b"\n", evidence.bad_tail_offset)
    assert newline >= 0
    replacement = raw[evidence.bad_tail_offset : newline + 1]
    assert len(replacement) > len(evidence.bad_tail)
    assert replacement.startswith(evidence.bad_tail)
    _parse_checked_line(replacement, L1_SEARCH_FIELDS)
    _read_checked(path, L1_SEARCH_FIELDS)


@pytest.fixture(scope="session")
def reference_run(tmp_path_factory: pytest.TempPathFactory) -> ReferenceRun:
    root = tmp_path_factory.mktemp("e_prime_reference")
    spec = _write_spec(root / "mini.json")
    run_dir = _run_complete(root, "reference", spec)
    _assert_resume_events(run_dir, 0)
    return ReferenceRun(spec, run_dir)


@pytest.mark.parametrize("point_id", [1, 2, 3])
def test_resume_is_bitwise_equivalent_after_each_killpoint(
    point_id: int, reference_run: ReferenceRun, tmp_path: Path
) -> None:
    evidence = _kill_at(tmp_path, "resumed", reference_run.spec, point_id)
    run_dir = _run_complete(tmp_path, "resumed", reference_run.spec, resume=True)
    _assert_scientific_outputs_equal(reference_run.run_dir, run_dir)
    _assert_resume_events(run_dir, 1)
    if point_id == 3:
        _assert_bad_tail_replaced(run_dir, evidence)


def test_same_generation_can_be_killed_twice_and_reissued_twice(
    reference_run: ReferenceRun, tmp_path: Path
) -> None:
    first = _kill_at(tmp_path, "double", reference_run.spec, 2)
    second = _kill_at(tmp_path, "double", reference_run.spec, 2, resume=True)
    assert (first.restart_id, first.generation_id) == (
        second.restart_id,
        second.generation_id,
    )
    run_dir = _run_complete(tmp_path, "double", reference_run.spec, resume=True)
    _assert_scientific_outputs_equal(reference_run.run_dir, run_dir)
    events = _assert_resume_events(run_dir, 2)
    assert len({(row["restart_id"], row["gen_id"]) for row in events}) == 1
