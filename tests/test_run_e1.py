from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import stat
import subprocess
import sys
from typing import Any

import pytest


CODE = Path(__file__).resolve().parents[1] / "src" / "experiments"
REPO = CODE.parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent / "core"))

import run_e1
from search import IntegrityError


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _mini_spec(path: Path, **overrides: Any) -> Path:
    spec = {"n_dim": 1, "n_systems": 2, "r_cap": 10.0}
    spec.update(overrides)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def _run_mini(
    root: Path,
    run_id: str,
    spec: Path,
    *,
    resume: bool = False,
    workers: int = 2,
    receipt: Path | None = None,
) -> subprocess.CompletedProcess[str]:
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
        str(workers),
    ]
    if resume:
        command.append("--resume")
    if receipt is not None:
        command.extend(("--input-receipt", str(receipt)))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )


@pytest.fixture(scope="session")
def mini_result(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("d_prime")
    spec = _mini_spec(root / "mini.json")
    result = _run_mini(root, "mini-main", spec)
    assert result.returncode == 0, result.stdout + result.stderr
    return root / "mini-main", spec


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# run_manifest_hash=")
    reader = csv.DictReader(lines[1:])
    return list(reader.fieldnames or ()), list(reader)


def test_01_a1_worker_thread_audit_and_direct_spawn(
    mini_result: tuple[Path, Path],
) -> None:
    run_dir, _ = mini_result
    env = json.loads((run_dir / "logs" / "env.json").read_text(encoding="utf-8"))
    profiles = env["worker_threadpool_info"]
    assert len(profiles) == 2
    for profile in profiles:
        assert profile["pools"]
        assert all(pool["num_threads"] == 1 for pool in profile["pools"])

    ctx, queue, pool = run_e1.create_pool(1)
    try:
        audited = run_e1.collect_handshake(queue, 1, deadline_seconds=30)
        assert run_e1.live_pool_pids(pool) == set(audited)
        assert all(
            lib["num_threads"] == 1 for lib in audited[next(iter(audited))]["pools"]
        )
    finally:
        pool.terminate()
        pool.join()


def test_02_a3_default_workers_and_static_bans() -> None:
    source = (CODE / "run_e1.py").read_text(encoding="utf-8")
    assert run_e1.DEFAULT_WORKERS == 12
    assert "os." + "cpu_count" not in source
    assert 'multiprocessing.get_context("spawn").Pool' in source
    assert "ProcessPoolExecutor" not in source


def test_03_manifest_hash_and_stable_projection(mini_result: tuple[Path, Path]) -> None:
    run_dir, _ = mini_result
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "config",
        "codebooks",
        "environment",
        "code",
        "inputs",
    }
    expected_hash = hashlib.sha256(_canonical(manifest)).hexdigest()
    first = (
        (run_dir / "logs" / "l1_search.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first == f"# run_manifest_hash={expected_hash}"
    serialized = json.dumps(manifest)
    assert not re.search(r'"(?:pid|filepath|timestamp)"\s*:', serialized)
    environment = manifest["environment"]
    assert "threadpool_info" not in environment
    assert run_e1.validate_threadpool_projection(environment["threadpool_projection"])


def test_04_config_authority_and_read_only(mini_result: tuple[Path, Path]) -> None:
    run_dir, _ = mini_result
    config_path = run_dir / "logs" / "config.json"
    manifest_path = run_dir / "run_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert _canonical(config) == _canonical(manifest["config"])
    assert not (config_path.stat().st_mode & stat.S_IWUSR)
    assert not (manifest_path.stat().st_mode & stat.S_IWUSR)


def test_05_l1_hash_repr_and_complete_identity(mini_result: tuple[Path, Path]) -> None:
    run_dir, _ = mini_result
    _, rows = _csv_rows(run_dir / "logs" / "l1_search.csv")
    assert rows
    for row in rows[:10]:
        assert run_e1.checked_row_hash(row) == row["h"]
        for name in ("R0", "R1", "R2"):
            assert repr(float(row[name])) == row[name]
    keys = [tuple(row[name] for name in run_e1.L1_SEARCH_FIELDS[:9]) for row in rows]
    assert len(keys) == len(set(keys))
    assert max(int(row["gen_id"]) for row in rows) >= 2


def test_06_l1_tail_truncation_middle_damage_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "l1.csv"
    run_e1.initialize_checked_csv(path, "a" * 64, run_e1.L1_SEARCH_FIELDS)
    row = run_e1.test_search_row(candidate_id=0)
    run_e1.append_checked_row(path, run_e1.L1_SEARCH_FIELDS, row)
    with path.open("ab") as handle:
        handle.write(b"1,2,broken")
    recovered = run_e1.scan_checked_csv(
        path, run_e1.L1_SEARCH_FIELDS, truncate_tail=True
    )
    assert len(recovered) == 1
    run_e1.append_checked_row(
        path, run_e1.L1_SEARCH_FIELDS, run_e1.test_search_row(candidate_id=1)
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace(",0.0,", ",9.0,", 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="middle"):
        run_e1.scan_checked_csv(path, run_e1.L1_SEARCH_FIELDS, truncate_tail=True)

    dup = tmp_path / "dup.csv"
    run_e1.initialize_checked_csv(dup, "a" * 64, run_e1.L1_SEARCH_FIELDS)
    run_e1.append_checked_row(dup, run_e1.L1_SEARCH_FIELDS, row)
    run_e1.append_checked_row(dup, run_e1.L1_SEARCH_FIELDS, row)
    with pytest.raises(IntegrityError, match="duplicate"):
        run_e1.scan_checked_csv(dup, run_e1.L1_SEARCH_FIELDS, truncate_tail=True)


def test_07_generation_buffer_rejects_missing_result() -> None:
    results = [
        {"candidate_id": 0, "m": 0},
        {"candidate_id": 1, "m": 0},
    ]
    with pytest.raises(IntegrityError, match="generation"):
        run_e1.validate_generation_buffer(
            results[:-1], candidate_ids={0, 1}, system_ids={0}
        )


def test_08_production_hard_door_and_preflight(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CODE / "run_e1.py"), "--mode", "production", "--armed"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 10
    assert "PRODUCTION_ENABLED=False" in result.stdout + result.stderr
    assert run_e1.PRODUCTION_ENABLED is False
    assert run_e1._production_preflight(REPO, None, {"returncode": 0}, "")[0] != 0
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"verification_digest":"abc"}', encoding="utf-8")
    assert run_e1._production_preflight(REPO, receipt, {"returncode": 1}, "")[0] != 0
    assert (
        run_e1._production_preflight(
            REPO, receipt, {"returncode": 0, "verification_digest": "def"}, ""
        )[0]
        != 0
    )
    assert (
        run_e1._production_preflight(
            REPO, receipt, {"returncode": 0, "verification_digest": "abc"}, " M x"
        )[0]
        != 0
    )
    assert run_e1._production_preflight(
        REPO, receipt, {"returncode": 0, "verification_digest": "abc"}, ""
    ) == (0, "ok")


def test_09_mini_seed_namespace_is_component_13(mini_result: tuple[Path, Path]) -> None:
    run_dir, _ = mini_result
    seeds = json.loads((run_dir / "logs" / "seeds.json").read_text(encoding="utf-8"))
    assert seeds
    assert all(item["key"][1] == 13 for item in seeds)


def test_10_two_fresh_minis_are_bitwise_deterministic(
    mini_result: tuple[Path, Path], tmp_path: Path
) -> None:
    first, spec = mini_result
    second_result = _run_mini(tmp_path, "mini-copy", spec)
    assert second_result.returncode == 0, second_result.stdout + second_result.stderr
    second = tmp_path / "mini-copy"
    for name in ("l1_search.csv", "l1_eval.csv"):
        assert (first / "logs" / name).read_bytes() == (
            second / "logs" / name
        ).read_bytes()
    first_cp = sorted((first / "logs" / "checkpoints").glob("*.pkl"))
    second_cp = sorted((second / "logs" / "checkpoints").glob("*.pkl"))
    assert len(first_cp) == len(second_cp)
    for left, right in zip(first_cp, second_cp, strict=True):
        a, b = pickle.loads(left.read_bytes()), pickle.loads(right.read_bytes())
        assert a["budget"] == b["budget"]
        assert a["terminal"] == b["terminal"]
        for ar, br in zip(a["gen_records"], b["gen_records"], strict=True):
            assert ar["J"] == br["J"]
            assert (ar["x_box"] == br["x_box"]).all()


def test_11_search_trace_rebuild_and_tamper_detection(
    mini_result: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, _ = mini_result
    trace = next((run_dir / "search_trace").glob("*.csv"))
    fields, rows = _csv_rows(trace)
    assert fields == run_e1.trace_fields(1)
    ordinary = [row for row in rows if row["row_kind"] in {"0", "1"}]
    assert all(not row["selection_kind"] and not row["final_ref"] for row in ordinary)
    final = [row for row in rows if row["row_kind"] == "2"]
    assert len(final) == 1 and final[0]["selection_kind"] == "0"
    assert final[0]["cum_scope"] == "search"
    damaged = tmp_path / "trace.csv"
    damaged.write_bytes(trace.read_bytes())
    text = damaged.read_text(encoding="utf-8")
    damaged.write_text(text.replace(",search,", ",restart,", 1), encoding="utf-8")
    with pytest.raises(IntegrityError):
        run_e1.validate_trace(
            damaged, run_dir / "logs" / "checkpoints", run_dir / "logs" / "selection", 1
        )


def test_12_fresh_resume_exclusion_and_immutable_core(
    mini_result: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, spec = mini_result
    root = run_dir.parent
    fresh_again = _run_mini(root, run_dir.name, spec)
    assert fresh_again.returncode != 0
    missing = _run_mini(tmp_path, "missing", spec, resume=True)
    assert missing.returncode != 0
    core = [
        run_dir / "logs" / "config.json",
        run_dir / "run_manifest.json",
        run_dir / "logs" / "env.json",
    ]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in core]
    resumed = _run_mini(root, run_dir.name, spec, resume=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in core]
    assert (run_dir / "logs" / "input_verification" / "resume_0001").is_dir()
    assert (run_dir / "logs" / "resume_env" / "event_0001.json").is_file()


def test_13_single_pool_projection_and_source_boundaries(
    mini_result: tuple[Path, Path],
) -> None:
    run_dir, _ = mini_result
    env = json.loads((run_dir / "logs" / "env.json").read_text(encoding="utf-8"))
    executed = set(env["executed_worker_pids"])
    audited = {item["pid"] for item in env["worker_threadpool_info"]}
    assert executed == audited
    source = (CODE / "run_e1.py").read_text(encoding="utf-8")
    assert "ProcessPoolExecutor" not in source
    with pytest.raises(IntegrityError, match="empty"):
        run_e1.validate_threadpool_projection(
            {"n_workers": 1, "distinct_worker_profiles": [{"count": 1, "libs": []}]}
        )


def test_14_resume_recomputes_inputs_receipt_and_config(tmp_path: Path) -> None:
    spec = _mini_spec(tmp_path / "mini.json")
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"verification_digest":"abc"}', encoding="utf-8")
    first = _run_mini(tmp_path, "resume-checks", spec, receipt=receipt)
    assert first.returncode == 0, first.stdout + first.stderr
    spec.write_text('{"n_dim":1,"n_systems":3,"r_cap":10.0}', encoding="utf-8")
    assert (
        _run_mini(
            tmp_path, "resume-checks", spec, resume=True, receipt=receipt
        ).returncode
        != 0
    )
    _mini_spec(spec)
    receipt.write_text('{"verification_digest":"abd"}', encoding="utf-8")
    assert (
        _run_mini(
            tmp_path, "resume-checks", spec, resume=True, receipt=receipt
        ).returncode
        != 0
    )
    receipt.write_text('{"verification_digest":"abc"}', encoding="utf-8")
    config = tmp_path / "resume-checks" / "logs" / "config.json"
    os.chmod(config, stat.S_IWRITE | stat.S_IREAD)
    config.write_text(
        config.read_text(encoding="utf-8").replace('"B": 2000', '"B": 1999'),
        encoding="utf-8",
    )
    assert (
        _run_mini(
            tmp_path, "resume-checks", spec, resume=True, receipt=receipt
        ).returncode
        != 0
    )


def test_15_runtime_worker_death_is_bounded() -> None:
    ctx, queue, pool = run_e1.create_pool(1)
    try:
        audited = run_e1.collect_handshake(queue, 1, deadline_seconds=30)
        with pytest.raises(IntegrityError, match="worker PID set changed"):
            run_e1.collect_results(
                pool, [{"kill_worker": True}], set(audited), poll_interval=0.1
            )
    finally:
        pool.terminate()
        pool.join()


@pytest.mark.parametrize(
    "messages,pattern",
    [
        ([("START", 1), ("START", 1)], "duplicate START"),
        (
            [
                ("START", 1),
                ("READY", 1, {"pools": [{"num_threads": 1}]}),
                ("READY", 1, {}),
            ],
            "duplicate READY",
        ),
        ([("READY", 1, {"pools": []})], "before START"),
        ([("START", 1), ("START", 2)], "more START"),
    ],
)
def test_16_handshake_state_machine(
    messages: list[tuple[Any, ...]], pattern: str
) -> None:
    with pytest.raises(IntegrityError, match=pattern):
        run_e1.collect_handshake_from_messages(messages, 1, complete=True)
    partial = run_e1.collect_handshake_from_messages([("START", 1)], 1, complete=False)
    assert partial == {1: "STARTED"}
    with pytest.raises(IntegrityError, match="STARTED=1"):
        run_e1.collect_handshake_from_messages([("START", 1)], 1, complete=True)


def test_17_private_pool_access_is_single_and_live() -> None:
    ctx, queue, pool = run_e1.create_pool(1)
    try:
        run_e1.collect_handshake(queue, 1, deadline_seconds=30)
        assert len(run_e1.live_pool_pids(pool)) == 1
    finally:
        pool.terminate()
        pool.join()
    source = (CODE / "run_e1.py").read_text(encoding="utf-8")
    assert source.count("." + "_pool") == 1


def test_18_real_directory_coordinator_scan(
    mini_result: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, _ = mini_result
    summary = run_e1.scan_coordinator(
        run_dir, run_e1.MINI_SEARCH_IDENTITY, 1, require_complete=True
    )
    assert summary == {
        "logical_eval_count": 300,
        "discarded_alloc": 0,
        "allocation": 300,
    }
    copied = tmp_path / "copy"
    run_e1.copy_coordinator_artifacts(run_dir, copied)
    checkpoints = sorted((copied / "logs" / "checkpoints").glob("*.pkl"))
    checkpoints[-1].unlink()
    run_e1.remove_restart_evidence(copied, 3)
    with pytest.raises(IntegrityError, match="missing restarts"):
        run_e1.scan_coordinator(
            copied, run_e1.MINI_SEARCH_IDENTITY, 1, require_complete=True
        )
    partial = run_e1.scan_coordinator(
        copied, run_e1.MINI_SEARCH_IDENTITY, 1, require_complete=False
    )
    assert partial["logical_eval_count"] == 200


def test_19_seed_registry_serialization_is_deterministic() -> None:
    first = run_e1.serialize_seed_registry()
    second = run_e1.serialize_seed_registry()
    assert _canonical(first) == _canonical(second)
    assert [(item["component"], item["seed_kind"]) for item in first] == sorted(
        (item["component"], item["seed_kind"]) for item in first
    )
    assert {item["component"] for item in first} == set(range(14))
    assert "legal_tuple" not in json.dumps(first)
    for item in first:
        for field in item["exact_shape"]:
            allowed = field["domain"]["allowed"]
            assert allowed is None or allowed == sorted(allowed)


def test_20_n_div_counts_systems_and_rebuilds_from_l1() -> None:
    train_rows = [
        {"m": "0", "eval_kind": "0", "st0": "1", "st1": "2", "st2": "3"},
        {"m": "1", "eval_kind": "0", "st0": "0", "st1": "1", "st2": "0"},
    ]
    assert run_e1.count_n_div(train_rows, 0) == 2
    val_rows = [
        {"m": "2", "eval_kind": "1", "st0": "1", "st1": "2", "st2": "3"},
        {"m": "3", "eval_kind": "1", "st0": "0", "st1": "1", "st2": "0"},
    ]
    assert run_e1.count_n_div(val_rows, 1) == 2
    final_rows = [
        {"m": "4", "eval_kind": "2", "st3": "1", "st4": "3"},
        {"m": "5", "eval_kind": "2", "st3": "0", "st4": "2"},
    ]
    assert run_e1.count_n_div(final_rows, 2) == 2
