from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence


P_VALUES = (1, 2, 5, 8, 12)
LAMBDA_VALUES = (1.0, 0.99, 0.95, 0.925, 0.90, 0.85)
EXPECTED_TRAIN_DELAY_COUNTS = {2: 67, 3: 67, 4: 67}

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CORE_SCRIPT = SCRIPT_DIR / "run_p_lambda_interaction_AUTO.py"
DEFAULT_PROJECT = WORKSPACE_ROOT
DEFAULT_FAMILY = WORKSPACE_ROOT / "data" / "systems.npz"
DEFAULT_ORIGINAL_RESULTS = (
    WORKSPACE_ROOT / "data" / "balanced_full_factorial_results.json"
)
DEFAULT_OUTDIR = WORKSPACE_ROOT / "data" / "factorial_results"
DEFAULT_SEED_RESULTS = WORKSPACE_ROOT / "data" / "full_train_gamma_audit_results.json"

_WORKER_CONTEXT: dict[str, Any] | None = None


def audit_cells() -> list[tuple[int, float]]:
    return [(p, lam) for p in P_VALUES for lam in LAMBDA_VALUES]


def cell_key(p: int, lambda_value: float) -> str:
    return f"p={int(p)}|lambda={str(float(lambda_value))}"


def select_full_train(systems: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = list(systems)
    if len(selected) != 201:
        raise ValueError(
            f"full-train audit requires exactly 201 systems; found {len(selected)}"
        )
    delay_counts = Counter(int(item["d"]) for item in selected)
    if dict(delay_counts) != EXPECTED_TRAIN_DELAY_COUNTS:
        raise ValueError(
            "full-train audit requires balanced delay counts "
            f"{EXPECTED_TRAIN_DELAY_COUNTS}; found {dict(delay_counts)}"
        )
    return selected


def protocol_meta(
    *,
    N: int,
    gamma_grid: Sequence[float],
    n_val: int,
    n_test: int,
    n_train_traj: int,
    n_eval_traj: int,
    top_k: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "analysis_type": "balanced_full_factorial_gamma_tuning",
        "p_values": list(P_VALUES),
        "lambda_values": list(LAMBDA_VALUES),
        "n_train": 201,
        "train_delay_counts": {
            str(delay): count for delay, count in EXPECTED_TRAIN_DELAY_COUNTS.items()
        },
        "N": int(N),
        "gamma_grid": [float(value) for value in gamma_grid],
        "n_val": int(n_val),
        "n_test": int(n_test),
        "n_train_traj": int(n_train_traj),
        "n_eval_traj": int(n_eval_traj),
        "top_k": int(top_k),
        "selection_split": "validation",
        "test_usage": "confirmation_only",
        "reuse_old_cells": False,
    }


def seed_compatible_cells(
    seed_results: Mapping[str, Any], expected_meta: Mapping[str, Any]
) -> dict[str, Any]:
    seed_meta = seed_results.get("meta", {})
    protocol_fields = (
        "N",
        "gamma_grid",
        "n_train",
        "train_delay_counts",
        "n_val",
        "n_test",
        "n_train_traj",
        "n_eval_traj",
        "top_k",
        "selection_split",
        "test_usage",
        "reuse_old_cells",
    )
    mismatched = [
        field
        for field in protocol_fields
        if seed_meta.get(field) != expected_meta.get(field)
    ]
    if mismatched:
        raise ValueError("seed protocol mismatch: " + ", ".join(mismatched))

    cells = seed_results.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("seed protocol mismatch: cells must be an object")
    allowed = {cell_key(p, lam) for p, lam in audit_cells()}
    seeded: dict[str, Any] = {}
    for key, record in cells.items():
        if key not in allowed:
            raise ValueError(f"seed protocol mismatch: unexpected cell {key}")
        if not isinstance(record, dict):
            raise ValueError(f"seed protocol mismatch: invalid cell {key}")
        p = int(record.get("p", -1))
        lam = float(record.get("lambda", -1.0))
        if cell_key(p, lam) != key:
            raise ValueError(f"seed protocol mismatch: inconsistent cell {key}")
        for split, expected_n in (
            ("val", int(expected_meta["n_val"])),
            ("test", int(expected_meta["n_test"])),
        ):
            summary = record.get(split, {})
            if int(summary.get("n_systems", -1)) != expected_n:
                raise ValueError(f"seed protocol mismatch: {key} {split} system count")
            risks = summary.get("system_risks", [])
            if len(risks) != expected_n:
                raise ValueError(f"seed protocol mismatch: {key} {split} risk vector")
        seeded[key] = record
    return seeded


def validate_checkpoint_meta(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if dict(actual) != dict(expected):
        raise ValueError("checkpoint protocol mismatch; refusing to resume")


def save_json_atomic(data: Any, path: Path) -> None:
    def strict_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strict_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict_json(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            strict_json(data),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.flush()
    temporary.replace(path)


def load_or_initialize(
    checkpoint_path: Path, expected_meta: Mapping[str, Any]
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"meta": dict(expected_meta), "cells": {}}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    validate_checkpoint_meta(saved.get("meta", {}), expected_meta)
    if not isinstance(saved.get("cells"), dict):
        raise ValueError("checkpoint protocol mismatch; cells must be an object")
    return saved


def remaining_cells(completed: Mapping[str, Any]) -> list[tuple[int, float]]:
    return [(p, lam) for p, lam in audit_cells() if cell_key(p, lam) not in completed]


def resolve_worker_count(requested: int, *, remaining: int, cpu_count: int) -> int:
    if requested < 1:
        raise ValueError("workers must be at least 1")
    if remaining < 1:
        return 0
    return min(int(requested), int(remaining), max(1, int(cpu_count)))


def _retune_worker(cell: tuple[int, float]) -> tuple[int, float, dict[str, Any], float]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("audit worker context was not initialized")
    p, lambda_value = cell
    context = _WORKER_CONTEXT
    started = time.perf_counter()
    record = context["core"].retune_gamma(
        p=p,
        lambda_value=lambda_value,
        gamma_grid=list(context["gamma_grid"]),
        top_k=int(context["top_k"]),
        train_pairs=context["train_pairs"],
        val_pairs=context["val_pairs"],
        test_pairs=context["test_pairs"],
        cfg=context["cfg"],
        misg_run=context["misg_run"],
        risk_R=context["risk_R"],
    )
    return p, lambda_value, record, float(time.perf_counter() - started)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_core_module(path: Path):
    spec = importlib.util.spec_from_file_location("misg_factorial_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import computational core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol_from_original(original_results: Mapping[str, Any]) -> dict[str, Any]:
    meta = original_results.get("meta", {})
    required = (
        "N",
        "gamma_grid",
        "n_val",
        "n_test",
        "n_train_traj",
        "n_eval_traj",
        "top_k",
    )
    missing = [field for field in required if field not in meta]
    if missing:
        raise ValueError("original results lack protocol fields: " + ", ".join(missing))
    return protocol_meta(
        N=int(meta["N"]),
        gamma_grid=[float(value) for value in meta["gamma_grid"]],
        n_val=int(meta["n_val"]),
        n_test=int(meta["n_test"]),
        n_train_traj=int(meta["n_train_traj"]),
        n_eval_traj=int(meta["n_eval_traj"]),
        top_k=int(meta["top_k"]),
    )


def validate_eval_split(systems: Sequence[Any], expected: int, name: str) -> list[Any]:
    selected = list(systems[:expected])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {name} systems; found {len(selected)}")
    return selected


def write_comparison_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "p",
        "lambda",
        "original_gamma",
        "audit_gamma",
        "gamma_same",
        "boundary",
        "val_median_R",
        "test_median_R",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_balanced_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "p",
        "lambda",
        "gamma_star",
        "boundary",
        "val_median_R",
        "val_mean_R",
        "test_median_R",
        "test_mean_R",
        "test_divergences",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    meta = protocol_meta(
        N=12000,
        gamma_grid=[0.02, 0.05, 0.08],
        n_val=102,
        n_test=201,
        n_train_traj=1,
        n_eval_traj=2,
        top_k=3,
    )
    assert meta["n_train"] == 201
    assert len(audit_cells()) == 30
    assert remaining_cells({}) == audit_cells()
    validate_checkpoint_meta(meta, meta)
    print("SELF-TEST PASSED", flush=True)


def run_audit(args: argparse.Namespace) -> tuple[Path, Path]:
    global _WORKER_CONTEXT

    core_path = args.core.expanduser().resolve()
    project_path = args.project.expanduser().resolve()
    family_path = args.family.expanduser().resolve()
    original_path = args.original_results.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    for path, label in (
        (core_path, "computational core"),
        (family_path, "system family"),
        (original_path, "original results"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    original = load_json(original_path)
    expected_meta = protocol_from_original(original)
    expected_meta.update(
        {
            "project": str(project_path),
            "family": str(family_path),
            "original_results": str(original_path),
            "computational_core": str(core_path),
        }
    )

    core = load_core_module(core_path)
    project = core.find_project_root(project_path)
    (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        load_family,
        simulate,
    ) = core.import_project(project)

    family = load_family(family_path)
    train_systems = select_full_train(family["train"])
    val_systems = validate_eval_split(
        family["val"], int(expected_meta["n_val"]), "validation"
    )
    test_systems = validate_eval_split(
        family["test"], int(expected_meta["n_test"]), "test"
    )
    cfg = replace(Config(), N=int(expected_meta["N"]))

    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "balanced_full_factorial_checkpoint.json"
    final_path = outdir / "balanced_full_factorial_results.json"
    comparison_path = outdir / "gamma_comparison.csv"
    summary_path = outdir / "balanced_full_factorial_summary.csv"
    checkpoint_exists = checkpoint_path.exists()
    results = load_or_initialize(checkpoint_path, expected_meta)
    if not checkpoint_exists and args.seed_results is not None:
        seed_path = args.seed_results.expanduser().resolve()
        if not seed_path.exists():
            raise FileNotFoundError(f"seed results not found: {seed_path}")
        results["cells"].update(
            seed_compatible_cells(load_json(seed_path), expected_meta)
        )
        save_json_atomic(results, checkpoint_path)
    results.setdefault("runtime", {})
    completed_before = len(results["cells"])
    todo = remaining_cells(results["cells"])

    print("=" * 72, flush=True)
    print("Balanced full-training factorial gamma tuning", flush=True)
    print(
        f"train/validation/test : 201/{len(val_systems)}/{len(test_systems)}",
        flush=True,
    )
    print(f"fixed cells           : {len(audit_cells())}", flush=True)
    print(f"checkpoint resumed    : {completed_before}", flush=True)
    print(f"remaining             : {len(todo)}", flush=True)
    print(f"output                : {outdir}", flush=True)
    print("=" * 72, flush=True)

    if todo:
        print("Building the fixed shared trajectories...", flush=True)
        train_pairs = core.build_pairs(
            train_systems,
            j_start=core.TRAIN_TRAJ_START,
            n_traj=int(expected_meta["n_train_traj"]),
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )
        val_pairs = core.build_pairs(
            val_systems,
            j_start=core.EVAL_TRAJ_START,
            n_traj=int(expected_meta["n_eval_traj"]),
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )
        test_pairs = core.build_pairs(
            test_systems,
            j_start=core.EVAL_TRAJ_START,
            n_traj=int(expected_meta["n_eval_traj"]),
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        worker_count = resolve_worker_count(
            int(args.workers),
            remaining=len(todo),
            cpu_count=os.cpu_count() or 1,
        )
        results["runtime"]["worker_count"] = worker_count
        _WORKER_CONTEXT = {
            "core": core,
            "gamma_grid": list(expected_meta["gamma_grid"]),
            "top_k": int(expected_meta["top_k"]),
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "cfg": cfg,
            "misg_run": misg_run,
            "risk_R": risk_R,
        }
        print(f"Computing with {worker_count} independent cell workers...", flush=True)
        run_started = time.perf_counter()

        def accept(
            completed_index: int, outcome: tuple[int, float, dict[str, Any], float]
        ) -> None:
            p, lambda_value, record, elapsed = outcome
            key = cell_key(p, lambda_value)
            record["balanced_cell_elapsed_seconds"] = float(elapsed)
            results["cells"][key] = record
            results["runtime"]["last_completed_cell"] = key
            results["runtime"]["last_cell_elapsed_seconds"] = float(elapsed)
            results["runtime"]["completed_cells"] = len(results["cells"])
            save_json_atomic(results, checkpoint_path)
            print(
                f"[{completed_index}/{len(todo)}] {key} complete: "
                f"gamma*={record['gamma']:.3g}, "
                f"val_R={record['val']['median_R']:.6g}, "
                f"test_R={record['test']['median_R']:.6g}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if worker_count == 1:
            for completed_index, outcome in enumerate(
                map(_retune_worker, todo), start=1
            ):
                accept(completed_index, outcome)
        else:
            fork_context = multiprocessing.get_context("fork")
            with fork_context.Pool(processes=worker_count) as pool:
                outcomes = pool.imap_unordered(_retune_worker, todo, chunksize=1)
                for completed_index, outcome in enumerate(outcomes, start=1):
                    accept(completed_index, outcome)

        results["runtime"]["this_invocation_elapsed_seconds"] = float(
            time.perf_counter() - run_started
        )
        _WORKER_CONTEXT = None

    missing = remaining_cells(results["cells"])
    if missing:
        raise RuntimeError(f"audit ended with missing cells: {missing}")
    rows = gamma_comparison_rows(results, original)
    results["gamma_comparison"] = rows
    results["runtime"]["status"] = "complete"
    results["runtime"]["completed_cells"] = len(results["cells"])
    save_json_atomic(results, checkpoint_path)
    save_json_atomic(results, final_path)
    write_comparison_csv(rows, comparison_path)
    write_balanced_summary_csv(balanced_summary_rows(results), summary_path)
    print(f"Balanced factorial tuning complete: {final_path}", flush=True)
    return final_path, comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run balanced full-training multiplier selection"
    )
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE_SCRIPT)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument(
        "--original-results", type=Path, default=DEFAULT_ORIGINAL_RESULTS
    )
    parser.add_argument(
        "--seed-results",
        type=Path,
        default=DEFAULT_SEED_RESULTS,
        help="Protocol-compatible balanced results used to seed completed cells.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Independent audit cells to compute concurrently (default: 3).",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def gamma_comparison_rows(
    audit_results: Mapping[str, Any], original_results: Mapping[str, Any]
) -> list[dict[str, Any]]:
    audit_map = audit_results["cells"]
    original_map = original_results["cells"]
    rows: list[dict[str, Any]] = []
    for p, lam in audit_cells():
        key = cell_key(p, lam)
        if key not in audit_map or key not in original_map:
            raise KeyError(f"missing required audit cell: {key}")
        audited = audit_map[key]
        original = original_map[key]
        audited_gamma = float(audited["gamma"])
        original_gamma = float(original["gamma"])
        rows.append(
            {
                "p": p,
                "lambda": lam,
                "original_gamma": original_gamma,
                "audit_gamma": audited_gamma,
                "gamma_same": audited_gamma == original_gamma,
                "boundary": bool(audited.get("boundary", False)),
                "val_median_R": float(audited["val"]["median_R"]),
                "test_median_R": float(audited["test"]["median_R"]),
            }
        )
    return rows


def balanced_summary_rows(
    balanced_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cells = balanced_results["cells"]
    rows: list[dict[str, Any]] = []
    for p, lam in audit_cells():
        key = cell_key(p, lam)
        if key not in cells:
            raise KeyError(f"missing required balanced cell: {key}")
        cell = cells[key]
        rows.append(
            {
                "p": p,
                "lambda": lam,
                "gamma_star": float(cell["gamma"]),
                "boundary": bool(cell.get("boundary", False)),
                "val_median_R": float(cell["val"]["median_R"]),
                "val_mean_R": float(cell["val"]["mean_R"]),
                "test_median_R": float(cell["test"]["median_R"]),
                "test_mean_R": float(cell["test"]["mean_R"]),
                "test_divergences": int(cell["test"]["divergences"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_audit(args)


if __name__ == "__main__":
    main()
