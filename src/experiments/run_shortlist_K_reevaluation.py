from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS = WORKSPACE_ROOT / "data" / "balanced_full_factorial_results.json"
DEFAULT_CORE = SCRIPT_DIR / "run_p_lambda_interaction_AUTO.py"
DEFAULT_PROJECT = WORKSPACE_ROOT
DEFAULT_FAMILY = WORKSPACE_ROOT / "data" / "systems.npz"
DEFAULT_OUTDIR = WORKSPACE_ROOT / "data" / "shortlist_results"
DEFAULT_K_VALUES = (1, 2, 3, 5)

_WORKER_CONTEXT: dict[str, Any] | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            strict_json(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.flush()
    temporary.replace(path)


def load_core_module(path: Path):
    spec = importlib.util.spec_from_file_location("misg_factorial_core_for_K", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import computational core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gamma_key(value: float) -> str:
    return format(float(value), ".12g")


def candidate_key(cell: str, gamma: float) -> str:
    return f"{cell}|gamma={gamma_key(gamma)}"


def ranked_training_candidates(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    scan = cell.get("train_scan")
    if not isinstance(scan, list) or not scan:
        raise ValueError("cell has no training gamma scan")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(scan):
        gamma = float(record["gamma"])
        mean_risk = float(record["mean_R"])
        if not math.isfinite(gamma) or not math.isfinite(mean_risk):
            raise ValueError("training scan contains a nonfinite gamma or mean risk")
        records.append(
            {
                "gamma": gamma,
                "train_mean_R": mean_risk,
                "scan_index": index,
            }
        )
    records.sort(key=lambda row: (row["train_mean_R"], row["scan_index"]))
    for rank, record in enumerate(records, start=1):
        record["train_rank"] = rank
    return records


def existing_validation_map(cell: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = cell.get("val_candidates")
    if not isinstance(candidates, list):
        raise ValueError("cell has no validation candidates")
    result: dict[str, dict[str, Any]] = {}
    for record in candidates:
        gamma = float(record["gamma"])
        validation = record.get("val")
        if not isinstance(validation, Mapping):
            raise ValueError("validation candidate lacks a validation summary")
        median = float(validation["median_R"])
        if not math.isfinite(median):
            raise ValueError("validation candidate has a nonfinite median risk")
        result[gamma_key(gamma)] = dict(validation)
    return result


def validate_source(payload: Mapping[str, Any], max_k: int) -> None:
    meta = payload.get("meta")
    cells = payload.get("cells")
    if not isinstance(meta, Mapping) or not isinstance(cells, Mapping):
        raise ValueError("balanced results must contain meta and cells objects")
    if meta.get("selection_split") != "validation":
        raise ValueError("source results do not declare validation-based selection")
    if int(meta.get("n_train", -1)) != 201:
        raise ValueError("K reevaluation requires the balanced 201-system training run")
    expected_cells = len(meta.get("p_values", [])) * len(meta.get("lambda_values", []))
    if expected_cells <= 0 or len(cells) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} factorial cells from metadata; found {len(cells)}"
        )
    for cell_name, cell in cells.items():
        ranked = ranked_training_candidates(cell)
        if len(ranked) < max_k:
            raise ValueError(
                f"{cell_name} has fewer than K={max_k} training candidates"
            )
        available = existing_validation_map(cell)
        expected_existing = ranked[: int(meta.get("top_k", 0))]
        for candidate in expected_existing:
            if gamma_key(candidate["gamma"]) not in available:
                raise ValueError(
                    f"{cell_name} is missing stored validation rank "
                    f"{candidate['train_rank']}"
                )


def checkpoint_meta(
    source_path: Path,
    core_path: Path,
    project_path: Path,
    family_path: Path,
    source_meta: Mapping[str, Any],
    max_k: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_type": "gamma_shortlist_K_validation_sensitivity",
        "source_results": str(source_path),
        "computational_core": str(core_path),
        "project": str(project_path),
        "family": str(family_path),
        "max_k": int(max_k),
        "N": int(source_meta["N"]),
        "n_train": int(source_meta["n_train"]),
        "n_val": int(source_meta["n_val"]),
        "n_eval_traj": int(source_meta["n_eval_traj"]),
        "selection_split": "validation",
        "test_usage": "not accessed",
    }


def load_checkpoint(path: Path, expected_meta: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"meta": dict(expected_meta), "evaluations": {}}
    saved = load_json(path)
    if saved.get("meta") != dict(expected_meta):
        raise ValueError("checkpoint protocol mismatch; use a new output directory")
    if not isinstance(saved.get("evaluations"), dict):
        raise ValueError("checkpoint evaluations must be an object")
    return saved


def missing_tasks(
    cells: Mapping[str, Any], checkpoint: Mapping[str, Any], max_k: int
) -> list[dict[str, Any]]:
    completed = checkpoint.get("evaluations", {})
    tasks: list[dict[str, Any]] = []
    for cell_name, cell in cells.items():
        existing = existing_validation_map(cell)
        for candidate in ranked_training_candidates(cell)[:max_k]:
            gamma = float(candidate["gamma"])
            key = candidate_key(cell_name, gamma)
            if gamma_key(gamma) in existing or key in completed:
                continue
            tasks.append(
                {
                    "cell": cell_name,
                    "p": int(cell["p"]),
                    "lambda": float(cell["lambda"]),
                    "gamma": gamma,
                    "train_rank": int(candidate["train_rank"]),
                    "train_mean_R": float(candidate["train_mean_R"]),
                }
            )
    return tasks


def _validation_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context is not initialized")
    context = _WORKER_CONTEXT
    started = time.perf_counter()
    validation = context["core"].evaluate(
        context["val_pairs"],
        p=int(task["p"]),
        gamma=float(task["gamma"]),
        lambda_value=float(task["lambda"]),
        cfg=context["cfg"],
        misg_run=context["misg_run"],
        risk_R=context["risk_R"],
    )
    return {
        **dict(task),
        "validation": validation,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def validation_for_candidate(
    cell_name: str,
    cell: Mapping[str, Any],
    gamma: float,
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    existing = existing_validation_map(cell)
    key = gamma_key(gamma)
    if key in existing:
        return existing[key]
    computed = checkpoint.get("evaluations", {}).get(candidate_key(cell_name, gamma))
    if not isinstance(computed, Mapping) or not isinstance(
        computed.get("validation"), Mapping
    ):
        raise RuntimeError(
            f"validation result is unavailable for {cell_name}, gamma={gamma}"
        )
    return computed["validation"]


def select_for_k(
    cell_name: str,
    cell: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    k: int,
) -> dict[str, Any]:
    candidates = ranked_training_candidates(cell)[:k]
    evaluated = []
    for candidate in candidates:
        validation = validation_for_candidate(
            cell_name, cell, float(candidate["gamma"]), checkpoint
        )
        evaluated.append(
            {
                **candidate,
                "val_median_R": float(validation["median_R"]),
                "val_mean_R": float(validation["mean_R"]),
                "val_divergences": int(validation["divergences"]),
            }
        )
    best = min(evaluated, key=lambda row: row["val_median_R"])
    return {
        "cell": cell_name,
        "p": int(cell["p"]),
        "lambda": float(cell["lambda"]),
        "K": int(k),
        "selected_gamma": float(best["gamma"]),
        "selected_train_rank": int(best["train_rank"]),
        "selected_train_mean_R": float(best["train_mean_R"]),
        "selected_val_median_R": float(best["val_median_R"]),
        "selected_val_mean_R": float(best["val_mean_R"]),
        "selected_val_divergences": int(best["val_divergences"]),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]
) -> list[dict[str, Any]]:
    by_cell_k = {(row["cell"], int(row["K"])): row for row in rows}
    cell_names = sorted({str(row["cell"]) for row in rows})
    reference_k = max(k_values)
    standard_k = 3 if 3 in k_values else reference_k
    summaries: list[dict[str, Any]] = []
    for k in k_values:
        current = [by_cell_k[(cell, int(k))] for cell in cell_names]
        reference = [by_cell_k[(cell, int(reference_k))] for cell in cell_names]
        standard = [by_cell_k[(cell, int(standard_k))] for cell in cell_names]
        relative_excess = [
            float(row["selected_val_median_R"]) / float(ref["selected_val_median_R"])
            - 1.0
            for row, ref in zip(current, reference)
        ]
        summaries.append(
            {
                "K": int(k),
                "n_cells": len(current),
                "n_gamma_changed_vs_K3": sum(
                    not math.isclose(
                        float(row["selected_gamma"]),
                        float(std["selected_gamma"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for row, std in zip(current, standard)
                ),
                "n_gamma_changed_vs_max_K": sum(
                    not math.isclose(
                        float(row["selected_gamma"]),
                        float(ref["selected_gamma"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for row, ref in zip(current, reference)
                ),
                "median_selected_val_R_across_cells": statistics.median(
                    float(row["selected_val_median_R"]) for row in current
                ),
                "median_relative_excess_val_R_vs_max_K": statistics.median(
                    relative_excess
                ),
                "maximum_relative_excess_val_R_vs_max_K": max(relative_excess),
            }
        )
    return summaries


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "cell",
        "p",
        "lambda",
        "K",
        "selected_gamma",
        "selected_train_rank",
        "selected_train_mean_R",
        "selected_val_median_R",
        "selected_val_mean_R",
        "selected_val_divergences",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def write_markdown(
    summaries: Sequence[Mapping[str, Any]],
    n_new_evaluations: int,
    path: Path,
) -> None:
    lines = [
        "# Gamma-shortlist size sensitivity",
        "",
        (
            "Gamma candidates were ranked by training mean system risk. Within each "
            "shortlist, gamma was selected by validation median system risk. Held-out "
            "test results were not accessed."
        ),
        "",
        f"New validation candidate evaluations performed: **{n_new_evaluations}**.",
        "",
        "| K | Cells changed vs K=3 | Cells changed vs largest K | Median validation risk | Median excess vs largest K | Maximum excess vs largest K |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['K']} | {row['n_gamma_changed_vs_K3']} / {row['n_cells']} | "
            f"{row['n_gamma_changed_vs_max_K']} / {row['n_cells']} | "
            f"{row['median_selected_val_R_across_cells']:.6g} | "
            f"{100.0 * row['median_relative_excess_val_R_vs_max_K']:.3f}% | "
            f"{100.0 * row['maximum_relative_excess_val_R_vs_max_K']:.3f}% |"
        )
    lines.extend(
        [
            "",
            (
                "These results assess the validation-stage shortlist rule. They do "
                "not authorize choosing K from held-out test performance. If a final K "
                "is fixed after this analysis, any further test evaluation should be "
                "reported as confirmation only."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    global _WORKER_CONTEXT

    source_path = args.results.expanduser().resolve()
    core_path = args.core.expanduser().resolve()
    project_path = args.project.expanduser().resolve()
    family_path = args.family.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    for path, label in (
        (source_path, "balanced factorial results"),
        (core_path, "computational core"),
        (family_path, "system family"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    k_values = sorted({int(value) for value in args.k_values})
    if not k_values or min(k_values) < 1:
        raise ValueError("K values must be positive integers")
    max_k = max(k_values)

    source = load_json(source_path)
    validate_source(source, max_k)
    source_meta = source["meta"]
    cells = source["cells"]
    expected_checkpoint_meta = checkpoint_meta(
        source_path,
        core_path,
        project_path,
        family_path,
        source_meta,
        max_k,
    )
    checkpoint_path = outdir / "shortlist_K_validation_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, expected_checkpoint_meta)
    todo = missing_tasks(cells, checkpoint, max_k)

    print("=" * 72)
    print("Gamma-shortlist K reevaluation")
    print(f"K values              : {k_values}")
    print(f"factorial cells       : {len(cells)}")
    print(f"stored K              : {source_meta['top_k']}")
    print(f"new validation runs   : {len(todo)}")
    print("held-out test usage   : none")
    print(f"output                : {outdir}")
    print("=" * 72)

    if args.dry_run:
        ranks = {}
        for task in todo:
            ranks[int(task["train_rank"])] = ranks.get(int(task["train_rank"]), 0) + 1
        print(f"Missing evaluations by training rank: {dict(sorted(ranks.items()))}")
        print("DRY RUN COMPLETE; no trajectories were evaluated")
        return checkpoint_path, checkpoint_path, checkpoint_path

    if todo and args.reuse_only:
        raise RuntimeError(
            f"{len(todo)} validation evaluations are missing; remove --reuse-only "
            "or omit K values larger than the stored shortlist"
        )

    new_evaluations = 0
    if todo:
        core = load_core_module(core_path)
        project = core.find_project_root(project_path)
        Config, traj_seed, misg_run, risk_R, load_family, simulate = (
            core.import_project(project)
        )
        family = load_family(family_path)
        val_systems = list(family["val"][: int(source_meta["n_val"])])
        if len(val_systems) != int(source_meta["n_val"]):
            raise ValueError(
                f"expected {source_meta['n_val']} validation systems; "
                f"found {len(val_systems)}"
            )
        cfg = replace(Config(), N=int(source_meta["N"]))
        print("Building the fixed validation trajectories...", flush=True)
        val_pairs = core.build_pairs(
            val_systems,
            j_start=core.EVAL_TRAJ_START,
            n_traj=int(source_meta["n_eval_traj"]),
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )
        worker_count = min(
            max(1, int(args.workers)), len(todo), max(1, os.cpu_count() or 1)
        )
        _WORKER_CONTEXT = {
            "core": core,
            "val_pairs": val_pairs,
            "cfg": cfg,
            "misg_run": misg_run,
            "risk_R": risk_R,
        }

        def accept(index: int, result: Mapping[str, Any]) -> None:
            nonlocal new_evaluations
            key = candidate_key(str(result["cell"]), float(result["gamma"]))
            checkpoint["evaluations"][key] = dict(result)
            atomic_json_dump(checkpoint, checkpoint_path)
            new_evaluations += 1
            print(
                f"[{index}/{len(todo)}] {result['cell']}, rank "
                f"{result['train_rank']}, gamma={result['gamma']:.3g}: "
                f"validation median R={result['validation']['median_R']:.6g} "
                f"({result['elapsed_seconds']:.1f}s)",
                flush=True,
            )

        if worker_count == 1:
            for index, result in enumerate(map(_validation_worker, todo), start=1):
                accept(index, result)
        else:
            context = multiprocessing.get_context("fork")
            with context.Pool(processes=worker_count) as pool:
                results = pool.imap_unordered(_validation_worker, todo, chunksize=1)
                for index, result in enumerate(results, start=1):
                    accept(index, result)
        _WORKER_CONTEXT = None

    still_missing = missing_tasks(cells, checkpoint, max_k)
    if still_missing:
        raise RuntimeError(
            f"analysis ended with {len(still_missing)} missing evaluations"
        )

    rows = [
        select_for_k(cell_name, cell, checkpoint, k)
        for cell_name, cell in cells.items()
        for k in k_values
    ]
    rows.sort(key=lambda row: (int(row["p"]), -float(row["lambda"]), int(row["K"])))
    summaries = summarize(rows, k_values)
    csv_path = outdir / "shortlist_K_cell_results.csv"
    json_path = outdir / "shortlist_K_reevaluation_results.json"
    markdown_path = outdir / "shortlist_K_reevaluation_summary.md"
    write_csv(rows, csv_path)
    atomic_json_dump(
        {
            "meta": {
                **expected_checkpoint_meta,
                "K_values": k_values,
                "selection_metric": "validation median system risk",
                "candidate_ranking_metric": "training mean system risk",
                "new_validation_evaluations_this_invocation": new_evaluations,
                "total_new_validation_evaluations": len(checkpoint["evaluations"]),
            },
            "summary_by_K": summaries,
            "cell_results": rows,
        },
        json_path,
    )
    write_markdown(summaries, len(checkpoint["evaluations"]), markdown_path)
    print("K reevaluation complete")
    print(f"  CSV      : {csv_path}")
    print(f"  JSON     : {json_path}")
    print(f"  Markdown : {markdown_path}")
    return csv_path, json_path, markdown_path


def self_test() -> None:
    cell = {
        "p": 1,
        "lambda": 0.95,
        "train_scan": [
            {"gamma": 0.1, "mean_R": 0.3},
            {"gamma": 0.2, "mean_R": 0.1},
            {"gamma": 0.3, "mean_R": 0.2},
        ],
        "val_candidates": [
            {
                "gamma": 0.2,
                "val": {"median_R": 0.20, "mean_R": 0.21, "divergences": 0},
            },
            {
                "gamma": 0.3,
                "val": {"median_R": 0.10, "mean_R": 0.11, "divergences": 0},
            },
        ],
    }
    ranked = ranked_training_candidates(cell)
    assert [row["gamma"] for row in ranked] == [0.2, 0.3, 0.1]
    checkpoint = {"evaluations": {}}
    selected1 = select_for_k("test", cell, checkpoint, 1)
    selected2 = select_for_k("test", cell, checkpoint, 2)
    assert selected1["selected_gamma"] == 0.2
    assert selected2["selected_gamma"] == 0.3
    print("SELF-TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reevaluate multiplier shortlist sizes"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Shortlist sizes to compare (default: 1 2 3 5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Independent validation candidates evaluated concurrently (default: 3).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
