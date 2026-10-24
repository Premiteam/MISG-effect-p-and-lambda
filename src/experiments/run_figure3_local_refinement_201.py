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


P_FIXED = 5
LOCAL_LAMBDAS = (0.90, 0.925, 0.95, 0.99, 0.995, 0.999, 1.0)
EXPECTED_TRAIN_DELAY_COUNTS = {2: 67, 3: 67, 4: 67}

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CORE_SCRIPT = SCRIPT_DIR / "run_p_lambda_interaction_AUTO.py"
DEFAULT_PROJECT = WORKSPACE_ROOT
DEFAULT_FAMILY = WORKSPACE_ROOT / "data" / "systems.npz"
DEFAULT_PROTOCOL_RESULTS = (
    WORKSPACE_ROOT / "data" / "balanced_full_factorial_results.json"
)
DEFAULT_OUTDIR = WORKSPACE_ROOT / "data" / "figure3_results"
DEFAULT_ALIGNMENT_AUDITOR = WORKSPACE_ROOT / "tools" / "audit_panel_alignment.py"

_WORKER_CONTEXT: dict[str, Any] | None = None


def lambda_key(value: float) -> str:
    return f"lambda={str(float(value))}"


def local_cells() -> list[float]:
    return list(LOCAL_LAMBDAS)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json_atomic(data: Any, path: Path) -> None:
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("misg_factorial_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import computational core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_balanced_training(
    systems: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected = list(systems)
    if len(selected) != 201:
        raise ValueError(
            "Figure 3 rerun requires exactly 201 training systems; "
            f"found {len(selected)}"
        )
    counts = Counter(int(item["d"]) for item in selected)
    if dict(counts) != EXPECTED_TRAIN_DELAY_COUNTS:
        raise ValueError(
            "Figure 3 rerun requires delay counts {2: 67, 3: 67, 4: 67}; "
            f"found {dict(counts)}"
        )
    return selected


def select_evaluation_split(
    systems: Sequence[Any], expected: int, name: str
) -> list[Any]:
    selected = list(systems[:expected])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {name} systems; found {len(selected)}")
    return selected


def protocol_from_balanced_results(data: Mapping[str, Any]) -> dict[str, Any]:
    meta = data.get("meta", {})
    required = (
        "N",
        "gamma_grid",
        "n_train",
        "train_delay_counts",
        "n_val",
        "n_test",
        "n_train_traj",
        "n_eval_traj",
        "top_k",
    )
    missing = [field for field in required if field not in meta]
    if missing:
        raise ValueError("protocol result is missing: " + ", ".join(missing))

    delay_counts = {
        int(key): int(value) for key, value in meta["train_delay_counts"].items()
    }
    if int(meta["n_train"]) != 201:
        raise ValueError("protocol result did not use 201 training systems")
    if delay_counts != EXPECTED_TRAIN_DELAY_COUNTS:
        raise ValueError(f"protocol result is not delay-balanced: {delay_counts}")
    if bool(meta.get("reuse_old_cells", False)):
        raise ValueError("protocol result permits reuse of old non-balanced cells")

    return {
        "schema_version": 1,
        "analysis_type": "figure3_local_lambda_refinement_201",
        "p_fixed": P_FIXED,
        "lambda_values": list(LOCAL_LAMBDAS),
        "N": int(meta["N"]),
        "gamma_grid": [float(value) for value in meta["gamma_grid"]],
        "n_train": 201,
        "train_delay_counts": {
            str(delay): count for delay, count in EXPECTED_TRAIN_DELAY_COUNTS.items()
        },
        "n_val": int(meta["n_val"]),
        "n_test": int(meta["n_test"]),
        "n_train_traj": int(meta["n_train_traj"]),
        "n_eval_traj": int(meta["n_eval_traj"]),
        "top_k": int(meta["top_k"]),
        "selection_split": "validation",
        "test_usage": "confirmation_only",
        "reuse_50_system_results": False,
    }


def validate_checkpoint_meta(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if dict(actual) != dict(expected):
        raise ValueError("checkpoint protocol mismatch; refusing to resume")


def load_or_initialize(path: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"meta": dict(meta), "cells": {}, "runtime": {}}
    saved = load_json(path)
    validate_checkpoint_meta(saved.get("meta", {}), meta)
    if not isinstance(saved.get("cells"), dict):
        raise ValueError("checkpoint cells must be a JSON object")
    return saved


def validate_result_record(record: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    if int(record.get("p", -1)) != P_FIXED:
        raise ValueError("seed record does not use p=5")
    if len(record.get("train_scan", [])) != len(meta["gamma_grid"]):
        raise ValueError("seed record does not contain the complete gamma scan")
    for split, expected in (("val", int(meta["n_val"])), ("test", int(meta["n_test"]))):
        summary = record.get(split, {})
        if int(summary.get("n_systems", -1)) != expected:
            raise ValueError(f"seed record has the wrong {split} system count")
        if len(summary.get("system_risks", [])) != expected:
            raise ValueError(f"seed record has the wrong {split} risk-vector length")


def seed_compatible_cells(
    seed_data: Mapping[str, Any], expected_meta: Mapping[str, Any]
) -> dict[str, Any]:
    seed_meta = seed_data.get("meta", {})
    checks = {
        "N": expected_meta["N"],
        "gamma_grid": expected_meta["gamma_grid"],
        "n_train": 201,
        "train_delay_counts": expected_meta["train_delay_counts"],
        "n_val": expected_meta["n_val"],
        "n_test": expected_meta["n_test"],
        "n_train_traj": expected_meta["n_train_traj"],
        "n_eval_traj": expected_meta["n_eval_traj"],
        "top_k": expected_meta["top_k"],
        "selection_split": "validation",
        "test_usage": "confirmation_only",
        "reuse_old_cells": False,
    }
    mismatched = [
        key for key, expected in checks.items() if seed_meta.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "seed results are not compatible with the 201-system protocol: "
            + ", ".join(mismatched)
        )

    seeded: dict[str, Any] = {}
    seed_cells = seed_data.get("cells", {})
    for value in LOCAL_LAMBDAS:
        source_key = f"p={P_FIXED}|lambda={str(float(value))}"
        if source_key not in seed_cells:
            continue
        record = dict(seed_cells[source_key])
        validate_result_record(record, expected_meta)
        record["local_refinement_provenance"] = "seeded_from_balanced_201_protocol"
        seeded[lambda_key(value)] = record
    return seeded


def remaining_lambdas(cells: Mapping[str, Any]) -> list[float]:
    return [value for value in LOCAL_LAMBDAS if lambda_key(value) not in cells]


def resolve_worker_count(requested: int, remaining: int) -> int:
    if requested < 1:
        raise ValueError("workers must be at least 1")
    if remaining == 0:
        return 0
    return min(requested, remaining, os.cpu_count() or 1)


def _worker(lambda_value: float) -> tuple[float, dict[str, Any], float]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context is not initialized")
    context = _WORKER_CONTEXT
    started = time.perf_counter()
    record = context["core"].retune_gamma(
        p=P_FIXED,
        lambda_value=lambda_value,
        gamma_grid=list(context["meta"]["gamma_grid"]),
        top_k=int(context["meta"]["top_k"]),
        train_pairs=context["train_pairs"],
        val_pairs=context["val_pairs"],
        test_pairs=context["test_pairs"],
        cfg=context["cfg"],
        misg_run=context["misg_run"],
        risk_R=context["risk_R"],
    )
    record["local_refinement_provenance"] = (
        "computed_with_201_balanced_training_systems"
    )
    return lambda_value, record, float(time.perf_counter() - started)


def write_source_data(results: Mapping[str, Any], path: Path) -> None:
    fields = (
        "lambda",
        "p",
        "gamma_star",
        "train_systems",
        "validation_systems",
        "test_systems",
        "validation_median_risk",
        "test_median_risk",
        "test_usage",
        "provenance",
    )
    rows = []
    for value in LOCAL_LAMBDAS:
        record = results["cells"][lambda_key(value)]
        rows.append(
            {
                "lambda": value,
                "p": P_FIXED,
                "gamma_star": record["gamma"],
                "train_systems": results["meta"]["n_train"],
                "validation_systems": record["val"]["n_systems"],
                "test_systems": record["test"]["n_systems"],
                "validation_median_risk": record["val"]["median_R"],
                "test_median_risk": record["test"]["median_R"],
                "test_usage": "confirmation_only",
                "provenance": record["local_refinement_provenance"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_panel(fig, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def make_figure_panels(results: Mapping[str, Any], outdir: Path) -> None:
    matplotlib_cache = outdir / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    cells = results["cells"]
    lambda_values = list(LOCAL_LAMBDAS)

    x = list(range(len(lambda_values)))
    val = [
        float(cells[lambda_key(value)]["val"]["median_R"]) for value in lambda_values
    ]
    test = [
        float(cells[lambda_key(value)]["test"]["median_R"]) for value in lambda_values
    ]
    gamma = [float(cells[lambda_key(value)]["gamma"]) for value in lambda_values]
    tick_labels = [f"{value:g}" for value in lambda_values]

    blue = "#2F6FA3"
    charcoal = "#4D4D4D"
    orange = "#D9822B"
    grid = "#D9D9D9"

    fig_a, ax_a = plt.subplots(figsize=(3.45, 2.55))
    ax_a.plot(
        x,
        val,
        color=blue,
        marker="o",
        markersize=3.2,
        linewidth=1.35,
        label="Validation median R",
    )
    ax_a.plot(
        x,
        test,
        color=charcoal,
        marker="s",
        markersize=2.8,
        linewidth=1.1,
        linestyle="--",
        label="Independent test median R",
    )
    ax_a.set_xlabel(r"Sampled forgetting factor $\lambda$")
    ax_a.set_ylabel("Median value of R")
    ax_a.set_title(r"Local error profile at $p=5$")
    ax_a.set_xticks(
        x,
        tick_labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax_a.grid(axis="y", color=grid, linewidth=0.5)
    ax_a.legend(loc="upper left")
    fig_a.tight_layout()
    save_panel(fig_a, outdir / "Figure3a_local_risk_201")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(3.45, 2.55))
    ax_b.plot(x, gamma, color=orange, marker="o", markersize=3.2, linewidth=1.35)
    ax_b.set_xlabel(r"Sampled forgetting factor $\lambda$")
    ax_b.set_ylabel(r"Validation-selected $\gamma$")
    ax_b.set_title(r"Validation-selected step-size multiplier at $p=5$")
    ax_b.set_xticks(
        x,
        tick_labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax_b.grid(axis="y", color=grid, linewidth=0.5)
    fig_b.tight_layout()
    save_panel(fig_b, outdir / "Figure3b_selected_gamma_201")
    plt.close(fig_b)

    combined, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.05))
    left.plot(
        x,
        val,
        color=blue,
        marker="o",
        markersize=3.2,
        linewidth=1.35,
        label="Validation median R",
    )
    left.plot(
        x,
        test,
        color=charcoal,
        marker="s",
        markersize=2.8,
        linewidth=1.1,
        linestyle="--",
        label="Independent test median R",
    )
    left.set_xlabel(r"Sampled forgetting factor $\lambda$")
    left.set_ylabel("Median value of R")
    left.set_title(r"Local error profile at $p=5$")
    left.set_xticks(
        x,
        tick_labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    left.grid(axis="y", color=grid, linewidth=0.5)
    left.legend(loc="upper left")

    right.plot(
        x,
        gamma,
        color=orange,
        marker="o",
        markersize=3.2,
        linewidth=1.35,
    )
    right.set_xlabel(r"Sampled forgetting factor $\lambda$")
    right.set_ylabel(r"Validation-selected $\gamma$")
    right.set_title(r"Validation-selected step-size multiplier at $p=5$")
    right.set_xticks(
        x,
        tick_labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    right.grid(axis="y", color=grid, linewidth=0.5)

    combined.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.30,
        top=0.88,
        wspace=0.34,
    )
    if DEFAULT_ALIGNMENT_AUDITOR.exists():
        auditor = load_module(DEFAULT_ALIGNMENT_AUDITOR)
        auditor.require_matplotlib_panel_alignment(
            combined,
            json_out=outdir / "Figure3_local_refinement_201_combined.alignment.json",
            overlay_svg=outdir / "Figure3_local_refinement_201_combined.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
            axes=[left, right],
            panel_ids=["a", "b"],
            row_groups=[["a", "b"]],
        )
    save_panel(combined, outdir / "Figure3_local_refinement_201_combined")
    plt.close(combined)


def self_test(protocol_path: Path) -> None:
    assert P_FIXED == 5
    assert len(LOCAL_LAMBDAS) == 7
    assert len(set(LOCAL_LAMBDAS)) == len(LOCAL_LAMBDAS)
    assert LOCAL_LAMBDAS == tuple(sorted(LOCAL_LAMBDAS))
    assert sum(EXPECTED_TRAIN_DELAY_COUNTS.values()) == 201
    protocol = load_json(protocol_path)
    meta = protocol_from_balanced_results(protocol)
    seeded = seed_compatible_cells(protocol, meta)
    assert set(seeded).issubset({lambda_key(value) for value in LOCAL_LAMBDAS})
    assert lambda_key(0.995) not in seeded
    assert lambda_key(0.999) not in seeded
    old_protocol = dict(protocol)
    old_protocol["meta"] = dict(protocol["meta"])
    old_protocol["meta"]["n_train"] = 50
    try:
        seed_compatible_cells(old_protocol, meta)
    except ValueError as error:
        assert "n_train" in str(error)
    else:
        raise AssertionError("a 50-system seed was incorrectly accepted")
    print(f"SELF-TEST PASSED: {len(seeded)} existing 201-system cells are reusable")


def run(args: argparse.Namespace) -> Path:
    global _WORKER_CONTEXT

    core_path = args.core.expanduser().resolve()
    project_path = args.project.expanduser().resolve()
    family_path = args.family.expanduser().resolve()
    protocol_path = args.protocol_results.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    for path, label in (
        (core_path, "factorial computational core"),
        (project_path, "state-estimation project"),
        (family_path, "system family"),
        (protocol_path, "201-system protocol results"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    protocol_data = load_json(protocol_path)
    expected_meta = protocol_from_balanced_results(protocol_data)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "figure3_local_refinement_201_checkpoint.json"
    final_path = outdir / "figure3_local_refinement_201_results.json"
    results = load_or_initialize(checkpoint_path, expected_meta)

    if not checkpoint_path.exists() and args.seed_results is not None:
        seed_path = args.seed_results.expanduser().resolve()
        if not seed_path.exists():
            raise FileNotFoundError(f"seed results not found: {seed_path}")
        seeded = seed_compatible_cells(load_json(seed_path), expected_meta)
        results["cells"].update(seeded)
        results["runtime"]["seeded_201_cells"] = sorted(seeded)
        save_json_atomic(results, checkpoint_path)

    todo = remaining_lambdas(results["cells"])
    print("=" * 72, flush=True)
    print("Figure 3 local lambda refinement with 201 training systems", flush=True)
    print("train/validation/test : 201/102/201", flush=True)
    print(f"training delays       : {EXPECTED_TRAIN_DELAY_COUNTS}", flush=True)
    print(f"local lambdas         : {list(LOCAL_LAMBDAS)}", flush=True)
    print(f"completed/resumed     : {len(results['cells'])}", flush=True)
    print(f"remaining             : {len(todo)}", flush=True)
    print(f"output                : {outdir}", flush=True)
    print("=" * 72, flush=True)

    if args.analysis_only and todo:
        raise RuntimeError(
            "analysis-only requested, but the checkpoint is missing: "
            + ", ".join(str(value) for value in todo)
        )

    if todo:
        core = load_module(core_path)
        project = core.find_project_root(project_path)
        Config, traj_seed, misg_run, risk_R, load_family, simulate = (
            core.import_project(project)
        )
        family = load_family(family_path)
        train_systems = select_balanced_training(family["train"])
        val_systems = select_evaluation_split(
            family["val"], int(expected_meta["n_val"]), "validation"
        )
        test_systems = select_evaluation_split(
            family["test"], int(expected_meta["n_test"]), "test"
        )
        cfg = replace(Config(), N=int(expected_meta["N"]))

        print("Building fixed shared trajectories...", flush=True)
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

        workers = resolve_worker_count(args.workers, len(todo))
        results["runtime"]["worker_count"] = workers
        _WORKER_CONTEXT = {
            "core": core,
            "meta": expected_meta,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "cfg": cfg,
            "misg_run": misg_run,
            "risk_R": risk_R,
        }
        started = time.perf_counter()

        def accept(index: int, outcome: tuple[float, dict[str, Any], float]) -> None:
            value, record, elapsed = outcome
            key = lambda_key(value)
            record["local_cell_elapsed_seconds"] = elapsed
            results["cells"][key] = record
            results["runtime"]["last_completed_lambda"] = value
            results["runtime"]["completed_cells"] = len(results["cells"])
            save_json_atomic(results, checkpoint_path)
            print(
                f"[{index}/{len(todo)}] lambda={value:g}: "
                f"gamma*={record['gamma']:.3g}, "
                f"val_R={record['val']['median_R']:.6g}, "
                f"test_R={record['test']['median_R']:.6g}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if workers == 1:
            for index, outcome in enumerate(map(_worker, todo), start=1):
                accept(index, outcome)
        else:
            context = multiprocessing.get_context("fork")
            with context.Pool(processes=workers) as pool:
                for index, outcome in enumerate(
                    pool.imap_unordered(_worker, todo, chunksize=1), start=1
                ):
                    accept(index, outcome)

        results["runtime"]["this_invocation_elapsed_seconds"] = (
            time.perf_counter() - started
        )
        _WORKER_CONTEXT = None

    missing = remaining_lambdas(results["cells"])
    if missing:
        raise RuntimeError("run ended with missing lambdas: " + str(missing))
    for record in results["cells"].values():
        validate_result_record(record, expected_meta)

    results["runtime"]["completed_cells"] = len(results["cells"])
    results["runtime"]["status"] = "complete"
    save_json_atomic(results, checkpoint_path)
    save_json_atomic(results, final_path)
    write_source_data(results, outdir / "figure3_local_refinement_201_source_data.csv")
    make_figure_panels(results, outdir)
    print(f"Complete results: {final_path}", flush=True)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 201-system local forgetting refinement"
    )
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE_SCRIPT)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument(
        "--protocol-results", type=Path, default=DEFAULT_PROTOCOL_RESULTS
    )
    parser.add_argument(
        "--seed-results",
        type=Path,
        default=None,
        help=(
            "Optional 201-system balanced result JSON. Compatible p=5 cells "
            "are reused; 50-system records are rejected."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.protocol_results.expanduser().resolve())
        return
    run(args)


if __name__ == "__main__":
    main()
