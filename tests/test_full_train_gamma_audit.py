from __future__ import annotations

from collections import Counter
import json

import pytest

import run_full_train_gamma_audit as audit


def _balanced_train() -> list[dict[str, int]]:
    systems = []
    m = 0
    for delay in (2, 3, 4):
        for _ in range(67):
            systems.append({"m": m, "d": delay})
            m += 1
    return systems


def test_audit_cells_cover_the_complete_balanced_factorial_grid() -> None:
    assert audit.audit_cells() == [
        (1, 1.0),
        (1, 0.99),
        (1, 0.95),
        (1, 0.925),
        (1, 0.9),
        (1, 0.85),
        (2, 1.0),
        (2, 0.99),
        (2, 0.95),
        (2, 0.925),
        (2, 0.9),
        (2, 0.85),
        (5, 1.0),
        (5, 0.99),
        (5, 0.95),
        (5, 0.925),
        (5, 0.9),
        (5, 0.85),
        (8, 1.0),
        (8, 0.99),
        (8, 0.95),
        (8, 0.925),
        (8, 0.9),
        (8, 0.85),
        (12, 1.0),
        (12, 0.99),
        (12, 0.95),
        (12, 0.925),
        (12, 0.9),
        (12, 0.85),
    ]


def test_select_full_train_requires_all_201_balanced_systems() -> None:
    selected = audit.select_full_train(_balanced_train())

    assert len(selected) == 201
    assert Counter(int(item["d"]) for item in selected) == {
        2: 67,
        3: 67,
        4: 67,
    }

    with pytest.raises(ValueError, match="201"):
        audit.select_full_train(_balanced_train()[:-1])


def test_checkpoint_meta_rejects_old_fifty_system_protocol() -> None:
    expected = audit.protocol_meta(
        N=12000,
        gamma_grid=[0.02, 0.05, 0.08],
        n_val=102,
        n_test=201,
        n_train_traj=1,
        n_eval_traj=2,
        top_k=3,
    )

    audit.validate_checkpoint_meta(expected, expected)
    assert expected["analysis_type"] == "balanced_full_factorial_gamma_tuning"
    assert expected["lambda_values"] == [1.0, 0.99, 0.95, 0.925, 0.9, 0.85]

    stale = dict(expected)
    stale["n_train"] = 50
    with pytest.raises(ValueError, match="checkpoint protocol mismatch"):
        audit.validate_checkpoint_meta(stale, expected)


def test_gamma_summary_compares_every_audit_cell_to_original() -> None:
    audit_results = {
        "cells": {
            audit.cell_key(p, lam): {
                "p": p,
                "lambda": lam,
                "gamma": float(p) / 10.0 + lam,
                "boundary": False,
                "val": {"median_R": 0.2},
                "test": {"median_R": 0.3},
            }
            for p, lam in audit.audit_cells()
        }
    }
    original_results = {
        "cells": {
            audit.cell_key(p, lam): {
                "gamma": float(p) / 10.0 + lam,
            }
            for p, lam in audit.audit_cells()
        }
    }

    rows = audit.gamma_comparison_rows(audit_results, original_results)

    assert len(rows) == 30
    assert all(row["gamma_same"] for row in rows)
    assert {(row["p"], row["lambda"]) for row in rows} == set(audit.audit_cells())


def test_checkpoint_round_trip_preserves_completed_cells(tmp_path) -> None:
    expected = audit.protocol_meta(
        N=12000,
        gamma_grid=[0.02, 0.05, 0.08],
        n_val=102,
        n_test=201,
        n_train_traj=1,
        n_eval_traj=2,
        top_k=3,
    )
    checkpoint = tmp_path / "checkpoint.json"
    data = {
        "meta": expected,
        "cells": {audit.cell_key(1, 1.0): {"gamma": 0.2}},
    }

    audit.save_json_atomic(data, checkpoint)
    resumed = audit.load_or_initialize(checkpoint, expected)

    assert resumed == data
    assert not (tmp_path / "checkpoint.json.tmp").exists()


def test_checkpoint_with_old_train_size_is_never_reused(tmp_path) -> None:
    expected = audit.protocol_meta(
        N=12000,
        gamma_grid=[0.02, 0.05, 0.08],
        n_val=102,
        n_test=201,
        n_train_traj=1,
        n_eval_traj=2,
        top_k=3,
    )
    stale_meta = dict(expected)
    stale_meta["n_train"] = 50
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"meta": stale_meta, "cells": {"stale": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint protocol mismatch"):
        audit.load_or_initialize(checkpoint, expected)


def test_remaining_cells_covers_every_unfinished_balanced_factorial_cell() -> None:
    completed = {audit.cell_key(1, 1.0): {"gamma": 0.2}}

    assert audit.remaining_cells(completed) == audit.audit_cells()[1:]


def test_default_output_is_separate_from_the_retrospective_ten_cell_archive() -> None:
    assert audit.DEFAULT_OUTDIR.name == "balanced_full_factorial_results"


def test_seed_cells_accepts_only_protocol_compatible_balanced_results() -> None:
    expected = audit.protocol_meta(
        N=12000,
        gamma_grid=[0.02, 0.05, 0.08],
        n_val=102,
        n_test=201,
        n_train_traj=1,
        n_eval_traj=2,
        top_k=3,
    )
    seed_meta = dict(expected)
    seed_meta["lambda_values"] = [1.0, 0.95]
    seed_meta["analysis_type"] = "retrospective_full_train_gamma_tuning_robustness"
    key = audit.cell_key(1, 1.0)
    seed = {
        "meta": seed_meta,
        "cells": {
            key: {
                "p": 1,
                "lambda": 1.0,
                "gamma": 2.5,
                "val": {"n_systems": 102, "system_risks": [0.2] * 102},
                "test": {"n_systems": 201, "system_risks": [0.3] * 201},
            }
        },
    }

    assert audit.seed_compatible_cells(seed, expected) == seed["cells"]

    incompatible = {"meta": dict(seed_meta, N=3000), "cells": seed["cells"]}
    with pytest.raises(ValueError, match="seed protocol mismatch"):
        audit.seed_compatible_cells(incompatible, expected)


def test_balanced_summary_contains_one_row_per_factorial_cell() -> None:
    results = {
        "cells": {
            audit.cell_key(p, lam): {
                "p": p,
                "lambda": lam,
                "gamma": 1.0,
                "boundary": False,
                "val": {"median_R": 0.01, "mean_R": 0.02},
                "test": {
                    "median_R": 0.03,
                    "mean_R": 0.04,
                    "divergences": 0,
                },
            }
            for p, lam in audit.audit_cells()
        }
    }

    rows = audit.balanced_summary_rows(results)

    assert len(rows) == 30
    assert rows[0] == {
        "p": 1,
        "lambda": 1.0,
        "gamma_star": 1.0,
        "boundary": False,
        "val_median_R": 0.01,
        "val_mean_R": 0.02,
        "test_median_R": 0.03,
        "test_mean_R": 0.04,
        "test_divergences": 0,
    }


def test_worker_count_is_bounded_by_work_and_machine() -> None:
    assert audit.resolve_worker_count(3, remaining=10, cpu_count=8) == 3
    assert audit.resolve_worker_count(20, remaining=2, cpu_count=8) == 2
    assert audit.resolve_worker_count(8, remaining=10, cpu_count=4) == 4

    with pytest.raises(ValueError, match="workers"):
        audit.resolve_worker_count(0, remaining=10, cpu_count=8)


def test_atomic_json_converts_nonfinite_diagnostics_to_null(tmp_path) -> None:
    path = tmp_path / "strict.json"

    audit.save_json_atomic({"finite": 1.0, "missing": float("nan")}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "missing": None,
    }
