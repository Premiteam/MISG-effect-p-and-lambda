from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "src" / "experiments"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
SRC = CODE.parent / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_e0_checks as checks
from config import Config
from systems import load_family


def _small_config() -> Config:
    return replace(
        Config(),
        n_train=6,
        n_val=6,
        n_test=6,
        n_pilot=6,
        N=200,
    )


def test_complete_small_run_and_data_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_misg_run = checks.misg_run
    called_splits: list[str] = []

    def recording_misg_run(system: dict, *args, **kwargs):
        called_splits.append(str(system["split"]))
        return original_misg_run(system, *args, **kwargs)

    monkeypatch.setattr(checks, "misg_run", recording_misg_run)
    outdir = tmp_path / "small_e0"
    result = checks.run_e0_checks(
        _small_config(),
        outdir,
        q_n_per_d=1,
        timing_reps=2,
        timing_warmup=1,
        timing_N=50,
        timing_n_per_d=1,
    )

    required = (
        "systems.npz",
        "manifest.csv",
        "split_by_d.csv",
        "gram_diagnostics.csv",
        "figures/family_hist.png",
        "q_sensitivity.csv",
        "logs/divergence_log.csv",
        "timing.csv",
        "logs/env.json",
        "logs/seeds.json",
        "summary.md",
    )
    assert all((outdir / relative).is_file() for relative in required)

    with (outdir / "split_by_d.csv").open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    assert len(split_rows) == 12
    assert all(int(row["count"]) == 2 for row in split_rows)

    with (outdir / "logs" / "divergence_log.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        divergence_rows = list(csv.DictReader(handle))
    expected_reasons = ["nan", "nan", "norm", "spec", ""]
    expected_at = [1, 1, 1, 50, None]
    expected_status = ["diverged", "diverged", "diverged", "diverged", "ok"]
    assert len(divergence_rows) == 5
    for index, row in enumerate(divergence_rows):
        assert row["expected_reason"] == expected_reasons[index]
        assert row["div_reason"] == expected_reasons[index]
        parsed_expected = int(row["expected_at"]) if row["expected_at"] else None
        parsed_actual = int(row["diverged_at"]) if row["diverged_at"] else None
        assert parsed_expected == expected_at[index]
        assert parsed_actual == expected_at[index]
        assert row["status"] == expected_status[index]
        assert row["match"] == "True"

    loaded = load_family(outdir / "systems.npz")
    for split, systems in result["family"].items():
        assert len(loaded[split]) == len(systems)
        for expected, actual in zip(systems, loaded[split]):
            for key in ("A", "B", "f", "rho_star"):
                assert np.array_equal(expected[key], actual[key])

    assert "test" not in called_splits


def test_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    outdir = tmp_path / "already_exists"
    outdir.mkdir()
    with pytest.raises(FileExistsError):
        checks.run_e0_checks(
            _small_config(),
            outdir,
            q_n_per_d=1,
            timing_reps=2,
            timing_warmup=1,
            timing_N=50,
            timing_n_per_d=1,
        )
