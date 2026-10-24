from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "src" / "experiments"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
SRC = CODE.parent / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_power as power
import systems
from config import Config


def _fake_system(m: int, d: int) -> dict:
    A = np.array([[0.0, 1.0], [-0.1, -0.2]])
    B = np.zeros((2, 2))
    f = np.array([0.5, -0.5])
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": d,
        "n": 2,
        "snr_db": 10.0,
        "sigma_v": 0.1,
        "rho_star": systems.pack_rho(A, B, f, d),
        "rho_F": systems.spectral_radius(systems.build_F(A, B, d)),
        "n_reject": 0,
        "m": m,
        "split": "pilot",
    }


def _write_fake_family(path: Path, delay_counts: tuple[int, int, int]) -> None:
    pilot: list[dict] = []
    m = 0
    for d, count in zip((2, 3, 4), delay_counts):
        for _ in range(count):
            pilot.append(_fake_system(m, d))
            m += 1
    systems.save_family(
        {"train": [], "val": [], "test": [], "pilot": pilot},
        path,
    )


def test_pilot_lineage_and_trajectory_seed(tmp_path: Path) -> None:
    archive = tmp_path / "valid.npz"
    _write_fake_family(archive, (17, 17, 17))
    pilot = power.validate_pilot(systems.load_family(archive), Config())
    assert all(system["split"] == "pilot" for system in pilot)
    for system in pilot:
        seed = power.pilot_trajectory_seed(Config(), system, 3)
        assert list(seed.entropy) == [
            Config().seed_base,
            4,
            int(system["m"]),
            3,
        ]


def test_loo_beta_can_differ_from_global_optimum() -> None:
    beta_grid = np.array([-0.5, 0.5])
    Jsearch = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, -1.0, -1.0],
        ]
    )
    global_beta = beta_grid[np.argmin(np.mean(Jsearch, axis=1))]
    chosen = power.choose_loo_beta(Jsearch, beta_grid)
    assert global_beta == -0.5
    assert chosen[0] == 0.5


def test_shifted_power_properties_and_equivariance() -> None:
    cfg = Config()
    rng = np.random.default_rng(20260730)
    delta = rng.normal(0.15, 0.1, 51)
    delta0 = delta - np.median(delta)
    strata = np.repeat(np.array(cfg.d_choices), 17)
    med0, lo0 = power.simulate_null(
        delta0,
        201,
        strata,
        n_rep=200,
        b_boot=100,
        cfg=cfg,
    )
    rows = power.shifted_power(med0, lo0)
    by_effect = {row["e"]: row for row in rows}
    assert by_effect[0.10]["power_i"] >= 0.90
    assert by_effect[0.00]["power_i"] < 0.10
    assert 0.30 <= by_effect[0.10]["power_ii"] <= 0.75
    power_i = np.asarray([row["power_i"] for row in rows])
    assert np.all(np.diff(power_i) >= -0.03)

    bootstrap_rng = np.random.default_rng(44)
    bootstrap_indices = np.stack(
        [
            power.metrics._resample_indices(51, bootstrap_rng, strata=strata)
            for _ in range(100)
        ]
    )
    median_at_zero = np.median(delta0)
    lo_at_zero = np.percentile(np.median(delta0[bootstrap_indices], axis=1), 2.5)
    for effect in (0.07, 0.13):
        shifted = delta0 + effect
        shifted_median = np.median(shifted)
        shifted_lo = np.percentile(np.median(shifted[bootstrap_indices], axis=1), 2.5)
        assert np.allclose(
            shifted_median, median_at_zero + effect, rtol=0.0, atol=1e-12
        )
        assert np.allclose(shifted_lo, lo_at_zero + effect, rtol=0.0, atol=1e-12)


def test_module_level_registered_constants() -> None:
    assert power.E_GRID == (
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
        0.30,
    )
    assert power.N_REP == 2000
    assert power.B_BOOT == 500
    assert np.array_equal(power.BETA_GRID, np.linspace(-0.5, 0.5, 11))


def test_invalid_pilot_archives_are_rejected(tmp_path: Path) -> None:
    cfg = Config()
    too_small = tmp_path / "too_small.npz"
    _write_fake_family(too_small, (2, 2, 2))
    with pytest.raises(ValueError):
        power.run_power_analysis(
            cfg,
            too_small,
            tmp_path / "too_small_output",
            n_rep=200,
            b_boot=100,
        )

    wrong_strata = tmp_path / "wrong_strata.npz"
    _write_fake_family(wrong_strata, (20, 17, 14))
    with pytest.raises(ValueError):
        power.run_power_analysis(
            cfg,
            wrong_strata,
            tmp_path / "wrong_strata_output",
            n_rep=200,
            b_boot=100,
        )


def test_main_flow_uses_loaded_pilot_and_records_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "systems.npz"
    _write_fake_family(archive, (17, 17, 17))

    def forbidden(*args, **kwargs):
        raise AssertionError("system resampling is forbidden")

    monkeypatch.setattr(systems, "sample_family", forbidden)
    monkeypatch.setattr(systems, "sample_system", forbidden)
    synthetic_delta = np.linspace(-0.12, 0.12, 51)
    synthetic_beta = np.zeros(51)
    monkeypatch.setattr(
        power,
        "estimate_pilot_delta",
        lambda pilot, cfg: (synthetic_delta, synthetic_beta),
    )

    outdir = tmp_path / "power_result"
    result = power.run_power_analysis(
        Config(),
        archive,
        outdir,
        n_rep=200,
        b_boot=100,
    )
    assert all(system["split"] == "pilot" for system in result["pilot"])
    expected_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = (outdir / "power.md").read_text(encoding="utf-8")
    assert expected_hash in report
    assert "功效只针对 (i)" in report
    assert "(ii) `median >= 0.10` 是实际意义门槛" in report
