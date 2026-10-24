from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

from config import (
    Config,
    pilot_seed,
    pilot_traj_seed,
    power_seed,
    q_select_seed,
    repro_seed,
    sys_seed,
    timing_select_seed,
    traj_seed,
)
from systems import (
    build_F,
    companion_A,
    family_manifest,
    get_u,
    get_x,
    gram_diagnostics,
    load_family,
    output_variance,
    pack_rho,
    phi_lags,
    rho_dim,
    sample_family,
    sample_system,
    save_family,
    simulate,
    spectral_radius,
    unpack_rho,
)


def _manual_phi(traj: dict, n: int, d: int, k: int) -> np.ndarray:
    x_lags, u_lags = phi_lags(n, d)
    return np.concatenate(
        [get_x(traj, k - lag) for lag in x_lags]
        + [np.asarray([get_u(traj, k - lag)]) for lag in u_lags]
    )


def _example_2017() -> dict:
    A = np.array([[0.0, 1.0], [-0.01, -0.22]])
    B = np.array([[0.19, -0.08], [0.16, -0.12]])
    f = np.array([0.8, -0.8])
    d = 2
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": d,
        "n": 2,
        "snr_db": 10.0 * np.log10(1.384708 / 0.09),
        "sigma_v": 0.30,
        "rho_star": pack_rho(A, B, f, d),
        "rho_F": spectral_radius(build_F(A, B, d)),
        "n_reject": 0,
        "m": 0,
        "split": "repro",
    }


def test_companion_A() -> None:
    expected = np.array([[0.0, 1.0], [0.1, 0.2]])
    assert np.array_equal(companion_A(np.array([0.1, 0.2])), expected)


def test_build_F_manual_layout() -> None:
    A = np.array([[0.0, 1.0], [0.1, 0.2]])
    B = np.array([[0.3, 0.4], [0.5, 0.6]])
    expected = np.block([[A, B], [np.eye(2), np.zeros((2, 2))]])
    actual = build_F(A, B, d=1)
    assert actual.shape == (4, 4)
    assert np.array_equal(actual, expected)


def test_sample_systems_are_stable() -> None:
    cfg = Config()
    rng = np.random.default_rng(1729)
    for index in range(200):
        d = cfg.d_choices[index % len(cfg.d_choices)]
        system = sample_system(rng, cfg, d)
        assert spectral_radius(build_F(system["A"], system["B"], d)) < cfg.rho_max


def test_sample_system_is_deterministic() -> None:
    cfg = Config()
    first = sample_system(np.random.default_rng(2468), cfg, d=3)
    second = sample_system(np.random.default_rng(2468), cfg, d=3)
    for key in ("A", "B", "f", "rho_star"):
        assert np.array_equal(first[key], second[key])
    for key in ("d", "n", "snr_db", "sigma_v", "rho_F", "n_reject"):
        assert first[key] == second[key]


def test_family_stratification_and_invalid_count() -> None:
    cfg = Config()
    family = sample_family(cfg)
    _, counts = family_manifest(family)
    assert counts == {
        "train": {2: 67, 3: 67, 4: 67},
        "val": {2: 34, 3: 34, 4: 34},
        "test": {2: 67, 3: 67, 4: 67},
        "pilot": {2: 17, 3: 17, 4: 17},
    }
    with pytest.raises(ValueError):
        sample_family(replace(cfg, n_train=200))


def test_seed_namespaces_and_pilot_independence() -> None:
    cfg = Config()
    assert list(sys_seed(cfg, 5).entropy) == [cfg.seed_base, 0, 5]
    assert list(traj_seed(cfg, 5, 1).entropy) == [cfg.seed_base, 1, 5, 1]
    assert list(repro_seed(cfg, 5).entropy) == [cfg.seed_base, 2, 5]
    assert list(pilot_seed(cfg, 5).entropy) == [cfg.seed_base, 3, 5]
    assert list(pilot_traj_seed(cfg, 5, 1).entropy) == [
        cfg.seed_base,
        4,
        5,
        1,
    ]
    assert list(power_seed(cfg, 5).entropy) == [cfg.seed_base, 5, 5]
    assert list(q_select_seed(cfg).entropy) == [cfg.seed_base, 6]
    assert list(timing_select_seed(cfg).entropy) == [cfg.seed_base, 7]

    main_system = sample_system(np.random.default_rng(sys_seed(cfg, 5)), cfg, d=2)
    pilot_system = sample_system(np.random.default_rng(pilot_seed(cfg, 5)), cfg, d=2)
    assert not np.array_equal(main_system["A"], pilot_system["A"])

    main_traj = simulate(main_system, np.random.default_rng(traj_seed(cfg, 5, 1)), N=64)
    pilot_traj = simulate(
        pilot_system, np.random.default_rng(pilot_traj_seed(cfg, 5, 1)), N=64
    )
    assert not np.array_equal(main_traj["u"], pilot_traj["u"])


def test_sample_system_requires_d() -> None:
    with pytest.raises(TypeError):
        sample_system(np.random.default_rng(1), Config())


@pytest.mark.parametrize(
    ("n", "d"),
    [
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 1),
        (4, 3),
    ],
)
def test_regression_identity(n: int, d: int) -> None:
    cfg = replace(Config(), n=n)
    system = sample_system(np.random.default_rng(10_000 + 100 * n + d), cfg, d)
    trajectory = simulate(system, np.random.default_rng(20_000 + n + d), N=400)
    rho = pack_rho(system["A"], system["B"], system["f"], d)
    for k in range(1, 401):
        phi = _manual_phi(trajectory, n, d, k)
        assert abs(get_x(trajectory, k)[0] - phi @ rho) < 1e-12


def test_rho_dimensions() -> None:
    assert rho_dim(2, 2) == 8
    assert rho_dim(2, 1) == 6
    assert rho_dim(3, 1) == 12
    assert rho_dim(3, 3) == 15


def test_unpack_rho_branches() -> None:
    rng = np.random.default_rng(321)
    A = companion_A(rng.normal(size=3))
    B = rng.normal(size=(3, 3))
    f = rng.normal(size=3)
    with pytest.raises(NotImplementedError, match="only identifiable as a sum"):
        unpack_rho(pack_rho(A, B, f, d=1), n=3, d=1)

    restored = unpack_rho(pack_rho(A, B, f, d=3), n=3, d=3)
    for actual, expected in zip(restored, (A, B, f), strict=True):
        assert np.array_equal(actual, expected)


def test_family_npz_roundtrip(tmp_path: Path) -> None:
    cfg = replace(Config(), n_train=6, n_val=6, n_test=6, n_pilot=6)
    family = sample_family(cfg)
    path = tmp_path / "systems.npz"
    save_family(family, path)
    restored = load_family(path)
    assert restored.keys() == family.keys()
    for split in family:
        for original, loaded in zip(family[split], restored[split], strict=True):
            for key in ("A", "B", "f", "rho_star"):
                assert np.array_equal(original[key], loaded[key])
            for key in (
                "m",
                "split",
                "d",
                "n",
                "snr_db",
                "sigma_v",
                "rho_F",
                "n_reject",
            ):
                assert original[key] == loaded[key]


def test_output_variance_and_snr_for_2017_example() -> None:
    system = _example_2017()
    variance = output_variance(system["A"], system["B"], system["f"], 2)
    assert abs(variance - 1.384708) < 1e-5
    snr_db = 10.0 * np.log10(1.384708 / 0.09)
    assert 11.8 <= snr_db <= 11.95


def test_sampled_snr_end_to_end() -> None:
    cfg = Config()
    sampling_rng = np.random.default_rng(86420)
    errors: list[float] = []
    for index in range(200):
        d = cfg.d_choices[index % len(cfg.d_choices)]
        system = sample_system(sampling_rng, cfg, d)
        trajectory = simulate(system, np.random.default_rng(90_000 + index), N=20_000)
        y_clean = trajectory["x"][1:20_001, 0]
        empirical_snr = 10.0 * np.log10(np.var(y_clean) / float(system["sigma_v"]) ** 2)
        errors.append(empirical_snr - float(system["snr_db"]))
    assert abs(float(np.median(errors))) < 0.3


def test_gram_diagnostics_full_rank_and_nonmutating() -> None:
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(13579), N=3000)
    system_before = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in system.items()
    }
    trajectory_before = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in trajectory.items()
    }

    diagnostics = gram_diagnostics(system, trajectory)
    assert diagnostics["rank"] == 8

    for key, value in system_before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(system[key], value)
        else:
            assert system[key] == value
    for key, value in trajectory_before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(trajectory[key], value)
        else:
            assert trajectory[key] == value
