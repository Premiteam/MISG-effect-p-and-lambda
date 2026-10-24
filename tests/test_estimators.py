from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import linalg


SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

from config import Config, repro_seed
from estimators import misg_run
from systems import (
    build_F,
    get_u,
    get_x,
    get_y,
    pack_rho,
    phi_lags,
    rho_dim,
    sample_system,
    simulate,
    unpack_rho,
)
from weights import eta0, make_weights


def _example_2017(sigma_v: float = 0.30) -> dict:
    A = np.array([[0.0, 1.0], [-0.01, -0.22]])
    B = np.array([[0.19, -0.08], [0.16, -0.12]])
    f = np.array([0.8, -0.8])
    rho_star = pack_rho(A, B, f, d=2)
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": 2,
        "n": 2,
        "sigma_v": sigma_v,
        "rho_star": rho_star,
    }


def _phi_reference(
    time: int,
    n: int,
    d: int,
    history: dict[int, np.ndarray],
    traj: dict,
) -> np.ndarray:
    if time <= 0:
        return np.zeros(rho_dim(n, d))
    x_lags, u_lags = phi_lags(n, d)
    return np.concatenate(
        [history[time - lag] for lag in x_lags]
        + [np.asarray([get_u(traj, time - lag)]) for lag in u_lags]
    )


def _plain_equal_weight_observer(
    sys: dict, traj: dict, p: int, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    n = int(sys["n"])
    d = int(sys["d"])
    N = int(traj["N"])
    m = rho_dim(n, d)
    history = {time: np.ones(n) / cfg.p0_init for time in range(1 - d - n, 2)}
    rho = np.ones(m) / cfg.p0_init
    r_value = 1.0
    rho_traj = np.empty((N, m))
    x_traj = np.empty((N, n))
    for k in range(1, N + 1):
        x_traj[k - 1] = history[k]
        phi = _phi_reference(k, n, d, history, traj)
        Phi = np.column_stack(
            [_phi_reference(k - offset, n, d, history, traj) for offset in range(p)]
        )
        Y = np.asarray([get_y(traj, k - offset) for offset in range(p)])
        E = Y - Phi.T @ rho
        r_value = r_value + np.linalg.norm(Phi, "fro") ** 2
        rho = rho + (Phi @ E) / r_value
        rho_traj[k - 1] = rho
        if k < N:
            A_hat, B_hat, f_hat = unpack_rho(rho, n, d)
            history[k + 1] = (
                A_hat @ history[k] + B_hat @ history[k - d] + f_hat * get_u(traj, k)
            )
    return rho_traj, x_traj


def _run_parameter_case(p: int) -> tuple[dict, np.ndarray, np.ndarray, Config]:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(repro_seed(cfg, 11)), N=80)
    reference_rho, reference_x = _plain_equal_weight_observer(
        system, trajectory, p, cfg
    )
    return system, reference_rho, reference_x, cfg


@pytest.mark.parametrize("p", (1, 2, 5, 8))
def test_literal_equal_weight_degeneracy(p: int) -> None:
    system, reference_rho, reference_x, cfg = _run_parameter_case(p)
    trajectory = simulate(system, np.random.default_rng(repro_seed(cfg, 11)), N=80)
    actual = misg_run(
        system,
        trajectory,
        np.ones(p),
        1.0,
        "cumulative",
        "observer",
        cfg,
    )
    assert actual["status"] == "ok"
    assert np.array_equal(actual["rho_hat_traj"], reference_rho)
    assert np.array_equal(actual["x_hat_traj"], reference_x)


@pytest.mark.parametrize("p", (1, 2, 5, 8))
def test_parameterized_baseline_degeneracy(p: int) -> None:
    system, reference_rho, reference_x, cfg = _run_parameter_case(p)
    trajectory = simulate(system, np.random.default_rng(repro_seed(cfg, 11)), N=80)
    w, gamma = make_weights(np.zeros(p), eta0(cfg.eps), cfg)
    actual = misg_run(system, trajectory, w, gamma, "cumulative", "observer", cfg)
    assert actual["status"] == "ok"
    assert np.allclose(actual["rho_hat_traj"], reference_rho, rtol=1e-12, atol=1e-14)
    assert np.allclose(actual["x_hat_traj"], reference_x, rtol=1e-12, atol=1e-14)


def test_p_one_matches_single_innovation_reference() -> None:
    system, reference_rho, reference_x, cfg = _run_parameter_case(1)
    trajectory = simulate(system, np.random.default_rng(repro_seed(cfg, 11)), N=80)
    actual = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "observer",
        cfg,
    )
    assert np.array_equal(actual["rho_hat_traj"], reference_rho)
    assert np.array_equal(actual["x_hat_traj"], reference_x)


def test_reproducible_run_is_bitwise_identical() -> None:
    cfg = Config()
    system = _example_2017()
    first_traj = simulate(system, np.random.default_rng(repro_seed(cfg, 12)), N=100)
    second_traj = simulate(system, np.random.default_rng(repro_seed(cfg, 12)), N=100)
    first = misg_run(system, first_traj, np.ones(5), 1.0, "cumulative", "kalman", cfg)
    second = misg_run(system, second_traj, np.ones(5), 1.0, "cumulative", "kalman", cfg)
    assert np.array_equal(first["rho_hat_traj"], second["rho_hat_traj"])
    assert np.array_equal(first["x_hat_traj"], second["x_hat_traj"])


def test_true_state_mode_is_self_consistent() -> None:
    cfg = Config()
    system = _example_2017(sigma_v=0.0)
    trajectory = simulate(system, np.random.default_rng(101), N=200)
    result = misg_run(
        system,
        trajectory,
        np.ones(5),
        1.0,
        "cumulative",
        "true",
        cfg,
        rho_hat0=system["rho_star"],
    )
    expected = np.broadcast_to(system["rho_star"], result["rho_hat_traj"].shape)
    assert result["status"] == "ok"
    assert np.allclose(result["rho_hat_traj"], expected, rtol=0.0, atol=1e-10)


def test_k0_exact_with_true_initial_history() -> None:
    cfg = Config()
    system = _example_2017(sigma_v=0.0)
    trajectory = simulate(system, np.random.default_rng(102), N=200)
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "observer",
        cfg,
        rho_hat0=system["rho_star"],
        freeze_rho=True,
        x_hat_init=np.zeros(system["n"]),
    )
    truth = trajectory["x"][1:201]
    assert result["status"] == "ok"
    assert np.allclose(result["x_hat_traj"], truth, rtol=0.0, atol=1e-12)


def test_k0_error_recurrence_and_final_decay() -> None:
    cfg = Config()
    system = _example_2017(sigma_v=0.0)
    trajectory = simulate(system, np.random.default_rng(103), N=200)
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "observer",
        cfg,
        rho_hat0=system["rho_star"],
        freeze_rho=True,
        x_hat_init=np.ones(system["n"]) / cfg.p0_init,
    )
    n = system["n"]
    d = system["d"]
    initial = np.ones(n) / cfg.p0_init
    errors = []
    for k in range(1, 201):
        estimated_blocks = []
        for block in range(d + 1):
            time = k - block
            if time >= 1:
                estimated_blocks.append(result["x_hat_traj"][time - 1])
            else:
                estimated_blocks.append(initial)
        z_estimated = np.concatenate(estimated_blocks)
        z_true = np.concatenate(
            [get_x(trajectory, k - block) for block in range(d + 1)]
        )
        errors.append(z_estimated - z_true)
    F = build_F(system["A"], system["B"], d)
    for index in range(199):
        assert np.allclose(errors[index + 1], F @ errors[index], rtol=0.0, atol=1e-14)
    assert np.linalg.norm(errors[-1]) < 1e-6


def _zero_noise_realization(system: dict, cfg: Config, N: int) -> dict:
    generator_system = {**system, "sigma_v": 0.0}
    return simulate(generator_system, np.random.default_rng(repro_seed(cfg, 20)), N=N)


def test_k1_augmented_state_indexing() -> None:
    cfg = Config()
    system = _example_2017()
    trajectory = _zero_noise_realization(system, cfg, N=120)
    n = system["n"]
    d = system["d"]
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "kalman",
        cfg,
        rho_hat0=system["rho_star"],
        freeze_rho=True,
        x_hat_init=np.zeros((d + n + 1, n)),
        P0=1e-12 * np.eye(n * (d + 1)),
        return_diagnostics=True,
    )
    assert result["status"] == "ok"
    for k in range(1, 121):
        for block in range(d + 1):
            actual = result["z_hat_traj"][k - 1, block * n : (block + 1) * n]
            assert np.allclose(
                actual, get_x(trajectory, k - block), rtol=0.0, atol=1e-10
            )


def _assert_covariances_healthy(P_traj: np.ndarray) -> None:
    for covariance in P_traj:
        assert np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_k1_covariance_symmetric_and_psd() -> None:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(104), N=150)
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "kalman",
        cfg,
        rho_hat0=system["rho_star"],
        freeze_rho=True,
        return_diagnostics=True,
    )
    assert result["status"] == "ok"
    _assert_covariances_healthy(result["P_traj"])


def test_k1_covariance_converges_to_dare() -> None:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(105), N=500)
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "kalman",
        cfg,
        rho_hat0=system["rho_star"],
        freeze_rho=True,
        return_diagnostics=True,
    )
    F = build_F(system["A"], system["B"], system["d"])
    H = np.zeros((1, F.shape[0]))
    H[0, 0] = 1.0
    Q = cfg.kalman_q * np.eye(F.shape[0])
    R = np.array([[system["sigma_v"] ** 2]])
    P_inf = linalg.solve_discrete_are(F.T, H.T, Q, R)
    relative_error = np.linalg.norm(result["P_traj"][-1] - P_inf) / np.linalg.norm(
        P_inf
    )
    assert relative_error < 1e-6


@pytest.mark.parametrize("q", (1e-6, 1e-4, 1e-2))
def test_k1_q_sensitivity_is_numerically_healthy(q: float) -> None:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(106), N=120)
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "kalman",
        cfg,
        rho_hat0=system["rho_star"],
        freeze_rho=True,
        q=q,
        return_diagnostics=True,
    )
    assert result["status"] == "ok"
    _assert_covariances_healthy(result["P_traj"])


def test_golden_trace_if_frozen() -> None:
    golden_path = Path(__file__).resolve().parents[1] / "data" / "estimators_golden.npz"
    if not golden_path.exists():
        pytest.skip("golden not frozen yet")
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(repro_seed(cfg, 0)), N=20)
    observer = misg_run(
        system,
        trajectory,
        np.ones(5),
        1.0,
        "cumulative",
        "observer",
        cfg,
        r_window="newest",
    )
    kalman = misg_run(
        system,
        trajectory,
        np.ones(5),
        1.0,
        "cumulative",
        "kalman",
        cfg,
        r_window="newest",
        return_diagnostics=True,
    )
    with np.load(golden_path, allow_pickle=False) as golden:
        assert np.allclose(
            observer["x_hat_traj"],
            golden["observer_x_hat_traj"],
            rtol=1e-12,
            atol=1e-14,
        )
        assert np.allclose(
            kalman["x_hat_traj"],
            golden["kalman_x_hat_traj"],
            rtol=1e-12,
            atol=1e-14,
        )
        assert np.allclose(
            kalman["z_hat_traj"],
            golden["kalman_z_hat_traj"],
            rtol=1e-12,
            atol=1e-14,
        )
        assert np.allclose(
            kalman["P_traj"],
            golden["kalman_P_traj"],
            rtol=1e-12,
            atol=1e-14,
        )


def test_spec_divergence_is_captured_at_exact_patience() -> None:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(107), N=80)
    unstable_rho = np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "true",
        cfg,
        rho_hat0=unstable_rho,
        freeze_rho=True,
    )
    assert result["status"] == "diverged"
    assert result["div_reason"] == "spec"
    assert result["diverged_at"] == cfg.div_spec_patience


@pytest.mark.parametrize(
    ("rho_hat0", "expected_reason"),
    [
        (np.full(8, 200.0 * 1.18785521 / np.sqrt(8.0)), "norm"),
        (np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "nan"),
    ],
)
def test_norm_and_nan_divergence_reasons(
    rho_hat0: np.ndarray, expected_reason: str
) -> None:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(108), N=10)
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "true",
        cfg,
        rho_hat0=rho_hat0,
        freeze_rho=True,
    )
    assert result["status"] == "diverged"
    assert result["div_reason"] == expected_reason
    assert result["diverged_at"] == 1


def test_coupled_branch_protection() -> None:
    cfg = replace(Config(), n=2)
    system = sample_system(np.random.default_rng(109), cfg, d=1)
    trajectory = simulate(system, np.random.default_rng(110), N=40)
    for mode in ("observer", "kalman"):
        with pytest.raises(NotImplementedError, match="only identifiable as a sum"):
            misg_run(
                system,
                trajectory,
                np.ones(1),
                1.0,
                "cumulative",
                mode,
                cfg,
            )
    result = misg_run(
        system,
        trajectory,
        np.ones(1),
        1.0,
        "cumulative",
        "true",
        cfg,
    )
    assert result["status"] == "ok"
    assert result["spec_check"] == "skipped"
