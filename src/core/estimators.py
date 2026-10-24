from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from config import Config
from systems import (
    build_F,
    get_u,
    get_x,
    get_y,
    phi_lags,
    rho_dim,
    spectral_radius,
    unpack_rho,
)


def _initial_history(
    n: int, d: int, p0: float, x_hat_init: np.ndarray | None
) -> dict[int, np.ndarray]:
    if x_hat_init is None:
        values = np.ones(n) / p0
        return {time: values.copy() for time in range(1 - d - n, 2)}

    initial = np.asarray(x_hat_init, dtype=float)
    if initial.shape == (n,):
        return {time: initial.copy() for time in range(1 - d - n, 2)}
    if initial.shape != (d + n + 1, n):
        raise ValueError(
            "x_hat_init must have shape (n,) or (d+n+1, n), with "
            "x_hat_init[i] corresponding to x_hat(1-i)"
        )
    return {1 - index: initial[index].copy() for index in range(d + n + 1)}


def _estimated_phi(
    time: int,
    n: int,
    d: int,
    history: Mapping[int, np.ndarray],
    traj: Mapping[str, Any],
) -> np.ndarray:
    m = rho_dim(n, d)
    if time <= 0:
        return np.zeros(m)
    x_lags, u_lags = phi_lags(n, d)
    blocks: list[np.ndarray] = [history[time - lag] for lag in x_lags]
    blocks.extend(np.asarray([get_u(traj, time - lag)]) for lag in u_lags)
    return np.concatenate(blocks)


def misg_run(
    sys: Mapping[str, Any],
    traj: Mapping[str, Any],
    w: np.ndarray,
    gamma: float,
    r_mode: str,
    state_mode: str,
    cfg: Config,
    *,
    lam_g: float = 1.0,
    r_window: str = "full",
    rho_hat0: np.ndarray | None = None,
    freeze_rho: bool = False,
    x_hat_init: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    q: float | None = None,
    return_diagnostics: bool = False,
) -> dict[str, Any]:
    n = int(sys["n"])
    d = int(sys["d"])
    N = int(traj["N"])
    m = rho_dim(n, d)
    weights = np.asarray(w, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("w must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("w must contain only finite values")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    if r_mode not in {"cumulative", "forgetting"}:
        raise ValueError("r_mode must be 'cumulative' or 'forgetting'")
    if state_mode not in {"true", "observer", "kalman"}:
        raise ValueError("state_mode must be 'true', 'observer', or 'kalman'")
    if r_mode == "forgetting" and not np.isfinite(lam_g):
        raise ValueError("lam_g must be finite")
    if r_window not in {"newest", "full"}:
        raise ValueError("r_window must be 'newest' or 'full'")
    if d <= n - 1 and state_mode in {"observer", "kalman"}:
        raise NotImplementedError(
            "d <= n-1: a and b_{n-d} are only identifiable as a sum; "
            "the observer cannot be formed."
        )

    p = weights.size
    if rho_hat0 is None:
        rho = np.ones(m) / cfg.p0_init
    else:
        rho = np.asarray(rho_hat0, dtype=float).copy()
        if rho.shape != (m,):
            raise ValueError(f"rho_hat0 must have shape ({m},)")

    history = _initial_history(n, d, cfg.p0_init, x_hat_init)
    if state_mode == "true":
        for time in range(1 - d - n, 2):
            history[time] = get_x(traj, time).copy()

    augmented_size = n * (d + 1)
    z_hat: np.ndarray | None = None
    covariance: np.ndarray | None = None
    H: np.ndarray | None = None
    Q: np.ndarray | None = None
    R: float | None = None
    if state_mode == "kalman":
        z_hat = np.concatenate([history[1 - block] for block in range(d + 1)])
        if P0 is None:
            covariance = cfg.kalman_p0 * np.eye(augmented_size)
        else:
            covariance = np.asarray(P0, dtype=float).copy()
            if covariance.shape != (augmented_size, augmented_size):
                raise ValueError(
                    f"P0 must have shape ({augmented_size}, {augmented_size})"
                )
        H = np.zeros(augmented_size)
        H[0] = 1.0
        q_value = cfg.kalman_q if q is None else float(q)
        if q_value < 0.0 or not np.isfinite(q_value):
            raise ValueError("q must be finite and non-negative")
        Q = q_value * np.eye(augmented_size)
        R = float(sys["sigma_v"]) ** 2

    rho_hat_traj = np.full((N, m), np.nan)
    x_hat_traj = np.full((N, n), np.nan)
    z_hat_traj = (
        np.full((N, augmented_size), np.nan)
        if return_diagnostics and state_mode == "kalman"
        else None
    )
    P_traj = (
        np.full((N, augmented_size, augmented_size), np.nan)
        if return_diagnostics and state_mode == "kalman"
        else None
    )

    r_value = 1.0
    spec_counter = 0
    status = "ok"
    diverged_at: int | None = None
    div_reason: str | None = None
    phi_buffer = np.zeros((p, m))
    phi_cursor = 0

    for k in range(1, N + 1):
        if state_mode == "true":
            history[k] = get_x(traj, k).copy()

        current_state = history[k]
        x_hat_traj[k - 1] = current_state
        if z_hat_traj is not None:
            assert z_hat is not None
            z_hat_traj[k - 1] = z_hat
        if P_traj is not None:
            assert covariance is not None
            P_traj[k - 1] = covariance

        phi = _estimated_phi(k, n, d, history, traj)
        phi_buffer[phi_cursor] = phi
        Phi_hat = np.column_stack(
            [phi_buffer[(phi_cursor - offset) % p] for offset in range(p)]
        )
        phi_cursor = (phi_cursor + 1) % p
        Y = np.asarray([get_y(traj, k - offset) for offset in range(p)])
        E = Y - Phi_hat.T @ rho
        window_energy = (
            np.linalg.norm(Phi_hat, "fro") ** 2
            if r_window == "full"
            else np.linalg.norm(phi) ** 2
        )
        if r_mode == "cumulative":
            r_value = r_value + window_energy
        else:
            r_value = lam_g * r_value + window_energy
        if not freeze_rho:
            rho = rho + gamma * (Phi_hat @ (weights * E)) / r_value
        rho_hat_traj[k - 1] = rho

        if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(current_state)):
            status = "diverged"
            diverged_at = k
            div_reason = "nan"
            break
        if np.linalg.norm(rho) > cfg.div_rho_factor * np.linalg.norm(
            np.asarray(sys["rho_star"])
        ):
            status = "diverged"
            diverged_at = k
            div_reason = "norm"
            break

        unpacked: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        F_hat: np.ndarray | None = None
        if d >= n:
            unpacked = unpack_rho(rho, n, d)
            A_hat, B_hat, _ = unpacked
            F_hat = build_F(A_hat, B_hat, d)
            inf_norm = float(np.abs(F_hat).sum(axis=1).max())
            if (
                inf_norm > cfg.div_spec_radius
                and spectral_radius(F_hat) > cfg.div_spec_radius
            ):
                spec_counter += 1
            else:
                spec_counter = 0
            if spec_counter >= cfg.div_spec_patience:
                status = "diverged"
                diverged_at = k
                div_reason = "spec"
                break

        if k == N:
            continue

        if state_mode == "observer":
            assert unpacked is not None
            A_hat, B_hat, f_hat = unpacked
            history[k + 1] = (
                A_hat @ current_state + B_hat @ history[k - d] + f_hat * get_u(traj, k)
            )
        elif state_mode == "kalman":
            assert unpacked is not None
            assert z_hat is not None
            assert covariance is not None
            assert H is not None
            assert Q is not None
            assert R is not None
            A_hat, B_hat, f_hat = unpacked
            assert F_hat is not None
            Gf_hat = np.zeros(augmented_size)
            Gf_hat[:n] = f_hat
            S_k = float(H @ covariance @ H.T + R)
            L_k = F_hat @ covariance @ H.T / S_k
            z_hat = (
                F_hat @ z_hat
                + Gf_hat * get_u(traj, k)
                + L_k * (get_y(traj, k) - H @ z_hat)
            )
            covariance = F_hat @ covariance @ F_hat.T + Q - np.outer(L_k, L_k) * S_k
            covariance = (covariance + covariance.T) / 2.0
            history[k + 1] = z_hat[:n].copy()
        else:
            history[k + 1] = get_x(traj, k + 1).copy()

    result: dict[str, Any] = {
        "rho_hat_traj": rho_hat_traj,
        "x_hat_traj": x_hat_traj,
        "status": status,
        "diverged_at": diverged_at,
        "div_reason": div_reason,
    }
    if d <= n - 1:
        result["spec_check"] = "skipped"
    if return_diagnostics:
        result["z_hat_traj"] = z_hat_traj
        result["P_traj"] = P_traj
    return result
