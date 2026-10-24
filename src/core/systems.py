from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg

from config import Config, pilot_seed, sys_seed


System = dict[str, Any]
Family = dict[str, list[System]]


def companion_A(a: np.ndarray) -> np.ndarray:
    row = np.asarray(a)
    if row.ndim != 1 or row.size == 0:
        raise ValueError("a must be a non-empty one-dimensional array")
    n = row.size
    matrix = np.zeros((n, n), dtype=np.result_type(row.dtype, float))
    if n > 1:
        matrix[:-1, 1:] = np.eye(n - 1)
    matrix[-1, :] = row
    return matrix


def build_F(A: np.ndarray, B: np.ndarray, d: int) -> np.ndarray:
    A_array = np.asarray(A)
    B_array = np.asarray(B)
    if A_array.ndim != 2 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError("A must be square")
    if B_array.shape != A_array.shape:
        raise ValueError("B must have the same shape as A")
    if d < 1:
        raise ValueError("d must be a positive integer")

    n = A_array.shape[0]
    size = n * (d + 1)
    matrix = np.zeros(
        (size, size), dtype=np.result_type(A_array.dtype, B_array.dtype, float)
    )
    matrix[:n, :n] = A_array
    matrix[:n, n * d : n * (d + 1)] = B_array
    for block in range(1, d + 1):
        matrix[block * n : (block + 1) * n, (block - 1) * n : block * n] = np.eye(n)
    return matrix


def spectral_radius(M: np.ndarray) -> float:
    matrix = np.asarray(M)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("M must be square")
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def output_variance(A: np.ndarray, B: np.ndarray, f: np.ndarray, d: int) -> float:
    A_array = np.asarray(A)
    f_array = np.asarray(f)
    if f_array.shape != (A_array.shape[0],):
        raise ValueError("f must have shape (n,)")
    F = build_F(A_array, B, d)
    Gf = np.zeros((F.shape[0], 1), dtype=np.result_type(F.dtype, f_array.dtype))
    Gf[: A_array.shape[0], 0] = f_array
    sigma = linalg.solve_discrete_lyapunov(F, Gf @ Gf.T)
    return float(sigma[0, 0])


def rho_dim(n: int, d: int) -> int:
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    return n**2 + 2 * n if d >= n else n**2 + n


def pack_rho(A: np.ndarray, B: np.ndarray, f: np.ndarray, d: int) -> np.ndarray:
    A_array = np.asarray(A)
    B_array = np.asarray(B)
    f_array = np.asarray(f)
    if A_array.ndim != 2 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError("A must be square")
    n = A_array.shape[0]
    if B_array.shape != (n, n) or f_array.shape != (n,):
        raise ValueError("B and f must have shapes (n, n) and (n,)")
    if d < 1:
        raise ValueError("d must be positive")

    a = A_array[-1, :]
    blocks = [B_array[index, :] for index in range(n)]
    if d >= n:
        return np.concatenate([a, *blocks, f_array])

    coupled_blocks = [block.copy() for block in blocks]
    coupled_index = n - d - 1
    coupled_blocks[coupled_index] = coupled_blocks[coupled_index] + a
    return np.concatenate([*coupled_blocks, f_array])


def unpack_rho(
    rho: np.ndarray, n: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d <= n - 1:
        raise NotImplementedError(
            "d <= n-1: a and b_{n-d} are only identifiable as a sum; "
            "the observer cannot be formed."
        )
    packed = np.asarray(rho)
    if packed.ndim != 1 or packed.size != rho_dim(n, d):
        raise ValueError("rho has the wrong shape")
    a = packed[:n]
    B = packed[n : n + n**2].reshape(n, n).copy()
    f = packed[n + n**2 :].copy()
    return companion_A(a), B, f


def phi_lags(n: int, d: int) -> tuple[list[int], list[int]]:
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    if d >= n:
        x_lags = [n, *range(d + 1, d + n + 1)]
    else:
        x_lags = list(range(d + 1, d + n + 1))
    return x_lags, list(range(1, n + 1))


def sample_system(rng: np.random.Generator, cfg: Config, d: int) -> System:
    if d < 1:
        raise ValueError("d must be positive")
    n_reject = 0
    while True:
        a = rng.uniform(-1.0, 1.0, cfg.n)
        A = companion_A(a)
        B = rng.standard_normal((cfg.n, cfg.n)) * cfg.b_scale
        rho_F = spectral_radius(build_F(A, B, d))
        if rho_F < cfg.rho_max:
            break
        n_reject += 1

    f = rng.standard_normal(cfg.n)
    snr_db = float(rng.uniform(cfg.snr_lo_db, cfg.snr_hi_db))
    clean_variance = output_variance(A, B, f, d)
    sigma_v = float(np.sqrt(clean_variance / 10.0 ** (snr_db / 10.0)))
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": d,
        "n": cfg.n,
        "snr_db": snr_db,
        "sigma_v": sigma_v,
        "rho_star": pack_rho(A, B, f, d),
        "rho_F": rho_F,
        "n_reject": n_reject,
    }


def sample_family(cfg: Config) -> Family:
    splits = (
        ("train", cfg.n_train),
        ("val", cfg.n_val),
        ("test", cfg.n_test),
        ("pilot", cfg.n_pilot),
    )
    n_strata = len(cfg.d_choices)
    if n_strata == 0:
        raise ValueError("d_choices must not be empty")
    for split, count in splits:
        if count % n_strata:
            raise ValueError(
                f"n_{split}={count} is not divisible by len(d_choices)={n_strata}"
            )

    family: Family = {split: [] for split, _ in splits}
    main_m = 0
    for split, count in splits:
        per_delay = count // n_strata
        for d in cfg.d_choices:
            for _ in range(per_delay):
                if split == "pilot":
                    m = len(family[split])
                    seed = pilot_seed(cfg, m)
                else:
                    m = main_m
                    main_m += 1
                    seed = sys_seed(cfg, m)
                system = sample_system(np.random.default_rng(seed), cfg, d)
                system["m"] = m
                system["split"] = split
                family[split].append(system)
    return family


def family_manifest(family: Mapping[str, list[System]]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    counts: dict[str, dict[int, int]] = {}
    for split, systems in family.items():
        split_counts: dict[int, int] = {}
        for system in systems:
            d = int(system["d"])
            split_counts[d] = split_counts.get(d, 0) + 1
            rows.append(
                {
                    "m": int(system["m"]),
                    "split": split,
                    "d": d,
                    "rho_F": float(system["rho_F"]),
                    "snr_db": float(system["snr_db"]),
                    "sigma_v": float(system["sigma_v"]),
                    "rho_star_norm": float(np.linalg.norm(system["rho_star"])),
                    "n_reject": int(system["n_reject"]),
                }
            )
        counts[split] = split_counts
    return rows, counts


def save_family(family: Mapping[str, list[System]], path: str | Path) -> None:
    flattened: list[tuple[str, System]] = [
        (split, system) for split, systems in family.items() for system in systems
    ]
    payload: dict[str, np.ndarray] = {
        "split_names": np.asarray(list(family.keys()), dtype=str),
        "item_splits": np.asarray([split for split, _ in flattened], dtype=str),
        "m": np.asarray([system["m"] for _, system in flattened], dtype=np.int64),
        "d": np.asarray([system["d"] for _, system in flattened], dtype=np.int64),
        "n": np.asarray([system["n"] for _, system in flattened], dtype=np.int64),
        "snr_db": np.asarray(
            [system["snr_db"] for _, system in flattened], dtype=float
        ),
        "sigma_v": np.asarray(
            [system["sigma_v"] for _, system in flattened], dtype=float
        ),
        "rho_F": np.asarray([system["rho_F"] for _, system in flattened], dtype=float),
        "n_reject": np.asarray(
            [system["n_reject"] for _, system in flattened], dtype=np.int64
        ),
    }
    for index, (_, system) in enumerate(flattened):
        suffix = f"{index:06d}"
        payload[f"A_{suffix}"] = np.asarray(system["A"])
        payload[f"B_{suffix}"] = np.asarray(system["B"])
        payload[f"f_{suffix}"] = np.asarray(system["f"])
        payload[f"rho_star_{suffix}"] = np.asarray(system["rho_star"])
    np.savez(Path(path), **payload)


def load_family(path: str | Path) -> Family:
    with np.load(Path(path), allow_pickle=False) as archive:
        split_names = [str(value) for value in archive["split_names"]]
        item_splits = [str(value) for value in archive["item_splits"]]
        item_count = len(item_splits)
        scalar_names = ("m", "d", "n", "snr_db", "sigma_v", "rho_F", "n_reject")
        if any(len(archive[name]) != item_count for name in scalar_names):
            raise ValueError("family archive has inconsistent metadata lengths")

        family: Family = {split: [] for split in split_names}
        for index, split in enumerate(item_splits):
            if split not in family:
                raise ValueError(f"unknown split {split!r} in family archive")
            suffix = f"{index:06d}"
            A = np.array(archive[f"A_{suffix}"], copy=True)
            B = np.array(archive[f"B_{suffix}"], copy=True)
            f = np.array(archive[f"f_{suffix}"], copy=True)
            rho_star = np.array(archive[f"rho_star_{suffix}"], copy=True)
            d = int(archive["d"][index])
            if not np.array_equal(rho_star, pack_rho(A, B, f, d)):
                raise ValueError(
                    f"rho_star does not match pack_rho for split={split}, "
                    f"m={int(archive['m'][index])}"
                )
            family[split].append(
                {
                    "A": A,
                    "B": B,
                    "f": f,
                    "d": d,
                    "n": int(archive["n"][index]),
                    "snr_db": float(archive["snr_db"][index]),
                    "sigma_v": float(archive["sigma_v"][index]),
                    "rho_star": rho_star,
                    "rho_F": float(archive["rho_F"][index]),
                    "n_reject": int(archive["n_reject"][index]),
                    "m": int(archive["m"][index]),
                    "split": split,
                }
            )
    return family


def get_x(traj: Mapping[str, Any], k: int) -> np.ndarray:
    states = np.asarray(traj["x"])
    if k <= 0:
        return np.zeros(states.shape[1], dtype=states.dtype)
    return states[k]


def get_u(traj: Mapping[str, Any], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(np.asarray(traj["u"])[k])


def get_y(traj: Mapping[str, Any], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(np.asarray(traj["y"])[k])


def simulate(
    sys: Mapping[str, Any], rng: np.random.Generator, N: int
) -> dict[str, Any]:
    if N < 1:
        raise ValueError("N must be positive")
    A = np.asarray(sys["A"])
    B = np.asarray(sys["B"])
    f = np.asarray(sys["f"])
    d = int(sys["d"])
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n) or f.shape != (n,):
        raise ValueError("system arrays have incompatible shapes")

    u = np.zeros(N + 1, dtype=float)
    y = np.zeros(N + 1, dtype=float)
    v = np.zeros(N + 1, dtype=float)
    x = np.zeros((N + 2, n), dtype=np.result_type(A.dtype, B.dtype, f.dtype, float))
    u[1:] = rng.standard_normal(N)
    v[1:] = rng.normal(0.0, float(sys["sigma_v"]), N)
    for k in range(1, N + 1):
        y[k] = x[k, 0] + v[k]
        delayed = get_x({"x": x}, k - d)
        x[k + 1] = A @ x[k] + B @ delayed + f * u[k]
    return {"u": u, "v": v, "y": y, "x": x, "N": N}


def _phi_from_traj(traj: Mapping[str, Any], n: int, d: int, k: int) -> np.ndarray:
    x_lags, u_lags = phi_lags(n, d)
    blocks: list[np.ndarray] = [get_x(traj, k - lag) for lag in x_lags]
    blocks.extend(np.asarray([get_u(traj, k - lag)]) for lag in u_lags)
    return np.concatenate(blocks)


def gram_diagnostics(sys: Mapping[str, Any], traj: Mapping[str, Any]) -> dict:
    n = int(sys["n"])
    d = int(sys["d"])
    N = int(traj["N"])
    Phi = np.stack([_phi_from_traj(traj, n, d, k) for k in range(1, N + 1)])
    gram = Phi.T @ Phi / N
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "rank": int(np.linalg.matrix_rank(gram)),
        "cond": float(np.linalg.cond(gram)),
        "eig_min": float(eigenvalues[0]),
        "eig_max": float(eigenvalues[-1]),
    }
