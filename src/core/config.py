from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    n: int = 2
    d_choices: tuple[int, ...] = (2, 3, 4)
    b_scale: float = 0.3
    rho_max: float = 0.95
    snr_lo_db: float = 5.0
    snr_hi_db: float = 25.0
    N: int = 3000
    n_traj: int = 5
    n_train: int = 201
    n_val: int = 102
    n_test: int = 201
    n_pilot: int = 51
    A_bound: float = 2.302585092994046
    eps: float = 0.1
    p_choices: tuple[int, ...] = (1, 2, 5, 8)
    p_main: int = 5
    kalman_q: float = 1e-4
    kalman_p0: float = 1.0
    p0_init: float = 1e6
    div_rho_factor: float = 100.0
    div_spec_radius: float = 1.2
    div_spec_patience: int = 50
    r_cap: float = 10.0
    bootstrap_B: int = 2000
    seed_base: int = 20260729


def sys_seed(cfg: Config, m: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 0, m])


def traj_seed(cfg: Config, m: int, j: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 1, m, j])


def repro_seed(cfg: Config, j: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 2, j])


def pilot_seed(cfg: Config, m: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 3, m])


def pilot_traj_seed(cfg: Config, m: int, j: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 4, m, j])


def power_seed(cfg: Config, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 5, rep])


def q_select_seed(cfg: Config) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 6])


def timing_select_seed(cfg: Config) -> np.random.SeedSequence:
    return np.random.SeedSequence([cfg.seed_base, 7])
