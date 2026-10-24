from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config
from estimators import misg_run
from systems import pack_rho, simulate


def _example_system() -> dict[str, Any]:
    A = np.array([[0.0, 1.0], [-0.01, -0.22]])
    B = np.array([[0.19, -0.08], [0.16, -0.12]])
    f = np.array([0.8, -0.8])
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": 2,
        "n": 2,
        "sigma_v": 0.30,
        "rho_star": pack_rho(A, B, f, d=2),
    }


def test_cached_phi_window_scales_sublinearly_with_p() -> None:
    cfg = Config()
    system = _example_system()
    trajectory = simulate(system, np.random.default_rng(12345), N=3000)
    arguments = {
        p: {
            "sys": system,
            "traj": trajectory,
            "w": np.ones(p),
            "gamma": 1.0,
            "r_mode": "cumulative",
            "state_mode": "kalman",
            "cfg": cfg,
        }
        for p in (1, 5, 8)
    }

    for p in (1, 5, 8):
        warmup = misg_run(**arguments[p])
        assert warmup["status"] == "ok"

    timings = {p: [] for p in (1, 5, 8)}
    for _ in range(3):
        for p in (8, 5, 1):
            start = time.perf_counter()
            result = misg_run(**arguments[p])
            timings[p].append(time.perf_counter() - start)
            assert result["status"] == "ok"

    fastest = {p: min(values) for p, values in timings.items()}
    assert fastest[5] <= 1.6 * fastest[1], fastest
    assert fastest[8] / fastest[1] < 2.0, fastest
