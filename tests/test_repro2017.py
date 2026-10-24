from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "src" / "experiments"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
SRC = CODE.parent / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_repro2017 import (
    A_2017,
    B_2017,
    F_2017,
    RHO_STAR_2017,
    run_reproduction,
)
from systems import output_variance, pack_rho


def test_registered_rho_star_matches_pack_layout_exactly() -> None:
    packed = pack_rho(A_2017, B_2017, F_2017, d=2)
    assert np.array_equal(RHO_STAR_2017, packed)


def test_registered_output_variance() -> None:
    variance = output_variance(A_2017, B_2017, F_2017, d=2)
    assert abs(variance - 1.384708) < 1e-5


def test_three_seed_smoke_run_has_only_finite_deltas() -> None:
    result = run_reproduction(N=200, seed_indices=(0, 1, 2))
    deltas = np.asarray([row["delta_3000"] for row in result["rows"]])
    assert deltas.shape == (9,)
    assert np.all(np.isfinite(deltas))
