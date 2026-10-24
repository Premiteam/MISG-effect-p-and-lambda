from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

from config import Config
from weights import (
    A_TILDE_BOX,
    ETA_BOX,
    eta0,
    make_weights,
    w_envelope,
)


@pytest.mark.parametrize("p", (1, 2, 5, 8))
def test_baseline_consistent_initialization(p: int) -> None:
    cfg = Config()
    w, gamma = make_weights(np.zeros(p), eta0(cfg.eps), cfg)
    assert np.allclose(w, 1.0, rtol=1e-12, atol=0.0)
    assert abs(gamma - 1.0) < 1e-12


def test_eta0_registered_value() -> None:
    assert abs(eta0(0.1) - 0.105360515657826) < 1e-15


@pytest.mark.parametrize("scale", (1.0, 10.0, 1e6))
@pytest.mark.parametrize("eta", (-1000.0, -10.0, 0.0, 10.0, 1000.0))
def test_weights_sum_to_p_for_extreme_inputs(scale: float, eta: float) -> None:
    cfg = Config()
    rng = np.random.default_rng(1000 + int(np.log10(scale)))
    for p in cfg.p_choices:
        a_tilde = rng.normal(size=p) * scale
        w, _ = make_weights(a_tilde, eta, cfg)
        assert abs(float(np.sum(w)) - p) < 1e-10


def test_step_size_constraint_and_strict_box_constraint() -> None:
    cfg = Config()
    rng = np.random.default_rng(20260729)
    for p in cfg.p_choices:
        for _ in range(100):
            a_tilde = rng.normal(size=p) * rng.choice((1.0, 10.0, 1e6))
            eta = float(rng.normal() * 100.0)
            w, gamma = make_weights(a_tilde, eta, cfg)
            assert gamma * float(np.max(w)) <= 2.0 - cfg.eps + 1e-12

            eta_in_box = float(rng.uniform(*ETA_BOX))
            w_box, gamma_box = make_weights(a_tilde, eta_in_box, cfg)
            assert gamma_box * float(np.max(w_box)) < 2.0 - cfg.eps


@pytest.mark.parametrize("p", (2, 5, 8))
def test_weight_ratio_bound_and_extreme_limit(p: int) -> None:
    cfg = Config()
    rng = np.random.default_rng(3100 + p)
    bound = np.exp(2.0 * cfg.A_bound)
    for scale in (1.0, 10.0, 1e6):
        for _ in range(50):
            w, _ = make_weights(rng.normal(size=p) * scale, 0.0, cfg)
            assert float(np.max(w) / np.min(w)) <= bound + 1e-9

    extreme = np.zeros(p)
    extreme[0] = 1e6
    extreme[1] = -1e6
    w_extreme, _ = make_weights(extreme, 0.0, cfg)
    ratio = float(np.max(w_extreme) / np.min(w_extreme))
    assert np.allclose(ratio, bound, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("p", (1, 2, 5, 8))
def test_coordinate_envelope(p: int) -> None:
    cfg = Config()
    patterns = (
        np.linspace(-1e6, 1e6, p),
        np.full(p, 1e6),
        np.full(p, -1e6),
    )
    lower, upper = w_envelope(p, cfg.A_bound)
    for pattern in patterns:
        w, _ = make_weights(pattern, 0.0, cfg)
        assert np.all(w >= lower - 1e-14)
        assert np.all(w <= upper + 1e-14)

    if p == 5:
        assert abs(lower - 0.012468827930174564) < 1e-12
        assert abs(upper - 4.8076923076923075) < 1e-12
    if p == 8:
        assert abs(lower - 0.011412268188302425) < 1e-12
        assert abs(upper - 7.4766355140186915) < 1e-12


@pytest.mark.parametrize("p", (2, 5, 8))
def test_gradient_is_nonzero_at_uniform_weights(p: int) -> None:
    cfg = Config()
    h = 1e-6
    numerical = np.empty((p, p))
    for j in range(p):
        plus = np.zeros(p)
        minus = np.zeros(p)
        plus[j] = h
        minus[j] = -h
        w_plus, _ = make_weights(plus, 0.0, cfg)
        w_minus, _ = make_weights(minus, 0.0, cfg)
        numerical[:, j] = (w_plus - w_minus) / (2.0 * h)

    analytical = cfg.A_bound * (np.eye(p) - np.full((p, p), 1.0 / p))
    assert np.allclose(numerical, analytical, rtol=0.0, atol=1e-8)
    assert np.any(np.abs(numerical) > 0.0)


def test_p_one_weight_is_identically_one() -> None:
    cfg = Config()
    for value in (-1e12, -10.0, 0.0, 10.0, 1e12):
        w, _ = make_weights(np.array([value]), value, cfg)
        assert np.array_equal(w, np.array([1.0]))


def test_search_boxes_are_registered_constants() -> None:
    assert A_TILDE_BOX == (-5.0, 5.0)
    assert ETA_BOX == (-10.0, 10.0)
