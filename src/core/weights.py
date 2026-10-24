from __future__ import annotations

from typing import Protocol

import numpy as np


A_TILDE_BOX = (-5.0, 5.0)
ETA_BOX = (-10.0, 10.0)


class WeightConfig(Protocol):
    A_bound: float
    eps: float


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def eta0(eps: float) -> float:
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between 0 and 1")
    probability = 1.0 / (2.0 - eps)
    return float(np.log(probability / (1.0 - probability)))


def make_weights(
    a_tilde: np.ndarray, eta: float, cfg: WeightConfig
) -> tuple[np.ndarray, float]:
    coordinates = np.asarray(a_tilde, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("a_tilde must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("a_tilde must contain only finite values")
    if not np.isfinite(eta):
        raise ValueError("eta must be finite")
    if cfg.A_bound <= 0.0:
        raise ValueError("A_bound must be positive")
    if not 0.0 < cfg.eps < 1.0:
        raise ValueError("eps must lie strictly between 0 and 1")

    p = coordinates.size
    a = cfg.A_bound * np.tanh(coordinates)
    shifted = a - np.max(a)
    exponentials = np.exp(shifted)
    w = p * exponentials / np.sum(exponentials)
    gamma = (2.0 - cfg.eps) / np.max(w) * _sigmoid(float(eta))
    return w, float(gamma)


def w_envelope(p: int, A_bound: float) -> tuple[float, float]:
    if p < 1:
        raise ValueError("p must be positive")
    if A_bound <= 0.0:
        raise ValueError("A_bound must be positive")
    exp_span = np.exp(2.0 * A_bound)
    w_min = p / (1.0 + (p - 1) * exp_span)
    w_max = p * exp_span / (exp_span + p - 1)
    return float(w_min), float(w_max)
