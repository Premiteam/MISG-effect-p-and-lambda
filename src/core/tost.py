from __future__ import annotations

from collections.abc import Callable

import numpy as np

from metrics import bootstrap_ci, paired_improvement


Statistic = Callable[[np.ndarray], float]


def paired_equivalence(
    base: np.ndarray,
    test: np.ndarray,
    rng: np.random.Generator,
    *,
    delta: float = 0.05,
    stat: Statistic = np.median,
    B: int = 2000,
    alpha: float = 0.05,
    strata: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    if delta < 0.0 or not np.isfinite(delta):
        raise ValueError("delta must be finite and non-negative")
    improvements = paired_improvement(base, test)
    point = float(stat(improvements))
    ci_lo, ci_hi = bootstrap_ci(
        improvements, rng, stat=stat, B=B, alpha=alpha, strata=strata
    )
    equivalent = bool(ci_lo > -delta and ci_hi < delta)
    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "delta": float(delta),
        "equivalent": equivalent,
        "n_systems": int(improvements.size),
    }


def _resample_with_strata(
    values: np.ndarray,
    rng: np.random.Generator,
    strata: np.ndarray | None,
    size_per_stratum: int | None,
) -> np.ndarray:
    n = values.size
    if strata is None:
        return values[rng.integers(0, n, size=n)]
    labels = np.asarray(strata)
    if labels.shape != values.shape:
        raise ValueError("strata must align with values")
    samples = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        draw_size = members.size if size_per_stratum is None else size_per_stratum
        samples.append(values[rng.choice(members, size=draw_size, replace=True)])
    return np.concatenate(samples)


def simulate_equivalence(
    delta_sample: np.ndarray,
    n: int,
    strata_pilot: np.ndarray,
    rng_factory: Callable[[int], np.random.Generator],
    *,
    delta: float = 0.05,
    n_rep: int = 2000,
    B: int = 500,
    size_per_stratum: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(delta_sample, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("delta_sample must be a finite non-empty vector")
    if np.asarray(strata_pilot).shape != sample.shape:
        raise ValueError("strata_pilot must align with delta_sample")
    if n < 1 or n_rep < 1 or B < 1:
        raise ValueError("n, n_rep, and B must be positive")

    centered = sample - np.median(sample)
    strata_out = np.repeat(
        np.unique(np.asarray(strata_pilot)),
        n // len(np.unique(np.asarray(strata_pilot))),
    )
    if strata_out.size != n:
        raise ValueError("n must be divisible by the number of strata")

    interval_half_widths = np.empty(n_rep)
    for rep in range(n_rep):
        rng = rng_factory(rep)
        resampled = _resample_with_strata(centered, rng, strata_pilot, size_per_stratum)
        bootstrap_stats = np.empty(B)
        for b in range(B):
            bootstrap_stats[b] = np.median(
                _resample_with_strata(resampled, rng, strata_out, None)
            )
        lo = np.percentile(bootstrap_stats, 2.5)
        hi = np.percentile(bootstrap_stats, 97.5)
        interval_half_widths[rep] = max(abs(lo), abs(hi))
    return interval_half_widths, strata_out


def equivalence_power_grid(
    half_widths: np.ndarray,
    delta: float,
    effect_grid: np.ndarray,
) -> np.ndarray:
    widths = np.asarray(half_widths, dtype=float)
    grid = np.asarray(effect_grid, dtype=float)
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("half_widths must be a non-empty vector")
    return np.asarray(
        [float(np.mean((widths + abs(effect)) < delta)) for effect in grid]
    )
