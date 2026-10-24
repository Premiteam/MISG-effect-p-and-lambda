from __future__ import annotations

import math

import numpy as np
import pytest

cma = pytest.importorskip("cma", reason="D3 需要 cma；见 code/environment.yml")

from scipy.stats import qmc

DIMS = (1, 2, 6)
SIGMA0 = 0.3
BOX = [0, 1]


@pytest.fixture(scope="module")
def bt() -> "cma.BoundTransform":
    return cma.BoundTransform(BOX)


def _make_es(n: int, seed: int = 12345) -> "cma.CMAEvolutionStrategy":
    return cma.CMAEvolutionStrategy([0.5] * n, SIGMA0, {"seed": seed, "verbose": -9})


def test_builtin_boundary_handler_is_broken_at_n1() -> None:
    es = cma.CMAEvolutionStrategy(
        [0.5],
        SIGMA0,
        {
            "bounds": BOX,
            "BoundaryHandler": cma.BoundTransform,
            "seed": 1,
            "verbose": -9,
        },
    )
    X = es.ask()
    with pytest.raises(ValueError, match="not yet initialized"):
        es.tell(X, [float(np.asarray(x)[0]) for x in X])


def test_bound_transform_image_covers_the_whole_box(bt) -> None:
    xs = np.linspace(-200.0, 200.0, 200_001)
    ys = np.array([bt.repair(np.array([x]))[0] for x in xs])
    assert ys.min() <= 1e-9, f"下端点不可达，最小像 {ys.min()}"
    assert ys.max() >= 1.0 - 1e-9, f"上端点不可达，最大像 {ys.max()}"
    assert np.all(ys >= 0.0) and np.all(ys <= 1.0)


def test_bound_transform_is_not_idempotent(bt) -> None:
    once = np.asarray(bt.repair(np.array([0.0])))
    twice = np.asarray(bt.repair(once.copy()))
    assert not np.array_equal(once, twice), "repair 竟然是幂等的？D3 §3 的告诫可以放宽"
    assert math.isclose(float(once[0]), 0.0125, rel_tol=1e-9)


def test_box_centre_is_a_fixed_point(bt) -> None:
    for n in DIMS:
        centre = np.full(n, 0.5)
        assert np.allclose(np.asarray(bt.repair(centre.copy())), centre, atol=0, rtol=0)


@pytest.mark.parametrize("n", DIMS)
def test_ask_tell_invariant_and_evaluated_points_in_box(n: int, bt) -> None:
    es = _make_es(n)

    for _ in range(3):
        X = es.ask()
        X_before = [np.array(x, copy=True) for x in X]

        P = [np.asarray(bt.repair(np.array(x, copy=True))) for x in X]
        for p in P:
            assert np.all(p >= 0.0) and np.all(p <= 1.0), "评估点落在箱外"

        es.tell(X, [float(np.sum((p - 0.9) ** 2)) for p in P])

        for before, after in zip(X_before, X):
            assert np.array_equal(before, after), "tell 修改了 ask 返回的点"

    assert es.countevals == 3 * es.popsize


@pytest.mark.parametrize("n", DIMS)
def test_optimum_at_upper_endpoint_is_reachable(n: int, bt) -> None:
    gens = {1: 75, 2: 50, 6: 33}[n]
    es = _make_es(n, seed=11)
    for _ in range(gens):
        X = es.ask()
        P = [np.asarray(bt.repair(np.array(x, copy=True))) for x in X]
        es.tell(X, [float(np.sum((p - 1.0) ** 2)) for p in P])
    best = np.asarray(bt.repair(np.asarray(es.result.xbest).copy()))
    assert np.all(best > 0.99), f"n={n} 未能逼近上端点：{best}"


@pytest.mark.parametrize(("n", "expected"), [(1, 4), (2, 6), (6, 9)])
def test_default_popsize_matches_formula(n: int, expected: int) -> None:
    assert _make_es(n).popsize == expected == 4 + math.floor(3 * math.log(n))


def test_sobol_rejects_seedsequence_but_accepts_generator() -> None:
    with pytest.raises(ValueError):
        qmc.Sobol(d=2, scramble=True, seed=np.random.SeedSequence([1, 2])).random(2)

    def points() -> np.ndarray:
        ss = np.random.SeedSequence([20260729, 10, 3, 1])
        return qmc.Sobol(d=2, scramble=True, seed=np.random.default_rng(ss)).random(2)

    p1, p2 = points(), points()
    assert np.array_equal(p1, p2), "同一 SeedSequence 两次构造的 Sobol 点不逐位相同"
    assert np.all(p1 >= 0.0) and np.all(p1 < 1.0)


@pytest.mark.parametrize("n", DIMS)
def test_sobol_points_are_not_the_box_centre(n: int) -> None:
    ss = np.random.SeedSequence([20260729, 10, 7, 2])
    pts = qmc.Sobol(d=n, scramble=True, seed=np.random.default_rng(ss)).random(2)
    centre = np.full(n, 0.5)
    for i, p in enumerate(pts):
        assert np.linalg.norm(p - centre) > 1e-9, f"Sobol 第 {i} 点与箱中心重合"


class SearchFailed(RuntimeError):
    pass


def _clamp_R(R: float, R_cap: float) -> float:
    if np.isnan(R) or R == np.inf:
        return R_cap
    if R < 0:
        raise SearchFailed("negative_R")
    return min(R, R_cap)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(np.nan, 10.0), (np.inf, 10.0), (20.0, 10.0), (5.0, 5.0), (0.0, 0.0)],
)
def test_clamp_R_maps_nonfinite_to_cap(value: float, expected: float) -> None:
    assert _clamp_R(value, 10.0) == expected


@pytest.mark.parametrize("value", [-np.inf, -1e-12, -1.0])
def test_clamp_R_raises_on_negative(value: float) -> None:
    with pytest.raises(SearchFailed):
        _clamp_R(value, 10.0)


def test_numpy_minimum_would_let_nan_through() -> None:
    assert np.isnan(np.minimum(np.nan, 10.0)), "np.minimum 竟然没让 NaN 穿透？"
    assert _clamp_R(np.nan, 10.0) == 10.0
