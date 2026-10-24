from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

from config import Config
import metrics
import rng as rng_keys
from metrics import (
    RESAMPLE_ALPHA,
    RESAMPLE_B,
    RESAMPLE_METHOD,
    RESAMPLE_STRATIFY_BY,
    ResampleMatrix,
    _resample_indices,
    bootstrap_ci,
    bootstrap_ci_from_matrix,
    delta_at,
    descriptive_R,
    divergence_rate,
    holm_bonferroni,
    kalman_flops,
    misg_flops,
    make_resample_matrix,
    paired_across_cells_from_matrix,
    paired_bootstrap,
    paired_bootstrap_from_matrix,
    paired_improvement,
    read_resample_set,
    resample_set_payload,
    resample_set_path,
    risk_R,
    risk_capped,
    system_risk,
    wilcoxon_signed_rank,
    write_resample_set,
)


def _legacy_bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    stat=np.median,
    B: int = 2000,
    alpha: float = 0.05,
    strata: np.ndarray | None = None,
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must contain only finite values")
    if B < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("B must be positive and alpha must lie in (0, 1)")
    if strata is not None and np.asarray(strata).shape != sample.shape:
        raise ValueError("strata must align with values")

    statistics = np.empty(B)
    for replicate in range(B):
        indices = _resample_indices(sample.size, rng, strata=strata)
        statistics[replicate] = float(stat(sample[indices]))
    lower, upper = np.percentile(
        statistics, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )
    return float(lower), float(upper)


def _variant_reseed_each_replicate(
    values: np.ndarray, seed: int, *, B: int
) -> tuple[float, float]:
    statistics = np.empty(B)
    for replicate in range(B):
        local_rng = np.random.default_rng(seed)
        row = _resample_indices(values.size, local_rng)
        statistics[replicate] = float(np.median(values[row]))
    lower, upper = np.percentile(statistics, [2.5, 97.5])
    return float(lower), float(upper)


def _variant_unpaired_point(base: np.ndarray, test: np.ndarray) -> float:
    return float(np.median(base) - np.median(test))


def _i_prime_population() -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(range(10, 20)), (2, 2, 2, 3, 3, 3, 4, 4, 4, 4)


def _i_prime_matrix(
    *, split_code: int = 2, systemset_code: int = 0, B: int = 64
) -> ResampleMatrix:
    ids, strata = _i_prime_population()
    key = rng_keys.resample_matrix_key(split_code, systemset_code)
    return make_resample_matrix(
        ids,
        strata,
        key.entropy,
        split_code=split_code,
        systemset_code=systemset_code,
        B=B,
    )


def test_risk_R_constant_relative_error() -> None:
    rho_star = np.array([1.0, -2.0, 3.0])
    trajectory = np.tile(rho_star * 1.1, (100, 1))
    expected = 0.01 * np.linalg.norm(rho_star) ** 2 / np.linalg.norm(rho_star) ** 2
    assert abs(risk_R(trajectory, rho_star, N=100) - expected) < 1e-12


def test_delta_at_is_single_time_unsquared_percent() -> None:
    rho_star = np.array([1.0, -2.0, 3.0])
    trajectory = np.tile(rho_star * 1.1, (100, 1))
    risk = risk_R(trajectory, rho_star, N=100)
    delta = delta_at(trajectory, rho_star, k=100)
    assert abs(delta - 10.0) < 1e-12

    assert abs(delta / 100.0 - np.sqrt(risk)) < 1e-12


def test_bootstrap_ci_for_normal_median_has_plausible_width() -> None:
    rng = np.random.default_rng(12345)
    values = rng.normal(0.0, 1.0, 10_000)
    ci_lo, ci_hi = bootstrap_ci(values, rng)
    width = ci_hi - ci_lo
    assert 0.02 <= width <= 0.12


def test_stratified_resampling_preserves_exact_counts() -> None:
    strata = np.asarray([2] * 67 + [3] * 67 + [4] * 67)
    rng = np.random.default_rng(54321)
    for _ in range(100):
        indices = _resample_indices(strata.size, rng, strata=strata)
        sampled_labels = strata[indices]
        assert np.sum(sampled_labels == 2) == 67
        assert np.sum(sampled_labels == 3) == 67
        assert np.sum(sampled_labels == 4) == 67
        for label in (2, 3, 4):
            source_members = set(np.flatnonzero(strata == label))
            sampled_members = set(indices[sampled_labels == label])
            assert sampled_members <= source_members


def test_stratified_resampling_supports_expansion() -> None:
    strata = np.asarray([2] * 17 + [3] * 17 + [4] * 17)
    indices = _resample_indices(
        strata.size,
        np.random.default_rng(321),
        strata=strata,
        size_per_stratum=67,
    )
    sampled = strata[indices]
    assert indices.size == 201
    assert [int(np.sum(sampled == label)) for label in (2, 3, 4)] == [
        67,
        67,
        67,
    ]


def test_paired_bootstrap_resamples_one_shared_system_index() -> None:
    base = np.linspace(1.0, 10.0, 201)
    test = base * 0.8
    result = paired_bootstrap(base, test, np.random.default_rng(456), B=500)
    assert abs(float(result["point"]) - 0.2) < 1e-15
    assert float(result["ci_hi"]) - float(result["ci_lo"]) < 1e-9

    rng = np.random.default_rng(456)
    independent = np.empty(500)
    for replicate in range(independent.size):
        base_idx = rng.integers(0, base.size, size=base.size)
        test_idx = rng.integers(0, test.size, size=test.size)
        independent[replicate] = np.median(
            (base[base_idx] - test[test_idx]) / base[base_idx]
        )
    independent_width = np.percentile(independent, 97.5) - np.percentile(
        independent, 2.5
    )
    assert independent_width > 1e-3


def test_system_risk_is_always_defined() -> None:
    cfg = Config()
    assert abs(system_risk({3: 0.2, 4: 0.4}, {3: "ok", 4: "ok"}, cfg) - 0.3) < 1e-15
    assert (
        abs(system_risk({3: 0.2, 4: 0.4}, {3: "ok", 4: "diverged"}, cfg) - 5.1) < 1e-15
    )
    assert (
        system_risk(
            {3: np.nan, 4: np.nan},
            {3: "diverged", 4: "diverged"},
            cfg,
        )
        == cfg.r_cap
    )
    assert abs(system_risk({3: 50.0, 4: 0.2}, {3: "ok", 4: "ok"}, cfg) - 5.1) < 1e-15


def test_descriptive_statistics_are_isolated_from_inference() -> None:
    cfg = Config()
    risks = np.array([0.2, np.nan, 50.0])
    statuses = ["ok", "diverged", "ok"]
    description = descriptive_R(risks, statuses, cfg)
    assert description["median_R_ok"] == 25.1
    assert description["n_ok"] == 2
    assert description["divergence_rate"] == 1.0 / 3.0
    assert description["n_capped"] == 2

    base = np.array([0.2, cfg.r_cap, cfg.r_cap])
    test = np.array([0.1, cfg.r_cap, 5.0])
    inference = paired_bootstrap(base, test, np.random.default_rng(654), B=200)
    assert inference["n_systems"] == 3


def test_wilcoxon_fixed_seed_cases() -> None:
    rng = np.random.default_rng(777)
    a = rng.normal(0.0, 1.0, 30)
    _, strong_p = wilcoxon_signed_rank(a, a + 5.0)
    _, null_p = wilcoxon_signed_rank(a, a + rng.normal(0.0, 1e-6, 30))

    assert strong_p < 1e-5
    assert null_p > 0.05


def test_holm_bonferroni_manual_example() -> None:
    reject, adjusted = holm_bonferroni(np.array([0.01, 0.02, 0.03, 0.04]), alpha=0.05)
    assert np.array_equal(reject, np.array([True, False, False, False]))
    assert np.allclose(
        adjusted,
        np.array([0.04, 0.06, 0.06, 0.06]),
        rtol=0.0,
        atol=0.0,
    )


def test_flop_counts() -> None:
    assert misg_flops(6, 1) == 37
    assert misg_flops(6, 5) == 137
    mz = 6
    assert kalman_flops(mz) == 2 * mz**3 + 5 * mz**2 + 2 * mz


def test_risk_capped_cases() -> None:
    cfg = Config()
    assert risk_capped(np.nan, cfg) == cfg.r_cap
    assert risk_capped(3.0, cfg) == 3.0
    assert risk_capped(50.0, cfg) == cfg.r_cap


def test_risk_nan_and_divergence_rate() -> None:
    rho_star = np.array([1.0, 2.0])
    trajectory = np.tile(rho_star, (20, 1))
    trajectory[-1, 0] = np.nan
    assert np.isnan(risk_R(trajectory, rho_star, N=20))
    assert divergence_rate(["ok", "diverged", "ok", "diverged"]) == 0.5


def test_paired_improvement_manual_values() -> None:
    base = np.array([1.0, 2.0, 4.0])
    test = np.array([0.8, 1.0, 5.0])
    assert np.allclose(
        paired_improvement(base, test),
        np.array([0.2, 0.5, -0.25]),
        rtol=0.0,
        atol=1e-15,
    )


def test_i_prime_shared_matrix_across_methods_and_cells() -> None:
    ids, _ = _i_prime_population()
    method_a = _i_prime_matrix(B=96)
    method_b = _i_prime_matrix(B=96)
    values_a = np.linspace(1.0, 10.0, len(ids))
    values_b = values_a[::-1].copy()

    ci_a = bootstrap_ci_from_matrix(values_a, method_a, ids)
    ci_b = bootstrap_ci_from_matrix(values_b, method_b, ids)
    assert len(ci_a) == len(ci_b) == 2
    assert np.array_equal(method_a.matrix, method_b.matrix)
    assert method_a.sha256_system_ids == method_b.sha256_system_ids
    assert method_a.sha256_strata_d == method_b.sha256_strata_d


def test_i_prime_legal_key_change_has_official_golden() -> None:
    assert np.__version__ == "2.4.2"
    full = _i_prime_matrix(split_code=2, systemset_code=0, B=32)
    survivors = _i_prime_matrix(split_code=2, systemset_code=1, B=32)
    assert full.matrix[0, :8].tolist() == [0, 2, 1, 3, 4, 4, 6, 7]
    assert survivors.matrix[0, :8].tolist() == [1, 1, 1, 4, 4, 3, 8, 9]
    assert not np.array_equal(full.matrix, survivors.matrix)


def test_i_prime_shared_rng_advances_between_replicates_golden() -> None:
    assert np.__version__ == "2.4.2"
    full = _i_prime_matrix(split_code=2, systemset_code=0, B=32)
    survivors = _i_prime_matrix(split_code=2, systemset_code=1, B=32)

    assert full.matrix[1, :8].tolist() == [2, 1, 2, 3, 3, 5, 9, 9]
    assert survivors.matrix[1, :8].tolist() == [0, 2, 1, 4, 5, 4, 8, 9]


def test_i_prime_matrix_ci_uses_two_sided_alpha() -> None:
    assert np.__version__ == "2.4.2"
    values = np.arange(100.0)
    indices = np.arange(100, dtype=int).reshape(100, 1)

    ci = metrics._ci_from_indices(values, indices, np.mean, 0.05)

    assert ci == (2.475, 96.52499999999999)


def test_i_prime_paired_bootstrap_point_is_median_of_system_improvements() -> None:
    ids, _ = _i_prime_population()
    matrix = _i_prime_matrix(B=32)
    base = np.asarray([1, 2, 3, 10, 1, 2, 3, 10, 1, 10], dtype=float)
    test = np.asarray([0, 0, 0, 9, 0, 0, 0, 9, 0, 9], dtype=float)

    result = paired_bootstrap_from_matrix(base, test, matrix, ids)

    assert result["point"] == 1.0
    assert float(np.median(base) - np.median(test)) == 2.5


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("system_ids", "sha256_system_ids"),
        ("strata_d", "sha256_strata_d"),
    ],
)
def test_i_prime_in_memory_hash_tamper_is_rejected(field: str, message: str) -> None:
    matrix = _i_prime_matrix(B=24)
    if field == "system_ids":
        tampered_ids = list(matrix.system_ids)
        tampered_ids[:2] = tampered_ids[1::-1]
        tampered = dataclasses.replace(matrix, system_ids=tuple(tampered_ids))
    else:
        tampered_strata = list(matrix.strata_d)
        tampered_strata[0] += 1
        tampered = dataclasses.replace(matrix, strata_d=tuple(tampered_strata))

    values = np.linspace(0.0, 1.0, len(tampered.system_ids))
    with pytest.raises(ValueError, match=message):
        bootstrap_ci_from_matrix(values, tampered, tampered.system_ids)


@pytest.mark.parametrize(
    "consumer",
    [
        bootstrap_ci_from_matrix,
        paired_bootstrap_from_matrix,
        paired_across_cells_from_matrix,
    ],
)
def test_i_prime_consumer_stat_defaults_are_median(consumer) -> None:
    assert inspect.signature(consumer).parameters["stat"].default is np.median


def test_i_prime_hash_format_has_literal_goldens() -> None:
    ids, strata = _i_prime_population()

    assert metrics._sha256_int_list(ids) == (
        "8a04a7f0ef0645599fcf0a0518057466832d620fd673cd727fd004e62d19c109"
    )
    assert metrics._sha256_int_list(strata) == (
        "3ccafea3a5dcf9a56ec1c92e4c2708d94bfed8b6fdb31fa64cdd1de5150416db"
    )


def test_i_prime_write_resample_set_uses_tmp_then_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _i_prime_matrix(B=RESAMPLE_B)
    output_path = resample_set_path(tmp_path, 2, 0)
    temporary_path = Path(f"{output_path}.tmp")
    events = []
    real_write_text = Path.write_text
    real_replace = metrics.os.replace

    def record_write_text(path, *args, **kwargs):
        events.append(("write", Path(path)))
        return real_write_text(path, *args, **kwargs)

    def record_replace(src, dst):
        events.append(("replace", Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(Path, "write_text", record_write_text)
    monkeypatch.setattr(metrics.os, "replace", record_replace)

    written = write_resample_set(matrix, tmp_path)

    assert written == output_path
    assert events == [
        ("write", temporary_path),
        ("replace", temporary_path, output_path),
    ]
    assert output_path.exists()
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    "strata",
    [None, np.asarray([2, 2, 3, 3, 4, 4], dtype=int)],
    ids=["unstratified", "stratified"],
)
def test_i_prime_legacy_bootstrap_is_bitwise_unchanged(
    strata: np.ndarray | None,
) -> None:
    values = np.asarray([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    current = bootstrap_ci(values, np.random.default_rng(12345), B=256, strata=strata)
    legacy = _legacy_bootstrap_ci(
        values, np.random.default_rng(12345), B=256, strata=strata
    )
    assert np.array_equal(np.asarray(current), np.asarray(legacy))


def test_i_prime_legacy_oracle_detects_reseed_each_replicate() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 10.0, 30.0])
    reference = _legacy_bootstrap_ci(values, np.random.default_rng(12345), B=256)
    broken = _variant_reseed_each_replicate(values, 12345, B=256)
    assert reference[0] < reference[1]
    assert broken[0] == broken[1]
    assert not np.array_equal(np.asarray(reference), np.asarray(broken))


def test_i_prime_legacy_oracle_detects_unpaired_point() -> None:
    base = np.asarray([1.0, 2.0, 3.0, 10.0])
    test = np.asarray([0.0, 0.0, 0.0, 9.0])
    paired = float(np.median(base - test))
    unpaired = _variant_unpaired_point(base, test)
    assert paired == 1.5
    assert unpaired == 2.5
    assert paired != unpaired


def test_i_prime_legacy_oracle_detects_dropped_strata_golden() -> None:
    assert np.__version__ == "2.4.2"
    values = np.asarray([1e6] + [0.0] * 99)
    strata = np.asarray([2] + [3] * 99)
    stratified = _legacy_bootstrap_ci(
        values,
        np.random.default_rng(20260729),
        stat=np.mean,
        B=2000,
        strata=strata,
    )
    no_strata = _legacy_bootstrap_ci(
        values,
        np.random.default_rng(20260729),
        stat=np.mean,
        B=2000,
        strata=None,
    )
    assert stratified == (10000.0, 10000.0)
    assert no_strata == (0.0, 30000.0)
    assert no_strata[0] < no_strata[1]


def test_i_prime_matrix_is_exactly_recreatable() -> None:
    first = _i_prime_matrix(B=80)
    second = _i_prime_matrix(B=80)
    assert np.array_equal(first.matrix, second.matrix)
    assert first == dataclasses.replace(second, matrix=first.matrix)


def test_i_prime_matrix_is_independent_of_call_order() -> None:
    def build(order: tuple[str, str]) -> dict[str, ResampleMatrix]:
        result: dict[str, ResampleMatrix] = {}
        for method in order:
            result[method] = _i_prime_matrix(B=72)
        return result

    a_then_b = build(("A", "B"))
    b_then_a = build(("B", "A"))
    assert np.array_equal(a_then_b["A"].matrix, b_then_a["A"].matrix)
    assert np.array_equal(a_then_b["B"].matrix, b_then_a["B"].matrix)


def test_i_prime_every_row_preserves_all_stratum_counts() -> None:
    _, strata_tuple = _i_prime_population()
    strata = np.asarray(strata_tuple)
    matrix = _i_prime_matrix(B=90)
    sampled_labels = strata[matrix.matrix]
    unique = np.unique(strata)
    assert len(np.unique(sampled_labels)) == len(unique)
    for row in sampled_labels:
        for label in unique:
            assert int(np.sum(row == label)) == int(np.sum(strata == label))
    collapsed = np.full_like(strata, unique[0])
    assert len(np.unique(collapsed)) != len(unique)


def test_i_prime_consumers_reject_reordered_system_ids() -> None:
    ids, _ = _i_prime_population()
    reordered = ids[::-1]
    matrix = _i_prime_matrix(B=24)
    values = np.linspace(1.0, 2.0, len(ids))
    with pytest.raises(ValueError):
        bootstrap_ci_from_matrix(values, matrix, reordered)
    with pytest.raises(ValueError):
        paired_bootstrap_from_matrix(values + 1.0, values, matrix, reordered)
    with pytest.raises(ValueError):
        paired_across_cells_from_matrix(
            values, values / 2.0, matrix, reordered, reordered
        )


def test_i_prime_matrix_shape_dtype_and_domain() -> None:
    ids, _ = _i_prime_population()
    matrix = _i_prime_matrix(B=47)
    assert matrix.matrix.shape == (47, len(ids))
    assert np.issubdtype(matrix.matrix.dtype, np.integer)
    assert int(np.min(matrix.matrix)) >= 0
    assert int(np.max(matrix.matrix)) < len(ids)


def test_i_prime_dataclass_and_metadata_match_d10() -> None:
    expected = {
        "matrix",
        "split_code",
        "systemset_code",
        "system_ids",
        "strata_d",
        "sha256_system_ids",
        "sha256_strata_d",
        "entropy",
        "B",
        "method",
        "stratify_by",
    }
    field_names = {field.name for field in dataclasses.fields(ResampleMatrix)}
    assert field_names == expected
    assert "statistic" not in field_names
    matrix = _i_prime_matrix(B=2000)
    key = rng_keys.resample_matrix_key(2, 0)
    assert matrix.B == 2000
    assert matrix.method == "percentile"
    assert matrix.stratify_by == "d"
    assert matrix.entropy == tuple(key.entropy)


def test_i_prime_resample_set_roundtrip(tmp_path: Path) -> None:
    matrix = _i_prime_matrix(B=RESAMPLE_B)
    logs_dir = tmp_path / "logs"
    path = write_resample_set(matrix, logs_dir)
    expected_path = logs_dir / "resample_sets" / "2_0.json"
    assert path == expected_path
    assert path == resample_set_path(logs_dir, 2, 0)

    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert (
        raw
        == json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )
    assert set(payload) == {
        "split_code",
        "systemset_code",
        "system_ids",
        "strata_d",
        "sha256_system_ids",
        "sha256_strata_d",
        "entropy",
        "B",
        "method",
        "stratify_by",
    }
    assert "matrix" not in payload
    assert payload == resample_set_payload(matrix)
    rebuilt = read_resample_set(path)
    assert np.array_equal(rebuilt.matrix, matrix.matrix)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("system_ids", "sha256_system_ids"),
        ("strata_d", "sha256_strata_d"),
        ("method", "method"),
        ("B", "B"),
        ("stratify_by", "stratify_by"),
    ],
)
def test_i_prime_tamper_fails_before_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    matrix = _i_prime_matrix(B=RESAMPLE_B)
    path = write_resample_set(matrix, tmp_path / "logs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "system_ids":
        changed = list(payload["system_ids"])
        changed[0], changed[1] = changed[1], changed[0]
        assert metrics._sha256_int_list(changed) != payload["sha256_system_ids"]
        payload["system_ids"] = changed
    elif mutation == "strata_d":
        changed = list(payload["strata_d"])
        changed[0] = 99
        assert metrics._sha256_int_list(changed) != payload["sha256_strata_d"]
        payload["strata_d"] = changed
    elif mutation == "method":
        payload["method"] = "bca"
        assert payload["method"] != RESAMPLE_METHOD
    elif mutation == "B":
        payload["B"] = 1000
        assert payload["B"] != RESAMPLE_B
    else:
        payload["stratify_by"] = "p"
        assert payload["stratify_by"] != RESAMPLE_STRATIFY_BY
    path.write_text(json.dumps(payload), encoding="utf-8")

    def rebuild_must_not_run(*args, **kwargs):
        raise RuntimeError("rebuild ran before metadata validation")

    monkeypatch.setattr(metrics, "make_resample_matrix", rebuild_must_not_run)
    with pytest.raises(ValueError, match=message):
        read_resample_set(path)


def test_i_prime_nonregistered_B_cannot_be_written(tmp_path: Path) -> None:
    matrix = _i_prime_matrix(B=1000)
    with pytest.raises(ValueError, match="B=2000"):
        write_resample_set(matrix, tmp_path / "logs")


def test_i_prime_constants_and_signature_defaults_are_frozen() -> None:
    assert RESAMPLE_B == 2000
    assert RESAMPLE_METHOD == "percentile"
    assert RESAMPLE_ALPHA == 0.05
    assert RESAMPLE_STRATIFY_BY == "d"
    make_signature = inspect.signature(make_resample_matrix)
    B_parameter = make_signature.parameters["B"]
    assert B_parameter.kind is inspect.Parameter.KEYWORD_ONLY

    assert B_parameter.default is RESAMPLE_B
    assert B_parameter.default == RESAMPLE_B
    ids, strata = _i_prime_population()
    default_matrix = make_resample_matrix(
        ids,
        strata,
        rng_keys.resample_matrix_key(2, 0).entropy,
        split_code=2,
        systemset_code=0,
    )
    assert default_matrix.B == RESAMPLE_B
    for function in (
        bootstrap_ci_from_matrix,
        paired_bootstrap_from_matrix,
        paired_across_cells_from_matrix,
    ):
        alpha_default = inspect.signature(function).parameters["alpha"].default
        assert alpha_default is RESAMPLE_ALPHA
        assert alpha_default == RESAMPLE_ALPHA


def test_i_prime_did_pairs_s2_minus_s1_by_system() -> None:
    ids, _ = _i_prime_population()
    matrix = _i_prime_matrix(B=128)
    values_s2 = np.asarray([1.0, 2.0, 3.0, 10.0, 8.0, 5.0, 7.0, 9.0, 4.0, 6.0])
    values_s1 = np.asarray([0.0, 0.0, 0.0, 9.0, 3.0, 1.0, 6.0, 3.0, 2.0, 7.0])
    result = paired_across_cells_from_matrix(values_s2, values_s1, matrix, ids, ids)
    assert np.array_equal(matrix.matrix, matrix.matrix)
    assert tuple(ids) == matrix.system_ids == tuple(ids)
    assert result["point"] == float(np.median(values_s2 - values_s1))


def test_i_prime_median_of_differences_is_not_difference_of_medians() -> None:
    a = np.asarray([1.0, 2.0, 3.0, 10.0])
    b = np.asarray([0.0, 0.0, 0.0, 9.0])
    paired = float(np.median(a - b))
    separate = float(np.median(a) - np.median(b))
    assert paired == 1.5
    assert separate == 2.5
    assert paired != separate


def test_i_prime_did_rejects_second_cell_reordering() -> None:
    ids, _ = _i_prime_population()
    matrix = _i_prime_matrix(B=32)
    values = np.arange(len(ids), dtype=float)
    with pytest.raises(ValueError):
        paired_across_cells_from_matrix(values, values + 1.0, matrix, ids, ids[::-1])


def test_i_prime_p8_minus_p5_pairs_by_system_independently() -> None:
    ids, _ = _i_prime_population()
    matrix = _i_prime_matrix(B=128)
    values_p8 = np.asarray([8.0, 2.0, 7.0, 1.0, 9.0, 3.0, 6.0, 4.0, 10.0, 5.0])
    values_p5 = np.asarray([5.0, 1.0, 4.0, 2.0, 3.0, 3.0, 7.0, 0.0, 8.0, 6.0])
    result = paired_across_cells_from_matrix(values_p8, values_p5, matrix, ids, ids)
    assert np.array_equal(matrix.matrix, matrix.matrix)
    assert tuple(ids) == matrix.system_ids
    assert result["point"] == float(np.median(values_p8 - values_p5))


def test_i_prime_metrics_ast_hygiene() -> None:
    source = Path(metrics.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_mutables = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, (ast.List, ast.Dict, ast.Set)):
                module_mutables.append(node)
    assert module_mutables == []
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "run_e1" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "run_e1"
