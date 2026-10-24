from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


RESAMPLE_B: int = 2000
RESAMPLE_METHOD: str = "percentile"
RESAMPLE_ALPHA: float = 0.05
RESAMPLE_STRATIFY_BY: str = "d"
RESAMPLE_SETS_DIRNAME = "resample_sets"

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class ResampleMatrix:
    matrix: np.ndarray
    split_code: int
    systemset_code: int
    system_ids: tuple[int, ...]
    strata_d: tuple[int, ...]
    sha256_system_ids: str
    sha256_strata_d: str
    entropy: tuple[int, ...]
    B: int
    method: str
    stratify_by: str


def risk_R(rho_hat_traj: np.ndarray, rho_star: np.ndarray, N: int) -> float:
    estimates = np.asarray(rho_hat_traj, dtype=float)
    truth = np.asarray(rho_star, dtype=float)
    if estimates.ndim != 2 or truth.ndim != 1:
        raise ValueError("rho_hat_traj must be 2D and rho_star must be 1D")
    if estimates.shape[1] != truth.size or estimates.shape[0] < N or N < 1:
        raise ValueError("trajectory, truth, and N have incompatible shapes")
    if not np.all(np.isfinite(estimates[:N])):
        return float(np.nan)
    denominator = float(np.linalg.norm(truth) ** 2)
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("rho_star must have a finite nonzero norm")

    evaluation = estimates[N // 2 : N]
    squared_errors = np.sum((evaluation - truth) ** 2, axis=1)
    return float(np.mean(squared_errors) / denominator)


def delta_at(rho_hat_traj: np.ndarray, rho_star: np.ndarray, k: int) -> float:
    estimates = np.asarray(rho_hat_traj, dtype=float)
    truth = np.asarray(rho_star, dtype=float)
    if estimates.ndim != 2 or truth.ndim != 1:
        raise ValueError("rho_hat_traj must be 2D and rho_star must be 1D")
    if estimates.shape[1] != truth.size or not 1 <= k <= estimates.shape[0]:
        raise ValueError("k or array shapes are invalid")
    estimate = estimates[k - 1]
    if not np.all(np.isfinite(estimate)):
        return float(np.nan)
    denominator = float(np.linalg.norm(truth))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("rho_star must have a finite nonzero norm")
    return float(np.linalg.norm(estimate - truth) / denominator * 100.0)


def risk_capped(R: float, cfg: Any) -> float:
    value = float(R)
    if not np.isfinite(value):
        return float(cfg.r_cap)
    return float(min(value, cfg.r_cap))


def system_risk(
    R_traj: Mapping[int, float], statuses: Mapping[int, str], cfg: Any
) -> float:
    if not R_traj or set(R_traj) != set(statuses):
        raise ValueError("R_traj and statuses must have the same non-empty keys")
    capped: list[float] = []
    for trajectory_id in R_traj:
        risk = float(R_traj[trajectory_id])
        if statuses[trajectory_id] == "diverged" or not np.isfinite(risk):
            capped.append(float(cfg.r_cap))
        else:
            capped.append(float(min(risk, cfg.r_cap)))
    return float(np.mean(capped))


def paired_improvement(base: np.ndarray, test: np.ndarray) -> np.ndarray:
    baseline = np.asarray(base, dtype=float)
    candidate = np.asarray(test, dtype=float)
    if baseline.ndim != 1 or candidate.ndim != 1 or baseline.shape != candidate.shape:
        raise ValueError("base and test must be aligned one-dimensional arrays")
    if np.any(baseline <= 0.0):
        raise ValueError("all baseline system risks must be strictly positive")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("base and test must contain only finite values")
    return (baseline - candidate) / baseline


def _resample_indices(
    n: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
    size_per_stratum: int | None = None,
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    if strata is None:
        if size_per_stratum is not None:
            raise ValueError("size_per_stratum requires strata")
        return rng.integers(0, n, size=n)

    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size != n:
        raise ValueError("strata must be one-dimensional and aligned with n items")
    if size_per_stratum is not None and size_per_stratum < 1:
        raise ValueError("size_per_stratum must be positive")

    sampled: list[np.ndarray] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        draw_size = members.size if size_per_stratum is None else size_per_stratum
        sampled.append(rng.choice(members, size=draw_size, replace=True))
    return np.concatenate(sampled)


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    stat: Statistic = np.median,
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


def paired_bootstrap(
    base: np.ndarray,
    test: np.ndarray,
    rng: np.random.Generator,
    stat: Statistic = np.median,
    B: int = 2000,
    alpha: float = 0.05,
    strata: np.ndarray | None = None,
) -> dict[str, float | int]:
    improvements = paired_improvement(base, test)
    point = float(stat(improvements))
    ci_lo, ci_hi = bootstrap_ci(
        improvements, rng, stat=stat, B=B, alpha=alpha, strata=strata
    )
    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_systems": improvements.size,
    }


def _canonical_int_tuple(
    values: Sequence[int], name: str, *, nonempty: bool = True
) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer sequence") from exc
    if nonempty and not items:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in items
    ):
        raise ValueError(f"{name} must contain only integers")
    return tuple(int(value) for value in items)


def _validate_resample_codes(split_code: int, systemset_code: int) -> tuple[int, int]:
    if isinstance(split_code, bool) or not isinstance(split_code, Integral):
        raise ValueError("split_code must be an integer")
    if isinstance(systemset_code, bool) or not isinstance(systemset_code, Integral):
        raise ValueError("systemset_code must be an integer")
    split = int(split_code)
    systemset = int(systemset_code)
    if split not in {1, 2}:
        raise ValueError("split_code must be 1 (val) or 2 (test)")
    if systemset not in {0, 1}:
        raise ValueError("systemset_code must be 0 (full) or 1 (S3-drift survivors)")
    return split, systemset


def _validate_B(B: int) -> int:
    if isinstance(B, bool) or not isinstance(B, Integral) or int(B) < 1:
        raise ValueError("B must be a positive integer")
    return int(B)


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return value


def _sha256_int_list(values: Sequence[int]) -> str:
    payload = json.dumps(
        [int(value) for value in values], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_resample_matrix(
    system_ids: Sequence[int],
    strata_d: Sequence[int],
    seed_key: Sequence[int],
    *,
    split_code: int,
    systemset_code: int,
    B: int = RESAMPLE_B,
) -> ResampleMatrix:
    split, systemset = _validate_resample_codes(split_code, systemset_code)
    replicate_count = _validate_B(B)
    ids = _canonical_int_tuple(system_ids, "system_ids")
    strata = _canonical_int_tuple(strata_d, "strata_d")
    entropy = _canonical_int_tuple(seed_key, "seed_key")
    if len(ids) != len(strata):
        raise ValueError("system_ids and strata_d must have the same length")
    if len(set(ids)) != len(ids):
        raise ValueError("system_ids must be unique")

    generator = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    labels = np.asarray(strata, dtype=np.int64)
    index_matrix = np.empty((replicate_count, len(ids)), dtype=np.int64)
    for replicate in range(replicate_count):
        index_matrix[replicate] = _resample_indices(len(ids), generator, strata=labels)
    return ResampleMatrix(
        matrix=index_matrix,
        split_code=split,
        systemset_code=systemset,
        system_ids=ids,
        strata_d=strata,
        sha256_system_ids=_sha256_int_list(ids),
        sha256_strata_d=_sha256_int_list(strata),
        entropy=entropy,
        B=replicate_count,
        method=RESAMPLE_METHOD,
        stratify_by=RESAMPLE_STRATIFY_BY,
    )


def _validated_matrix_indices(matrix: ResampleMatrix) -> np.ndarray:
    if not isinstance(matrix, ResampleMatrix):
        raise ValueError("matrix must be a ResampleMatrix")
    _validate_resample_codes(matrix.split_code, matrix.systemset_code)
    replicate_count = _validate_B(matrix.B)
    system_ids = _canonical_int_tuple(matrix.system_ids, "matrix.system_ids")
    strata_d = _canonical_int_tuple(matrix.strata_d, "matrix.strata_d")
    _canonical_int_tuple(matrix.entropy, "matrix.entropy")
    if len(set(system_ids)) != len(system_ids):
        raise ValueError("matrix.system_ids must be unique")
    if matrix.method != RESAMPLE_METHOD:
        raise ValueError(f"matrix method must equal {RESAMPLE_METHOD!r}")
    if matrix.stratify_by != RESAMPLE_STRATIFY_BY:
        raise ValueError(f"matrix stratify_by must equal {RESAMPLE_STRATIFY_BY!r}")
    if matrix.sha256_system_ids != _sha256_int_list(matrix.system_ids):
        raise ValueError("sha256_system_ids does not match system_ids")
    if matrix.sha256_strata_d != _sha256_int_list(matrix.strata_d):
        raise ValueError("sha256_strata_d does not match strata_d")
    if len(matrix.system_ids) != len(matrix.strata_d):
        raise ValueError("matrix metadata is not aligned")
    indices = np.asarray(matrix.matrix)
    expected_shape = (replicate_count, len(matrix.system_ids))
    if indices.ndim != 2 or indices.shape != expected_shape:
        raise ValueError(f"matrix must have shape {expected_shape}")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("matrix indices must have an integer dtype")
    if np.any(indices < 0) or np.any(indices >= len(matrix.system_ids)):
        raise ValueError("matrix indices are outside the system population")
    return indices


def _validated_matrix_sample(
    values: np.ndarray,
    matrix: ResampleMatrix,
    system_ids: Sequence[int],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(matrix, ResampleMatrix):
        raise ValueError("matrix must be a ResampleMatrix")
    supplied_ids = _canonical_int_tuple(system_ids, "system_ids")
    if supplied_ids != matrix.system_ids:
        raise ValueError("system_ids order does not match the resample matrix")
    indices = _validated_matrix_indices(matrix)
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size != len(matrix.system_ids):
        raise ValueError(f"{name} must be one-dimensional and aligned with system_ids")
    if not np.all(np.isfinite(sample)):
        raise ValueError(f"{name} must contain only finite values")
    return sample, indices


def _ci_from_indices(
    values: np.ndarray,
    indices: np.ndarray,
    stat: Statistic,
    alpha: float,
) -> tuple[float, float]:
    tail = _validate_alpha(alpha)
    statistics = np.empty(indices.shape[0])
    for replicate, row in enumerate(indices):
        statistics[replicate] = float(stat(values[row]))
    lower, upper = np.percentile(
        statistics, [100.0 * tail / 2.0, 100.0 * (1.0 - tail / 2.0)]
    )
    return float(lower), float(upper)


def bootstrap_ci_from_matrix(
    values: np.ndarray,
    matrix: ResampleMatrix,
    system_ids: Sequence[int],
    stat: Statistic = np.median,
    alpha: float = RESAMPLE_ALPHA,
) -> tuple[float, float]:
    sample, indices = _validated_matrix_sample(values, matrix, system_ids, "values")
    return _ci_from_indices(sample, indices, stat, alpha)


def paired_bootstrap_from_matrix(
    base: np.ndarray,
    test: np.ndarray,
    matrix: ResampleMatrix,
    system_ids: Sequence[int],
    stat: Statistic = np.median,
    alpha: float = RESAMPLE_ALPHA,
) -> dict[str, float | int]:
    baseline, indices = _validated_matrix_sample(base, matrix, system_ids, "base")
    candidate = np.asarray(test, dtype=float)
    if candidate.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError("test must be one-dimensional and aligned with base")
    improvements = paired_improvement(baseline, candidate)
    point = float(stat(improvements))
    ci_lo, ci_hi = _ci_from_indices(improvements, indices, stat, alpha)
    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_systems": improvements.size,
    }


def paired_across_cells_from_matrix(
    values_a: np.ndarray,
    values_b: np.ndarray,
    matrix: ResampleMatrix,
    system_ids_a: Sequence[int],
    system_ids_b: Sequence[int],
    stat: Statistic = np.median,
    alpha: float = RESAMPLE_ALPHA,
) -> dict[str, float | int]:
    sample_a, indices = _validated_matrix_sample(
        values_a, matrix, system_ids_a, "values_a"
    )
    ids_b = _canonical_int_tuple(system_ids_b, "system_ids_b")
    if ids_b != matrix.system_ids:
        raise ValueError("both cells must use the matrix's exact system_id order")
    sample_b = np.asarray(values_b, dtype=float)
    if sample_b.ndim != 1 or sample_b.shape != sample_a.shape:
        raise ValueError("values_b must be one-dimensional and aligned with values_a")
    if not np.all(np.isfinite(sample_b)):
        raise ValueError("values_b must contain only finite values")
    differences = sample_a - sample_b
    point = float(stat(differences))
    ci_lo, ci_hi = _ci_from_indices(differences, indices, stat, alpha)
    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_systems": differences.size,
    }


def resample_set_path(logs_dir: Path, split_code: int, systemset_code: int) -> Path:
    split, systemset = _validate_resample_codes(split_code, systemset_code)
    return Path(logs_dir) / RESAMPLE_SETS_DIRNAME / f"{split}_{systemset}.json"


def resample_set_payload(matrix: ResampleMatrix) -> dict[str, Any]:
    _validated_matrix_indices(matrix)
    return {
        "split_code": matrix.split_code,
        "systemset_code": matrix.systemset_code,
        "system_ids": list(matrix.system_ids),
        "strata_d": list(matrix.strata_d),
        "sha256_system_ids": matrix.sha256_system_ids,
        "sha256_strata_d": matrix.sha256_strata_d,
        "entropy": list(matrix.entropy),
        "B": matrix.B,
        "method": matrix.method,
        "stratify_by": matrix.stratify_by,
    }


def write_resample_set(matrix: ResampleMatrix, logs_dir: Path) -> Path:
    if matrix.B != RESAMPLE_B:
        raise ValueError(f"only B={RESAMPLE_B} resample sets may be written")
    payload = resample_set_payload(matrix)
    path = resample_set_path(logs_dir, matrix.split_code, matrix.systemset_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    temporary = Path(f"{path}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_resample_set(path: Path) -> ResampleMatrix:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
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
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            "resample-set JSON must contain exactly the registered ten keys"
        )

    system_ids = _canonical_int_tuple(payload["system_ids"], "system_ids")
    strata_d = _canonical_int_tuple(payload["strata_d"], "strata_d")
    if not isinstance(payload["sha256_system_ids"], str):
        raise ValueError("sha256_system_ids must be a string")
    if not isinstance(payload["sha256_strata_d"], str):
        raise ValueError("sha256_strata_d must be a string")
    if payload["sha256_system_ids"] != _sha256_int_list(system_ids):
        raise ValueError("sha256_system_ids does not match system_ids")
    if payload["sha256_strata_d"] != _sha256_int_list(strata_d):
        raise ValueError("sha256_strata_d does not match strata_d")

    B = _validate_B(payload["B"])
    if B != RESAMPLE_B:
        raise ValueError(f"resample-set B must equal {RESAMPLE_B}")
    if payload["method"] != RESAMPLE_METHOD:
        raise ValueError(f"resample-set method must equal {RESAMPLE_METHOD!r}")
    if payload["stratify_by"] != RESAMPLE_STRATIFY_BY:
        raise ValueError(
            f"resample-set stratify_by must equal {RESAMPLE_STRATIFY_BY!r}"
        )
    entropy = _canonical_int_tuple(payload["entropy"], "entropy")
    return make_resample_matrix(
        system_ids,
        strata_d,
        entropy,
        split_code=payload["split_code"],
        systemset_code=payload["systemset_code"],
        B=B,
    )


def divergence_rate(statuses: Sequence[str]) -> float:
    if len(statuses) == 0:
        raise ValueError("statuses must not be empty")
    return float(np.mean(np.asarray(statuses, dtype=str) == "diverged"))


def descriptive_R(
    R_values: np.ndarray, statuses: list[str], cfg: Any
) -> dict[str, float | int]:
    risks = np.asarray(R_values, dtype=float)
    if risks.ndim != 1 or risks.size != len(statuses) or risks.size == 0:
        raise ValueError("R_values and statuses must be aligned and non-empty")
    status_array = np.asarray(statuses, dtype=str)
    ok_mask = status_array != "diverged"
    ok_risks = risks[ok_mask]
    median_ok = float(np.median(ok_risks)) if ok_risks.size else float(np.nan)
    capped_mask = (
        (status_array == "diverged") | ~np.isfinite(risks) | (risks > float(cfg.r_cap))
    )
    return {
        "median_R_ok": median_ok,
        "n_ok": int(np.sum(ok_mask)),
        "divergence_rate": divergence_rate(statuses),
        "n_capped": int(np.sum(capped_mask)),
    }


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    result = stats.wilcoxon(
        np.asarray(a),
        np.asarray(b),
        alternative="two-sided",
        zero_method="wilcox",
    )
    return float(result.statistic), float(result.pvalue)


def holm_bonferroni(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    order = np.argsort(values)
    sorted_values = values[order]
    multipliers = values.size - np.arange(values.size)
    adjusted_sorted = np.maximum.accumulate(sorted_values * multipliers)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


def misg_flops(m: int, p: int) -> int:
    if m < 1 or p < 1:
        raise ValueError("m and p must be positive")
    return (4 * p + 2) * m + p


def kalman_flops(mz: int) -> int:
    if mz < 1:
        raise ValueError("mz must be positive")
    return 2 * mz**3 + 2 * mz**2 + mz**2 + 2 * mz**2 + 2 * mz
