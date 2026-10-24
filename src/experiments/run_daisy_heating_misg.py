from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DATA_URL = "https://ftp.esat.kuleuven.be/pub/SISTA/data/thermic/heating_system.dat.gz"
DEFAULT_P = (1, 2, 5, 8, 12)
DEFAULT_LAMBDA = (1.0, 0.99, 0.95, 0.925, 0.90, 0.85)
DEFAULT_GAMMA = (
    0.02,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.80,
    1.00,
    1.20,
    1.50,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
    5.00,
    6.00,
)
DEFAULT_DELAYS = (2, 3, 4)


@dataclass(frozen=True)
class Split:
    train_end: int
    validation_end: int
    total: int


@dataclass
class RunResult:
    prediction: np.ndarray
    rho: np.ndarray
    status: str
    diverged_at: int | None
    maximum_spectral_radius: float


def parse_number_list(text: str, cast: type) -> tuple:
    values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list must contain at least one value")
    return values


def download_dataset(path: Path, url: str = DATA_URL) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MISG-DaISy-heating-validation/1.0"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        values = np.loadtxt(handle, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("DaISy Heating data must contain time, input and output")
    values = values[:, :3]
    if not np.all(np.isfinite(values)):
        raise ValueError("dataset contains nonfinite values")
    time, input_signal, output_signal = values.T
    if len(values) < 100:
        raise ValueError("dataset is unexpectedly short")
    increments = np.diff(time)
    if np.any(increments <= 0) or not np.allclose(
        increments, np.median(increments), rtol=1e-6, atol=1e-9
    ):
        raise ValueError("time samples must be strictly increasing and uniform")
    return time, input_signal, output_signal


def chronological_split(
    n: int, train_fraction: float, validation_fraction: float
) -> Split:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie in (0, 1)")
    if not (0.0 < validation_fraction < 1.0 - train_fraction):
        raise ValueError("validation_fraction leaves no independent test interval")
    train_end = int(math.floor(n * train_fraction))
    validation_end = int(math.floor(n * (train_fraction + validation_fraction)))
    if min(train_end, validation_end - train_end, n - validation_end) < 40:
        raise ValueError("each chronological interval must contain at least 40 samples")
    return Split(train_end=train_end, validation_end=validation_end, total=n)


def standardize_from_training(
    input_signal: np.ndarray, output_signal: np.ndarray, train_end: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    u_mean = float(np.mean(input_signal[:train_end]))
    u_scale = float(np.std(input_signal[:train_end], ddof=1))
    y_mean = float(np.mean(output_signal[:train_end]))
    y_scale = float(np.std(output_signal[:train_end], ddof=1))
    if u_scale <= 0.0 or y_scale <= 0.0:
        raise ValueError("training input and output must both vary")
    u = (input_signal - u_mean) / u_scale
    y = (output_signal - y_mean) / y_scale
    return (
        u,
        y,
        {
            "input_mean": u_mean,
            "input_scale": u_scale,
            "output_mean": y_mean,
            "output_scale": y_scale,
        },
    )


def delayed_io_features(
    u: np.ndarray, y: np.ndarray, d: int, indices: np.ndarray
) -> np.ndarray:
    columns = (
        y[indices - 1],
        y[indices - 2],
        y[indices - d - 1],
        y[indices - d - 2],
        y[indices - 2 * d - 2],
        u[indices - 1],
        u[indices - 2],
        u[indices - d - 2],
        np.ones(indices.size),
    )
    return np.column_stack(columns)


def select_effective_delay(
    u: np.ndarray,
    y: np.ndarray,
    train_end: int,
    delay_values: Sequence[int],
    ridge: float = 1e-6,
) -> tuple[int, list[dict[str, float]]]:
    inner_end = int(math.floor(0.75 * train_end))
    rows: list[dict[str, float]] = []
    for d in delay_values:
        if d < 2:
            raise ValueError("the paper's observer construction requires d >= 2")
        first = 2 * d + 2
        fit_index = np.arange(first, inner_end)
        holdout_index = np.arange(max(first, inner_end), train_end)
        if fit_index.size < 30 or holdout_index.size < 20:
            raise ValueError(f"not enough training samples to assess d={d}")
        x_fit = delayed_io_features(u, y, d, fit_index)
        x_holdout = delayed_io_features(u, y, d, holdout_index)
        penalty = ridge * np.eye(x_fit.shape[1])
        penalty[-1, -1] = 0.0
        beta = np.linalg.solve(x_fit.T @ x_fit + penalty, x_fit.T @ y[fit_index])
        prediction = x_holdout @ beta
        rmse = float(np.sqrt(np.mean((prediction - y[holdout_index]) ** 2)))
        scale = float(np.std(y[holdout_index], ddof=1))
        nrmse = rmse / scale if scale > 0.0 else math.inf
        rows.append({"delay": int(d), "training_holdout_nrmse": nrmse})
    selected = min(rows, key=lambda row: (row["training_holdout_nrmse"], row["delay"]))
    return int(selected["delay"]), rows


def companion_matrix(a: np.ndarray) -> np.ndarray:
    return np.array([[0.0, 1.0], [a[0], a[1]]], dtype=float)


def unpack_rho(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rho.shape != (8,):
        raise ValueError("the two-state delayed model requires eight parameters")
    return companion_matrix(rho[:2]), rho[2:6].reshape(2, 2), rho[6:8]


def augmented_transition(A: np.ndarray, B: np.ndarray, d: int) -> np.ndarray:
    matrix = np.zeros((2 * (d + 1), 2 * (d + 1)), dtype=float)
    matrix[:2, :2] = A
    matrix[:2, 2 * d : 2 * (d + 1)] = B
    for block in range(1, d + 1):
        matrix[2 * block : 2 * (block + 1), 2 * (block - 1) : 2 * block] = np.eye(2)
    return matrix


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def run_misg(
    u: np.ndarray,
    y: np.ndarray,
    *,
    stop: int,
    d: int,
    p: int,
    forgetting_factor: float,
    gamma: float,
    freeze_from: int | None = None,
    initialization_scale: float = 1e6,
    normalizer_initial: float = 1.0,
    divergence_norm: float = 1e6,
    spectral_threshold: float = 1.2,
    spectral_patience: int = 50,
) -> RunResult:
    if stop > len(y) or stop > len(u):
        raise ValueError("stop exceeds the available record")
    if p < 1 or d < 2 or gamma <= 0.0 or not (0.0 < forgetting_factor <= 1.0):
        raise ValueError("invalid MISG configuration")

    rho = np.ones(8, dtype=float) / initialization_scale
    history = {
        time: np.ones(2, dtype=float) / initialization_scale
        for time in range(1 - d - 2, 2)
    }
    prediction = np.full(stop, np.nan, dtype=float)
    phi_buffer = np.zeros((p, 8), dtype=float)
    phi_cursor = 0
    r_value = float(normalizer_initial)
    spectral_counter = 0
    max_radius = 0.0
    status = "ok"
    diverged_at: int | None = None

    def get_u(time: int) -> float:
        return 0.0 if time <= 0 else float(u[time - 1])

    def get_y(time: int) -> float:
        return 0.0 if time <= 0 else float(y[time - 1])

    for k in range(1, stop + 1):
        phi = np.concatenate(
            (
                history[k - 2],
                history[k - d - 1],
                history[k - d - 2],
                np.array([get_u(k - 1), get_u(k - 2)]),
            )
        )
        prediction[k - 1] = float(phi @ rho)

        phi_buffer[phi_cursor] = phi
        phi_window = np.column_stack(
            [phi_buffer[(phi_cursor - offset) % p] for offset in range(p)]
        )
        phi_cursor = (phi_cursor + 1) % p
        output_block = np.array([get_y(k - offset) for offset in range(p)])
        innovation = output_block - phi_window.T @ rho
        r_value = forgetting_factor * r_value + float(np.sum(phi_window**2))
        correction = gamma * (phi_window @ innovation) / r_value
        if freeze_from is None or (k - 1) < freeze_from:
            rho = rho + correction

        if not np.all(np.isfinite(rho)) or np.linalg.norm(rho) > divergence_norm:
            status = "diverged_parameter"
            diverged_at = k
            break

        A_hat, B_hat, f_hat = unpack_rho(rho)
        radius = spectral_radius(augmented_transition(A_hat, B_hat, d))
        max_radius = max(max_radius, radius)
        if radius > spectral_threshold:
            spectral_counter += 1
        else:
            spectral_counter = 0
        if spectral_counter >= spectral_patience:
            status = "diverged_spectral"
            diverged_at = k
            break

        if k < stop:
            next_state = A_hat @ history[k] + B_hat @ history[k - d] + f_hat * get_u(k)
            if (
                not np.all(np.isfinite(next_state))
                or np.linalg.norm(next_state) > divergence_norm
            ):
                status = "diverged_state"
                diverged_at = k
                break
            history[k + 1] = next_state

    return RunResult(
        prediction=prediction,
        rho=rho,
        status=status,
        diverged_at=diverged_at,
        maximum_spectral_radius=max_radius,
    )


def fixed_parameter_one_step_prediction(
    u: np.ndarray,
    y: np.ndarray,
    *,
    rho: np.ndarray,
    d: int,
    stop: int,
) -> np.ndarray:
    if stop > len(y) or stop > len(u):
        raise ValueError("stop exceeds the available record")
    A_hat, B_hat, f_hat = unpack_rho(rho)
    del A_hat
    history = {time: np.zeros(2, dtype=float) for time in range(1 - d - 2, 2)}
    prediction = np.full(stop, np.nan, dtype=float)

    def get_u(time: int) -> float:
        return 0.0 if time <= 0 else float(u[time - 1])

    def get_y(time: int) -> float:
        return 0.0 if time <= 0 else float(y[time - 1])

    for k in range(1, stop + 1):
        state_time = k - 2
        if state_time >= 1:
            delayed_state = history[state_time - d]
            history[state_time] = np.array(
                [
                    get_y(state_time),
                    get_y(state_time + 1)
                    - float(B_hat[0] @ delayed_state)
                    - float(f_hat[0] * get_u(state_time)),
                ],
                dtype=float,
            )
        phi = np.concatenate(
            (
                history[k - 2],
                history[k - d - 1],
                history[k - d - 2],
                np.array([get_u(k - 1), get_u(k - 2)]),
            )
        )
        prediction[k - 1] = float(phi @ rho)
    return prediction


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if (
        actual.size == 0
        or predicted.shape != actual.shape
        or not np.all(np.isfinite(predicted))
    ):
        return {
            "rmse": math.inf,
            "nrmse": math.inf,
            "mae": math.inf,
            "fit_percent": -math.inf,
        }
    residual = predicted - actual
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = float(np.std(actual, ddof=1))
    denominator = float(np.linalg.norm(actual - np.mean(actual)))
    return {
        "rmse": rmse,
        "nrmse": rmse / scale if scale > 0.0 else math.inf,
        "mae": float(np.mean(np.abs(residual))),
        "fit_percent": 100.0 * (1.0 - float(np.linalg.norm(residual)) / denominator)
        if denominator > 0.0
        else -math.inf,
    }


def score_interval(
    result: RunResult, y: np.ndarray, start: int, stop: int
) -> dict[str, float]:
    if result.status != "ok":
        return prediction_metrics(y[start:stop], np.full(stop - start, np.nan))
    return prediction_metrics(y[start:stop], result.prediction[start:stop])


def select_cell_multipliers(
    u: np.ndarray,
    y: np.ndarray,
    split: Split,
    d: int,
    p_values: Sequence[int],
    lambdas: Sequence[float],
    gamma_values: Sequence[float],
    shortlist_size: int,
) -> tuple[dict[tuple[int, float], float], list[dict[str, object]]]:
    selected: dict[tuple[int, float], float] = {}
    rows: list[dict[str, object]] = []
    training_score_start = split.train_end // 2

    for p in p_values:
        for lam in lambdas:
            training_candidates: list[tuple[float, float, str]] = []
            for gamma in gamma_values:
                run = run_misg(
                    u,
                    y,
                    stop=split.train_end,
                    d=d,
                    p=p,
                    forgetting_factor=lam,
                    gamma=gamma,
                )
                metrics = score_interval(run, y, training_score_start, split.train_end)
                training_candidates.append((metrics["nrmse"], gamma, run.status))
            training_candidates.sort(key=lambda item: (item[0], item[1]))
            shortlist = training_candidates[
                : min(shortlist_size, len(training_candidates))
            ]

            validation_candidates: list[tuple[float, float, str, float]] = []
            for training_nrmse, gamma, _ in shortlist:
                run = run_misg(
                    u,
                    y,
                    stop=split.validation_end,
                    d=d,
                    p=p,
                    forgetting_factor=lam,
                    gamma=gamma,
                )
                metrics = score_interval(run, y, split.train_end, split.validation_end)
                validation_candidates.append(
                    (metrics["nrmse"], gamma, run.status, training_nrmse)
                )
            validation_candidates.sort(key=lambda item: (item[0], item[1]))
            best_validation, best_gamma, best_status, best_training = (
                validation_candidates[0]
            )
            selected[(p, float(lam))] = float(best_gamma)
            rows.append(
                {
                    "p": p,
                    "lambda": float(lam),
                    "selected_gamma": float(best_gamma),
                    "training_nrmse": float(best_training),
                    "validation_nrmse": float(best_validation),
                    "validation_status": best_status,
                    "training_shortlist": ";".join(
                        f"{item[1]:g}" for item in shortlist
                    ),
                    "selected_at_gamma_grid_boundary": best_gamma
                    in {min(gamma_values), max(gamma_values)},
                }
            )
    return selected, rows


def evaluate_selected_cells(
    u: np.ndarray,
    y: np.ndarray,
    split: Split,
    d: int,
    selected: dict[tuple[int, float], float],
    output_mean: float,
    output_scale: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[int, float, str], np.ndarray],
]:
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    prediction_map: dict[tuple[int, float, str], np.ndarray] = {}

    for (p, lam), gamma in sorted(selected.items()):
        online = run_misg(
            u,
            y,
            stop=split.total,
            d=d,
            p=p,
            forgetting_factor=lam,
            gamma=gamma,
        )
        frozen = run_misg(
            u,
            y,
            stop=split.total,
            d=d,
            p=p,
            forgetting_factor=lam,
            gamma=gamma,
            freeze_from=split.validation_end,
        )
        fixed_one_step_prediction = fixed_parameter_one_step_prediction(
            u,
            y,
            rho=frozen.rho,
            d=d,
            stop=split.total,
        )
        frozen_A, frozen_B, _ = unpack_rho(frozen.rho)
        test_boundary_spectral_radius = spectral_radius(
            augmented_transition(frozen_A, frozen_B, d)
        )
        fixed_one_step = RunResult(
            prediction=fixed_one_step_prediction,
            rho=frozen.rho.copy(),
            status="ok",
            diverged_at=None,
            maximum_spectral_radius=frozen.maximum_spectral_radius,
        )
        for mode, run in (
            ("online", online),
            ("frozen_open_loop", frozen),
            ("fixed_parameter_one_step", fixed_one_step),
        ):
            metrics = score_interval(run, y, split.validation_end, split.total)
            rows.append(
                {
                    "p": p,
                    "lambda": lam,
                    "gamma": gamma,
                    "mode": mode,
                    **metrics,
                    "status": run.status,
                    "diverged_at": "" if run.diverged_at is None else run.diverged_at,
                    "maximum_spectral_radius": run.maximum_spectral_radius,
                    "test_boundary_spectral_radius": test_boundary_spectral_radius,
                }
            )
            prediction_map[(p, lam, mode)] = run.prediction.copy()

        for index in range(split.validation_end, split.total):
            prediction_rows.append(
                {
                    "sample_index": index + 1,
                    "p": p,
                    "lambda": lam,
                    "gamma": gamma,
                    "observed_output": y[index] * output_scale + output_mean,
                    "online_prediction": online.prediction[index] * output_scale
                    + output_mean,
                    "frozen_open_loop_prediction": frozen.prediction[index]
                    * output_scale
                    + output_mean,
                    "fixed_parameter_one_step_prediction": fixed_one_step.prediction[
                        index
                    ]
                    * output_scale
                    + output_mean,
                }
            )
    return rows, prediction_rows, prediction_map


def circular_block_ratio_interval(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    finite = np.isfinite(residual_a) & np.isfinite(residual_b)
    a = residual_a[finite]
    b = residual_b[finite]
    if a.size < 20:
        return math.nan, math.nan, math.nan
    observed = float(np.sqrt(np.mean(a**2) / np.mean(b**2)))
    number_of_blocks = int(math.ceil(a.size / block_length))
    ratios = np.empty(resamples, dtype=float)
    offsets = np.arange(block_length)
    for draw in range(resamples):
        starts = rng.integers(0, a.size, size=number_of_blocks)
        index = ((starts[:, None] + offsets[None, :]) % a.size).ravel()[: a.size]
        ratios[draw] = np.sqrt(np.mean(a[index] ** 2) / np.mean(b[index] ** 2))
    lower, upper = np.percentile(ratios, [5.0, 95.0])
    return observed, float(lower), float(upper)


def principal_contrasts(
    y: np.ndarray,
    split: Split,
    prediction_map: dict[tuple[int, float, str], np.ndarray],
    lambdas: Sequence[float],
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    block_length = max(5, int(round(math.sqrt(split.total - split.validation_end))))
    actual = y[split.validation_end : split.total]
    for lam in lambdas:
        for mode in (
            "online",
            "frozen_open_loop",
            "fixed_parameter_one_step",
        ):
            key_1 = (1, float(lam), mode)
            key_5 = (5, float(lam), mode)
            if key_1 not in prediction_map or key_5 not in prediction_map:
                continue
            residual_1 = (
                prediction_map[key_1][split.validation_end : split.total] - actual
            )
            residual_5 = (
                prediction_map[key_5][split.validation_end : split.total] - actual
            )
            ratio, lower, upper = circular_block_ratio_interval(
                residual_1,
                residual_5,
                block_length=block_length,
                resamples=resamples,
                rng=rng,
            )
            rows.append(
                {
                    "lambda": float(lam),
                    "mode": mode,
                    "p1_to_p5_rmse_ratio": ratio,
                    "block_bootstrap_90pct_lower": lower,
                    "block_bootstrap_90pct_upper": upper,
                    "block_length": block_length,
                    "resamples": resamples,
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def build_frozen_diagnostics(
    input_original: np.ndarray,
    output_original: np.ndarray,
    split: Split,
    test_rows: Sequence[dict[str, object]],
    prediction_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    segment_rows: list[dict[str, object]] = []
    for name, start, stop in (
        ("training", 0, split.train_end),
        ("validation", split.train_end, split.validation_end),
        ("test", split.validation_end, split.total),
    ):
        segment_rows.append(
            {
                "segment": name,
                "samples": stop - start,
                "input_mean": float(np.mean(input_original[start:stop])),
                "input_standard_deviation": float(
                    np.std(input_original[start:stop], ddof=1)
                ),
                "output_mean": float(np.mean(output_original[start:stop])),
                "output_standard_deviation": float(
                    np.std(output_original[start:stop], ddof=1)
                ),
            }
        )

    metric_lookup = {
        (int(row["p"]), float(row["lambda"]), str(row["mode"])): row
        for row in test_rows
    }
    grouped_predictions: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in prediction_rows:
        grouped_predictions.setdefault(
            (int(row["p"]), float(row["lambda"])), []
        ).append(row)

    horizon_rows: list[dict[str, object]] = []
    available = split.total - split.validation_end
    horizons = tuple(
        sorted({h for h in (10, 20, 40, 80, 120, available) if h <= available})
    )
    for (p, lam), rows in sorted(grouped_predictions.items()):
        rows = sorted(rows, key=lambda row: int(row["sample_index"]))
        actual = np.array([float(row["observed_output"]) for row in rows])
        predicted = np.array(
            [float(row["frozen_open_loop_prediction"]) for row in rows]
        )
        for horizon in horizons:
            residual = predicted[:horizon] - actual[:horizon]
            denominator = float(
                np.linalg.norm(actual[:horizon] - np.mean(actual[:horizon]))
            )
            horizon_rows.append(
                {
                    "p": p,
                    "lambda": lam,
                    "horizon_samples": horizon,
                    "rmse_output_units": float(np.sqrt(np.mean(residual**2))),
                    "mean_error_output_units": float(np.mean(residual)),
                    "fit_percent": 100.0
                    * (1.0 - float(np.linalg.norm(residual)) / denominator)
                    if denominator > 0.0
                    else -math.inf,
                    "test_boundary_spectral_radius": float(
                        metric_lookup[(p, lam, "fixed_parameter_one_step")][
                            "test_boundary_spectral_radius"
                        ]
                    ),
                }
            )
    return segment_rows, horizon_rows


def make_figure(
    path_stem: Path,
    time: np.ndarray,
    y_original: np.ndarray,
    split: Split,
    contrast_rows: list[dict[str, object]],
    prediction_map: dict[tuple[int, float, str], np.ndarray],
    output_mean: float,
    output_scale: float,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as exc:
        raise RuntimeError(
            "audit_panel_alignment.py must be beside this script before figure export"
        ) from exc

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14.5,
            "axes.labelsize": 14.5,
            "axes.titlesize": 14.5,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "legend.fontsize": 14.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    blue = "#2C6DB2"
    orange = "#E1812C"
    neutral = "#333333"

    figure_size = (5.9055, 3.1496)

    test_slice = slice(split.validation_end, split.total)
    fig_a, ax_a = plt.subplots(figsize=figure_size)
    ax_a.plot(
        time[test_slice],
        y_original[test_slice],
        color=neutral,
        lw=2.0,
        label="Observed",
    )
    for p, color in ((1, blue), (5, orange)):
        key = (p, 0.95, "fixed_parameter_one_step")
        if key in prediction_map:
            predicted = prediction_map[key] * output_scale + output_mean
            ax_a.plot(
                time[test_slice],
                predicted[test_slice],
                color=color,
                lw=1.8,
                label=f"p = {p}, λ = 0.95",
            )
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Temperature (°C)")
    ax_a.set_title("Fixed parameter one step prediction", loc="left")
    ax_a.legend(loc="best", handlelength=2.2)
    fig_a.subplots_adjust(left=0.16, right=0.985, bottom=0.22, top=0.87)
    fig_a.canvas.draw()
    panel_a_stem = path_stem.parent / f"{path_stem.name}_a"
    require_matplotlib_panel_alignment(
        fig_a,
        json_out=str(panel_a_stem) + ".alignment.json",
        overlay_svg=str(panel_a_stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig_a.savefig(panel_a_stem.with_suffix(".pdf"))
    fig_a.savefig(panel_a_stem.with_suffix(".svg"))
    fig_a.savefig(panel_a_stem.with_suffix(".png"), dpi=600)
    fig_a.savefig(panel_a_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig_a)

    contrast_lookup = {
        (float(row["lambda"]), str(row["mode"])): row for row in contrast_rows
    }
    fig_b, ax_b = plt.subplots(figsize=figure_size)
    x = np.array([0.0, 1.0])
    for mode, color, offset, marker_face, display_label in (
        ("online", blue, -0.06, blue, "Adaptive"),
        (
            "fixed_parameter_one_step",
            orange,
            0.06,
            orange,
            "Fixed one step",
        ),
    ):
        rows = [contrast_lookup.get((lam, mode)) for lam in (1.0, 0.95)]
        centers = np.array(
            [float(row["p1_to_p5_rmse_ratio"]) if row else math.nan for row in rows]
        )
        lower = np.array(
            [
                float(row["block_bootstrap_90pct_lower"]) if row else math.nan
                for row in rows
            ]
        )
        upper = np.array(
            [
                float(row["block_bootstrap_90pct_upper"]) if row else math.nan
                for row in rows
            ]
        )
        ax_b.errorbar(
            x + offset,
            centers,
            yerr=np.vstack((centers - lower, upper - centers)),
            marker="o",
            ls="-",
            color=color,
            markerfacecolor=marker_face,
            capsize=4.0,
            lw=1.8,
            ms=7.0,
            label=display_label,
        )
    ax_b.axhline(1.0, color="#888888", lw=1.2, ls=":")
    ax_b.set_xticks(x, ["λ = 1", "λ = 0.95"])
    ax_b.set_ylabel("Test RMSE ratio, p = 1 / p = 5")
    ax_b.set_title("Innovation length contrast", loc="left")
    ax_b.legend(loc="best", handlelength=2.0)
    fig_b.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.87)
    fig_b.canvas.draw()
    panel_b_stem = path_stem.parent / f"{path_stem.name}_b"
    require_matplotlib_panel_alignment(
        fig_b,
        json_out=str(panel_b_stem) + ".alignment.json",
        overlay_svg=str(panel_b_stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig_b.savefig(panel_b_stem.with_suffix(".pdf"))
    fig_b.savefig(panel_b_stem.with_suffix(".svg"))
    fig_b.savefig(panel_b_stem.with_suffix(".png"), dpi=600)
    fig_b.savefig(panel_b_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig_b)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DaISy heating evaluation")
    parser.add_argument(
        "--data-file", type=Path, default=None, help="local .dat or .dat.gz file"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--p-values", default=",".join(map(str, DEFAULT_P)))
    parser.add_argument("--lambdas", default=",".join(map(str, DEFAULT_LAMBDA)))
    parser.add_argument("--gamma-values", default=",".join(map(str, DEFAULT_GAMMA)))
    parser.add_argument("--delay-values", default=",".join(map(str, DEFAULT_DELAYS)))
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test with p={1,5}, lambda={1,0.95}, fewer gamma values and 500 resamples",
    )
    parser.add_argument("--skip-figure", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    data_file = args.data_file or root / "data" / "heating_system.dat.gz"
    output_dir = args.output_dir or root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.data_file is None:
        print(f"Downloading/using official DaISy data: {data_file}")
        download_dataset(data_file)
    elif not data_file.exists():
        raise FileNotFoundError(data_file)

    p_values = parse_number_list(args.p_values, int)
    lambdas = parse_number_list(args.lambdas, float)
    gamma_values = parse_number_list(args.gamma_values, float)
    delay_values = parse_number_list(args.delay_values, int)
    bootstrap_resamples = args.bootstrap_resamples
    if args.quick:
        p_values = (1, 5)
        lambdas = (1.0, 0.95)
        gamma_values = (0.05, 0.2, 0.5, 1.0, 2.0, 3.5)
        bootstrap_resamples = 500

    time, input_original, output_original = load_dataset(data_file)
    split = chronological_split(
        len(time), args.train_fraction, args.validation_fraction
    )
    u, y, scaling = standardize_from_training(
        input_original, output_original, split.train_end
    )
    selected_delay, delay_rows = select_effective_delay(
        u, y, split.train_end, delay_values
    )
    print(f"Selected effective delay d = {selected_delay} from training data only")

    selected_gamma, selection_rows = select_cell_multipliers(
        u,
        y,
        split,
        selected_delay,
        p_values,
        lambdas,
        gamma_values,
        args.shortlist_size,
    )
    boundary_cells = [
        {
            "p": int(row["p"]),
            "lambda": float(row["lambda"]),
            "gamma": float(row["selected_gamma"]),
        }
        for row in selection_rows
        if bool(row["selected_at_gamma_grid_boundary"])
    ]
    if boundary_cells:
        print(
            "WARNING: one or more cells selected an endpoint of the gamma grid. "
            "Treat those cells as tuning-boundary results and expand the grid in a "
            "separate prespecified sensitivity run before drawing a numerical conclusion."
        )
    test_rows, prediction_rows, prediction_map = evaluate_selected_cells(
        u,
        y,
        split,
        selected_delay,
        selected_gamma,
        scaling["output_mean"],
        scaling["output_scale"],
    )
    contrast_lambdas = tuple(lam for lam in (1.0, 0.95) if lam in lambdas)
    contrast_rows = principal_contrasts(
        y,
        split,
        prediction_map,
        contrast_lambdas,
        bootstrap_resamples,
        args.seed,
    )
    segment_rows, frozen_horizon_rows = build_frozen_diagnostics(
        input_original,
        output_original,
        split,
        test_rows,
        prediction_rows,
    )

    write_csv(output_dir / "delay_scan.csv", delay_rows)
    write_csv(output_dir / "cell_selection.csv", selection_rows)
    write_csv(output_dir / "test_metrics.csv", test_rows)
    write_csv(output_dir / "test_predictions.csv", prediction_rows)
    write_csv(output_dir / "principal_contrasts.csv", contrast_rows)
    write_csv(output_dir / "segment_distribution.csv", segment_rows)
    write_csv(output_dir / "frozen_open_loop_horizons.csv", frozen_horizon_rows)

    summary = {
        "dataset": {
            "name": "DaISy Heating System",
            "source_url": DATA_URL,
            "samples": len(time),
            "sampling_interval_seconds": float(np.median(np.diff(time))),
        },
        "split": {
            "training_samples": split.train_end,
            "validation_samples": split.validation_end - split.train_end,
            "test_samples": split.total - split.validation_end,
        },
        "scaling_from_training_only": scaling,
        "selected_effective_delay": selected_delay,
        "selected_gamma": {
            f"p={p},lambda={lam:g}": gamma for (p, lam), gamma in selected_gamma.items()
        },
        "gamma_grid_boundary_cells": boundary_cells,
        "principal_contrasts": contrast_rows,
        "prediction_modes": {
            "online": (
                "The MISG parameters continue to adapt throughout the test interval."
            ),
            "frozen_open_loop": (
                "The parameter vector and open-loop observer are propagated without "
                "measurement correction after the validation boundary."
            ),
            "fixed_parameter_one_step": (
                "The parameter vector is frozen before testing. Lagged states are "
                "causally reconstructed with outputs through k-1, and y(k) is not used "
                "to predict itself. This diagnostic changes the state reconstruction "
                "mapping and is not the paper-matched open-loop observer."
            ),
        },
        "segment_distribution": segment_rows,
        "interpretation_boundary": (
            "This is a single measured SISO record with unknown physical parameters. "
            "The delay is an effective model delay, and block-bootstrap intervals describe "
            "time-block uncertainty rather than variation across independent systems. "
            "The fixed-parameter one-step diagnostic is separated from the paper-matched "
            "open-loop observer and must not be presented as the same estimator."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if not args.skip_figure:
        make_figure(
            output_dir / "Figure_DaISy_heating",
            time,
            output_original,
            split,
            contrast_rows,
            prediction_map,
            scaling["output_mean"],
            scaling["output_scale"],
        )

    print(f"Results written to: {output_dir}")
    for row in contrast_rows:
        print(
            f"lambda={row['lambda']:g}, {row['mode']}: "
            f"RMSE ratio p=1/p=5 = {row['p1_to_p5_rmse_ratio']:.4f} "
            f"[{row['block_bootstrap_90pct_lower']:.4f}, "
            f"{row['block_bootstrap_90pct_upper']:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
