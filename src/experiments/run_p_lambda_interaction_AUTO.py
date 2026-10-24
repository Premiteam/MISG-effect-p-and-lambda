from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_MODE = "observer"

P_VALUES = [1, 2, 5, 8, 12]

LAMBDA_VALUES = [
    1.00,
    0.99,
    0.95,
    0.925,
    0.90,
    0.85,
]

FINAL_SCALAR_LAMBDA = 0.95

DEFAULT_GAMMA_GRID = [
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
]

TRAIN_TRAJ_START = 0
EVAL_TRAJ_START = 3


def find_project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if (root / "src" / "core").is_dir():
            return root
        raise SystemExit(
            f"--project is not a valid state-estimation project root: {root}"
        )

    candidates: list[Path] = []

    for base in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidates.append(base)
        candidates.append(base / "state-estimation")

    seen: set[Path] = set()

    for root in candidates:
        try:
            root = root.resolve()
        except OSError:
            continue

        if root in seen:
            continue
        seen.add(root)

        if (
            (root / "src" / "core" / "config.py").exists()
            and (root / "src" / "core" / "systems.py").exists()
            and (root / "src" / "core" / "estimators.py").exists()
            and (root / "src" / "core" / "metrics.py").exists()
        ):
            return root

    raise SystemExit(
        "Could not auto-discover the state-estimation project.\n"
        "Keep this script under the same 'State Estimation' folder, or run:\n"
        "  python run_p_lambda_interaction_FINAL.py "
        "--project '/full/path/to/state-estimation'"
    )


def import_project(project: Path):
    src = project / "src" / "core"
    sys.path.insert(0, str(src))

    from config import Config, traj_seed
    from estimators import misg_run
    from metrics import risk_R
    from systems import load_family, simulate

    return Config, traj_seed, misg_run, risk_R, load_family, simulate


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )


def float_key(value: float) -> str:
    return str(float(value))


def cell_key(p: int, lambda_value: float) -> str:
    return f"p={int(p)}|lambda={float_key(lambda_value)}"


def find_base_profile(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()

        if not path.exists():
            raise SystemExit(f"Base lambda-profile JSON not found: {path}")

        return path

    candidates = [
        SCRIPT_DIR / "lambda_profile_p5_results.json",
        SCRIPT_DIR / "lambda_profile_p5_results" / "lambda_profile_p5_results.json",
        SCRIPT_DIR.parent / "01_lambda_selection" / "lambda_profile_p5_results.json",
        SCRIPT_DIR.parent
        / "01_lambda_selection"
        / "lambda_profile_p5_results"
        / "lambda_profile_p5_results.json",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise SystemExit(
        "Could not find lambda_profile_p5_results.json automatically.\n"
        "Expected either:\n"
        f"  {SCRIPT_DIR / 'lambda_profile_p5_results.json'}\n"
        "or the recommended sibling layout:\n"
        f"  {SCRIPT_DIR.parent / '01_lambda_selection' / 'lambda_profile_p5_results.json'}\n"
        "You can also pass:\n"
        "  --base-profile '/full/path/to/lambda_profile_p5_results.json'"
    )


def find_closeout_results(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()

        if not path.exists():
            raise SystemExit(f"Close-out results JSON not found: {path}")

        return path

    candidates = [
        SCRIPT_DIR / "lambda_closeout_steps1_2" / "lambda_closeout_results.json",
        SCRIPT_DIR / "lambda_closeout_results.json",
        SCRIPT_DIR.parent
        / "01_lambda_selection"
        / "lambda_closeout_steps1_2"
        / "lambda_closeout_results.json",
        SCRIPT_DIR.parent / "01_lambda_selection" / "lambda_closeout_results.json",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def validate_base_profile(base: dict[str, Any]) -> None:
    if "meta" not in base or "per_lambda" not in base:
        raise SystemExit("Base profile JSON is invalid: missing meta/per_lambda.")

    meta = base["meta"]

    required_meta = [
        "N",
        "p_fixed",
        "gamma_grid",
        "n_train",
        "n_val",
        "n_test",
        "n_train_traj",
        "n_eval_traj",
        "top_k",
        "family",
    ]

    missing = [name for name in required_meta if name not in meta]

    if missing:
        raise SystemExit("Base profile JSON is missing metadata: " + ", ".join(missing))

    if int(meta["p_fixed"]) != 5:
        raise SystemExit(f"Base profile p_fixed={meta['p_fixed']}; expected p=5.")

    for value in [1.0, 0.99, 0.95, 0.90, 0.85]:
        key = float_key(value)

        if key not in base["per_lambda"]:
            raise SystemExit(f"Base profile is missing lambda={value:g}.")


def protocol_signature(
    *,
    N: int,
    gamma_grid: list[float],
    n_train: int,
    n_val: int,
    n_test: int,
    n_train_traj: int,
    n_eval_traj: int,
    top_k: int,
) -> dict[str, Any]:
    return {
        "N": int(N),
        "gamma_grid": [float(x) for x in gamma_grid],
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "n_train_traj": int(n_train_traj),
        "n_eval_traj": int(n_eval_traj),
        "top_k": int(top_k),
    }


def same_float_list(a: Any, b: Any) -> bool:
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
    except Exception:
        return False

    return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=0.0, rtol=0.0))


def validate_closeout_for_reuse(
    closeout: dict[str, Any],
    base_meta: dict[str, Any],
) -> None:
    if "meta" not in closeout:
        raise ValueError("Close-out JSON has no meta block.")

    meta = closeout["meta"]

    scalar_fields = [
        "N",
        "n_train",
        "n_val",
        "n_test",
        "n_train_traj",
        "n_eval_traj",
        "top_k",
    ]

    for field in scalar_fields:
        if int(meta.get(field, -1)) != int(base_meta[field]):
            raise ValueError(
                f"Close-out protocol mismatch for {field}: "
                f"{meta.get(field)} vs {base_meta[field]}"
            )

    if not same_float_list(
        meta.get("gamma_grid", []),
        base_meta["gamma_grid"],
    ):
        raise ValueError("Close-out gamma grid differs from base profile.")


def build_pairs(
    systems,
    *,
    j_start: int,
    n_traj: int,
    cfg,
    traj_seed,
    simulate,
):
    pairs = []

    for system_index, system_item in enumerate(systems):
        for j in range(j_start, j_start + n_traj):
            rng = np.random.default_rng(traj_seed(cfg, system_item["m"], j))

            trajectory = simulate(
                system_item,
                rng,
                cfg.N,
            )

            pairs.append(
                (
                    system_index,
                    system_item,
                    trajectory,
                )
            )

    return pairs


def capped_risk(
    result,
    system_item,
    cfg,
    risk_R,
) -> float:
    value = risk_R(
        result["rho_hat_traj"],
        system_item["rho_star"],
        cfg.N,
    )

    if not np.isfinite(value):
        return float(cfg.r_cap)

    return min(
        float(value),
        float(cfg.r_cap),
    )


def relative_error_series(
    rho_hat_traj: np.ndarray,
    rho_star: np.ndarray,
    N: int,
) -> np.ndarray:
    rho = np.asarray(
        rho_hat_traj,
        dtype=float,
    )

    truth = np.asarray(
        rho_star,
        dtype=float,
    )

    denominator = float(np.linalg.norm(truth))

    if denominator <= 0.0:
        return np.full(
            N,
            np.nan,
        )

    if rho.ndim != 2 or rho.shape[1] != truth.size:
        raise ValueError(
            f"Unexpected rho_hat_traj shape {rho.shape}; truth dimension={truth.size}"
        )

    if rho.shape[0] >= N + 1:
        sequence = rho[1 : N + 1]
    elif rho.shape[0] == N:
        sequence = rho
    else:
        raise ValueError(f"rho_hat_traj has {rho.shape[0]} rows but N={N}.")

    return (
        np.linalg.norm(
            sequence - truth[None, :],
            axis=1,
        )
        / denominator
    )


def first_hit_time(
    errors: np.ndarray,
    threshold: float,
) -> float:
    indices = np.flatnonzero(np.isfinite(errors) & (errors <= threshold))

    if indices.size == 0:
        return float("nan")

    return float(indices[0] + 1)


def mean_or_nan(
    values: list[float],
) -> float:
    arr = np.asarray(
        values,
        dtype=float,
    )

    finite = np.isfinite(arr)

    if not np.any(finite):
        return float("nan")

    return float(np.mean(arr[finite]))


def evaluate(
    pairs,
    *,
    p: int,
    gamma: float,
    lambda_value: float,
    cfg,
    misg_run,
    risk_R,
) -> dict[str, Any]:
    if p < 1:
        raise ValueError("p must be >= 1.")

    if not (0.0 < lambda_value <= 1.0):
        raise ValueError("lambda must satisfy 0 < lambda <= 1.")

    weights = np.ones(
        int(p),
        dtype=float,
    )

    r_mode = (
        "cumulative"
        if math.isclose(
            lambda_value,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        else "forgetting"
    )

    risk_by_system: dict[int, list[float]] = {}
    hit10_by_system: dict[int, list[float]] = {}
    hit30_by_system: dict[int, list[float]] = {}

    divergences = 0

    for (
        system_index,
        system_item,
        trajectory,
    ) in pairs:
        result = misg_run(
            system_item,
            trajectory,
            weights,
            float(gamma),
            r_mode,
            STATE_MODE,
            cfg,
            lam_g=float(lambda_value),
        )

        diverged = result.get("status") == "diverged"

        if diverged:
            divergences += 1

        risk = capped_risk(
            result,
            system_item,
            cfg,
            risk_R,
        )

        risk_by_system.setdefault(
            system_index,
            [],
        ).append(risk)

        if diverged:
            hit10 = float("nan")
            hit30 = float("nan")
        else:
            errors = relative_error_series(
                result["rho_hat_traj"],
                system_item["rho_star"],
                cfg.N,
            )

            hit10 = first_hit_time(
                errors,
                0.10,
            )

            hit30 = first_hit_time(
                errors,
                0.30,
            )

        hit10_by_system.setdefault(
            system_index,
            [],
        ).append(hit10)

        hit30_by_system.setdefault(
            system_index,
            [],
        ).append(hit30)

    system_risks = np.asarray(
        [np.mean(values) for _, values in sorted(risk_by_system.items())],
        dtype=float,
    )

    system_hit10 = np.asarray(
        [mean_or_nan(values) for _, values in sorted(hit10_by_system.items())],
        dtype=float,
    )

    system_hit30 = np.asarray(
        [mean_or_nan(values) for _, values in sorted(hit30_by_system.items())],
        dtype=float,
    )

    finite10 = np.isfinite(system_hit10)

    finite30 = np.isfinite(system_hit30)

    return {
        "median_R": float(np.median(system_risks)),
        "mean_R": float(np.mean(system_risks)),
        "divergences": int(divergences),
        "system_risks": system_risks.tolist(),
        "hit_times_10pct": system_hit10.tolist(),
        "hit_times_30pct": system_hit30.tolist(),
        "median_hit_10pct_success_only": (
            float(np.median(system_hit10[finite10]))
            if np.any(finite10)
            else float("nan")
        ),
        "median_hit_30pct_success_only": (
            float(np.median(system_hit30[finite30]))
            if np.any(finite30)
            else float("nan")
        ),
        "n_failed_10pct": int(np.sum(~finite10)),
        "n_failed_30pct": int(np.sum(~finite30)),
        "failure_rate_10pct": float(np.mean(~finite10)),
        "failure_rate_30pct": float(np.mean(~finite30)),
        "n_systems": int(system_risks.size),
    }


def retune_gamma(
    *,
    p: int,
    lambda_value: float,
    gamma_grid: list[float],
    top_k: int,
    train_pairs,
    val_pairs,
    test_pairs,
    cfg,
    misg_run,
    risk_R,
) -> dict[str, Any]:
    train_scan = []

    for gamma in gamma_grid:
        train_eval = evaluate(
            train_pairs,
            p=p,
            gamma=float(gamma),
            lambda_value=lambda_value,
            cfg=cfg,
            misg_run=misg_run,
            risk_R=risk_R,
        )

        train_scan.append(
            {
                "gamma": float(gamma),
                "mean_R": float(train_eval["mean_R"]),
                "median_R": float(train_eval["median_R"]),
                "divergences": int(train_eval["divergences"]),
            }
        )

    order = np.argsort([record["mean_R"] for record in train_scan])

    candidate_indices = [
        int(index)
        for index in order[
            : min(
                top_k,
                len(order),
            )
        ]
    ]

    val_candidates = []
    best = None

    for index in candidate_indices:
        gamma = float(train_scan[index]["gamma"])

        val_eval = evaluate(
            val_pairs,
            p=p,
            gamma=gamma,
            lambda_value=lambda_value,
            cfg=cfg,
            misg_run=misg_run,
            risk_R=risk_R,
        )

        candidate = {
            "gamma": gamma,
            "train_mean_R": float(train_scan[index]["mean_R"]),
            "val": val_eval,
        }

        val_candidates.append(candidate)

        if best is None or val_eval["median_R"] < best["val"]["median_R"]:
            best = candidate

    if best is None:
        raise RuntimeError("No validation gamma candidate was selected.")

    gamma_star = float(best["gamma"])

    gamma_index = gamma_grid.index(gamma_star)

    boundary = gamma_index in (
        0,
        len(gamma_grid) - 1,
    )

    test_eval = evaluate(
        test_pairs,
        p=p,
        gamma=gamma_star,
        lambda_value=lambda_value,
        cfg=cfg,
        misg_run=misg_run,
        risk_R=risk_R,
    )

    return {
        "p": int(p),
        "lambda": float(lambda_value),
        "gamma": gamma_star,
        "boundary": bool(boundary),
        "train_mean_R": float(best["train_mean_R"]),
        "val_selection_risk": float(best["val"]["median_R"]),
        "train_scan": train_scan,
        "val_candidates": val_candidates,
        "val": best["val"],
        "test": test_eval,
        "source": "computed",
    }


def reusable_p5_record(
    *,
    lambda_value: float,
    base: dict[str, Any],
    closeout: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    key = float_key(lambda_value)

    if closeout is not None and key in closeout.get(
        "per_lambda",
        {},
    ):
        record = dict(closeout["per_lambda"][key])

        return (
            record,
            "reused_p5_closeout",
        )

    if key in base["per_lambda"]:
        record = dict(base["per_lambda"][key])

        return (
            record,
            "reused_p5_profile",
        )

    return None, None


def normalize_reused_record(
    record: dict[str, Any],
    *,
    lambda_value: float,
    source: str,
) -> dict[str, Any]:
    required = [
        "p",
        "gamma",
        "boundary",
        "val",
        "test",
    ]

    missing = [name for name in required if name not in record]

    if missing:
        raise ValueError(
            "Cannot reuse p=5 record; missing fields: " + ", ".join(missing)
        )

    if int(record["p"]) != 5:
        raise ValueError(f"Attempted p=5 reuse from a p={record['p']} record.")

    if not math.isclose(
        float(
            record.get(
                "lambda",
                lambda_value,
            )
        ),
        float(lambda_value),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Reused record has the wrong lambda.")

    normalized = dict(record)

    normalized["p"] = 5
    normalized["lambda"] = float(lambda_value)
    normalized["source"] = source

    return normalized


def build_summary_dataframe(
    results: dict[str, Any],
):
    import pandas as pd

    rows = []

    for record in results["cells"].values():
        test = record["test"]
        val = record["val"]

        rows.append(
            {
                "p": int(record["p"]),
                "lambda": float(record["lambda"]),
                "gamma_star": float(record["gamma"]),
                "boundary": bool(record["boundary"]),
                "source": str(
                    record.get(
                        "source",
                        "unknown",
                    )
                ),
                "val_median_R": float(val["median_R"]),
                "val_mean_R": float(val["mean_R"]),
                "test_median_R": float(test["median_R"]),
                "test_mean_R": float(test["mean_R"]),
                "test_divergences": int(test["divergences"]),
                "T30_success_only": float(test["median_hit_30pct_success_only"]),
                "failure_rate_30pct": float(test["failure_rate_30pct"]),
                "T10_success_only": float(test["median_hit_10pct_success_only"]),
                "failure_rate_10pct": float(test["failure_rate_10pct"]),
            }
        )

    df = pd.DataFrame(rows)

    expected_rows = len(results["meta"]["p_values"]) * len(
        results["meta"]["lambda_values"]
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Interaction summary has {len(df)} rows; expected {expected_rows}."
        )

    return df.sort_values(
        ["p", "lambda"],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def validation_selection_tables(
    df,
):
    import pandas as pd

    best_rows = []
    gain_rows = []

    for p in sorted(df["p"].unique()):
        subset = df[df["p"] == p].copy()

        best_index = subset["val_median_R"].idxmin()

        best = df.loc[best_index]

        best_rows.append(
            {
                "p": int(p),
                "best_lambda_validation": float(best["lambda"]),
                "best_gamma_validation": float(best["gamma_star"]),
                "best_val_median_R": float(best["val_median_R"]),
                "test_median_R_confirmation": float(best["test_median_R"]),
                "test_failure30_confirmation": float(best["failure_rate_30pct"]),
                "test_failure10_confirmation": float(best["failure_rate_10pct"]),
            }
        )

        baseline = subset[
            np.isclose(
                subset["lambda"],
                1.0,
            )
        ]

        if len(baseline) != 1:
            raise RuntimeError(
                f"Expected exactly one lambda=1 cell for p={p}; got {len(baseline)}."
            )

        baseline_risk = float(baseline.iloc[0]["val_median_R"])

        for _, row in subset.iterrows():
            risk = float(row["val_median_R"])

            gain_rows.append(
                {
                    "p": int(p),
                    "lambda": float(row["lambda"]),
                    "val_median_R": risk,
                    "lambda1_val_median_R": baseline_risk,
                    "relative_risk_vs_lambda1": (
                        risk / baseline_risk if baseline_risk > 0.0 else float("nan")
                    ),
                    "relative_improvement_vs_lambda1": (
                        1.0 - risk / baseline_risk
                        if baseline_risk > 0.0
                        else float("nan")
                    ),
                }
            )

    return (
        pd.DataFrame(best_rows),
        pd.DataFrame(gain_rows),
    )


def matrix_from_df(
    df,
    *,
    value_column: str,
    p_values: list[int],
    lambda_values: list[float],
) -> np.ndarray:
    matrix = np.full(
        (
            len(p_values),
            len(lambda_values),
        ),
        np.nan,
    )

    for i, p in enumerate(p_values):
        for j, lam in enumerate(lambda_values):
            match = df[
                (df["p"] == p)
                & np.isclose(
                    df["lambda"],
                    lam,
                )
            ]

            if len(match) != 1:
                raise RuntimeError(
                    f"Expected one cell for p={p}, lambda={lam}; found {len(match)}."
                )

            matrix[i, j] = float(match.iloc[0][value_column])

    return matrix


def plot_heatmap(
    *,
    matrix: np.ndarray,
    p_values: list[int],
    lambda_values: list[float],
    title: str,
    value_label: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="Blues_r",
    )

    ax.set_xticks(
        range(len(lambda_values)),
        labels=[f"{x:g}" for x in lambda_values],
    )

    ax.set_yticks(
        range(len(p_values)),
        labels=[str(x) for x in p_values],
    )

    ax.set_xlabel("Forgetting factor lambda")

    ax.set_ylabel("Innovation window p")

    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]

            if np.isfinite(value):
                rgba = image.cmap(image.norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    j,
                    i,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    color=("white" if luminance < 0.52 else "black"),
                )

    fig.colorbar(
        image,
        ax=ax,
        label=value_label,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_outputs(
    results: dict[str, Any],
    outdir: Path,
) -> None:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    df = build_summary_dataframe(results)

    summary_path = outdir / "p_lambda_interaction_summary.csv"

    df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    (
        best_df,
        gain_df,
    ) = validation_selection_tables(df)

    best_df.to_csv(
        outdir / "best_lambda_by_p_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gain_df.to_csv(
        outdir / "gain_vs_lambda1_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    p_values = [int(x) for x in results["meta"]["p_values"]]

    lambda_values = [float(x) for x in results["meta"]["lambda_values"]]

    heatmaps = [
        (
            "val_median_R",
            "Validation median risk",
            "Validation median risk",
            "01_validation_risk_heatmap.png",
        ),
        (
            "test_median_R",
            "Test median risk (confirmation only)",
            "Test median risk",
            "02_test_risk_heatmap.png",
        ),
        (
            "gamma_star",
            "Selected gamma* for every (p, lambda)",
            "gamma*",
            "03_gamma_star_heatmap.png",
        ),
        (
            "failure_rate_30pct",
            "30% threshold failure rate",
            "Failure rate",
            "04_failure30_heatmap.png",
        ),
        (
            "failure_rate_10pct",
            "10% threshold failure rate",
            "Failure rate",
            "05_failure10_heatmap.png",
        ),
    ]

    for (
        column,
        title,
        label,
        filename,
    ) in heatmaps:
        matrix = matrix_from_df(
            df,
            value_column=column,
            p_values=p_values,
            lambda_values=lambda_values,
        )

        plot_heatmap(
            matrix=matrix,
            p_values=p_values,
            lambda_values=lambda_values,
            title=title,
            value_label=label,
            output=outdir / filename,
        )

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for p in p_values:
        subset = df[df["p"] == p].sort_values("lambda")

        ax.plot(
            subset["lambda"],
            subset["val_median_R"],
            label=f"p={p}",
        )

    ax.set_xlabel("Forgetting factor lambda")

    ax.set_ylabel("Validation median risk")

    ax.set_title("p-lambda interaction: validation profiles")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "06_validation_profiles_by_p.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.plot(
        best_df["p"],
        best_df["best_lambda_validation"],
    )

    ax.axhline(
        FINAL_SCALAR_LAMBDA,
        label=(f"scalar reference lambda={FINAL_SCALAR_LAMBDA}"),
    )

    ax.set_xlabel("Innovation window p")

    ax.set_ylabel("Best lambda on validation")

    ax.set_title("Does preferred lambda change with p?")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "07_best_lambda_by_p.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def self_test() -> None:
    assert float_key(0.90) == "0.9"

    assert (
        cell_key(
            5,
            0.95,
        )
        == "p=5|lambda=0.95"
    )

    assert same_float_list(
        [0.1, 0.2],
        [0.1, 0.2],
    )

    assert not same_float_list(
        [0.1],
        [0.1, 0.2],
    )

    fake_base = {
        "per_lambda": {
            "0.95": {
                "lambda": 0.95,
                "p": 5,
                "gamma": 1.0,
                "boundary": False,
                "val": {
                    "median_R": 0.01,
                    "mean_R": 0.02,
                },
                "test": {
                    "median_R": 0.02,
                    "mean_R": 0.03,
                    "divergences": 0,
                    "median_hit_30pct_success_only": 10.0,
                    "failure_rate_30pct": 0.1,
                    "median_hit_10pct_success_only": 20.0,
                    "failure_rate_10pct": 0.2,
                },
            }
        }
    }

    record, source = reusable_p5_record(
        lambda_value=0.95,
        base=fake_base,
        closeout=None,
    )

    assert record is not None
    assert source == "reused_p5_profile"

    normalized = normalize_reused_record(
        record,
        lambda_value=0.95,
        source=source,
    )

    assert normalized["p"] == 5
    assert math.isclose(
        normalized["lambda"],
        0.95,
    )

    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the innovation-length and forgetting-factor grid"
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--family",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--base-profile",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--closeout-results",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=(SCRIPT_DIR / "p_lambda_interaction_results"),
    )

    parser.add_argument(
        "--no-reuse-p5",
        action="store_true",
        help=("Do not reuse completed p=5 profile/close-out cells."),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    base_path = find_base_profile(args.base_profile)

    base = load_json(base_path)

    validate_base_profile(base)

    base_meta = base["meta"]

    N = int(base_meta["N"])

    gamma_grid = [
        float(x)
        for x in base_meta.get(
            "gamma_grid",
            DEFAULT_GAMMA_GRID,
        )
    ]

    n_train = int(base_meta["n_train"])

    n_val = int(base_meta["n_val"])

    n_test = int(base_meta["n_test"])

    n_train_traj = int(base_meta["n_train_traj"])

    n_eval_traj = int(base_meta["n_eval_traj"])

    top_k = int(base_meta["top_k"])

    closeout_path = (
        None if args.no_reuse_p5 else find_closeout_results(args.closeout_results)
    )

    closeout = None

    if closeout_path is not None:
        closeout = load_json(closeout_path)

        try:
            validate_closeout_for_reuse(
                closeout,
                base_meta,
            )
        except ValueError as exc:
            raise SystemExit(f"Close-out results cannot be safely reused: {exc}")

    if 0.925 in LAMBDA_VALUES and not args.no_reuse_p5 and closeout is None:
        raise SystemExit(
            "The default interaction grid contains lambda=0.925, "
            "but lambda=0.925 exists only in the close-out results.\n"
            "Either keep lambda_closeout_steps1_2/lambda_closeout_results.json "
            "available, pass --closeout-results, or run with --no-reuse-p5 "
            "to recompute every p=5 cell."
        )

    project = find_project_root(args.project)

    (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        load_family,
        simulate,
    ) = import_project(project)

    if args.family is not None:
        family_path = args.family.expanduser().resolve()
    else:
        stored_family = Path(str(base_meta["family"])).expanduser()

        if stored_family.exists():
            family_path = stored_family.resolve()
        else:
            family_path = project / "data" / "systems.npz"

    if not family_path.exists():
        raise SystemExit(f"System family not found: {family_path}")

    cfg0 = Config()

    cfg = replace(
        cfg0,
        N=N,
    )

    family = load_family(family_path)

    train_systems = family["train"][:n_train]

    val_systems = family["val"][:n_val]

    test_systems = family["test"][:n_test]

    if len(train_systems) != n_train:
        raise SystemExit(
            f"Expected {n_train} train systems; found {len(train_systems)}."
        )

    if len(val_systems) != n_val:
        raise SystemExit(
            f"Expected {n_val} validation systems; found {len(val_systems)}."
        )

    if len(test_systems) != n_test:
        raise SystemExit(f"Expected {n_test} test systems; found {len(test_systems)}.")

    outdir = args.outdir.expanduser().resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = outdir / "p_lambda_interaction_checkpoint.json"

    final_json_path = outdir / "p_lambda_interaction_results.json"

    signature = protocol_signature(
        N=N,
        gamma_grid=gamma_grid,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_train_traj=n_train_traj,
        n_eval_traj=n_eval_traj,
        top_k=top_k,
    )

    results: dict[str, Any] = {
        "meta": {
            **signature,
            "project": str(project),
            "family": str(family_path),
            "base_profile": str(base_path),
            "closeout_results": (
                str(closeout_path) if closeout_path is not None else None
            ),
            "p_values": [int(x) for x in P_VALUES],
            "lambda_values": [float(x) for x in LAMBDA_VALUES],
            "scalar_lambda_reference": FINAL_SCALAR_LAMBDA,
            "selection_split": "validation",
            "test_usage": "confirmation only",
        },
        "cells": {},
    }

    if checkpoint_path.exists():
        saved = load_json(checkpoint_path)

        saved_meta = saved.get(
            "meta",
            {},
        )

        protocol_matches = (
            int(
                saved_meta.get(
                    "N",
                    -1,
                )
            )
            == N
            and saved_meta.get("p_values") == [int(x) for x in P_VALUES]
            and saved_meta.get("lambda_values") == [float(x) for x in LAMBDA_VALUES]
            and same_float_list(
                saved_meta.get(
                    "gamma_grid",
                    [],
                ),
                gamma_grid,
            )
            and int(
                saved_meta.get(
                    "n_train",
                    -1,
                )
            )
            == n_train
            and int(
                saved_meta.get(
                    "n_val",
                    -1,
                )
            )
            == n_val
            and int(
                saved_meta.get(
                    "n_test",
                    -1,
                )
            )
            == n_test
            and int(
                saved_meta.get(
                    "n_train_traj",
                    -1,
                )
            )
            == n_train_traj
            and int(
                saved_meta.get(
                    "n_eval_traj",
                    -1,
                )
            )
            == n_eval_traj
            and int(
                saved_meta.get(
                    "top_k",
                    -1,
                )
            )
            == top_k
        )

        if protocol_matches:
            results = saved

            print(
                "Checkpoint loaded: "
                f"{len(results['cells'])} / "
                f"{len(P_VALUES) * len(LAMBDA_VALUES)} "
                "cells complete."
            )
        else:
            print(
                "Existing checkpoint does not match "
                "the current interaction protocol; "
                "starting fresh."
            )

    if not args.no_reuse_p5:
        for lambda_value in LAMBDA_VALUES:
            key = cell_key(
                5,
                lambda_value,
            )

            if key in results["cells"]:
                continue

            record, source = reusable_p5_record(
                lambda_value=lambda_value,
                base=base,
                closeout=closeout,
            )

            if record is None or source is None:
                continue

            normalized = normalize_reused_record(
                record,
                lambda_value=lambda_value,
                source=source,
            )

            results["cells"][key] = normalized

        json_dump(
            results,
            checkpoint_path,
        )

    total_cells = len(P_VALUES) * len(LAMBDA_VALUES)

    print("=" * 78)
    print("MISG p × lambda interaction")
    print(f"project          : {project}")
    print(f"family           : {family_path}")
    print(f"output           : {outdir}")
    print(f"N                : {N}")
    print(f"p grid           : {P_VALUES}")
    print(f"lambda grid      : {LAMBDA_VALUES}")
    print(f"scalar reference : {FINAL_SCALAR_LAMBDA}")
    print(f"gamma grid       : {gamma_grid}")
    print(f"splits           : train={n_train}, val={n_val}, test={n_test}")
    print(f"reused cells     : {len(results['cells'])}")
    print(f"total cells      : {total_cells}")
    print("=" * 78)

    remaining_cells = [
        (
            p,
            lambda_value,
        )
        for p in P_VALUES
        for lambda_value in LAMBDA_VALUES
        if cell_key(
            p,
            lambda_value,
        )
        not in results["cells"]
    ]

    if remaining_cells:
        print("\nBuilding shared trajectories once...")

        train_pairs = build_pairs(
            train_systems,
            j_start=TRAIN_TRAJ_START,
            n_traj=n_train_traj,
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        val_pairs = build_pairs(
            val_systems,
            j_start=EVAL_TRAJ_START,
            n_traj=n_eval_traj,
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        test_pairs = build_pairs(
            test_systems,
            j_start=EVAL_TRAJ_START,
            n_traj=n_eval_traj,
            cfg=cfg,
            traj_seed=traj_seed,
            simulate=simulate,
        )

        start_all = time.perf_counter()

        for (
            p,
            lambda_value,
        ) in remaining_cells:
            key = cell_key(
                p,
                lambda_value,
            )

            print("\n" + "#" * 78)

            print(f"p={p}, lambda={lambda_value:g}")

            print("#" * 78)

            start = time.perf_counter()

            record = retune_gamma(
                p=p,
                lambda_value=lambda_value,
                gamma_grid=gamma_grid,
                top_k=top_k,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                test_pairs=test_pairs,
                cfg=cfg,
                misg_run=misg_run,
                risk_R=risk_R,
            )

            results["cells"][key] = record

            json_dump(
                results,
                checkpoint_path,
            )

            test = record["test"]

            print(
                f"gamma*={record['gamma']:.3g}  "
                f"val_R={record['val']['median_R']:.6f}  "
                f"test_R={test['median_R']:.6f}  "
                f"T30={test['median_hit_30pct_success_only']:.1f}  "
                f"fail30={test['failure_rate_30pct']:.3f}  "
                f"T10={test['median_hit_10pct_success_only']:.1f}  "
                f"fail10={test['failure_rate_10pct']:.3f}  "
                f"div={test['divergences']}  "
                f"boundary={record['boundary']}  "
                f"time={time.perf_counter() - start:.1f}s"
            )

        results["meta"]["new_computation_elapsed_seconds"] = float(
            time.perf_counter() - start_all
        )
    else:
        print("\nAll interaction cells already exist.")

    missing = [
        cell_key(
            p,
            lambda_value,
        )
        for p in P_VALUES
        for lambda_value in LAMBDA_VALUES
        if cell_key(
            p,
            lambda_value,
        )
        not in results["cells"]
    ]

    if missing:
        raise RuntimeError(
            "Interaction run ended with missing cells: " + ", ".join(missing)
        )

    boundary_hits = []

    for record in results["cells"].values():
        if record["boundary"]:
            boundary_hits.append(
                {
                    "p": int(record["p"]),
                    "lambda": float(record["lambda"]),
                    "gamma": float(record["gamma"]),
                }
            )

    results["meta"]["boundary_hits"] = boundary_hits

    results["meta"]["all_gamma_optima_interior"] = not bool(boundary_hits)

    write_outputs(
        results,
        outdir,
    )

    json_dump(
        results,
        final_json_path,
    )

    print("\n" + "=" * 78)

    print("DONE")

    print(f"JSON : {final_json_path}")

    print(f"CSV  : {outdir / 'p_lambda_interaction_summary.csv'}")

    print(f"Best lambda by p: {outdir / 'best_lambda_by_p_validation.csv'}")

    if boundary_hits:
        print("\nWARNING: gamma* hit the search-grid boundary for these cells:")

        for item in boundary_hits:
            print(f"  p={item['p']}, lambda={item['lambda']}, gamma*={item['gamma']}")

        print("Expand gamma grid for those cells before treating them as final.")
    else:
        print("\nAll selected gamma* values are interior to the gamma grid.")

    print("=" * 78)


if __name__ == "__main__":
    main()
