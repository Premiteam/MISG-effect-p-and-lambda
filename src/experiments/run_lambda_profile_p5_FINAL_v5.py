from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

P_FIXED = 5
STATE_MODE = "observer"

LAMBDA_VALUES = [
    1.00,
    0.99,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
]

GAMMA_GRID = [
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
            f"--project does not look like the state-estimation project root: {root}"
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
        "Could not find the state-estimation project automatically.\n"
        "Either keep this script under the same 'State Estimation' folder, or run:\n"
        "python run_lambda_profile_p5_FINAL.py --project '/full/path/to/state-estimation'"
    )


def import_project(project: Path):
    src = project / "src" / "core"
    sys.path.insert(0, str(src))

    from config import Config, traj_seed
    from estimators import misg_run
    from metrics import risk_R
    from systems import load_family, simulate

    return Config, traj_seed, misg_run, risk_R, load_family, simulate


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
            trajectory = simulate(system_item, rng, cfg.N)
            pairs.append((system_index, system_item, trajectory))

    return pairs


def capped_risk(result, system_item, cfg, risk_R) -> float:
    value = risk_R(
        result["rho_hat_traj"],
        system_item["rho_star"],
        cfg.N,
    )

    if not np.isfinite(value):
        return float(cfg.r_cap)

    return min(float(value), float(cfg.r_cap))


def relative_error_series(
    rho_hat_traj: np.ndarray,
    rho_star: np.ndarray,
    N: int,
) -> np.ndarray:
    rho = np.asarray(rho_hat_traj, dtype=float)
    truth = np.asarray(rho_star, dtype=float)

    denominator = float(np.linalg.norm(truth))

    if denominator <= 0.0:
        return np.full(N, np.nan)

    if rho.ndim != 2 or rho.shape[1] != truth.size:
        raise ValueError(
            f"Unexpected rho_hat_traj shape {rho.shape}; "
            f"expected parameter dimension {truth.size}."
        )

    if rho.shape[0] >= N + 1:
        sequence = rho[1 : N + 1]
    elif rho.shape[0] == N:
        sequence = rho
    else:
        raise ValueError(f"rho_hat_traj has {rho.shape[0]} rows but N={N}.")

    return np.linalg.norm(sequence - truth[None, :], axis=1) / denominator


def first_hit_time(
    errors: np.ndarray,
    threshold: float,
) -> float:
    indices = np.flatnonzero(np.isfinite(errors) & (errors <= threshold))

    if indices.size == 0:
        return float("nan")

    return float(indices[0] + 1)


def mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)

    if not np.any(finite):
        return float("nan")

    return float(np.mean(arr[finite]))


def evaluate(
    pairs,
    *,
    gamma: float,
    lambda_value: float,
    cfg,
    misg_run,
    risk_R,
) -> dict[str, Any]:
    weights = np.ones(P_FIXED, dtype=float)

    r_mode = "cumulative" if abs(lambda_value - 1.0) < 1e-12 else "forgetting"

    risk_by_system: dict[int, list[float]] = {}
    hit10_by_system: dict[int, list[float]] = {}
    hit30_by_system: dict[int, list[float]] = {}

    divergences = 0

    for system_index, system_item, trajectory in pairs:
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

            hit10 = first_hit_time(errors, 0.10)
            hit30 = first_hit_time(errors, 0.30)

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

    train_order = np.argsort([record["mean_R"] for record in train_scan])

    candidate_indices = [
        int(index) for index in train_order[: min(top_k, len(train_order))]
    ]

    validation_candidates = []
    best_candidate = None

    for index in candidate_indices:
        gamma = float(train_scan[index]["gamma"])

        val_eval = evaluate(
            val_pairs,
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

        validation_candidates.append(candidate)

        if (
            best_candidate is None
            or val_eval["median_R"] < best_candidate["val"]["median_R"]
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No gamma candidate was selected.")

    gamma_star = float(best_candidate["gamma"])

    selected_index = gamma_grid.index(gamma_star)

    boundary = selected_index in (
        0,
        len(gamma_grid) - 1,
    )

    test_eval = evaluate(
        test_pairs,
        gamma=gamma_star,
        lambda_value=lambda_value,
        cfg=cfg,
        misg_run=misg_run,
        risk_R=risk_R,
    )

    return {
        "lambda": float(lambda_value),
        "p": P_FIXED,
        "gamma": gamma_star,
        "boundary": bool(boundary),
        "train_mean_R": float(best_candidate["train_mean_R"]),
        "val_selection_risk": float(best_candidate["val"]["median_R"]),
        "train_scan": train_scan,
        "val_candidates": validation_candidates,
        "val": best_candidate["val"],
        "test": test_eval,
    }


def json_dump(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )


def save_summary_csv(
    results: dict[str, Any],
    outdir: Path,
):
    import pandas as pd

    rows = []

    for lambda_key, record in results["per_lambda"].items():
        test = record["test"]

        rows.append(
            {
                "lambda": float(lambda_key),
                "p": P_FIXED,
                "gamma_star": record["gamma"],
                "boundary": record["boundary"],
                "val_median_R": record["val"]["median_R"],
                "test_median_R": test["median_R"],
                "test_mean_R": test["mean_R"],
                "test_divergences": test["divergences"],
                "median_hit_30pct_success_only": test["median_hit_30pct_success_only"],
                "n_failed_30pct": test["n_failed_30pct"],
                "failure_rate_30pct": test["failure_rate_30pct"],
                "median_hit_10pct_success_only": test["median_hit_10pct_success_only"],
                "n_failed_10pct": test["n_failed_10pct"],
                "failure_rate_10pct": test["failure_rate_10pct"],
                "n_systems": test["n_systems"],
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "lambda",
        ascending=False,
    )

    df.to_csv(
        outdir / "lambda_profile_p5_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_results(
    results: dict[str, Any],
    outdir: Path,
):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    lambda_values = sorted(float(key) for key in results["per_lambda"].keys())

    def values_from(path: list[str]) -> list[float]:
        output = []

        for value in lambda_values:
            obj: Any = results["per_lambda"][str(value)]

            for key in path:
                obj = obj[key]

            output.append(float(obj))

        return output

    figures = [
        (
            ["test", "median_R"],
            "Median test risk R",
            "Lambda profile: test risk",
            "01_lambda_risk.png",
        ),
        (
            ["test", "median_hit_30pct_success_only"],
            "Median steps to reach 30% error",
            "Lambda profile: 30% convergence",
            "02_lambda_T30.png",
        ),
        (
            ["test", "median_hit_10pct_success_only"],
            "Median steps to reach 10% error",
            "Lambda profile: 10% convergence",
            "03_lambda_T10.png",
        ),
        (
            ["gamma"],
            "Selected gamma*",
            "Lambda profile: selected gamma*",
            "04_lambda_gamma.png",
        ),
    ]

    for path, ylabel, title, filename in figures:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        ax.plot(
            lambda_values,
            values_from(path),
            marker="o",
        )

        ax.set_xlabel("Forgetting factor lambda")

        ax.set_ylabel(ylabel)

        ax.set_title(f"{title} (fixed p={P_FIXED})")

        ax.grid(True)

        fig.tight_layout()

        fig.savefig(
            outdir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        lambda_values,
        values_from(["test", "failure_rate_30pct"]),
        marker="o",
        label="30% threshold",
    )

    ax.plot(
        lambda_values,
        values_from(["test", "failure_rate_10pct"]),
        marker="o",
        label="10% threshold",
    )

    ax.set_xlabel("Forgetting factor lambda")

    ax.set_ylabel("Failure rate")

    ax.set_title(f"Lambda profile: threshold failure rates (fixed p={P_FIXED})")

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "05_lambda_failure_rates.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Profile forgetting factors for p=5")

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
        "--outdir",
        type=Path,
        default=SCRIPT_DIR,
        help=("Output directory. Default: the same folder as this script."),
    )

    parser.add_argument(
        "--N",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--n-train-traj",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--n-eval-traj",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--gamma-grid",
        nargs="+",
        type=float,
        default=GAMMA_GRID,
    )

    parser.add_argument(
        "--n-val",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=("Smoke test with fewer systems, N<=2000, and a reduced gamma grid."),
    )

    args = parser.parse_args()

    if args.N < 100:
        parser.error("--N is unrealistically small.")

    if not LAMBDA_VALUES:
        parser.error("Lambda grid is empty.")

    if any(not (0.0 < value <= 1.0) for value in LAMBDA_VALUES):
        parser.error("Every lambda value must satisfy 0 < lambda <= 1.")

    if not args.gamma_grid:
        parser.error("Gamma grid is empty.")

    if min(args.gamma_grid) <= 0.0:
        parser.error("Every gamma value must be positive.")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    project = find_project_root(args.project)

    (
        Config,
        traj_seed,
        misg_run,
        risk_R,
        load_family,
        simulate,
    ) = import_project(project)

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.family is None:
        family_path = project / "data" / "systems.npz"
    else:
        family_path = args.family.expanduser().resolve()

    if not family_path.exists():
        raise SystemExit(f"Family NPZ not found: {family_path}")

    if args.quick:
        args.N = min(args.N, 2000)
        args.n_train = min(args.n_train, 12)

        args.n_val = 12 if args.n_val is None else min(args.n_val, 12)

        args.n_test = 12 if args.n_test is None else min(args.n_test, 12)

        args.gamma_grid = [
            0.05,
            0.10,
            0.20,
            0.30,
            0.50,
            1.00,
            2.00,
            4.00,
        ]

    cfg0 = Config()
    cfg = replace(
        cfg0,
        N=int(args.N),
    )

    family = load_family(family_path)

    train_systems = family["train"][: int(args.n_train)]

    val_systems = family["val"]
    test_systems = family["test"]

    if args.n_val is not None:
        val_systems = val_systems[: int(args.n_val)]

    if args.n_test is not None:
        test_systems = test_systems[: int(args.n_test)]

    print("=" * 72)
    print("MISG lambda profile experiment")
    print(f"project      : {project}")
    print(f"family       : {family_path}")
    print(f"output       : {outdir}")
    print(f"N            : {cfg.N}")
    print(f"fixed p      : {P_FIXED}")
    print(f"lambda grid  : {LAMBDA_VALUES}")
    print(f"gamma grid   : {args.gamma_grid}")
    print(
        "splits       : "
        f"train={len(train_systems)}, "
        f"val={len(val_systems)}, "
        f"test={len(test_systems)}"
    )
    print("=" * 72)

    print("\nBuilding shared trajectories...")

    train_pairs = build_pairs(
        train_systems,
        j_start=TRAIN_TRAJ_START,
        n_traj=args.n_train_traj,
        cfg=cfg,
        traj_seed=traj_seed,
        simulate=simulate,
    )

    val_pairs = build_pairs(
        val_systems,
        j_start=EVAL_TRAJ_START,
        n_traj=args.n_eval_traj,
        cfg=cfg,
        traj_seed=traj_seed,
        simulate=simulate,
    )

    test_pairs = build_pairs(
        test_systems,
        j_start=EVAL_TRAJ_START,
        n_traj=args.n_eval_traj,
        cfg=cfg,
        traj_seed=traj_seed,
        simulate=simulate,
    )

    checkpoint_path = outdir / "lambda_profile_p5_checkpoint.json"

    results: dict[str, Any] = {
        "meta": {
            "project": str(project),
            "family": str(family_path),
            "N": int(cfg.N),
            "p_fixed": P_FIXED,
            "lambda_values": [float(value) for value in LAMBDA_VALUES],
            "gamma_grid": [float(value) for value in args.gamma_grid],
            "n_train": len(train_systems),
            "n_val": len(val_systems),
            "n_test": len(test_systems),
            "n_train_traj": int(args.n_train_traj),
            "n_eval_traj": int(args.n_eval_traj),
            "top_k": int(args.top_k),
            "quick": bool(args.quick),
        },
        "per_lambda": {},
    }

    if checkpoint_path.exists():
        try:
            with checkpoint_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                saved = json.load(f)

            if (
                saved.get("meta", {}).get("p_fixed") == P_FIXED
                and saved.get("meta", {}).get("N") == cfg.N
                and "per_lambda" in saved
            ):
                results["per_lambda"] = saved["per_lambda"]

                print(
                    "Resume checkpoint loaded: "
                    f"{len(results['per_lambda'])} "
                    "lambda values already completed."
                )
        except Exception as exc:
            print(f"Checkpoint could not be loaded; starting fresh. ({exc})")

    total_start = time.perf_counter()

    for lambda_value in LAMBDA_VALUES:
        lambda_value = float(lambda_value)

        key = str(lambda_value)

        if key in results["per_lambda"]:
            print(f"\nlambda={lambda_value:g} already completed -> skip.")
            continue

        print("\n" + "#" * 72)
        print(f"lambda={lambda_value:g}, p={P_FIXED}")
        print("#" * 72)

        start = time.perf_counter()

        record = retune_gamma(
            lambda_value=lambda_value,
            gamma_grid=[float(value) for value in args.gamma_grid],
            top_k=int(args.top_k),
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            cfg=cfg,
            misg_run=misg_run,
            risk_R=risk_R,
        )

        results["per_lambda"][key] = record

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
            f"fail30={test['n_failed_30pct']}  "
            f"T10={test['median_hit_10pct_success_only']:.1f}  "
            f"fail10={test['n_failed_10pct']}  "
            f"div={test['divergences']}  "
            f"boundary={record['boundary']}  "
            f"time={time.perf_counter() - start:.1f}s"
        )

    results["meta"]["elapsed_seconds"] = float(time.perf_counter() - total_start)

    boundary_hits = []

    for key, record in results["per_lambda"].items():
        if record["boundary"]:
            boundary_hits.append(
                {
                    "lambda": float(key),
                    "gamma": float(record["gamma"]),
                }
            )

    results["meta"]["boundary_hits"] = boundary_hits

    results["meta"]["all_gamma_optima_interior"] = not bool(boundary_hits)

    final_json = outdir / "lambda_profile_p5_results.json"

    json_dump(
        results,
        final_json,
    )

    save_summary_csv(
        results,
        outdir,
    )

    plot_results(
        results,
        outdir,
    )

    print("\n" + "=" * 72)
    print("DONE")
    print(f"JSON    : {final_json}")
    print(f"CSV     : {outdir / 'lambda_profile_p5_summary.csv'}")
    print(
        "Figures : "
        "01_lambda_risk.png, "
        "02_lambda_T30.png, "
        "03_lambda_T10.png, "
        "04_lambda_gamma.png, "
        "05_lambda_failure_rates.png"
    )

    if boundary_hits:
        print("\nWARNING: selected gamma* reached the gamma-grid boundary:")

        for item in boundary_hits:
            print(f"  lambda={item['lambda']}, gamma*={item['gamma']}")

        print("Expand the gamma grid before treating those lambda points as final.")
    else:
        print("\nAll selected gamma* values are inside the gamma grid.")

    print("=" * 72)


if __name__ == "__main__":
    main()
