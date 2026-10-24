from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (
    Config,
    pilot_seed,
    pilot_traj_seed,
    q_select_seed,
    repro_seed,
    sys_seed,
    timing_select_seed,
    traj_seed,
)
from estimators import misg_run
from metrics import divergence_rate, risk_R
from systems import (
    family_manifest,
    gram_diagnostics,
    load_family,
    sample_family,
    save_family,
    simulate,
)


Q_VALUES = (1e-6, 1e-4, 1e-2)
STATE_MODES = ("true", "observer", "kalman")
MANIFEST_FIELDS = (
    "m",
    "split",
    "d",
    "rho_F",
    "snr_db",
    "sigma_v",
    "rho_star_norm",
    "n_reject",
)


def _write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_components(seed: np.random.SeedSequence) -> list[int]:
    entropy = seed.entropy
    if isinstance(entropy, Sequence):
        return [int(value) for value in entropy]
    return [int(entropy)]


def _select_stratified(
    systems: Sequence[dict[str, Any]],
    d_choices: Sequence[int],
    n_per_d: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if n_per_d < 1:
        raise ValueError("n_per_d must be positive")
    selected: list[dict[str, Any]] = []
    for d in d_choices:
        candidates = [system for system in systems if int(system["d"]) == int(d)]
        if n_per_d > len(candidates):
            raise ValueError(
                f"requested {n_per_d} systems for d={d}, only {len(candidates)} exist"
            )
        indices = rng.choice(len(candidates), size=n_per_d, replace=False)
        selected.extend(candidates[int(index)] for index in np.atleast_1d(indices))
    return selected


def _example_2017() -> dict[str, Any]:
    A = np.array([[0.0, 1.0], [-0.01, -0.22]])
    B = np.array([[0.19, -0.08], [0.16, -0.12]])
    f = np.array([0.8, -0.8])
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": 2,
        "n": 2,
        "sigma_v": 0.30,
        "rho_star": np.array([-0.01, -0.22, 0.19, -0.08, 0.16, -0.12, 0.8, -0.8]),
        "split": "repro2017",
        "m": 0,
    }


def _family_outputs(
    cfg: Config,
    outdir: Path,
    figures_dir: Path,
    seed_log: dict[str, Any],
) -> dict[str, Any]:
    family = sample_family(cfg)
    systems_path = outdir / "systems.npz"
    save_family(family, systems_path)
    manifest_rows, counts = family_manifest(family)

    for split in ("train", "val", "test", "pilot"):
        expected = getattr(cfg, f"n_{split}") // len(cfg.d_choices)
        assert counts[split] == {d: expected for d in cfg.d_choices}

    _write_csv(outdir / "manifest.csv", MANIFEST_FIELDS, manifest_rows)
    split_rows = [
        {"split": split, "d": d, "count": counts[split][d]}
        for split in ("train", "val", "test", "pilot")
        for d in cfg.d_choices
    ]
    _write_csv(outdir / "split_by_d.csv", ("split", "d", "count"), split_rows)

    gram_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test", "pilot"):
        for system in family[split]:
            m = int(system["m"])
            if split == "pilot":
                system_seed = pilot_seed(cfg, m)
                trajectory_seed = pilot_traj_seed(cfg, m, 0)
            else:
                system_seed = sys_seed(cfg, m)
                trajectory_seed = traj_seed(cfg, m, 0)
            seed_log["system_generation"].append(
                {
                    "split": split,
                    "m": m,
                    "components": _seed_components(system_seed),
                }
            )
            seed_log["gram_trajectories"].append(
                {
                    "split": split,
                    "m": m,
                    "j": 0,
                    "components": _seed_components(trajectory_seed),
                }
            )
            trajectory = simulate(
                system,
                np.random.default_rng(trajectory_seed),
                N=cfg.N,
            )
            diagnostics = gram_diagnostics(system, trajectory)
            gram_rows.append(
                {
                    "m": m,
                    "split": split,
                    "d": int(system["d"]),
                    "rank": diagnostics["rank"],
                    "cond": diagnostics["cond"],
                    "eig_min": diagnostics["eig_min"],
                    "eig_max": diagnostics["eig_max"],
                }
            )
    _write_csv(
        outdir / "gram_diagnostics.csv",
        ("m", "split", "d", "rank", "cond", "eig_min", "eig_max"),
        gram_rows,
    )

    cond_values = np.asarray([row["cond"] for row in gram_rows], dtype=float)
    rho_F = np.asarray([row["rho_F"] for row in manifest_rows], dtype=float)
    snr_db = np.asarray([row["snr_db"] for row in manifest_rows], dtype=float)
    rho_norm = np.asarray([row["rho_star_norm"] for row in manifest_rows], dtype=float)
    finite_log_cond = np.log10(cond_values[np.isfinite(cond_values)])
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    panels = (
        (rho_F, r"$\rho(F)$"),
        (snr_db, "SNR (dB)"),
        (rho_norm, r"$\|\rho^\star\|$"),
        (finite_log_cond, r"$\log_{10}\,\mathrm{cond}(G)$"),
    )
    for axis, (values, label) in zip(axes.flat, panels):
        axis.hist(values, bins=30, color="#4472C4", alpha=0.82)
        axis.set_xlabel(label)
        axis.set_ylabel("count")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / "family_hist.png", dpi=180)
    plt.close(figure)

    full_rank = {
        int(d): int(cfg.n**2 + 2 * cfg.n) if int(d) >= cfg.n else 0
        for d in cfg.d_choices
    }
    rank_deficient = sum(
        int(row["rank"]) < full_rank[int(row["d"])] for row in gram_rows
    )
    return {
        "family": family,
        "counts": counts,
        "manifest_rows": manifest_rows,
        "gram_rows": gram_rows,
        "rank_deficient": int(rank_deficient),
        "cond_median": float(np.median(cond_values)),
        "cond_q99": float(np.quantile(cond_values, 0.99)),
    }


def _p_min_eigenvalue(P_traj: np.ndarray, stop: int) -> float:
    covariances = np.asarray(P_traj[:stop])
    if covariances.size == 0 or not np.all(np.isfinite(covariances)):
        return float("nan")
    return float(np.min(np.linalg.eigvalsh(covariances)))


def _q_sensitivity(
    cfg: Config,
    family: Mapping[str, list[dict[str, Any]]],
    outdir: Path,
    q_n_per_d: int,
    seed_log: dict[str, Any],
) -> dict[str, Any]:
    selection_seed = q_select_seed(cfg)
    selected = _select_stratified(
        family["val"],
        cfg.d_choices,
        q_n_per_d,
        np.random.default_rng(selection_seed),
    )
    seed_log["q_selection"] = _seed_components(selection_seed)
    rows: list[dict[str, Any]] = []
    for system in selected:
        assert system["split"] != "test"
        m = int(system["m"])
        trajectory_seed = traj_seed(cfg, m, 3)
        seed_log["q_trajectories"].append(
            {
                "m": m,
                "d": int(system["d"]),
                "j": 3,
                "components": _seed_components(trajectory_seed),
            }
        )
        trajectory = simulate(
            system,
            np.random.default_rng(trajectory_seed),
            N=cfg.N,
        )
        seed_text = json.dumps(_seed_components(trajectory_seed), separators=(",", ":"))
        for q in Q_VALUES:
            assert system["split"] != "test"
            result = misg_run(
                system,
                trajectory,
                np.ones(cfg.p_main),
                1.0,
                "cumulative",
                "kalman",
                cfg,
                q=q,
                return_diagnostics=True,
            )
            stop = (
                cfg.N if result["diverged_at"] is None else int(result["diverged_at"])
            )
            rows.append(
                {
                    "m": m,
                    "d": int(system["d"]),
                    "j": 3,
                    "trajectory_seed": seed_text,
                    "q": q,
                    "R": risk_R(result["rho_hat_traj"], system["rho_star"], cfg.N),
                    "status": result["status"],
                    "P_min_eig_over_run": _p_min_eigenvalue(result["P_traj"], stop),
                }
            )
    _write_csv(
        outdir / "q_sensitivity.csv",
        (
            "m",
            "d",
            "j",
            "trajectory_seed",
            "q",
            "R",
            "status",
            "P_min_eig_over_run",
        ),
        rows,
    )

    summaries: dict[float, dict[str, float]] = {}
    for q in Q_VALUES:
        q_rows = [row for row in rows if float(row["q"]) == q]
        risks = np.asarray([row["R"] for row in q_rows], dtype=float)
        statuses = [str(row["status"]) for row in q_rows]
        summaries[q] = {
            "median_R": float(np.median(risks)),
            "divergence_rate": float(divergence_rate(statuses)),
        }
    return {
        "rows": rows,
        "selected": selected,
        "summaries": summaries,
    }


def _divergence_audit(
    cfg: Config,
    outdir: Path,
    seed_log: dict[str, Any],
) -> dict[str, Any]:
    system = _example_2017()
    N = max(60, cfg.div_spec_patience + 10)
    trajectory_seed = repro_seed(cfg, 0)
    seed_log["divergence_trajectory"] = _seed_components(trajectory_seed)
    trajectory = simulate(
        system,
        np.random.default_rng(trajectory_seed),
        N=N,
    )
    rho_star = np.asarray(system["rho_star"])

    rho_nan = rho_star.copy()
    rho_nan[0] = np.nan
    rho_inf = rho_star.copy()
    rho_inf[0] = np.inf
    rho_norm = np.full(8, 200 * 1.18785521 / np.sqrt(8))
    rho_spec = np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rho_ok = np.array([0.0, 1.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cases = (
        (1, "rho_star with element 0 = NaN", rho_nan, "nan", 1),
        (2, "rho_star with element 0 = Inf", rho_inf, "nan", 1),
        (3, "all entries = 200*1.18785521/sqrt(8)", rho_norm, "norm", 1),
        (4, "a=[0,1.5], B=0, f=0", rho_spec, "spec", cfg.div_spec_patience),
        (5, "a=[0,1.19], B=0, f=0", rho_ok, None, None),
    )

    rows: list[dict[str, Any]] = []
    for case, description, rho_hat0, expected_reason, expected_at in cases:
        assert system["split"] != "test"
        result = misg_run(
            system,
            trajectory,
            np.ones(1),
            1.0,
            "cumulative",
            "true",
            cfg,
            rho_hat0=rho_hat0,
            freeze_rho=True,
        )
        if case == 5:
            match = (
                result["status"] == "ok"
                and result["div_reason"] is None
                and result["diverged_at"] is None
            )
        else:
            match = (
                result["status"] == "diverged"
                and result["div_reason"] == expected_reason
                and result["diverged_at"] == expected_at
            )
        rows.append(
            {
                "case": case,
                "rho_hat0_desc": description,
                "expected_reason": expected_reason,
                "div_reason": result["div_reason"],
                "expected_at": expected_at,
                "diverged_at": result["diverged_at"],
                "status": result["status"],
                "match": bool(match),
            }
        )
    _write_csv(
        outdir / "logs" / "divergence_log.csv",
        (
            "case",
            "rho_hat0_desc",
            "expected_reason",
            "div_reason",
            "expected_at",
            "diverged_at",
            "status",
            "match",
        ),
        rows,
    )
    assert all(row["match"] for row in rows)
    return {"rows": rows, "passed": True}


def _timing_audit(
    cfg: Config,
    family: Mapping[str, list[dict[str, Any]]],
    outdir: Path,
    *,
    timing_reps: int,
    timing_warmup: int,
    timing_N: int,
    timing_n_per_d: int,
    seed_log: dict[str, Any],
) -> dict[str, Any]:
    if timing_reps < 1 or timing_warmup < 0 or timing_N < 1:
        raise ValueError("timing scales must be positive, with non-negative warmup")
    selection_seed = timing_select_seed(cfg)
    selected = _select_stratified(
        family["val"],
        cfg.d_choices,
        timing_n_per_d,
        np.random.default_rng(selection_seed),
    )
    seed_log["timing_selection"] = _seed_components(selection_seed)
    trajectories: dict[int, dict[str, Any]] = {}
    for system in selected:
        assert system["split"] != "test"
        m = int(system["m"])
        trajectory_seed = traj_seed(cfg, m, 4)
        seed_log["timing_trajectories"].append(
            {
                "m": m,
                "d": int(system["d"]),
                "j": 4,
                "components": _seed_components(trajectory_seed),
            }
        )
        trajectories[m] = simulate(
            system,
            np.random.default_rng(trajectory_seed),
            N=timing_N,
        )

    rows: list[dict[str, Any]] = []
    for state_mode in STATE_MODES:
        for p in cfg.p_choices:
            weights = np.ones(p)
            elapsed_values: list[float] = []
            for system in selected:
                assert system["split"] != "test"
                trajectory = trajectories[int(system["m"])]
                for _ in range(timing_warmup):
                    assert system["split"] != "test"
                    misg_run(
                        system,
                        trajectory,
                        weights,
                        1.0,
                        "cumulative",
                        state_mode,
                        cfg,
                        freeze_rho=False,
                        return_diagnostics=False,
                    )
                for _ in range(timing_reps):
                    assert system["split"] != "test"
                    start = time.perf_counter()
                    misg_run(
                        system,
                        trajectory,
                        weights,
                        1.0,
                        "cumulative",
                        state_mode,
                        cfg,
                        freeze_rho=False,
                        return_diagnostics=False,
                    )
                    elapsed_values.append(time.perf_counter() - start)
            rows.append(
                {
                    "state_mode": state_mode,
                    "p": int(p),
                    "N": timing_N,
                    "n_systems": len(selected),
                    "reps": timing_reps,
                    "median_sec": float(np.median(elapsed_values)),
                    "min_sec": float(np.min(elapsed_values)),
                    "max_sec": float(np.max(elapsed_values)),
                }
            )
    _write_csv(
        outdir / "timing.csv",
        (
            "state_mode",
            "p",
            "N",
            "n_systems",
            "reps",
            "median_sec",
            "min_sec",
            "max_sec",
        ),
        rows,
    )

    environment = {
        "processor": platform.processor(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
    }
    (outdir / "logs" / "env.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"rows": rows, "selected": selected, "environment": environment}


def _summary_text(
    cfg: Config,
    outdir: Path,
    family_result: Mapping[str, Any],
    q_result: Mapping[str, Any],
    divergence_result: Mapping[str, Any],
    timing_result: Mapping[str, Any],
    csv_hashes: Mapping[str, str],
) -> str:
    q_ids = [int(system["m"]) for system in q_result["selected"]]
    timing_ids = [int(system["m"]) for system in timing_result["selected"]]
    lines = [
        "# E0 infrastructure checks",
        "",
        f"- run-id: `{outdir.name}`",
        f"- NumPy: `{np.__version__}`",
        f"- SciPy: `{scipy.__version__}`",
        "",
        "## Config",
        "",
        "```python",
        pformat(asdict(cfg), sort_dicts=False),
        "```",
        "",
        "## (1) E0-2 system family and Gram diagnostics — PASS",
        "",
        f"- Split counts: `{family_result['counts']}`",
        f"- Rank-deficient systems: `{family_result['rank_deficient']}` "
        "(reported only; no system was filtered)",
        f"- Gram condition median: `{family_result['cond_median']:.8g}`",
        f"- Gram condition 99% quantile: `{family_result['cond_q99']:.8g}`",
        "",
        "## (2) E0-4b q sensitivity — PASS",
        "",
        f"- Stratified selected val m: `{q_ids}`",
        "",
        "| q | median R | divergence rate |",
        "|---:|---:|---:|",
    ]
    for q in Q_VALUES:
        summary = q_result["summaries"][q]
        lines.append(
            f"| {q:.0e} | {summary['median_R']:.8g} | "
            f"{summary['divergence_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "q=1e-4 是预注册值；本表只作稳健性报告，不得据此更换 q。",
            "",
            "## (3) E0-5 divergence logging — "
            f"{'PASS' if divergence_result['passed'] else 'FAIL'}",
            "",
            "| case | expected reason | actual reason | expected at | actual at | status | match |",
            "|---:|---|---|---:|---:|---|---|",
        ]
    )
    for row in divergence_result["rows"]:
        lines.append(
            f"| {row['case']} | {row['expected_reason']} | "
            f"{row['div_reason']} | {row['expected_at']} | "
            f"{row['diverged_at']} | {row['status']} | {row['match']} |"
        )
    lines.extend(
        [
            "",
            "## (4) E0-9 wall-clock timing — PASS",
            "",
            f"- Stratified selected val m: `{timing_ids}`",
            f"- `reps={timing_result['rows'][0]['reps']}` means repetitions "
            "per system; `n_systems` is the full selected-system count.",
            "",
            "| state mode | p | N | systems | reps/system | median s | objective eval s | K=100 h | K=300 h | K=600 h |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    rollouts_per_objective = cfg.n_train * 3
    for row in timing_result["rows"]:
        objective_seconds = float(row["median_sec"]) * rollouts_per_objective
        hours = {K: objective_seconds * K / 3600.0 for K in (100, 300, 600)}
        lines.append(
            f"| {row['state_mode']} | {row['p']} | {row['N']} | "
            f"{row['n_systems']} | {row['reps']} | "
            f"{row['median_sec']:.8g} | {objective_seconds:.8g} | "
            f"{hours[100]:.6g} | {hours[300]:.6g} | {hours[600]:.6g} |"
        )
    lines.extend(
        [
            "",
            f"One objective evaluation uses `{cfg.n_train} × 3 = "
            f"{rollouts_per_objective}` rollouts. This script reports timing "
            "only and does not choose K.",
            "",
            "## CSV SHA-256",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in csv_hashes.items())
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "- (1) PASS",
            "- (2) PASS",
            f"- (3) {'PASS' if divergence_result['passed'] else 'FAIL'}",
            "- (4) PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def run_e0_checks(
    cfg: Config,
    outdir: Path,
    *,
    q_n_per_d: int = 10,
    timing_reps: int = 20,
    timing_warmup: int = 3,
    timing_N: int = 3000,
    timing_n_per_d: int = 3,
) -> dict[str, Any]:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=False)
    figures_dir = output / "figures"
    logs_dir = output / "logs"
    figures_dir.mkdir()
    logs_dir.mkdir()
    if q_n_per_d < 1:
        raise ValueError("q_n_per_d must be positive")

    seed_log: dict[str, Any] = {
        "system_generation": [],
        "gram_trajectories": [],
        "q_selection": None,
        "q_trajectories": [],
        "divergence_trajectory": None,
        "timing_selection": None,
        "timing_trajectories": [],
    }
    family_result = _family_outputs(cfg, output, figures_dir, seed_log)
    q_result = _q_sensitivity(
        cfg,
        family_result["family"],
        output,
        q_n_per_d,
        seed_log,
    )
    print("q=1e-4 是预注册值；本表只作稳健性报告，不得据此更换 q。")
    divergence_result = _divergence_audit(cfg, output, seed_log)
    timing_result = _timing_audit(
        cfg,
        family_result["family"],
        output,
        timing_reps=timing_reps,
        timing_warmup=timing_warmup,
        timing_N=timing_N,
        timing_n_per_d=timing_n_per_d,
        seed_log=seed_log,
    )
    (logs_dir / "seeds.json").write_text(
        json.dumps(seed_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_paths = sorted(output.rglob("*.csv"))
    csv_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
        for path in csv_paths
    }
    summary = _summary_text(
        cfg,
        output,
        family_result,
        q_result,
        divergence_result,
        timing_result,
        csv_hashes,
    )
    (output / "summary.md").write_text(summary, encoding="utf-8")

    print("(1) E0-2 system family and diagnostics: PASS")
    print("(2) E0-4b q sensitivity: PASS")
    print("(3) E0-5 divergence logging: PASS")
    print("(4) E0-9 wall-clock timing: PASS")
    return {
        "outdir": output,
        "family": family_result["family"],
        "counts": family_result["counts"],
        "gram": family_result,
        "q_sensitivity": q_result,
        "divergence": divergence_result,
        "timing": timing_result,
        "csv_sha256": csv_hashes,
    }


def _default_run_id() -> str:
    return datetime.now().strftime("E0_%Y%m%d_%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist E0 infrastructure checks.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--q-n-per-d", type=int, default=10)
    parser.add_argument("--timing-reps", type=int, default=20)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-N", type=int, default=3000)
    parser.add_argument("--timing-n-per-d", type=int, default=3)
    arguments = parser.parse_args(argv)

    run_id = _default_run_id() if arguments.run_id is None else arguments.run_id
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be one non-empty path component")
    repo = Path(__file__).resolve().parent.parent
    outdir = repo / "results" / run_id
    run_e0_checks(
        Config(),
        outdir,
        q_n_per_d=arguments.q_n_per_d,
        timing_reps=arguments.timing_reps,
        timing_warmup=arguments.timing_warmup,
        timing_N=arguments.timing_N,
        timing_n_per_d=arguments.timing_n_per_d,
    )
    print(f"Results: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
