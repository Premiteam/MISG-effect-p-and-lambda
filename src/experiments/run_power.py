from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import metrics
import systems
from config import Config, pilot_traj_seed, power_seed
from estimators import misg_run


BETA_GRID = np.linspace(-0.5, 0.5, 11)
E_GRID = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30)
N_REP = 2000
B_BOOT = 500
POWER_TARGET = 0.80
MDE_TARGET = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pilot(
    family: dict[str, list[dict[str, Any]]], cfg: Config
) -> list[dict[str, Any]]:
    pilot = family["pilot"]
    assert all(system["split"] == "pilot" for system in pilot)
    if len(pilot) != cfg.n_pilot:
        raise ValueError(
            f"pilot must contain {cfg.n_pilot} systems, found {len(pilot)}"
        )
    counts = Counter(int(system["d"]) for system in pilot)
    expected = cfg.n_pilot // len(cfg.d_choices)
    expected_counts = {int(d): expected for d in cfg.d_choices}
    if dict(counts) != expected_counts:
        raise ValueError(
            f"pilot delay counts must be {expected_counts}, found {dict(counts)}"
        )
    return pilot


def pilot_trajectory_seed(
    cfg: Config, system: dict[str, Any], j: int
) -> np.random.SeedSequence:
    assert system["split"] == "pilot"
    return pilot_traj_seed(cfg, int(system["m"]), j)


def _pilot_trajectory(system: dict[str, Any], cfg: Config, j: int) -> dict[str, Any]:
    seed = pilot_trajectory_seed(cfg, system, j)
    return systems.simulate(system, np.random.default_rng(seed), N=cfg.N)


def _beta_weights(beta: float, p: int) -> np.ndarray:
    logits = beta * np.arange(p, dtype=float)
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return p * exponentials / np.sum(exponentials)


def choose_loo_beta(Jsearch: np.ndarray, beta_grid: np.ndarray) -> np.ndarray:
    risks = np.asarray(Jsearch, dtype=float)
    grid = np.asarray(beta_grid, dtype=float)
    if risks.ndim != 2 or risks.shape[0] != grid.size or risks.shape[1] < 2:
        raise ValueError("Jsearch must have shape (n_beta, n_systems>=2)")
    if not np.all(np.isfinite(risks)):
        raise ValueError("Jsearch must contain only finite capped risks")
    totals = np.sum(risks, axis=1)
    chosen = np.empty(risks.shape[1])
    for system_index in range(risks.shape[1]):
        loo_mean = (totals - risks[:, system_index]) / (risks.shape[1] - 1)
        chosen[system_index] = grid[int(np.argmin(loo_mean))]
    return chosen


def _run_risk(
    system: dict[str, Any],
    trajectory: dict[str, Any],
    weights: np.ndarray,
    cfg: Config,
) -> tuple[float, str]:
    assert system["split"] == "pilot"
    result = misg_run(
        system,
        trajectory,
        weights,
        1.0,
        "cumulative",
        "kalman",
        cfg,
    )
    return (
        metrics.risk_R(result["rho_hat_traj"], system["rho_star"], cfg.N),
        str(result["status"]),
    )


def estimate_pilot_delta(
    pilot: Sequence[dict[str, Any]],
    cfg: Config,
    beta_grid: np.ndarray = BETA_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    used_systems = list(pilot)
    assert all(system["split"] == "pilot" for system in used_systems)
    grid = np.asarray(beta_grid, dtype=float)
    p = cfg.p_main
    weights_by_beta = [_beta_weights(float(beta), p) for beta in grid]
    search_risk = np.empty((grid.size, len(used_systems)))

    for system_index, system in enumerate(used_systems):
        trajectories = [_pilot_trajectory(system, cfg, j) for j in (0, 1, 2)]
        for beta_index, weights in enumerate(weights_by_beta):
            capped: list[float] = []
            for trajectory in trajectories:
                risk, status = _run_risk(system, trajectory, weights, cfg)
                if status == "diverged":
                    capped.append(float(cfg.r_cap))
                else:
                    capped.append(metrics.risk_capped(risk, cfg))
            search_risk[beta_index, system_index] = float(np.mean(capped))

    beta_chosen = choose_loo_beta(search_risk, grid)
    baseline_system_risks = np.empty(len(used_systems))
    candidate_system_risks = np.empty(len(used_systems))
    baseline_weights = np.ones(p)
    for system_index, system in enumerate(used_systems):
        trajectories = {j: _pilot_trajectory(system, cfg, j) for j in (3, 4)}
        baseline_R: dict[int, float] = {}
        baseline_status: dict[int, str] = {}
        candidate_R: dict[int, float] = {}
        candidate_status: dict[int, str] = {}
        candidate_weights = _beta_weights(beta_chosen[system_index], p)
        for j, trajectory in trajectories.items():
            baseline_R[j], baseline_status[j] = _run_risk(
                system, trajectory, baseline_weights, cfg
            )
            candidate_R[j], candidate_status[j] = _run_risk(
                system, trajectory, candidate_weights, cfg
            )
        baseline_system_risks[system_index] = metrics.system_risk(
            baseline_R, baseline_status, cfg
        )
        candidate_system_risks[system_index] = metrics.system_risk(
            candidate_R, candidate_status, cfg
        )

    delta = metrics.paired_improvement(baseline_system_risks, candidate_system_risks)
    return delta, beta_chosen


def simulate_null(
    delta0: np.ndarray,
    n: int,
    strata_pilot: np.ndarray,
    n_rep: int = N_REP,
    b_boot: int = B_BOOT,
    *,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    configuration = Config() if cfg is None else cfg
    centered = np.asarray(delta0, dtype=float)
    strata = np.asarray(strata_pilot)
    if centered.ndim != 1 or centered.size == 0 or not np.all(np.isfinite(centered)):
        raise ValueError("delta0 must be a finite non-empty vector")
    if strata.shape != centered.shape:
        raise ValueError("strata_pilot must align with delta0")
    if n < 3 or n % len(configuration.d_choices):
        raise ValueError("n must be divisible by the number of delay strata")
    if n_rep < 1 or b_boot < 1:
        raise ValueError("n_rep and b_boot must be positive")

    size_per_stratum = n // len(configuration.d_choices)
    strata_out = np.repeat(np.asarray(configuration.d_choices), size_per_stratum)
    med0 = np.empty(n_rep)
    lo0 = np.empty(n_rep)
    for rep in range(n_rep):
        rng = np.random.default_rng(power_seed(configuration, rep))
        indices = metrics._resample_indices(
            len(centered),
            rng,
            strata=strata,
            size_per_stratum=size_per_stratum,
        )
        sample = centered[indices]
        bootstrap_indices = np.stack(
            [
                metrics._resample_indices(n, rng, strata=strata_out)
                for _ in range(b_boot)
            ]
        )
        med0[rep] = np.median(sample)
        lo0[rep] = np.percentile(np.median(sample[bootstrap_indices], axis=1), 2.5)
    return med0, lo0


def shifted_power(
    med0: np.ndarray,
    lo0: np.ndarray,
    e_grid: Sequence[float] = E_GRID,
) -> list[dict[str, float]]:
    medians = np.asarray(med0, dtype=float)
    lower_bounds = np.asarray(lo0, dtype=float)
    if medians.ndim != 1 or lower_bounds.shape != medians.shape or medians.size == 0:
        raise ValueError("med0 and lo0 must be aligned non-empty vectors")
    rows: list[dict[str, float]] = []
    for effect in e_grid:
        e = float(effect)
        significant = lower_bounds + e > 0.0
        rows.append(
            {
                "e": e,
                "power_i": float(np.mean(significant)),
                "power_ii": float(np.mean(significant & (medians + e >= MDE_TARGET))),
            }
        )
    return rows


def mde_sig(lo0: np.ndarray) -> float:
    lower_bounds = np.asarray(lo0, dtype=float)
    if lower_bounds.ndim != 1 or lower_bounds.size == 0:
        raise ValueError("lo0 must be a non-empty vector")
    return float(max(0.0, -np.quantile(lower_bounds, 0.20)))


def _sample_size_decision(
    delta0: np.ndarray,
    strata_pilot: np.ndarray,
    cfg: Config,
    *,
    n_rep: int,
    b_boot: int,
    initial_lo0: np.ndarray,
) -> dict[str, Any]:
    cache: dict[int, float] = {cfg.n_test: mde_sig(initial_lo0)}

    def value(n: int) -> float:
        if n not in cache:
            _, lower = simulate_null(
                delta0,
                n,
                strata_pilot,
                n_rep=n_rep,
                b_boot=b_boot,
                cfg=cfg,
            )
            cache[n] = mde_sig(lower)
        return cache[n]

    initial = cache[cfg.n_test]
    if initial <= MDE_TARGET:
        return {
            "status": "SAMPLE SIZE OK",
            "n_star": None,
            "mde_by_n": dict(sorted(cache.items())),
        }

    if value(501) > MDE_TARGET:
        return {
            "status": "INSUFFICIENT",
            "n_star": None,
            "mde_by_n": dict(sorted(cache.items())),
        }

    candidates = list(range(204, 502, 3))
    low = 0
    high = len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if value(candidates[middle]) <= MDE_TARGET:
            high = middle
        else:
            low = middle + 1
    n_star = candidates[low]
    for neighbor in (n_star - 3, n_star, n_star + 3):
        if 204 <= neighbor <= 501:
            value(neighbor)
    return {
        "status": "NEED MORE SYSTEMS",
        "n_star": n_star,
        "mde_by_n": dict(sorted(cache.items())),
    }


def _write_power_figure(rows: Sequence[dict[str, float]], path: Path) -> None:
    effects = np.asarray([row["e"] for row in rows])
    power_i = np.asarray([row["power_i"] for row in rows])
    power_ii = np.asarray([row["power_ii"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.plot(effects, power_i, marker="o", label="power_i: CI lower > 0")
    axis.plot(
        effects,
        power_ii,
        marker="s",
        label="power_ii: CI lower > 0 and median >= 0.10",
    )
    axis.axhline(POWER_TARGET, color="black", linestyle="--", linewidth=1)
    axis.axvline(MDE_TARGET, color="gray", linestyle=":", linewidth=1)
    axis.set_xlabel("True median effect e")
    axis.set_ylabel("Probability")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _power_markdown(
    *,
    systems_path: Path,
    systems_sha256: str,
    cfg: Config,
    delta: np.ndarray,
    beta_chosen: np.ndarray,
    rows: Sequence[dict[str, float]],
    mde: float,
    decision: dict[str, Any],
    n_rep: int,
    b_boot: int,
) -> str:
    q25, q75 = np.quantile(delta, [0.25, 0.75])
    beta_counts = Counter(float(beta) for beta in beta_chosen)
    lines = [
        "# E0-7 pilot minimum detectable effect analysis",
        "",
        "**功效只针对 (i)：95% CI 下界 > 0。**",
        "**(ii) `median >= 0.10` 是实际意义门槛，单独执行，不做功效分析。**",
        "**W0 vs W1 只是估计 delta 分布形状的代理，不是 E1 结果，"
        "不得作为任何方法优劣的证据。**",
        "",
        f"- Input systems: `{systems_path}`",
        f"- Input SHA-256: `{systems_sha256}`",
        f"- Pilot systems used: `{delta.size}`; test systems used: `0`",
        f"- Outer Monte Carlo: `{n_rep}`",
        f"- Inner bootstrap: `{b_boot}` (E1 uses 2000; the formal F default is 500)",
        "",
        "## Config",
        "",
        "```python",
        pformat(asdict(cfg), sort_dicts=False),
        "```",
        "",
        "## Cross-fitted pilot delta distribution",
        "",
        f"- median: `{np.median(delta):.8g}`",
        f"- IQR: `[{q25:.8g}, {q75:.8g}]`",
        f"- standard deviation: `{np.std(delta, ddof=1):.8g}`",
        f"- minimum / maximum: `[{np.min(delta):.8g}, {np.max(delta):.8g}]`",
        f"- LOO beta distribution: `{dict(sorted(beta_counts.items()))}`",
        "",
        "## Power table",
        "",
        "| e | power_i: CI lower > 0 | power_ii: additionally median >= 0.10 |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['e']:.2f} | {row['power_i']:.6f} | {row['power_ii']:.6f} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "`power_ii` 在 `e = 0.10` 处按构造只能约为 0.5；"
            "这不是问题，也不是功效目标。",
            "",
            "## MDE and sample-size decision",
            "",
            f"- `n_test={cfg.n_test}` 下 `MDE_sig = {mde:.8g}`",
            f"- `MDE_sig <= 0.10`: `{'YES' if mde <= MDE_TARGET else 'NO'}`",
            f"- Decision: **{decision['status']}**",
        ]
    )
    if mde == 0.0:
        lines.append("- pilot 上的代理效应已足够大，MDE 被夹到 0，该数值不可外推。")
    if decision["n_star"] is not None:
        lines.append(
            f"- Candidate expansion size: `{decision['n_star']}`；"
            "必须由人签字后才生效，脚本未修改 config。"
        )
    if decision["mde_by_n"]:
        lines.extend(
            [
                "",
                "| evaluated n | MDE_sig |",
                "|---:|---:|",
            ]
        )
        lines.extend(
            f"| {n} | {value:.8g} |" for n, value in decision["mde_by_n"].items()
        )
    return "\n".join(lines) + "\n"


def run_power_analysis(
    cfg: Config,
    systems_path: Path,
    outdir: Path,
    *,
    n_rep: int = N_REP,
    b_boot: int = B_BOOT,
) -> dict[str, Any]:
    source_path = Path(systems_path).resolve()
    source_sha256 = _sha256(source_path)
    family = systems.load_family(source_path)
    pilot = validate_pilot(family, cfg)
    used_systems = pilot
    assert all(system["split"] == "pilot" for system in used_systems)

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=False)
    figures_dir = output / "figures"
    logs_dir = output / "logs"
    figures_dir.mkdir()
    logs_dir.mkdir()

    delta, beta_chosen = estimate_pilot_delta(pilot, cfg)
    delta = np.asarray(delta, dtype=float)
    beta_chosen = np.asarray(beta_chosen, dtype=float)
    if (
        delta.shape != (cfg.n_pilot,)
        or beta_chosen.shape != delta.shape
        or not np.all(np.isfinite(delta))
        or not np.all(np.isfinite(beta_chosen))
    ):
        raise ValueError("pilot delta and beta outputs must be finite 51-vectors")

    strata_pilot = np.asarray([int(system["d"]) for system in pilot])
    delta0 = delta - np.median(delta)
    med0, lo0 = simulate_null(
        delta0,
        cfg.n_test,
        strata_pilot,
        n_rep=n_rep,
        b_boot=b_boot,
        cfg=cfg,
    )
    rows = shifted_power(med0, lo0)
    mde = mde_sig(lo0)
    decision = _sample_size_decision(
        delta0,
        strata_pilot,
        cfg,
        n_rep=n_rep,
        b_boot=b_boot,
        initial_lo0=lo0,
    )

    _write_power_figure(rows, figures_dir / "power.png")
    report = _power_markdown(
        systems_path=source_path,
        systems_sha256=source_sha256,
        cfg=cfg,
        delta=delta,
        beta_chosen=beta_chosen,
        rows=rows,
        mde=mde,
        decision=decision,
        n_rep=n_rep,
        b_boot=b_boot,
    )
    (output / "power.md").write_text(report, encoding="utf-8")

    seeds = {
        "systems_path": str(source_path),
        "systems_sha256": source_sha256,
        "n_rep": n_rep,
        "B_boot": b_boot,
        "power_seeds": [
            {
                "rep": rep,
                "components": [int(value) for value in power_seed(cfg, rep).entropy],
            }
            for rep in range(n_rep)
        ],
        "pilot_trajectory_seeds": [
            {
                "m": int(system["m"]),
                "j": j,
                "components": [
                    int(value)
                    for value in pilot_trajectory_seed(cfg, system, j).entropy
                ],
            }
            for system in pilot
            for j in range(cfg.n_traj)
        ],
        "evaluated_sample_sizes": sorted(decision["mde_by_n"]),
    }
    (logs_dir / "power_seeds.json").write_text(
        json.dumps(seeds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"n_test={cfg.n_test}, MDE_sig={mde:.8g}")
    print(decision["status"])
    return {
        "systems_path": source_path,
        "systems_sha256": source_sha256,
        "pilot": pilot,
        "delta": delta,
        "beta_chosen": beta_chosen,
        "med0": med0,
        "lo0": lo0,
        "power": rows,
        "mde_sig": mde,
        "decision": decision,
        "outdir": output,
    }


def _default_run_id() -> str:
    return datetime.now().strftime("E0_%Y%m%d_%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pilot-only E0 MDE analysis.")
    parser.add_argument("--systems", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-rep", type=int, default=N_REP)
    parser.add_argument("--b-boot", type=int, default=B_BOOT)
    arguments = parser.parse_args(argv)

    run_id = _default_run_id() if arguments.run_id is None else arguments.run_id
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be one non-empty path component")
    repo = Path(__file__).resolve().parent.parent
    outdir = repo / "results" / run_id
    run_power_analysis(
        Config(),
        Path(arguments.systems),
        outdir,
        n_rep=arguments.n_rep,
        b_boot=arguments.b_boot,
    )
    print(f"Results: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
