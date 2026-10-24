from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config, repro_seed
from estimators import misg_run
from metrics import delta_at
from systems import output_variance, simulate


A_2017 = np.array([[0.0, 1.0], [-0.01, -0.22]])
B_2017 = np.array([[0.19, -0.08], [0.16, -0.12]])
F_2017 = np.array([0.8, -0.8])
RHO_STAR_2017 = np.array([-0.01, -0.22, 0.19, -0.08, 0.16, -0.12, 0.8, -0.8])
P_VALUES = (1, 2, 5)
PAPER_VALUES = {1: 32.86, 2: 16.38, 5: 1.61}
MEDIAN_INTERVALS = {
    p: (0.5 * paper_value, 2.0 * paper_value) for p, paper_value in PAPER_VALUES.items()
}

ASSUMPTIONS = """## 我们补齐的四条原文未定义条件

| 项目 | 本次复现采用的条件 | 与原文的关系 |
|---|---|---|
| (1) 输入分布 | `u(k) ~ N(0,1)` i.i.d. | 我们的选择；原文只规定零均值、单位方差、不相关持续激励 |
| (2) 真状态初始历史 | `x(k) = 0 for k <= 0` | 我们选择的、与预样本约定一致的仿真条件；既非原文明定，也不是严格推论 |
| (3) 负时间数据 | `u(j)=0, y(j)=0, phi_hat(j)=0 for j <= 0`，窗口零补齐且不缩短 | 原文未定义 |
| (4) 每个数据种子重采样内容 | 同时重采样 `u` 与 `v` | 因此称为数据种子，而不是噪声种子 |
"""


def example_system_2017() -> dict[str, Any]:
    return {
        "A": A_2017.copy(),
        "B": B_2017.copy(),
        "f": F_2017.copy(),
        "d": 2,
        "n": 2,
        "c": np.array([1.0, 0.0]),
        "sigma_v": 0.30,
        "rho_star": RHO_STAR_2017.copy(),
    }


def run_reproduction(
    *,
    N: int = 3000,
    seed_indices: Sequence[int] = tuple(range(50)),
    p_values: Sequence[int] = P_VALUES,
    cfg: Config | None = None,
) -> dict[str, Any]:
    if N < 1:
        raise ValueError("N must be positive")
    seeds = tuple(int(seed) for seed in seed_indices)
    ps = tuple(int(p) for p in p_values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed_indices must be non-empty and unique")
    if not ps or any(p < 1 for p in ps):
        raise ValueError("p_values must contain positive integers")

    configuration = Config() if cfg is None else cfg
    system = example_system_2017()
    variance = output_variance(system["A"], system["B"], system["f"], int(system["d"]))
    if abs(variance - 1.384708) >= 1e-5:
        raise AssertionError("the fixed 2017 system has an unexpected output variance")

    rows: list[dict[str, Any]] = []
    curves_by_p: dict[int, list[np.ndarray]] = {p: [] for p in ps}
    for seed_j in seeds:
        rng = np.random.default_rng(repro_seed(configuration, seed_j))
        trajectory = simulate(system, rng, N=N)
        for p in ps:
            result = misg_run(
                system,
                trajectory,
                np.ones(p),
                1.0,
                "cumulative",
                "observer",
                configuration,
                r_window="newest",
            )
            curve = np.asarray(
                [
                    delta_at(
                        result["rho_hat_traj"],
                        system["rho_star"],
                        k,
                    )
                    for k in range(1, N + 1)
                ]
            )
            curves_by_p[p].append(curve)
            rows.append(
                {
                    "p": p,
                    "seed_j": seed_j,
                    "delta_3000": float(curve[-1]),
                    "status": str(result["status"]),
                }
            )

    curve_arrays = {p: np.stack(curves_by_p[p]) for p in ps}
    summaries: dict[int, dict[str, float | int]] = {}
    for p in ps:
        p_rows = [row for row in rows if row["p"] == p]
        final_deltas = np.asarray([row["delta_3000"] for row in p_rows])
        summaries[p] = {
            "median": float(np.quantile(final_deltas, 0.50)),
            "q05": float(np.quantile(final_deltas, 0.05)),
            "q95": float(np.quantile(final_deltas, 0.95)),
            "minimum": float(np.min(final_deltas)),
            "maximum": float(np.max(final_deltas)),
            "n_diverged": sum(row["status"] == "diverged" for row in p_rows),
        }

    formal_criteria = ps == P_VALUES
    interval_ok = formal_criteria and all(
        MEDIAN_INTERVALS[p][0]
        <= float(summaries[p]["median"])
        <= MEDIAN_INTERVALS[p][1]
        for p in P_VALUES
    )
    monotone_ok = formal_criteria and (
        float(summaries[1]["median"])
        > float(summaries[2]["median"])
        > float(summaries[5]["median"])
    )
    paper_in_quantiles = formal_criteria and all(
        float(summaries[p]["q05"]) <= PAPER_VALUES[p] <= float(summaries[p]["q95"])
        for p in P_VALUES
    )
    passed = bool(interval_ok and monotone_ok and paper_in_quantiles)

    return {
        "N": N,
        "seed_indices": seeds,
        "p_values": ps,
        "rows": rows,
        "curves_by_p": curve_arrays,
        "summaries": summaries,
        "criteria": {
            "median_interval": bool(interval_ok),
            "median_monotone": bool(monotone_ok),
            "paper_in_q05_q95": bool(paper_in_quantiles),
        },
        "passed": passed,
        "output_variance": float(variance),
        "snr_db": float(10.0 * np.log10(variance / 0.30**2)),
    }


def _summary_table(result: dict[str, Any]) -> str:
    lines = [
        "## 2017 Table 3 分布性复现结果",
        "",
        "| p | 原文 δ(%) | 中位数 | 5% | 95% | 最小值 | 最大值 | 发散数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in result["p_values"]:
        summary = result["summaries"][p]
        paper = PAPER_VALUES.get(p, float("nan"))
        lines.append(
            f"| {p} | {paper:.3f} | {summary['median']:.6f} | "
            f"{summary['q05']:.6f} | {summary['q95']:.6f} | "
            f"{summary['minimum']:.6f} | {summary['maximum']:.6f} | "
            f"{summary['n_diverged']} |"
        )
    criteria = result["criteria"]
    lines.extend(
        [
            "",
            "## 预注册判据",
            "",
            "| 判据 | 结果 |",
            "|---|---|",
            f"| 三个中位数均落在原文值的 0.5–2 倍内 | "
            f"{'PASS' if criteria['median_interval'] else 'FAIL'} |",
            f"| median(p=1) > median(p=2) > median(p=5) | "
            f"{'PASS' if criteria['median_monotone'] else 'FAIL'} |",
            f"| 三个原文值均落在各自 [5%,95%] 区间内 | "
            f"{'PASS' if criteria['paper_in_q05_q95'] else 'FAIL'} |",
            "",
            f"**{'PASS' if result['passed'] else 'FAIL'}**",
        ]
    )
    return "\n".join(lines)


def _write_csv(result: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("p", "seed_j", "delta_3000", "status"),
        )
        writer.writeheader()
        writer.writerows(result["rows"])


def _write_figure(result: dict[str, Any], path: Path) -> None:
    steps = np.arange(1, int(result["N"]) + 1)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for p in result["p_values"]:
        curves = result["curves_by_p"][p]
        median = np.quantile(curves, 0.50, axis=0)
        q05 = np.quantile(curves, 0.05, axis=0)
        q95 = np.quantile(curves, 0.95, axis=0)
        (line,) = axis.plot(steps, median, label=f"p={p}")
        axis.fill_between(steps, q05, q95, color=line.get_color(), alpha=0.18)
    axis.set_yscale("log")
    axis.set_xlabel("k")
    axis.set_ylabel(r"$\delta(k)$ (%)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_results(
    result: dict[str, Any],
    *,
    run_id: str,
    cfg: Config | None = None,
) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path component")
    configuration = Config() if cfg is None else cfg
    repo = Path(__file__).resolve().parent.parent
    outdir = repo / "results" / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    figures_dir = outdir / "figures"
    logs_dir = outdir / "logs"
    figures_dir.mkdir()
    logs_dir.mkdir()

    csv_path = outdir / "repro2017.csv"
    figure_path = figures_dir / "repro2017_delta_vs_k.png"
    config_path = logs_dir / "repro2017_config.json"
    readme_path = outdir / "README.md"
    _write_csv(result, csv_path)
    _write_figure(result, figure_path)

    seed_records = [
        {
            "seed_j": seed_j,
            "seed_sequence_entropy": list(repro_seed(configuration, seed_j).entropy),
        }
        for seed_j in result["seed_indices"]
    ]
    log = {
        "config": asdict(configuration),
        "system": {
            "A": A_2017.tolist(),
            "B": B_2017.tolist(),
            "f": F_2017.tolist(),
            "d": 2,
            "n": 2,
            "c": [1.0, 0.0],
            "rho_star": RHO_STAR_2017.tolist(),
            "sigma_v": 0.30,
        },
        "run": {
            "N": result["N"],
            "p_values": list(result["p_values"]),
            "state_mode": "observer",
            "weights": "ones(p)",
            "gamma": 1.0,
            "r_mode": "cumulative",
            "output_variance": result["output_variance"],
            "snr_db": result["snr_db"],
        },
        "data_seeds": seed_records,
        "criteria": result["criteria"],
        "passed": result["passed"],
        "csv_sha256": {"repro2017.csv": _sha256(csv_path)},
    }
    config_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    table = _summary_table(result)
    readme = (
        f"{ASSUMPTIONS}\n"
        f"{table}\n\n"
        "重跑命令："
        f"`python code/run_repro2017.py --run-id {run_id}_rerun`。"
        "输出目录不可覆盖，请使用新的 run-id。\n"
    )
    readme_path.write_text(readme, encoding="utf-8")
    return outdir


def _default_run_id() -> str:
    return datetime.now().strftime("E0_%Y%m%d_%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the E0 distributional reproduction of 2017 Table 3."
    )
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args(argv)
    run_id = _default_run_id() if arguments.run_id is None else arguments.run_id

    print(ASSUMPTIONS)
    result = run_reproduction()
    table = _summary_table(result)
    print(table)
    outdir = write_results(result, run_id=run_id)
    print(f"\n结果目录：{outdir}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
