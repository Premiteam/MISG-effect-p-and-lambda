from __future__ import annotations


import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "core"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TOOLS))

from config import Config, traj_seed
from estimators import misg_run
from systems import sample_system, simulate


D1_WORKERS = 12
WARMUP_SECONDS = 120.0
ROUND_SECONDS = 180.0
N_ROUNDS = 3
SERIAL_SECONDS = 60.0
SERIAL_WARMUP_SECONDS = 10.0


DISPERSION_GREEN = 1.10
DISPERSION_AMBER = 1.25
DRIFT_GREEN = 0.92

EXIT_GREEN, EXIT_AMBER, EXIT_RED, EXIT_SCRIPT_ERROR = 0, 1, 2, 3


ROLLOUT_KIND = "kalman · p=5 · d=2 · w=1 · gamma=1 · r_mode=cumulative"


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdout()


class _Tee(io.TextIOBase):
    def __init__(self, stream):
        self._stream = stream
        self.buf: list[str] = []

    def write(self, s: str) -> int:
        self.buf.append(s)
        self._stream.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()


def _timed_batch(args: tuple[int, float]) -> tuple[int, float]:
    seed, seconds = args
    cfg = Config()
    rng = np.random.default_rng(seed)
    system = sample_system(rng, cfg, d=2)

    n = 0
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        traj = simulate(system, np.random.default_rng(traj_seed(cfg, seed, n)), cfg.N)
        misg_run(system, traj, np.ones(5), 1.0, "cumulative", "kalman", cfg)
        n += 1
    return n, time.perf_counter() - start


def _run_round(pool, workers: int, seconds: float, seed_base: int) -> dict:
    args = [(seed_base + w, seconds) for w in range(workers)]
    start = time.perf_counter()
    out = pool.map(_timed_batch, args)
    wall = time.perf_counter() - start
    counts = [n for n, _ in out]
    total = sum(counts)
    return {
        "workers": workers,
        "wall_s": wall,
        "rollouts_total": total,
        "rollouts_per_worker_min": min(counts),
        "rollouts_per_worker_max": max(counts),
        "throughput_rollouts_per_s": total / wall,
    }


def _serial_rate(seconds: float, seed: int, warmup: float) -> dict:
    wu_n, wu_s = _timed_batch((seed + 500, warmup)) if warmup > 0 else (0, 0.0)
    n, elapsed = _timed_batch((seed, seconds))
    return {
        "warmup_discarded": {"rollouts": wu_n, "wall_s": wu_s},
        "wall_s": elapsed,
        "rollouts": n,
        "seconds_per_rollout": elapsed / n if n else float("nan"),
    }


def _verdict(rounds: list[dict]) -> dict:
    tps = [r["throughput_rollouts_per_s"] for r in rounds]
    dispersion = max(tps) / min(tps)
    monotone_down = all(tps[i] > tps[i + 1] for i in range(len(tps) - 1))
    drift = tps[-1] / tps[0]

    if dispersion > DISPERSION_AMBER:
        tier, code = "🔴 不可用", EXIT_RED
        why = (
            f"三轮吞吐 max/min = {dispersion:.3f} > {DISPERSION_AMBER}：机器明显未空载"
        )
    elif dispersion > DISPERSION_GREEN:
        tier, code = "🟡 需重测", EXIT_AMBER
        why = (
            f"三轮吞吐 max/min = {dispersion:.3f} 落在 "
            f"({DISPERSION_GREEN}, {DISPERSION_AMBER}]：疑似有后台负载"
        )
    elif monotone_down and drift < DRIFT_GREEN:
        tier, code = "🟡 需重测", EXIT_AMBER
        why = (
            f"三轮单调下降且 round3/round1 = {drift:.3f} < {DRIFT_GREEN}："
            f"预热不足，🔴 请把 --warmup-seconds 加倍后重测"
        )
    else:
        tier, code = "🟢 可用", EXIT_GREEN
        why = f"三轮吞吐 max/min = {dispersion:.3f} ≤ {DISPERSION_GREEN}，无未收敛的单调下降"

    return {
        "tier": tier,
        "exit_code": code,
        "why": why,
        "dispersion_ratio": dispersion,
        "monotone_decreasing": monotone_down,
        "drift_last_over_first": drift,
        "throughput_median_rollouts_per_s": statistics.median(tps),
    }


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E1 墙钟单价与 12-worker 吞吐量实测")
    ap.add_argument(
        "--workers",
        type=int,
        default=D1_WORKERS,
        help=f"并行进程数（默认 {D1_WORKERS} = 已签的 D1 口径）",
    )
    ap.add_argument("--warmup-seconds", type=float, default=WARMUP_SECONDS)
    ap.add_argument("--round-seconds", type=float, default=ROUND_SECONDS)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--serial-seconds", type=float, default=SERIAL_SECONDS)
    ap.add_argument(
        "--serial-warmup-seconds", type=float, default=SERIAL_WARMUP_SECONDS
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="额外扫 worker 数（🔴 仅诊断，结果不得回填 D1）",
    )
    ap.add_argument(
        "--total-rollouts",
        type=int,
        default=None,
        help="给定后才换算 CPU·h / 墙钟 h / 占用核时；🔴 不给则不算",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    return ap.parse_args()


def run(a: argparse.Namespace) -> tuple[int, dict]:
    total_logical = os.cpu_count() or 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 74)
    print("E1 墙钟实测 · bench_parallel.py")
    print("=" * 74)
    print(f"UTC 时间戳     : {stamp}")
    print(f"platform       : {platform.platform()}")
    print(f"processor      : {platform.processor()}")
    print(f"logical cores  : {total_logical}")
    print(f"python         : {sys.version.split()[0]}  ({sys.executable})")
    print(f"numpy          : {np.__version__}")
    print(f"BLAS 线程      : 每个 worker 钉为 1（import numpy 之前设置）")
    print(f"rollout 口径   : {ROLLOUT_KIND}")
    print()
    print(
        f"🔴 D1 口径 workers = {a.workers}"
        + (
            "（= 已签值）"
            if a.workers == D1_WORKERS
            else f"  ⚠️ **不是已签的 {D1_WORKERS}，本次结果不得回填 D1**"
        )
    )
    print(
        f"🔴 预热 {a.warmup_seconds:.0f} s（整段丢弃）"
        f" · 每轮 {a.round_seconds:.0f} s × {a.rounds} 轮"
        f" · 单 worker 标定 {a.serial_seconds:.0f} s（冷/热各一次）"
    )
    print(f"🔴 聚合规则（跑前写死）：**{a.rounds} 轮吞吐量的中位数**")
    est = a.serial_seconds * 2 + a.warmup_seconds + a.round_seconds * a.rounds
    print(f"预计总时长     : 约 {est / 60:.0f} 分钟")
    print()

    if a.workers > total_logical:
        print(f"⚠️ workers={a.workers} > 逻辑核 {total_logical}，超订会失真")

    result: dict = {
        "stamp_utc": stamp,
        "rollout_kind": ROLLOUT_KIND,
        "d1_workers_signed": D1_WORKERS,
        "params": {
            "workers": a.workers,
            "warmup_seconds": a.warmup_seconds,
            "round_seconds": a.round_seconds,
            "rounds": a.rounds,
            "serial_seconds": a.serial_seconds,
            "serial_warmup_seconds": a.serial_warmup_seconds,
        },
        "aggregation_rule": f"median of {a.rounds} round throughputs (pre-registered)",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cores": total_logical,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }

    print(
        f"① 冷态单 worker 标定（前置丢弃 {a.serial_warmup_seconds:.0f} s；此时其余 "
        f"{max(total_logical - 1, 0)} 核空闲，与旧环境的 0.4313 / 0.326 s 可比）"
    )
    cold = _serial_rate(a.serial_seconds, seed=90001, warmup=a.serial_warmup_seconds)
    result["serial_cold"] = cold
    print(
        f"   {cold['rollouts']} rollouts / {cold['wall_s']:.1f} s "
        f"⟹ **{cold['seconds_per_rollout']:.4f} s / rollout**\n"
    )

    with mp.Pool(a.workers) as pool:
        print(
            f"② 预热 {a.warmup_seconds:.0f} s（{a.workers} workers，🔴 结果整段丢弃）"
        )
        wu = _run_round(pool, a.workers, a.warmup_seconds, seed_base=91000)
        result["warmup_discarded"] = wu
        print(f"   （丢弃）吞吐 {wu['throughput_rollouts_per_s']:.2f} rollouts/s\n")

        print(f"③ 正式 {a.rounds} 轮 × {a.round_seconds:.0f} s（{a.workers} workers）")
        print(
            f"   {'轮':>3} {'wall s':>9} {'rollouts':>10} {'每worker min/max':>18} {'rollout/s':>11}"
        )
        rounds: list[dict] = []
        for r in range(1, a.rounds + 1):
            rr = _run_round(
                pool, a.workers, a.round_seconds, seed_base=92000 + 1000 * r
            )
            rounds.append(rr)
            print(
                f"   {r:>3} {rr['wall_s']:>9.2f} {rr['rollouts_total']:>10,} "
                f"{rr['rollouts_per_worker_min']:>8,}/{rr['rollouts_per_worker_max']:<9,}"
                f"{rr['throughput_rollouts_per_s']:>11.2f}"
            )
        result["rounds"] = rounds
        print()

    if a.sweep:
        print("④ 诊断性 worker 扫描 🔴 **结果不得回填 D1**")
        sweep = []
        for w in sorted({1, 2, 4, 6, 8, 10, 12, 16, total_logical}):
            if not 1 <= w <= total_logical:
                continue
            with mp.Pool(w) as p2:
                sw = _run_round(
                    p2, w, min(30.0, a.round_seconds), seed_base=93000 + 100 * w
                )
            sweep.append(sw)
            print(
                f"   workers={w:>3} 吞吐 {sw['throughput_rollouts_per_s']:>8.2f} rollouts/s"
            )
        result["sweep_diagnostic_only"] = sweep
        print()

    print("⑤ 热态单 worker 标定（紧接持续负载之后 —— 🔴 **这是 D1 的单价口径**）")
    warm = _serial_rate(a.serial_seconds, seed=94001, warmup=a.serial_warmup_seconds)
    result["serial_warm"] = warm
    print(
        f"   {warm['rollouts']} rollouts / {warm['wall_s']:.1f} s "
        f"⟹ **{warm['seconds_per_rollout']:.4f} s / rollout**"
    )
    ratio = warm["seconds_per_rollout"] / cold["seconds_per_rollout"]
    print(
        f"   热/冷 = {ratio:.3f}（两次都已各自预热 ⟹ 差异可归因于温度/频率，"
        f"不是首调开销）"
    )
    if ratio >= 1.05:
        print(f"   ⚠️ 持续负载后单价上升 {ratio - 1:.1%}，长跑会比冷态估计更慢")
    print()

    v = _verdict(rounds)
    result["verdict"] = v
    result["serial_rate_for_d1"] = warm["seconds_per_rollout"]

    print("=" * 74)
    print("判定（🔴 规则于跑前写死，见 `环境验收日志` §8）")
    print("=" * 74)
    print(
        f"  三轮吞吐        : "
        + ", ".join(f"{r['throughput_rollouts_per_s']:.2f}" for r in rounds)
    )
    print(
        f"  离散度 max/min  : {v['dispersion_ratio']:.3f}   (🟢 ≤{DISPERSION_GREEN} · "
        f"🟡 ≤{DISPERSION_AMBER} · 🔴 >{DISPERSION_AMBER})"
    )
    print(
        f"  round3/round1   : {v['drift_last_over_first']:.3f}"
        f"   (单调下降时要求 ≥{DRIFT_GREEN})"
    )
    print(f"  **{v['tier']}** —— {v['why']}")
    print()
    print("🔴 回填 D1 时用这两个数（其余一律不用）：")
    print(f"     单 rollout 秒数（热态）      = {warm['seconds_per_rollout']:.4f} s")
    print(
        f"     {a.workers}-worker 吞吐（中位数）  = {v['throughput_median_rollouts_per_s']:.3f} rollouts/s"
    )
    eff = (
        v["throughput_median_rollouts_per_s"] * warm["seconds_per_rollout"] / a.workers
    )
    result["parallel_efficiency_vs_warm_serial"] = eff
    print(f"     （并行效率 = {eff:.1%}，仅供参考，不进 D1）")
    if eff > 1.0:
        print(
            "     🔴 **并行效率 > 100% 是异常信号** —— 多半是热态串行标定与三轮负载"
            "之间机器状态不一致（例如中途降频恢复）。此时**不要回填 D1**，请重测。"
        )
    print()

    if a.total_rollouts:
        T = a.total_rollouts
        s = warm["seconds_per_rollout"]
        tp = v["throughput_median_rollouts_per_s"]
        cpu_h = T * s / 3600.0
        wall_h = T / tp / 3600.0
        occupied = a.workers * wall_h
        result["projection"] = {
            "total_rollouts": T,
            "serial_equivalent_cpu_h": cpu_h,
            "wall_clock_h": wall_h,
            "occupied_core_h": occupied,
        }
        print(f"⑦ 按给定总量 {T:,} rollouts 换算：")
        print(f"     ① 串行等效 CPU·h = {T:,} × {s:.4f} / 3600 = **{cpu_h:.1f}**")
        print(
            f"     ② 墙钟 h         = {T:,} / ({tp:.3f} × 3600) = **{wall_h:.1f}**（{wall_h / 24:.1f} 天）"
        )
        print(
            f"     ③ 占用核时       = {a.workers} × {wall_h:.1f} = **{occupied:.1f}**"
        )
        rel = "大于" if occupied > cpu_h else "小于"
        print(f"   🔴 ①≠③：并行效率 {eff:.0%} ⟹ 占用核时 {rel} 串行等效 CPU·h。")
        print(f"   🔴 论文/预算表写哪个必须写明是哪一个。")
    else:
        print(
            "⑦ 未给 --total-rollouts ⟹ **不做任何总量换算**"
            "（🔴 总量归 D1 管，本脚本不猜；旧版的「~28 个搜索任务」估算已删除）。"
        )
    print()
    print("⚠️ 本测量的适用边界：只测了 " + ROLLOUT_KIND + " 这一种 rollout。")
    print("   其它 $p$、其它场景、其它 `r_mode` 的单价**未测**，不得外推。")

    return v["exit_code"], result


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    a = _parse()
    tee = _Tee(sys.stdout)
    real_stdout = sys.stdout
    sys.stdout = tee
    try:
        code, result = run(a)
    except Exception:
        sys.stdout = real_stdout
        traceback.print_exc()
        print("\n🔴 脚本自身出错 ⟹ 退出码 3（**不是**任何一档测量结论）")
        return EXIT_SCRIPT_ERROR
    finally:
        sys.stdout = real_stdout

    outdir = a.outdir or (
        ROOT / "stage" / "E1" / "验收日志" / f"bench_{result['stamp_utc']}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    tail = f"\n退出码 {code}（0=🟢可用 · 1=🟡需重测 · 2=🔴不可用 · 3=脚本自身出错）\n"
    (outdir / "bench.log").write_text("".join(tee.buf) + tail, encoding="utf-8")
    (outdir / "bench.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        from snapshot_env import collect

        (outdir / "env.json").write_text(
            json.dumps(collect(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        (outdir / "env.json").write_text(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    lines = [
        f"{_sha256(p)}  {p.name}"
        for p in sorted(outdir.iterdir())
        if p.is_file() and p.name != "SHA256SUMS.txt"
    ]
    (outdir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n已落盘 → {outdir}")
    for name in ("bench.log", "bench.json", "env.json", "SHA256SUMS.txt"):
        print(f"   {name}")
    print(f"\n退出码 {code}（0=🟢可用 · 1=🟡需重测 · 2=🔴不可用 · 3=脚本自身出错）")
    return code


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
