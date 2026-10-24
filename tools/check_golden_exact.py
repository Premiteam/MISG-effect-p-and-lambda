from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdout()


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src" / "core"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config, repro_seed
from estimators import misg_run
from systems import pack_rho, simulate


RTOL = 1e-12
ATOL = 1e-14


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
        "rho_star": pack_rho(A, B, f, d=2),
    }


def _recompute() -> dict[str, np.ndarray]:
    cfg = Config()
    system = _example_2017()
    trajectory = simulate(system, np.random.default_rng(repro_seed(cfg, 0)), N=20)
    observer = misg_run(
        system, trajectory, np.ones(5), 1.0, "cumulative", "observer", cfg
    )
    kalman = misg_run(
        system,
        trajectory,
        np.ones(5),
        1.0,
        "cumulative",
        "kalman",
        cfg,
        return_diagnostics=True,
    )
    if observer["status"] != "ok" or kalman["status"] != "ok":
        raise RuntimeError(
            f"重算得到发散轨迹：observer={observer['status']} kalman={kalman['status']}"
        )
    return {
        "observer_x_hat_traj": np.asarray(observer["x_hat_traj"])[:20],
        "kalman_x_hat_traj": np.asarray(kalman["x_hat_traj"])[:20],
        "kalman_z_hat_traj": np.asarray(kalman["z_hat_traj"])[:20],
        "kalman_P_traj": np.asarray(kalman["P_traj"])[:20],
    }


def _compare(name: str, got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "shape_got": got.shape, "shape_ref": ref.shape}
    if got.shape != ref.shape:
        out.update(exact=False, close=False, note="形状不同")
        return out
    out["dtype_match"] = got.dtype == ref.dtype
    out["exact"] = bool(np.array_equal(got, ref))
    out["close"] = bool(np.allclose(got, ref, rtol=RTOL, atol=ATOL))

    diff = np.abs(got - ref)
    out["max_abs"] = float(np.max(diff)) if diff.size else 0.0
    denom = np.abs(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, diff / denom, 0.0)
    out["max_rel"] = float(np.max(rel)) if rel.size else 0.0

    if not out["exact"]:
        bad = np.argwhere(got != ref)
        first = tuple(int(i) for i in bad[0])
        out["n_differing"] = int(bad.shape[0])
        out["n_total"] = int(got.size)
        out["first_index"] = first
        out["first_got"] = float(got[first])
        out["first_ref"] = float(ref[first])

        try:
            out["first_ulp"] = int(
                abs(
                    np.asarray(got[first]).view(np.int64)
                    - np.asarray(ref[first]).view(np.int64)
                )
            )
        except Exception:
            out["first_ulp"] = None
    return out


def main() -> int:
    golden_path = CODE_DIR / "data" / "estimators_golden.npz"
    if not golden_path.exists():
        print(f"🔴 找不到黄金轨迹：{golden_path}")
        return 2

    print("# 黄金轨迹逐位比对")
    print(f"#   golden : {golden_path}")
    print(f"#   numpy  : {np.__version__}")
    import scipy

    print(f"#   scipy  : {scipy.__version__}")
    print(f"#   python : {sys.version.split()[0]}  ({sys.executable})")
    print()

    try:
        got = _recompute()
    except Exception as exc:
        print(f"🔴 重算失败：{type(exc).__name__}: {exc}")
        return 2

    rows = []
    with np.load(golden_path, allow_pickle=False) as golden:
        for key in (
            "observer_x_hat_traj",
            "kalman_x_hat_traj",
            "kalman_z_hat_traj",
            "kalman_P_traj",
        ):
            if key not in golden:
                print(f"🔴 黄金文件缺少数组 {key}")
                return 2
            rows.append(_compare(key, got[key], golden[key]))

    all_exact = all(r["exact"] for r in rows)
    all_close = all(r.get("close") for r in rows)

    for r in rows:
        mark = (
            "✅ 逐位相同"
            if r["exact"]
            else ("🟡 容差内" if r.get("close") else "🔴 超出容差")
        )
        print(f"{mark}  {r['name']}  shape={r['shape_got']}")
        print(f"      max|Δ| = {r['max_abs']:.3e}   max|Δ|/|ref| = {r['max_rel']:.3e}")
        if not r["exact"] and "first_index" in r:
            print(
                f"      不同元素 {r['n_differing']}/{r['n_total']}"
                f"，首处 {r['first_index']}: got={r['first_got']!r} ref={r['first_ref']!r}"
                f"  相差 {r['first_ulp']} ULP"
            )
    print()

    if all_exact:
        print("=" * 68)
        print("✅ 第一档：四条轨迹全部**逐位相同**（np.array_equal）。")
        print(
            "   ⟹ A4 ⑥ / D3 §7-a ④ 判绿，可把 BLAS 后端与线程信息补记进 logs/env.json。"
        )
        print("=" * 68)
        return 0

    if all_close:
        print("=" * 68)
        print("🟡 第二档：不逐位一致，但全部落在 allclose(rtol=1e-12, atol=1e-14) 内。")
        print("🔴 不得静默接受。按 A4 / D3 §7-a ④ 必须完成三件事后才能开跑：")
        print("   ① 把上面的最大绝对/相对偏差记入 stage/E1/；")
        print("   ② 判定 E0 已签结论是否受影响（尤其 E0-H 的 R 中位数与发散判定）；")
        print("   ③ 走 idea/00 §7.2 留档。")
        print("   📌 若首处只差 1–2 ULP，通常是编译器/指令选择的纯舍入差异，")
        print("      但结论仍须由 ② 得出，不得由「看起来很小」直接放行。")
        print("=" * 68)
        return 1

    print("=" * 68)
    print("🔴 第三档：存在超出 allclose 容差的差异 ⟹ 硬阻塞。")
    print("   先查明是解释器构建差异、wheel 差异，还是代码问题，不得绕过。")
    print("=" * 68)
    return 2


if __name__ == "__main__":
    try:
        _code = main()
    except Exception as _exc:
        import traceback

        traceback.print_exc()
        print()
        print("=" * 68)
        print(f"🔴 脚本自身出错：{type(_exc).__name__}: {_exc}")
        print("🔴 退出码 3 = 脚本错误，**不得解释为第一/二/三档中的任何一档**。")
        print("=" * 68)
        _code = 3
    raise SystemExit(_code)
