from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdout()


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"} if default is None else default


def _threadpool_info() -> Any:
    try:
        import threadpoolctl
    except ImportError:
        return {
            "error": "threadpoolctl 未安装",
            "hint": "pip install threadpoolctl；A1 判绿需要它，A4 也需要它回填版本号",
        }
    return {
        "threadpoolctl_version": threadpoolctl.__version__,
        "pools": threadpoolctl.threadpool_info(),
    }


def _worker_probe(_ignored: int) -> Any:
    import numpy

    info: dict[str, Any] = {
        "pid": os.getpid(),
        "env_vars": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    try:
        import threadpoolctl

        info["pools"] = threadpoolctl.threadpool_info()
    except ImportError:
        info["pools"] = {"error": "threadpoolctl 未安装"}
    return info


def _numpy_info() -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {"version": np.__version__, "file": np.__file__}

    def _cfg():
        return np.show_config(mode="dicts")

    out["show_config"] = _safe(_cfg)

    libs_dir = Path(np.__file__).parent.parent / "numpy.libs"
    out["numpy_libs_dir"] = str(libs_dir)
    out["numpy_libs_exists"] = libs_dir.is_dir()
    out["numpy_libs_files"] = (
        sorted(p.name for p in libs_dir.iterdir()) if libs_dir.is_dir() else []
    )
    return out


def _scipy_info() -> dict[str, Any]:
    import scipy

    out: dict[str, Any] = {"version": scipy.__version__, "file": scipy.__file__}
    out["show_config"] = _safe(lambda: scipy.show_config(mode="dicts"))
    libs_dir = Path(scipy.__file__).parent.parent / "scipy.libs"
    out["scipy_libs_exists"] = libs_dir.is_dir()
    out["scipy_libs_files"] = (
        sorted(p.name for p in libs_dir.iterdir()) if libs_dir.is_dir() else []
    )
    return out


def _run(cmd: list[str]) -> str:
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        return (
            res.stdout
            if res.returncode == 0
            else f"[rc={res.returncode}]\n{res.stdout}\n{res.stderr}"
        )
    except Exception as exc:
        return f"[failed] {type(exc).__name__}: {exc}"


def collect() -> dict[str, Any]:
    is_conda = (Path(sys.prefix) / "conda-meta").is_dir()
    snap: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "version_full": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_virtualenv": sys.prefix != sys.base_prefix,
            "is_conda_env": is_conda,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "sysconfig_platform": sysconfig.get_platform(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count_logical": os.cpu_count(),
        },
        "numpy": _safe(_numpy_info),
        "scipy": _safe(_scipy_info),
        "threadpool_main_process": _safe(_threadpool_info),
    }

    def _cma():
        import cma

        return {"version": cma.__version__, "file": cma.__file__}

    snap["cma"] = _safe(_cma)
    return snap


def collect_worker() -> Any:
    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            return pool.map(_worker_probe, [0])[0]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


README = """# 环境快照 · `{label}`

由 `code/tools/snapshot_env.py` 生成，**只读，不修改环境**。

| 文件 | 内容 |
|---|---|
| `env.json` | 结构化全量，供 `--diff` 比对 |
| `pip-freeze.txt` | `pip freeze` 原样（A4 回填 `threadpoolctl` / `matplotlib` 版本用这份） |
| `numpy-config.txt` | `numpy.show_config()` 的人读版 |

## 关键字段怎么读

- `python.is_conda_env` / `python.in_virtualenv` —— 区分 system / venv / conda 解释器。
  E0 用的是 **system**（`code/.vscode/settings.json` 记为 `ms-python.python:system`）。
- `numpy.numpy_libs_exists` —— **PyPI wheel 会带 `numpy.libs/`**（内含 OpenBLAS 的 DLL），
  conda 构建通常没有。这是判断 BLAS 来源最直接的一条证据。
- `threadpool_main_process.pools[*].internal_api` —— `openblas` 还是 `mkl`。
  🔴 若是 `openblas`，则 `idea/00` §10 已签的 `MKL_NUM_THREADS` 一直是空转的，
  真正吃力的是 `OPENBLAS_NUM_THREADS`（**属发现，须留档，不改签字内容**）。
- `threadpool_worker_process` —— 在 spawn 出来的**子进程内**取的同一组信息。
  A1 的判绿标准要求的正是这一条，主进程的值不算数。

## 下一步

两个环境各跑一次后：

```
python code/tools/snapshot_env.py --diff code/env-snapshot/e0-system code/env-snapshot/e1-conda
```
"""


def write_snapshot(label: str, outroot: Path) -> Path:
    outdir = outroot / label
    outdir.mkdir(parents=True, exist_ok=True)

    snap = collect()
    snap["threadpool_worker_process"] = collect_worker()
    snap["_meta"] = {
        "label": label,
        "generator": "code/tools/snapshot_env.py",
        "note": "只读快照；不设任何线程环境变量，记录的是环境的真实默认值",
    }

    (outdir / "env.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (outdir / "pip-freeze.txt").write_text(
        _run([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8"
    )
    (outdir / "numpy-config.txt").write_text(
        _run([sys.executable, "-c", "import numpy; numpy.show_config()"]),
        encoding="utf-8",
    )
    (outdir / "README.md").write_text(README.format(label=label), encoding="utf-8")
    return outdir


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def do_diff(a: Path, b: Path) -> int:
    fa = _flatten(json.loads((a / "env.json").read_text(encoding="utf-8")))
    fb = _flatten(json.loads((b / "env.json").read_text(encoding="utf-8")))
    keys = sorted(set(fa) | set(fb))
    diffs = [
        (k, fa.get(k, "<缺>"), fb.get(k, "<缺>"))
        for k in keys
        if not k.startswith("_meta")
        and k != "threadpool_worker_process.pid"
        and fa.get(k, "<缺>") != fb.get(k, "<缺>")
    ]
    print(f"# 环境快照比对\n#   A = {a}\n#   B = {b}\n")
    if not diffs:
        print("两份快照逐字段完全相同。")
        return 0
    print(f"共 {len(diffs)} 处差异：\n")
    for k, va, vb in diffs:
        print(f"  {k}\n      A: {va}\n      B: {vb}")
    print(
        "\n🔴 重点看 python.version / python.is_conda_env / numpy.numpy_libs_exists"
        "\n   以及 threadpool_*.pools[*].internal_api 与 .filepath —— "
        "\n   BLAS 后端若变了，黄金轨迹与墙钟都必须重新验收（A4 的三档处置规则）。"
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="环境快照 / 比对（只读）")
    ap.add_argument("--label", help="快照名，如 e0-system / e1-conda")
    ap.add_argument(
        "--outroot",
        default="code/env-snapshot",
        help="输出根目录（默认 code/env-snapshot）",
    )
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="比对两份已有快照目录")
    args = ap.parse_args()

    if args.diff:
        return do_diff(Path(args.diff[0]), Path(args.diff[1]))

    if not args.label:
        ap.error("需要 --label（或用 --diff）")

    outdir = write_snapshot(args.label, Path(args.outroot))
    print(f"✅ 快照已写入 {outdir}")
    print("   env.json · pip-freeze.txt · numpy-config.txt · README.md")

    snap = json.loads((outdir / "env.json").read_text(encoding="utf-8"))
    py = snap["python"]
    print(
        f"\nPython {py['version']}  conda={py['is_conda_env']}  venv={py['in_virtualenv']}"
    )
    print(
        f"NumPy  {snap['numpy'].get('version')}   numpy.libs 存在={snap['numpy'].get('numpy_libs_exists')}"
    )
    print(f"SciPy  {snap['scipy'].get('version')}")
    pools = snap.get("threadpool_main_process", {}).get("pools")
    if isinstance(pools, list):
        for p in pools:
            print(
                f"  BLAS: {p.get('internal_api')}  threads={p.get('num_threads')}  {p.get('filepath')}"
            )
    else:
        print(f"  BLAS: {pools}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        _code = main()
    except Exception as _exc:
        import traceback

        traceback.print_exc()
        print(f"\n[ERROR] 脚本自身出错：{type(_exc).__name__}: {_exc}")
        print("[ERROR] 退出码 3 = 脚本错误，不是「两份快照有差异」。")
        _code = 3
    raise SystemExit(_code)
