from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import scipy

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src" / "core"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config, repro_seed
from estimators import misg_run
from systems import pack_rho, simulate


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


def _git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "not_available"
    return completed.stdout.strip()


def main() -> None:
    golden_dir = CODE_DIR / "data"
    golden_path = golden_dir / "estimators_golden.npz"
    provenance_path = golden_dir / "PROVENANCE.md"
    if golden_path.exists():
        raise FileExistsError(
            f"{golden_path} already exists; refusing to overwrite frozen data"
        )
    if provenance_path.exists():
        raise FileExistsError(
            f"{provenance_path} already exists; refusing to overwrite provenance"
        )
    if not golden_dir.exists():
        golden_dir.mkdir(parents=True)

    cfg = Config()
    system = _example_2017()
    trajectory = simulate(
        system,
        np.random.default_rng(repro_seed(cfg, 0)),
        N=20,
    )
    observer = misg_run(
        system,
        trajectory,
        np.ones(5),
        1.0,
        "cumulative",
        "observer",
        cfg,
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
        raise RuntimeError("cannot freeze a diverged estimator trajectory")

    np.savez(
        golden_path,
        observer_x_hat_traj=observer["x_hat_traj"][:20],
        kalman_x_hat_traj=kalman["x_hat_traj"][:20],
        kalman_z_hat_traj=kalman["z_hat_traj"][:20],
        kalman_P_traj=kalman["P_traj"][:20],
    )

    project_root = CODE_DIR
    provenance = (
        "# Estimator golden trace provenance\n\n"
        f"- Frozen date: `{date.today().isoformat()}`\n"
        f"- NumPy version: `{np.__version__}`\n"
        f"- SciPy version: `{scipy.__version__}`\n"
        f"- Git commit: `{_git_commit(project_root)}`\n"
        "- Generator: `python tools/freeze_golden.py`\n\n"
        "## Config\n\n"
        "```python\n"
        f"{pformat(asdict(cfg), sort_dicts=False)}\n"
        "```\n"
    )
    provenance_path.write_text(provenance, encoding="utf-8")


if __name__ == "__main__":
    main()
