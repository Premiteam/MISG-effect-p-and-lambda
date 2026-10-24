from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rng import tost_power_key
from tost import equivalence_power_grid, simulate_equivalence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate paired equivalence-test power"
    )
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=201)
    parser.add_argument("--n-rep", type=int, default=2000)
    parser.add_argument("--bootstrap-B", type=int, default=500)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    values = []
    strata = []
    with open(args.paired_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"improvement", "d"}
        if not required.issubset(reader.fieldnames or set()):
            raise SystemExit("CSV must contain columns: improvement, d")
        for row in reader:
            values.append(float(row["improvement"]))
            strata.append(int(row["d"]))

    sample = np.asarray(values, dtype=float)
    strata_arr = np.asarray(strata, dtype=int)

    def rng_factory(rep: int):
        return np.random.default_rng(
            tost_power_key(args.scenario, args.condition)
            if rep == 0
            else np.random.SeedSequence([20260819, args.scenario, args.condition, rep])
        )

    widths, _ = simulate_equivalence(
        sample,
        args.n,
        strata_arr,
        rng_factory,
        delta=args.delta,
        n_rep=args.n_rep,
        B=args.bootstrap_B,
    )
    grid = np.linspace(-0.10, 0.10, 21)
    power = equivalence_power_grid(widths, args.delta, grid)

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "delta": args.delta,
        "n": args.n,
        "n_rep": args.n_rep,
        "effect_grid": grid.tolist(),
        "power": power.tolist(),
    }
    with open(args.outdir / "tost_power.json", "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, power, marker="o")
    ax.axhline(0.80, color="black", ls="--", lw=1)
    ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("true paired median improvement")
    ax.set_ylabel("equivalence-test power")
    ax.set_title("TOST power for profile-risk equivalence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "tost_power.png", dpi=180)
    plt.close(fig)
    print("Saved", args.outdir / "tost_power.json", args.outdir / "tost_power.png")


if __name__ == "__main__":
    main()
