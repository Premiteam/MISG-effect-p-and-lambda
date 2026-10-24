from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from audit_panel_alignment import require_matplotlib_panel_alignment


SUMMARY = ROOT / "data" / "figure4_source_data.csv"
TIMING = ROOT / "data" / "system_timing_summary.csv"
OUT = Path(__file__).resolve().parent / "Figure7_computational_tradeoff"
FIGURE_SIZE_IN = (15.0 / 2.54, 8.0 / 2.54)
RASTER_DPI = 600


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = read_rows(SUMMARY)
    timing = read_rows(TIMING)
    p = [int(row["p"]) for row in summary]
    operations = [int(row["update_flops_per_step"]) for row in summary]
    buffers = [int(row["buffer_bytes"]) for row in summary]
    runtime_p1 = np.asarray([float(row["median_sec_p1"]) for row in timing])
    runtime_p5 = np.asarray([float(row["median_sec_p5"]) for row in timing])

    navy = "#155A9C"
    pale_blue = "#9DC3E6"
    orange = "#E28B2D"
    grid = "#D9DEE5"

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_IN, constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.82, wspace=0.48)

    axes[0].plot(p, operations, marker="o", color=navy, linewidth=1.6, markersize=4.2)
    axes[0].fill_between(p, operations, color=pale_blue, alpha=0.28)
    axes[0].set_title("Innovation vector update", fontweight="bold")
    axes[0].set_xlabel("Innovation length, p")
    axes[0].set_ylabel("Operations per update")
    axes[0].set_xticks(p)
    axes[0].set_ylim(0, 450)

    axes[1].plot(p, buffers, marker="o", color=navy, linewidth=1.6, markersize=4.2)
    axes[1].fill_between(p, buffers, color=pale_blue, alpha=0.28)
    axes[1].set_title("Persistent regressor storage", fontweight="bold")
    axes[1].set_xlabel("Innovation length, p")
    axes[1].set_ylabel("Buffer size (bytes)")
    axes[1].set_xticks(p)
    axes[1].set_ylim(0, 820)

    for left, right in zip(runtime_p1, runtime_p5):
        axes[2].plot((1, 5), (left, right), color="#A7ACB3", linewidth=0.8, zorder=1)
    axes[2].scatter(np.ones_like(runtime_p1), runtime_p1, color=navy, s=16, zorder=2)
    axes[2].scatter(
        np.full_like(runtime_p5, 5), runtime_p5, color=orange, s=16, zorder=2
    )
    axes[2].plot(
        (1, 5),
        (np.median(runtime_p1), np.median(runtime_p5)),
        color="#222222",
        marker="D",
        markersize=4.2,
        linewidth=1.6,
        zorder=3,
    )
    axes[2].set_title("Matched trajectory runtime", fontweight="bold")
    axes[2].set_xlabel("Innovation length, p")
    axes[2].set_ylabel("System median runtime (s)")
    axes[2].set_xticks((1, 5))
    axes[2].set_ylim(0.28, 0.56)

    for axis in axes:
        axis.grid(axis="y", color=grid, linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(OUT) + ".alignment-layout.json",
        overlay_svg=str(OUT) + ".alignment-overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    fig.savefig(str(OUT) + ".pdf")
    fig.savefig(str(OUT) + ".svg")
    fig.savefig(str(OUT) + ".png", dpi=RASTER_DPI)
    fig.savefig(
        str(OUT) + ".tiff",
        dpi=RASTER_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
