from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 15.0,
        "axes.titlesize": 16.0,
        "axes.labelsize": 15.0,
        "legend.fontsize": 14.5,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "data" / "record_length_results"
FIGURES = HERE
COLORS = {1: "#2F6EA8", 5: "#D97A1B"}
LAMBDA_COLORS = {1.0: "#222222", 0.99: "#4C78A8", 0.95: "#D97A1B"}
FIGURE_SIZE_IN = (15.0 / 2.54, 8.0 / 2.54)
RASTER_DPI = 600

FIGURE_AUDIT_SCRIPTS = REPO / "tools"
if FIGURE_AUDIT_SCRIPTS.exists():
    sys.path.insert(0, str(FIGURE_AUDIT_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    require_matplotlib_panel_alignment = None


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.png", dpi=RASTER_DPI, facecolor="white")
    fig.savefig(
        FIGURES / f"{stem}.tiff",
        dpi=RASTER_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def make_partition_ecdf() -> None:
    data = rows(RESULTS / "trajectory_swap_system_risks.csv")
    styles = {
        "val_original": ("Validation, original", "#2F6EA8", "-"),
        "val_swapped": ("Validation, exchanged", "#2F6EA8", "--"),
        "test_original": ("Test, original", "#D97A1B", "-"),
        "test_swapped": ("Test, exchanged", "#D97A1B", "--"),
    }
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.78)
    for condition, (label, color, linestyle) in styles.items():
        values = np.asarray(
            [float(r["system_risk"]) for r in data if r["condition"] == condition]
        )
        if np.any(values <= 0):
            raise ValueError(
                "The logarithmic error axis requires positive system risks."
            )
        x, y = empirical_cdf(values)
        ax.step(
            x,
            y,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("System parameter error criterion, R")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0, 1.01)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            axes=[ax],
            panel_ids=["a"],
            json_out=RESULTS / "FigureA1_alignment.json",
            overlay_svg=RESULTS / "FigureA1_alignment_overlay.svg",
            require_panel_labels=False,
        )
    export(fig, "FigureA1_partition_ecdf")
    plt.close(fig)


def make_long_horizon_curves() -> None:
    data = rows(RESULTS / "long_horizon_curves.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=FIGURE_SIZE_IN, sharex=True, sharey=True, constrained_layout=False
    )
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.22, top=0.72, wspace=0.30)
    for ax, lam in zip(axes, (1.0, 0.99, 0.95)):
        for p in (1, 5):
            selected = sorted(
                [
                    r
                    for r in data
                    if int(r["p"]) == p and np.isclose(float(r["lambda"]), lam)
                ],
                key=lambda r: int(r["k"]),
            )
            k = np.asarray([int(r["k"]) for r in selected])
            median = np.asarray([float(r["median_delta_pct"]) for r in selected])
            q25 = np.asarray([float(r["q25_delta_pct"]) for r in selected])
            q75 = np.asarray([float(r["q75_delta_pct"]) for r in selected])
            if (
                np.any(k <= 0)
                or np.any(median <= 0)
                or np.any(q25 <= 0)
                or np.any(q75 <= 0)
            ):
                raise ValueError(
                    "The logarithmic horizon panels require positive coordinates."
                )
            ax.plot(k, median, color=COLORS[p], linewidth=1.45, label=f"p = {p}")
            ax.fill_between(k, q25, q75, color=COLORS[p], alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_title(f"λ = {lam:g}", loc="left", pad=2)
        if ax is axes[0]:
            ax.set_ylabel("Relative error (%)")
        ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.supxlabel("Observation index, k", y=0.045, fontsize=15.0)
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            axes=list(axes),
            panel_ids=["lambda1", "lambda099", "lambda095"],
            row_groups=[["lambda1", "lambda099", "lambda095"]],
            json_out=RESULTS / "Figure8a_alignment.json",
            overlay_svg=RESULTS / "Figure8a_alignment_overlay.svg",
            require_panel_labels=False,
        )
    export(fig, "Figure8a_long_horizon_curves")
    plt.close(fig)


def make_horizon_ratios() -> None:
    data = rows(RESULTS / "horizon_metric_comparisons.csv")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.80)
    for lam in (1.0, 0.99, 0.95):
        selected = sorted(
            [
                r
                for r in data
                if r["metric"] == "second_half_prefix_R"
                and np.isclose(float(r["lambda"]), lam)
            ],
            key=lambda r: int(r["horizon"]),
        )
        x = np.asarray([int(r["horizon"]) for r in selected])
        y = np.asarray([float(r["ratio_p1_over_p5"]) for r in selected])
        lo = np.asarray([float(r["ci90_lo"]) for r in selected])
        hi = np.asarray([float(r["ci90_hi"]) for r in selected])
        color = LAMBDA_COLORS[lam]
        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            markersize=3.6,
            linewidth=1.4,
            label=f"λ = {lam:g}",
        )
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.axhspan(1 / 1.02, 1.02, color="#999999", alpha=0.10, linewidth=0)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Paired median ratio, p = 1 / p = 5")
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            axes=[ax],
            panel_ids=["a"],
            json_out=RESULTS / "Figure8b_alignment.json",
            overlay_svg=RESULTS / "Figure8b_alignment_overlay.svg",
            require_panel_labels=False,
        )
    export(fig, "Figure8b_horizon_ratios")
    plt.close(fig)


def main() -> None:
    make_partition_ecdf()
    make_long_horizon_curves()
    make_horizon_ratios()


if __name__ == "__main__":
    main()
