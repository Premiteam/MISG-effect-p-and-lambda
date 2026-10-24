from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SKILL_SCRIPTS = HERE.parents[1] / "tools"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 15.0,
        "axes.labelsize": 15.0,
        "axes.titlesize": 16.0,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BLUE = "#0F4D92"
BLUE_LIGHT = "#A8C5E5"
GREY = "#767676"
FIGURE_SIZE_IN = (15.0 / 2.54, 8.0 / 2.54)
RASTER_DPI = 600


def export_figure(fig: plt.Figure, stem: str) -> None:
    output = HERE / stem
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=RASTER_DPI, facecolor="white")
    fig.savefig(
        output.with_suffix(".tiff"),
        dpi=RASTER_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def read_rows() -> list[dict[str, str]]:
    with (HERE.parents[1] / "data" / "figure4_source_data.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9:
        raise ValueError("Figure 4 requires five panel-a rows and four panel-b rows")
    return rows


def forest(
    ax: plt.Axes,
    labels: list[str],
    estimates: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    color: str,
    *,
    reference: float = 1.0,
    log_scale: bool = False,
) -> None:
    if not (
        np.all(np.isfinite(estimates))
        and np.all(np.isfinite(lows))
        and np.all(np.isfinite(highs))
    ):
        raise ValueError("Figure 4 contains nonfinite values")
    if not (np.all(lows <= estimates) and np.all(estimates <= highs)):
        raise ValueError("Every confidence interval must contain its point estimate")
    if log_scale and (
        np.any(lows <= 0) or np.any(estimates <= 0) or np.any(highs <= 0)
    ):
        raise ValueError("Log-scale forest values must be strictly positive")
    y = np.arange(len(labels))[::-1]
    for yi, estimate, low, high in zip(y, estimates, lows, highs):
        ax.plot(
            [low, high], [yi, yi], color=color, linewidth=1.5, solid_capstyle="round"
        )
        ax.scatter(
            estimate, yi, color=color, s=20, edgecolor="white", linewidth=0.5, zorder=3
        )
    ax.axvline(reference, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.6, zorder=-1)
    if log_scale:
        ax.set_xscale("log")


def values(
    rows: list[dict[str, str]], panel: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = [row for row in rows if row["panel"] == panel]
    return tuple(
        np.asarray([float(row[key]) for row in subset])
        for key in ("estimate", "ci_low", "ci_high")
    )


def main() -> None:
    rows = read_rows()
    panel_a = [row for row in rows if row["panel"] == "a"]
    panel_b = [row for row in rows if row["panel"] == "b"]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_IN,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 0.95]},
    )

    est_a, low_a, high_a = values(rows, "a")
    labels_a = [f"p={row['comparison'].rsplit('=', 1)[-1]}" for row in panel_a]
    forest(ax_a, labels_a, est_a, low_a, high_a, BLUE, log_scale=True)
    ax_a.set_title(
        "Tuned regime contrast by innovation length", fontweight="bold", pad=5
    )
    ax_a.set_xlabel("Paired median error ratio, tuned λ=0.95 / tuned λ=1")
    ax_a.set_xlim(0.03, 1.2)
    ax_a.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0], ["0.05", "0.1", "0.2", "0.5", "1"])

    est_b, low_b, high_b = values(rows, "b")
    labels_b = [f"p=1 vs {row['comparison'].rsplit('=', 1)[-1]}" for row in panel_b]
    forest(ax_b, labels_b, est_b, low_b, high_b, BLUE_LIGHT)
    ax_b.set_title("Contrast ratio by innovation length", fontweight="bold", pad=5)
    ax_b.set_xlabel("Ratio of tuned regime contrasts")
    ax_b.set_xlim(0.78, 1.02)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_a, ax_b],
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        exemptions=[
            {
                "panels": ["a", "b"],
                "checks": ["panel-width", "horizontal-gutter"],
                "reason": "Distinct label lengths and numerical ranges require slightly unequal panel widths.",
            }
        ],
        json_out=HERE / "Figure4_component_attribution_compact.alignment.json",
        overlay_svg=HERE / "Figure4_component_attribution_compact.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )

    export_figure(fig, "Figure4_component_attribution_compact")
    plt.close(fig)

    fig_a, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    forest(ax, labels_a, est_a, low_a, high_a, BLUE, log_scale=True)
    ax.set_title("Tuned regime contrast by innovation length", fontweight="bold", pad=5)
    ax.set_xlabel("Paired median error ratio, tuned λ=0.95 / tuned λ=1")
    ax.set_xlim(0.03, 1.2)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0], ["0.05", "0.1", "0.2", "0.5", "1"])
    require_matplotlib_panel_alignment(
        fig_a,
        axes=[ax],
        panel_ids=["a"],
        require_panel_labels=False,
        json_out=HERE / "Figure4a_regime_contrast.alignment.json",
        overlay_svg=HERE / "Figure4a_regime_contrast.alignment.svg",
    )
    export_figure(fig_a, "Figure4a_regime_contrast")
    plt.close(fig_a)

    fig_b, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    forest(ax, labels_b, est_b, low_b, high_b, BLUE_LIGHT)
    ax.set_title("Contrast ratio by innovation length", fontweight="bold", pad=5)
    ax.set_xlabel("Ratio of tuned regime contrasts")
    ax.set_xlim(0.78, 1.02)
    require_matplotlib_panel_alignment(
        fig_b,
        axes=[ax],
        panel_ids=["b"],
        require_panel_labels=False,
        json_out=HERE / "Figure4b_contrast_ratio.alignment.json",
        overlay_svg=HERE / "Figure4b_contrast_ratio.alignment.svg",
    )
    export_figure(fig_b, "Figure4b_contrast_ratio")
    plt.close(fig_b)


if __name__ == "__main__":
    main()
