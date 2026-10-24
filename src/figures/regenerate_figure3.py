from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "data" / "figure3_local_refinement_source.csv"
FIGURE_SIZE_IN = (15.0 / 2.54, 8.0 / 2.54)
RASTER_DPI = 600

SKILL_SCRIPTS = HERE.parents[1] / "tools"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 15.0,
        "axes.labelsize": 15.0,
        "axes.titlesize": 16.0,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "legend.fontsize": 14.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def export(fig: plt.Figure, stem: str) -> None:
    prefix = HERE / stem
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=RASTER_DPI, facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=RASTER_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def load_data() -> tuple[list[float], list[float], list[float], list[float]]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: float(row["lambda"]))
    if len(rows) != 7:
        raise ValueError(f"Expected seven local-refinement rows, found {len(rows)}")
    lambdas = [float(row["lambda"]) for row in rows]
    validation = [float(row["validation_median_risk"]) for row in rows]
    test = [float(row["test_median_risk"]) for row in rows]
    gamma = [float(row["gamma_star"]) for row in rows]
    return lambdas, validation, test, gamma


def style_risk(ax: plt.Axes, x, labels, validation, test) -> None:
    ax.plot(
        x,
        validation,
        color="#2F6FA3",
        marker="o",
        markersize=3.2,
        linewidth=1.35,
        label="Validation median R",
    )
    ax.plot(
        x,
        test,
        color="#4D4D4D",
        marker="s",
        markersize=2.8,
        linewidth=1.1,
        linestyle="--",
        label="Independent test median R",
    )
    ax.set_xlabel("Sampled forgetting factor, λ")
    ax.set_ylabel("Median value of R")
    ax.set_title("Local error profile at p=5", fontweight="bold")
    ax.set_xticks(x, labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)


def style_gamma(ax: plt.Axes, x, labels, gamma) -> None:
    ax.plot(x, gamma, color="#D9822B", marker="o", markersize=3.2, linewidth=1.35)
    ax.set_xlabel("Sampled forgetting factor, λ")
    ax.set_ylabel("Validation-selected γ")
    ax.set_title("Validation-selected step-size multiplier at p=5", fontweight="bold")
    ax.set_xticks(x, labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)


def main() -> None:
    lambdas, validation, test, gamma = load_data()
    x = list(range(len(lambdas)))
    labels = [f"{value:g}" for value in lambdas]

    fig_a, ax_a = plt.subplots(figsize=FIGURE_SIZE_IN, layout=None)
    fig_a.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.76)
    style_risk(ax_a, x, labels, validation, test)
    handles, legend_labels = ax_a.get_legend_handles_labels()
    fig_a.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
    )
    require_matplotlib_panel_alignment(
        fig_a,
        axes=[ax_a],
        panel_ids=["a"],
        require_panel_labels=False,
        json_out=HERE / "Figure3a_local_risk_201.alignment.json",
        overlay_svg=HERE / "Figure3a_local_risk_201.alignment.svg",
    )
    export(fig_a, "Figure3a_local_risk_201")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    style_gamma(ax_b, x, labels, gamma)
    require_matplotlib_panel_alignment(
        fig_b,
        axes=[ax_b],
        panel_ids=["b"],
        require_panel_labels=False,
        json_out=HERE / "Figure3b_selected_gamma_201.alignment.json",
        overlay_svg=HERE / "Figure3b_selected_gamma_201.alignment.svg",
    )
    export(fig_b, "Figure3b_selected_gamma_201")
    plt.close(fig_b)

    combined, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE_IN, layout=None)
    combined.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.72, wspace=0.38)
    style_risk(left, x, labels, validation, test)
    style_gamma(right, x, labels, gamma)
    handles, legend_labels = left.get_legend_handles_labels()
    combined.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
    )
    require_matplotlib_panel_alignment(
        combined,
        axes=[left, right],
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        require_panel_labels=False,
        json_out=HERE / "Figure3_local_refinement_201_combined.alignment.json",
        overlay_svg=HERE / "Figure3_local_refinement_201_combined.alignment.svg",
        strict=True,
    )
    export(combined, "Figure3_local_refinement_201_combined")
    plt.close(combined)


if __name__ == "__main__":
    main()
