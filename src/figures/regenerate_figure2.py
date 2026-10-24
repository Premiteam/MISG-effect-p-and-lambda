from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from audit_panel_alignment import require_matplotlib_panel_alignment


HERE = Path(__file__).resolve()
OUTPUT_DIR = HERE.parent
DATA_DIR = HERE.parents[2] / "data"
SOURCE_DIR = DATA_DIR
STEM = "Figure2_balanced_tuning_two_panel"

FIGURE_WIDTH_CM = 15.0
FIGURE_HEIGHT_CM = 8.0
FIGURE_SIZE_IN = (FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54)
RASTER_DPI = 600

COLORS = {
    "blue_dark": "#0B3C6F",
    "blue_mid": "#8FB8D8",
    "blue_light": "#F5F8FB",
    "orange": "#D58A32",
    "grey_dark": "#3F3F3F",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_source_data(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "panel",
        "p",
        "lambda",
        "validation_median_system_risk",
        "selected_gamma",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel_letter_below(ax: mpl.axes.Axes, letter: str) -> None:
    return None


def export_figure(fig: mpl.figure.Figure, stem: str) -> None:
    prefix = OUTPUT_DIR / stem
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=RASTER_DPI, facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=RASTER_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    data_path = DATA_DIR / "figure2_balanced_full_factorial_summary.csv"
    rows = read_csv(data_path)
    if len(rows) != 30:
        raise ValueError(f"Expected 30 factorial cells, found {len(rows)}")

    p_values = [1, 2, 5, 8, 12]
    lambda_values = [1.0, 0.99, 0.95, 0.925, 0.90, 0.85]
    lookup = {(int(row["p"]), float(row["lambda"])): row for row in rows}
    expected = {(p, lam) for p in p_values for lam in lambda_values}
    if set(lookup) != expected:
        raise ValueError(
            "The factorial grid is incomplete or contains unexpected cells"
        )

    risk = np.array(
        [
            [float(lookup[(p, lam)]["val_median_R"]) for lam in lambda_values]
            for p in p_values
        ]
    )
    gamma = np.array(
        [
            [float(lookup[(p, lam)]["gamma_star"]) for lam in lambda_values]
            for p in p_values
        ]
    )
    if not (np.all(np.isfinite(risk)) and np.all(risk > 0)):
        raise ValueError(
            "Risk values must be finite and strictly positive for the log scale"
        )
    if not np.all(np.isfinite(gamma)):
        raise ValueError("Selected gamma values must be finite")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, object]] = []
    for p in p_values:
        for lam in lambda_values:
            row = lookup[(p, lam)]
            source_rows.append(
                {
                    "panel": "a-b",
                    "p": p,
                    "lambda": lam,
                    "validation_median_system_risk": row["val_median_R"],
                    "selected_gamma": row["gamma_star"],
                }
            )
    write_source_data(SOURCE_DIR / f"{STEM}_source_data.csv", source_rows)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_IN,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    risk_cmap = LinearSegmentedColormap.from_list(
        "riskblue", [COLORS["blue_dark"], COLORS["blue_mid"], COLORS["blue_light"]]
    )
    risk_image = ax_a.imshow(
        risk,
        aspect="auto",
        cmap=risk_cmap,
        norm=LogNorm(vmin=float(risk.min()), vmax=float(risk.max())),
    )
    ax_a.set_title("Validation median system risk", pad=4, fontweight="bold")
    ax_a.set_xticks(
        range(len(lambda_values)), [f"{value:g}" for value in lambda_values]
    )
    ax_a.set_yticks(range(len(p_values)), [str(value) for value in p_values])
    ax_a.set_xlabel("Forgetting factor, λ")
    ax_a.set_ylabel("Innovation length, p")
    for row_index in range(risk.shape[0]):
        for column_index in range(risk.shape[1]):
            text_color = (
                "white" if risk[row_index, column_index] < 0.02 else COLORS["grey_dark"]
            )
            ax_a.text(
                column_index,
                row_index,
                f"{risk[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=14.5,
                color=text_color,
            )
    highlight = Rectangle(
        (1.5, -0.5),
        1.0,
        risk.shape[0],
        fill=False,
        edgecolor=COLORS["orange"],
        linewidth=1.4,
    )
    highlight.set_gid("lambda-095-highlight")
    ax_a.add_patch(highlight)
    risk_cbar = fig.colorbar(risk_image, ax=ax_a, fraction=0.046, pad=0.025)
    risk_cbar.set_ticks(
        [0.01, 0.02, 0.05, 0.10], labels=["0.01", "0.02", "0.05", "0.10"]
    )
    risk_cbar.ax.tick_params(labelsize=14.5, width=0.5)
    risk_cbar.set_label("Risk (log scale)", fontsize=15.0)
    panel_letter_below(ax_a, "a")

    gamma_cmap = LinearSegmentedColormap.from_list(
        "gamma", ["#F4F6F8", "#A8B8CC", "#4F6B8A"]
    )
    gamma_image = ax_b.imshow(
        gamma,
        aspect="auto",
        cmap=gamma_cmap,
        norm=Normalize(vmin=0, vmax=4),
    )
    ax_b.set_title("Cell-specific tuning parameter", pad=4, fontweight="bold")
    ax_b.set_xticks(
        range(len(lambda_values)), [f"{value:g}" for value in lambda_values]
    )
    ax_b.set_yticks(range(len(p_values)), [str(value) for value in p_values])
    ax_b.set_xlabel("Forgetting factor, λ")
    ax_b.set_ylabel("Innovation length, p")
    for row_index in range(gamma.shape[0]):
        for column_index in range(gamma.shape[1]):
            text_color = (
                "white"
                if gamma[row_index, column_index] >= 2.5
                else COLORS["grey_dark"]
            )
            ax_b.text(
                column_index,
                row_index,
                f"{gamma[row_index, column_index]:g}",
                ha="center",
                va="center",
                fontsize=14.5,
                color=text_color,
            )
    gamma_cbar = fig.colorbar(gamma_image, ax=ax_b, fraction=0.046, pad=0.025)
    gamma_cbar.ax.tick_params(labelsize=14.5, width=0.5)
    gamma_cbar.set_label("Selected γ", fontsize=15.0)
    panel_letter_below(ax_b, "b")

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_a, ax_b],
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        json_out=OUTPUT_DIR / f"{STEM}.alignment.json",
        overlay_svg=OUTPUT_DIR / f"{STEM}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )

    export_figure(fig, STEM)
    plt.close(fig)

    fig_a, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    image = ax.imshow(
        risk,
        aspect="auto",
        cmap=risk_cmap,
        norm=LogNorm(vmin=float(risk.min()), vmax=float(risk.max())),
    )
    ax.set_title("Validation median system risk", pad=4, fontweight="bold")
    ax.set_xticks(range(len(lambda_values)), [f"{value:g}" for value in lambda_values])
    ax.set_yticks(range(len(p_values)), [str(value) for value in p_values])
    ax.set_xlabel("Forgetting factor, λ")
    ax.set_ylabel("Innovation length, p")
    for row_index in range(risk.shape[0]):
        for column_index in range(risk.shape[1]):
            text_color = (
                "white" if risk[row_index, column_index] < 0.02 else COLORS["grey_dark"]
            )
            ax.text(
                column_index,
                row_index,
                f"{risk[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=14.5,
                color=text_color,
            )
    ax.add_patch(
        Rectangle(
            (1.5, -0.5),
            1.0,
            risk.shape[0],
            fill=False,
            edgecolor=COLORS["orange"],
            linewidth=1.4,
        )
    )
    cbar = fig_a.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_ticks([0.01, 0.02, 0.05, 0.10], labels=["0.01", "0.02", "0.05", "0.10"])
    cbar.ax.tick_params(labelsize=8.8, width=0.5)
    cbar.ax.tick_params(labelsize=14.5, width=0.5)
    cbar.set_label("Risk (log scale)", fontsize=15.0)
    require_matplotlib_panel_alignment(
        fig_a,
        axes=[ax],
        panel_ids=["a"],
        require_panel_labels=False,
        json_out=OUTPUT_DIR / "Figure2a_validation_risk.alignment.json",
        overlay_svg=OUTPUT_DIR / "Figure2a_validation_risk.alignment.svg",
    )
    export_figure(fig_a, "Figure2a_validation_risk")
    plt.close(fig_a)

    fig_b, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    image = ax.imshow(
        gamma, aspect="auto", cmap=gamma_cmap, norm=Normalize(vmin=0, vmax=4)
    )
    ax.set_title("Cell-specific tuning parameter", pad=4, fontweight="bold")
    ax.set_xticks(range(len(lambda_values)), [f"{value:g}" for value in lambda_values])
    ax.set_yticks(range(len(p_values)), [str(value) for value in p_values])
    ax.set_xlabel("Forgetting factor, λ")
    ax.set_ylabel("Innovation length, p")
    for row_index in range(gamma.shape[0]):
        for column_index in range(gamma.shape[1]):
            text_color = (
                "white"
                if gamma[row_index, column_index] >= 2.5
                else COLORS["grey_dark"]
            )
            ax.text(
                column_index,
                row_index,
                f"{gamma[row_index, column_index]:g}",
                ha="center",
                va="center",
                fontsize=14.5,
                color=text_color,
            )
    cbar = fig_b.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=8.8, width=0.5)
    cbar.ax.tick_params(labelsize=14.5, width=0.5)
    cbar.set_label("Selected γ", fontsize=15.0)
    require_matplotlib_panel_alignment(
        fig_b,
        axes=[ax],
        panel_ids=["b"],
        require_panel_labels=False,
        json_out=OUTPUT_DIR / "Figure2b_selected_gamma.alignment.json",
        overlay_svg=OUTPUT_DIR / "Figure2b_selected_gamma.alignment.svg",
    )
    export_figure(fig_b, "Figure2b_selected_gamma")
    plt.close(fig_b)
    print(f"Created 15 cm x 8 cm exports for {STEM} and its two panels")


if __name__ == "__main__":
    main()
