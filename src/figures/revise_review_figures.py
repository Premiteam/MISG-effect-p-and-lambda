from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = Path(__file__).resolve().parent
SIZE = (15 / 2.54, 8 / 2.54)
BLUE = "#1F5A91"
ORANGE = "#C76E1D"
CHARCOAL = "#2B2B2B"
MID_GREY = "#6E6E6E"
GRID = "#D9DEE3"
PALE_BLUE = "#DCE8F2"
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 16,
        "xtick.labelsize": 15.5,
        "ytick.labelsize": 15.5,
        "legend.fontsize": 15,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _line_identity(label):
    compact = label.replace(" ", "").lower()
    if compact in {"p=1", "p=1.0"}:
        return BLUE, "-", "o"
    if compact in {"p=5", "p=5.0"}:
        return ORANGE, (0, (5, 2.2)), "^"
    if "validation" in compact:
        return BLUE, "-", "o"
    if "independenttest" in compact or compact.startswith("test,"):
        return CHARCOAL, (0, (4, 2)), "s"
    if "observed" in compact:
        return CHARCOAL, "-", None
    if "adaptive" in compact:
        return BLUE, "-", "o"
    if "fixedonestep" in compact:
        return ORANGE, (0, (5, 2.2)), "^"
    if "λ=1" in compact and "0.99" not in compact and "0.95" not in compact:
        return CHARCOAL, "-", "o"
    if "λ=0.99" in compact:
        return BLUE, (0, (4, 1.6, 1.2, 1.6)), "s"
    if "λ=0.95" in compact:
        return ORANGE, (0, (5, 2.2)), "^"
    return None


def _apply_journal_style(fig, stem):
    for ax in fig.axes:
        ax.set_facecolor("white")
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CHARCOAL)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(direction="out", length=3.2, width=0.7, colors=CHARCOAL, pad=2.5)
        ax.xaxis.label.set_color(CHARCOAL)
        ax.yaxis.label.set_color(CHARCOAL)
        for line in ax.lines:
            identity = _line_identity(line.get_label())
            if identity is None:
                continue
            color, linestyle, marker = identity
            line.set_color(color)
            line.set_linestyle(linestyle)
            line.set_linewidth(1.8)
            if marker and not stem.startswith("Figure9a"):
                line.set_marker(marker)
                line.set_markersize(4.5)
                line.set_markerfacecolor("white")
                line.set_markeredgecolor(color)
                line.set_markeredgewidth(1.0)

        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                collection.set_alpha(0.12)

    if stem in {"Figure2a_validation_risk", "Figure2b_selected_gamma"}:
        cmap = LinearSegmentedColormap.from_list(
            "tsp_blue", ["#F4F7FA", "#C7D9E8", "#6E9FC4", "#214F78"]
        )
        for ax in fig.axes:
            for image in ax.images:
                image.set_cmap(cmap)
                array = np.asarray(image.get_array())
                for annotation in ax.texts:
                    x, y = annotation.get_position()
                    column, row = int(round(x)), int(round(y))
                    if 0 <= row < array.shape[0] and 0 <= column < array.shape[1]:
                        rgba = image.cmap(image.norm(float(array[row, column])))
                        luminance = (
                            0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                        )
                        annotation.set_color(CHARCOAL if luminance > 0.62 else "white")

    if stem == "Figure4a_regime_contrast":
        ax = fig.axes[0]
        for line in ax.lines:
            if line.get_linestyle() != "--" and np.asarray(line.get_xdata()).size == 2:
                line.set_color(BLUE)
                line.set_linewidth(2.0)
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes(np.array([34.0]))
                collection.set_facecolor(BLUE)
                collection.set_edgecolor("white")
                collection.set_linewidth(0.7)

    if stem == "Figure3b_selected_gamma_201":
        for line in fig.axes[0].lines:
            line.set_color(BLUE)
            line.set_linewidth(1.8)
            line.set_marker("o")
            line.set_markersize(4.5)
            line.set_markerfacecolor("white")
            line.set_markeredgewidth(1.0)

    for legend in list(fig.legends):
        legend.get_frame().set_linewidth(0)
        for handle, label in zip(
            legend.legend_handles, [text.get_text() for text in legend.get_texts()]
        ):
            identity = _line_identity(label)
            if identity is not None and isinstance(handle, Line2D):
                color, linestyle, marker = identity
                handle.set_color(color)
                handle.set_linestyle(linestyle)
                handle.set_linewidth(1.8)
                if marker:
                    handle.set_marker(marker)
                    handle.set_markerfacecolor("white")
                    handle.set_markeredgecolor(color)
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_linewidth(0)
            for handle, label in zip(
                legend.legend_handles, [text.get_text() for text in legend.get_texts()]
            ):
                identity = _line_identity(label)
                if identity is not None and isinstance(handle, Line2D):
                    color, linestyle, marker = identity
                    handle.set_color(color)
                    handle.set_linestyle(linestyle)
                    handle.set_linewidth(1.8)
                    if marker:
                        handle.set_marker(marker)
                        handle.set_markerfacecolor("white")
                        handle.set_markeredgecolor(color)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def export(fig, stem):
    if stem in {
        "Figure2_balanced_tuning_two_panel",
        "Figure4_component_attribution_compact",
        "Figure3_local_refinement_201_combined",
        "Figure4b_contrast_ratio",
    }:
        return
    _apply_journal_style(fig, stem)
    wide = stem.startswith("Figure8")
    size = 16
    for ax in fig.axes:
        if not wide:
            ax.set_title("")
        for title in (ax.title, ax._left_title, ax._right_title):
            if not wide:
                title.set_text("")
        if ax.get_ylabel() == "Risk (log scale)":
            ax.set_ylabel("Median R (log scale)")
        if "Paired median error ratio, tuned" in ax.get_xlabel():
            ax.set_xlabel("Paired median ratio, λ = 0.95 / λ = 1")
        if "cumulative probability" in ax.get_ylabel():
            ax.set_ylabel("Cumulative probability")
        if "System parameter error criterion" in ax.get_xlabel():
            ax.set_xlabel("System error criterion, R")
    for txt in fig.findobj(Text):
        txt.set_fontsize(size)
        if wide:
            txt.set_fontfamily("DejaVu Sans")
    if stem == "Figure2a_validation_risk":
        for patch in list(fig.axes[0].patches):
            patch.remove()
    if stem == "Figure8a_long_horizon_curves":
        for ax in fig.axes:
            ax.set_xticks([100, 100000], ["10²", "10⁵"])
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.24, top=0.75, wspace=0.40)
    if stem == "Figure8b_horizon_ratios":
        ax = fig.axes[0]
        ax.set_xticks([2000, 10000, 100000], ["2 × 10³", "10⁴", "10⁵"])
        ax.set_ylabel("Paired median ratio\np = 1 / p = 5", fontsize=size)
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.23, top=0.78)
    if stem == "Figure3a_local_risk_201":
        fig.legends[0].remove()
        fig.axes[0].legend(
            ["Validation", "Independent test"],
            loc="lower left",
            bbox_to_anchor=(0, 1.03),
            frameon=False,
            fontsize=size,
            ncol=2,
            borderaxespad=0,
            handlelength=1.2,
            columnspacing=0.8,
        )
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.29, top=0.79)
    if stem == "FigureA1_partition_ecdf":
        fig.axes[0].set_ylabel("Cumulative\nprobability", fontsize=size)
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.24, top=0.70)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT / f"{stem}.alignment.json",
        require_panel_labels=False,
        strict=True,
    )
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", facecolor="white", dpi=600)
    fig.savefig(
        OUT / f"{stem}.tiff",
        facecolor="white",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def heating(source):
    contrasts = read_rows(source / "principal_contrasts.csv")
    fig, ax = plt.subplots(figsize=SIZE)
    for mode, color, marker, shift, label in [
        ("online", BLUE, "o", -0.08, "Adaptive"),
        ("fixed_parameter_one_step", ORANGE, "^", 0.08, "Fixed one step"),
    ]:
        rs = [
            next(
                r for r in contrasts if r["mode"] == mode and float(r["lambda"]) == lam
            )
            for lam in (1.0, 0.95)
        ]
        y, lo, hi = [
            np.array([float(r[k]) for r in rs])
            for k in [
                "p1_to_p5_rmse_ratio",
                "block_bootstrap_90pct_lower",
                "block_bootstrap_90pct_upper",
            ]
        ]
        ax.errorbar(
            np.arange(2) + shift,
            y,
            yerr=[y - lo, hi - y],
            fmt=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=color,
            capsize=4,
            capthick=1.0,
            label=label,
        )
    ax.axhline(1, color="#777777", ls=":", lw=1)
    ax.set_xticks([0, 1], ["λ = 1", "λ = 0.95"])
    ax.set(xlim=(-0.35, 1.35), ylim=(0.5, 4.2), ylabel="RMSE ratio, p = 1 / p = 5")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.96)
    export(fig, "Figure9b_DaISy_heating_ratios")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--horizon-only", action="store_true")
    args = parser.parse_args()
    source = args.source_root
    for filename in (
        ()
        if args.horizon_only
        else (
            "regenerate_figure2.py",
            "regenerate_figure3.py",
            "regenerate_figure4_compact.py",
        )
    ):
        m = module(ROOT / "figures" / filename, filename[:-3])
        m.export_figure = export
        m.export = export
        m.main()
    m = module(
        source / "review_1_4_analysis" / "make_review_figures.py",
        "horizon_figure_source",
    )
    m.export = export

    def audit_local(*args, **kwargs):
        for key in ("json_out", "overlay_svg"):
            if key in kwargs:
                kwargs[key] = OUT / Path(kwargs[key]).name
        return require_matplotlib_panel_alignment(*args, **kwargs)

    m.require_matplotlib_panel_alignment = audit_local
    if args.horizon_only:
        m.make_long_horizon_curves()
        m.make_horizon_ratios()
    else:
        m.main()
        heating(
            source
            / "real_data_validation"
            / "daisy_heating"
            / "results_frozen_diagnostic"
        )


if __name__ == "__main__":
    main()
