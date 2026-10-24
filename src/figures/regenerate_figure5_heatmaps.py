import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "figure5_source_data.csv"
SOURCE_JSON = ROOT.parents[1] / "data" / "balanced_full_factorial_results.json"
PROVENANCE = ROOT.parents[1] / "data" / "figure5_source_provenance.json"
P_VALUES = [1, 2, 5, 8, 12]
LAMBDA_VALUES = [1.0, 0.99, 0.95, 0.925, 0.90, 0.85]
FIGURE_SIZE_IN = (15.0 / 2.54, 8.0 / 2.54)
RASTER_DPI = 600

PANELS = (
    (
        "test_median_R",
        "Independent test median parameter estimation error criterion",
        "Median identification-error index",
        "02_test_risk_heatmap.png",
    ),
    (
        "failure_rate_30pct",
        "30% threshold non-attainment rate",
        "Non-attainment rate",
        "04_failure30_heatmap.png",
    ),
    (
        "failure_rate_10pct",
        "10% threshold non-attainment rate",
        "Non-attainment rate",
        "05_failure10_heatmap.png",
    ),
)


def export_source_data() -> None:
    raw = SOURCE_JSON.read_bytes()
    payload = json.loads(raw)
    meta = payload.get("meta", {})
    expected = {
        "analysis_type": "balanced_full_factorial_gamma_tuning",
        "n_train": 201,
        "n_val": 102,
        "n_test": 201,
        "n_train_traj": 1,
        "n_eval_traj": 2,
        "reuse_old_cells": False,
        "test_usage": "confirmation_only",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Unexpected metadata {key}={meta.get(key)!r}")

    cells = payload.get("cells", {})
    expected_keys = {
        f"p={p_value}|lambda={float(lambda_value)}"
        for p_value in P_VALUES
        for lambda_value in LAMBDA_VALUES
    }
    if set(cells) != expected_keys:
        raise ValueError(
            "The balanced factorial JSON does not contain the expected 30 cells"
        )

    rows = []
    for p_value in P_VALUES:
        for lambda_value in LAMBDA_VALUES:
            cell = cells[f"p={p_value}|lambda={float(lambda_value)}"]
            test = cell["test"]
            if int(test["n_systems"]) != 201 or len(test["system_risks"]) != 201:
                raise ValueError(
                    f"Incomplete test cell for p={p_value}, lambda={lambda_value}"
                )
            rows.append(
                {
                    "p": p_value,
                    "lambda": lambda_value,
                    "gamma_star": cell["gamma"],
                    "val_median_R": cell["val"]["median_R"],
                    "test_median_R": test["median_R"],
                    "test_mean_R": test["mean_R"],
                    "test_divergences": test["divergences"],
                    "failure_rate_30pct": test["failure_rate_30pct"],
                    "failure_rate_10pct": test["failure_rate_10pct"],
                    "n_failed_30pct": test["n_failed_30pct"],
                    "n_failed_10pct": test["n_failed_10pct"],
                    "n_test_systems": test["n_systems"],
                    "source": "balanced_full_factorial_201",
                }
            )

    with DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "source_json": str(SOURCE_JSON),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "analysis_type": meta["analysis_type"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": meta["n_test"],
        "n_train_traj": meta["n_train_traj"],
        "n_eval_traj": meta["n_eval_traj"],
        "cell_count": len(rows),
        "reuse_old_cells": meta["reuse_old_cells"],
        "test_usage": meta["test_usage"],
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def matrix_from_frame(frame: pd.DataFrame, column: str) -> np.ndarray:
    matrix = np.full((len(P_VALUES), len(LAMBDA_VALUES)), np.nan, dtype=float)
    for i, p_value in enumerate(P_VALUES):
        for j, lambda_value in enumerate(LAMBDA_VALUES):
            row = frame[
                (frame["p"] == p_value) & np.isclose(frame["lambda"], lambda_value)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one row for p={p_value}, lambda={lambda_value}; "
                    f"found {len(row)}."
                )
            matrix[i, j] = float(row.iloc[0][column])
    return matrix


def save_heatmap(
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    output_name: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="Blues_r")
    ax.set_xticks(
        range(len(LAMBDA_VALUES)),
        labels=[f"{value:g}" for value in LAMBDA_VALUES],
    )
    ax.set_yticks(range(len(P_VALUES)), labels=[str(value) for value in P_VALUES])
    ax.set_xlabel(r"Forgetting factor $\lambda$")
    ax.set_ylabel(r"Innovation length $p$")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                j,
                i,
                f"{value:.3g}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="white" if luminance < 0.52 else "black",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    output_stem = Path(output_name).stem
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        exclude_axes=[colorbar.ax],
        panel_ids=["a"],
        require_panel_labels=False,
        json_out=ROOT / f"{output_stem}.alignment.json",
        overlay_svg=ROOT / f"{output_stem}.alignment.svg",
    )
    fig.savefig(ROOT / f"{output_stem}.svg")
    fig.savefig(ROOT / f"{output_stem}.pdf")
    fig.savefig(
        ROOT / f"{output_stem}.tiff",
        dpi=RASTER_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(ROOT / output_name, dpi=RASTER_DPI)
    plt.close(fig)


def main() -> None:
    export_source_data()
    frame = pd.read_csv(DATA, encoding="utf-8-sig")
    for column, title, colorbar_label, output_name in PANELS:
        save_heatmap(
            matrix_from_frame(frame, column),
            title,
            colorbar_label,
            output_name,
        )


if __name__ == "__main__":
    main()
