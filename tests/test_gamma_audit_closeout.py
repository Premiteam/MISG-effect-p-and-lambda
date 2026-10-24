from __future__ import annotations

import numpy as np

import run_gamma_audit_closeout as closeout


def test_joint_bootstrap_uses_bonferroni_9833_intervals() -> None:
    vectors = {
        "S1": np.log(np.array([0.10, 0.12, 0.09, 0.11])),
        "S2": np.log(np.array([0.20, 0.18, 0.22, 0.19])),
        "S3": np.log(np.array([1.20, 1.25, 1.18, 1.22])),
    }

    rows = closeout.joint_bootstrap_rows(vectors, B=1000, seed=7)

    assert len(rows) == 3
    assert all(row["simultaneous_ci_level"] == 1.0 - 0.05 / 3.0 for row in rows)
    assert all(row["n_systems"] == 4 for row in rows)
    assert (
        next(row for row in rows if row["comparison_id"] == "S1")[
            "simultaneous_conclusion"
        ]
        == "numerator lower paired risk"
    )
    assert (
        next(row for row in rows if row["comparison_id"] == "S3")[
            "simultaneous_conclusion"
        ]
        == "denominator lower paired risk"
    )


def test_equivalence_requires_entire_90pct_interval_inside_bounds() -> None:
    risk_a = np.array([1.00, 1.01, 0.99, 1.00, 1.01, 0.99])
    risk_b = np.ones(6)

    row = closeout.equivalence_row(risk_a, risk_b, B=1000, seed=11, delta=0.05)

    assert row["equivalent_90pct_CI"] is True
    assert row["ci90_ratio_lo"] > 1.0 / 1.05
    assert row["ci90_ratio_hi"] < 1.05
