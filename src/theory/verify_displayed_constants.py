from fractions import Fraction as F
from pathlib import Path
import json, sys
import certify_transverse as ct

ROOT = Path(__file__).resolve().parent


def main():
    sys.set_int_max_str_digits(20000)
    b = json.loads((ROOT / "quantitative_escape_basin.json").read_text())
    e = json.loads((ROOT / "basin_entry_verified.json").read_text())
    f = json.loads((ROOT / "escape_final_checks.json").read_text())
    assert F(b["power_128_infinity_upper"]) < F(318, 1000)
    assert F(b["adapted_norm_equivalence_upper_M"]) < 46000
    assert F(b["transverse_contraction_upper"]) < F(399, 400)
    assert F(b["common_L_upper"]) < 31000000
    assert F(e["center_distance_upper"]) < F(743, 100) * F(1, 10**66)
    assert F(e["transverse_adapted_norm_upper"]) < F(259, 100) * F(1, 10**74)
    assert F(e["true_endpoint_norm_upper"]) < F(181, 100) * F(1, 10**32)
    assert F(e["true_matrix_prefactor_upper"]) < 5
    assert F(e["amplitude_absolute_lower"]) > F(275, 100) * 10**22
    R = F(1, 10**20)
    L = F(31000000)
    kappa = F(399, 400)
    a = F(1, 10**32)
    S = F(1, 10**40)
    assert a + L * S < R
    drift = L * a / (1 - kappa) + L * (1 + L / (1 - kappa)) * S
    assert drift < F(1625, 1000) * F(1, 10**22) < R / 4
    alpha = F(1001, 1000)
    eta = F(201, 200)
    assert F(1, 10**49) / (1 - alpha / eta) == F(25125, 10000) * F(1, 10**47)
    assert F(1, 10**49) / (1 - alpha / eta) < S
    assert f["maximum_spacing_between_certified_resets"] < 50

    C, _ = ct.exact_matrix()
    old = json.loads((ROOT / "transverse_interval_certificate.json").read_text())
    assert [[str(a) for a in row] for row in C] == old["exact_jacobian_rationals"]
    print(
        "All displayed conservative inequalities passed exact rational checks; original certificate unchanged."
    )


if __name__ == "__main__":
    main()
