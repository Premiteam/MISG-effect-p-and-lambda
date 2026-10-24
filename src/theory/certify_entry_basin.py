from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import json
import interval_arithmetic as ia
import certify_transverse as ct

ROOT = Path(__file__).resolve().parent


def isum_upper(xs):
    ans = Decimal(0)
    for x in xs:
        ans = ia.HI.add(ans, x)
    return ans


def matrix_intervals(C):
    return [
        [
            (
                ct.rounded_fraction(a, ct.ROUND_FLOOR),
                ct.rounded_fraction(a, ct.ROUND_CEILING),
            )
            for a in row
        ]
        for row in C
    ]


def base_s(z):
    return (
        (z**-4 + z**-6 + z**-8)
        * sum(z ** (-2 * j) for j in range(5))
        / (z * z - F(19, 20))
    )


def chart_map(center, rad):
    A = ia.AD
    I = ia.I
    variables = []
    for j, val in enumerate(center + [F(0)] * 31):
        box = I(val) + I(-rad, rad)
        variables.append(A(box, [I(int(k == j)) for k in range(38)]))
    c = variables[:7]
    t = variables[7:26]
    w = variables[26:]
    z = (-c[3]).sqrt()
    rho = [c[0], c[1], c[2], c[3], c[4], z * z * (z - c[1]) + t[0], c[5], c[6]]
    h = []
    idx = 1
    for j in range(9):
        row = []
        for b in range(2):
            val = A(0) if b == 0 else z ** (-j)
            if not (j == 0 and b == 1):
                val += t[idx]
                idx += 1
            row.append(val)
        h.append(row)
    s = base_s(z) + t[18]
    phi = [h[j + 2] + h[j + 3] + h[j + 4] + [w[j + 1], w[j + 2]] for j in range(5)]
    den = F(19, 20) * s + sum(a * a for col in phi for a in col)
    err = [w[j + 7] - sum(a * b for a, b in zip(col, rho)) for j, col in enumerate(phi)]
    rr = [rho[b] + sum(col[b] * e for col, e in zip(phi, err)) / den for b in range(8)]
    nxt = [
        h[0][1] + rr[2] * h[2][0] + rr[3] * h[2][1] + rr[6] * w[0],
        rr[0] * h[0][0]
        + rr[1] * h[0][1]
        + rr[4] * h[2][0]
        + rr[5] * h[2][1]
        + rr[7] * w[0],
    ]
    g = nxt[1]
    hh = [[a / g for a in row] for row in [nxt] + h[:-1]]
    zz = (-rr[3]).sqrt()
    normal = [rr[5] - zz * zz * (zz - rr[1])]
    for j in range(9):
        for b in range(2):
            if not (j == 0 and b == 1):
                normal.append(hh[j][b] - (A(0) if b == 0 else zz ** (-j)))
    normal.append(den / (g * g) - base_s(zz))
    return [rr[i] for i in [0, 1, 2, 3, 4, 6, 7]], normal, g, den


def main():
    ia.precision(80)
    I = ia.I
    ref = json.loads((ROOT / "entry_reference_point.json").read_text())
    C, rho = ct.exact_matrix(ref["centers"], ref["z"])
    rho = [a.v for a in rho]
    center = [rho[i] for i in [0, 1, 2, 3, 4, 6, 7]]
    ci = matrix_intervals(C)
    dim = 19
    ident = [
        [(Decimal(int(i == j)), Decimal(int(i == j))) for j in range(dim)]
        for i in range(dim)
    ]
    powers = {0: ident, 1: ci}
    for p in [2, 4, 8, 16, 32, 64, 128]:
        powers[p] = ct.multiply(powers[p // 2], powers[p // 2])
    bound = ct.upper_norm(powers[128])
    assert bound < Decimal(".5")
    theta = F(199, 200)
    assert F(bound) < theta**128
    for j in range(2, 128):
        if j not in powers:
            bit = 1 << (j.bit_length() - 1)
            powers[j] = ct.multiply(powers[bit], powers[j - bit])
    M = isum_upper([(I(ct.upper_norm(powers[j])) / I(theta**j)).hi for j in range(128)])
    print("Power bound", bound, "adapted norm M", M, flush=True)
    radius = Decimal("1e-20")
    cout, tout, g, den = chart_map(center, radius)
    delta = max(
        isum_upper([(tout[i].g[j + 7] - I(C[i][j])).abs_upper() for j in range(19)])
        for i in range(19)
    )
    contraction = (I(theta) + I(M) * I(delta)).hi
    forcing_t = max(
        isum_upper([a.g[j].abs_upper() for j in range(26, 38)]) for a in tout
    )
    center_t = max(isum_upper([a.g[j].abs_upper() for j in range(7, 26)]) for a in cout)
    center_w = max(
        isum_upper([a.g[j].abs_upper() for j in range(26, 38)]) for a in cout
    )
    L = max(Decimal(1), (I(M) * I(forcing_t)).hi, center_t, center_w)
    kappa = F(399, 400)
    eta = F(201, 200)
    assert contraction < F(kappa.numerator, kappa.denominator)
    assert g.v.lo > I(eta).hi and den.v.lo > 0

    a_limit = min((I(radius) / 4).lo, ((1 - I(kappa)) * I(radius) / (4 * I(L))).lo)
    S_limit = min(
        radius,
        (I(radius) / (4 * I(L))).lo,
        (I(radius) / (4 * I(L) * (1 + I(L) / (1 - I(kappa))))).lo,
    )
    data = {
        "claim": "explicit sufficient local escape neighborhood for the stated exact reference point",
        "rho_rationals": list(map(str, rho)),
        "center_rationals": list(map(str, center)),
        "z_rational": ref["z"],
        "C_exact_rationals": [[str(a) for a in row] for row in C],
        "power_128_infinity_upper": str(bound),
        "theta": str(theta),
        "adapted_norm_equivalence_upper_M": str(M),
        "coordinate_box_radius": str(radius),
        "transverse_jacobian_difference_infinity_upper": str(delta),
        "transverse_contraction_upper": str(contraction),
        "chosen_kappa": str(kappa),
        "common_L_upper": str(L),
        "growth_interval": g.v.serial(),
        "denominator_interval": den.v.serial(),
        "eta": str(eta),
        "sufficient_initial_center_distance_strict_upper": str((I(radius) / 4).lo),
        "sufficient_initial_transverse_adapted_norm_strict_upper": str(a_limit),
        "sufficient_total_normalized_forcing_strict_upper": str(S_limit),
        "not_yet_claimed": "certified finite-prefix entry or satisfaction of the forcing budget",
    }
    out = ROOT / "quantitative_escape_basin.json"
    if out.exists():
        raise SystemExit("Refusing overwrite")
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(
        {
            k: data[k]
            for k in [
                "transverse_contraction_upper",
                "common_L_upper",
                "sufficient_initial_transverse_adapted_norm_strict_upper",
                "sufficient_total_normalized_forcing_strict_upper",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
