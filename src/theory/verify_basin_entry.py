from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import json
import numpy as np
import interval_arithmetic as ia
import certify_transverse as ct
from construct_entry_prefix import controls
from certify_entry_basin import matrix_intervals, base_s
import run_audit as audit

ROOT = Path(__file__).resolve().parent


def frombounds(b):
    return ia.I(Decimal(b[0]), Decimal(b[1]))


def norm_power_bound(mat):
    n = len(mat)
    powers = {
        0: [
            [(Decimal(int(i == j)), Decimal(int(i == j))) for j in range(n)]
            for i in range(n)
        ],
        1: matrix_intervals(mat),
    }
    m = 1
    while ct.upper_norm(powers[m]) >= 1:
        powers[2 * m] = ct.multiply(powers[m], powers[m])
        m *= 2
        if m > 512:
            raise ArithmeticError("No power bound")
    for j in range(2, m):
        if j not in powers:
            bit = 1 << (j.bit_length() - 1)
            powers[j] = ct.multiply(powers[bit], powers[j - bit])
    return m, ct.upper_norm(powers[m]), max(ct.upper_norm(powers[j]) for j in range(m))


def main():
    ia.precision(100)
    I = ia.I
    ep = json.loads((ROOT / "entry_prefix_certificate.json").read_text())
    bas = json.loads((ROOT / "quantitative_escape_basin.json").read_text())
    rho = list(map(frombounds, ep["rho"]))
    h = [[frombounds(a) for a in row] for row in ep["h_normalized"]]
    s = frombounds(ep["s_normalized"])
    q = frombounds(ep["q"])
    z = (-rho[3]).sqrt()
    t = [rho[5] - z * z * (z - rho[1])]
    for j in range(9):
        for b in range(2):
            if not (j == 0 and b == 1):
                t.append(h[j][b] - (I(0) if b == 0 else z ** (-j)))
    t.append(s - base_s(z))
    c = [rho[i] for i in [0, 1, 2, 3, 4, 6, 7]]
    cd = max((a - I(F(b))).abs_upper() for a, b in zip(c, bas["center_rationals"]))
    tinf = max(a.abs_upper() for a in t)
    astar = (I(bas["adapted_norm_equivalence_upper_M"]) * I(tinf)).hi
    assert cd < Decimal(bas["sufficient_initial_center_distance_strict_upper"])
    assert astar < Decimal(
        bas["sufficient_initial_transverse_adapted_norm_strict_upper"]
    )
    fam = np.load(audit.ARCHIVE, allow_pickle=False)
    true = [F(float(x)) for x in fam["rho_star_000319"]]
    f = [[F(0) for _ in range(6)] for _ in range(6)]
    f[0][1] = F(1)
    f[1][0] = true[0]
    f[1][1] = true[1]
    f[0][4] = true[2]
    f[0][5] = true[3]
    f[1][4] = true[4]
    f[1][5] = true[5]
    for j in range(4):
        f[j + 2][j] = F(1)
    decay = F(24, 25)
    m, power, MM = norm_power_bound([[a / decay for a in row] for row in f])

    u, y = controls(ep["zero_controls_after"])
    umax = max(F(float(abs(a))) for a in u)
    gnorm = max(abs(true[6]), abs(true[7]))
    zero_steps = ep["steps"] - ep["zero_controls_after"]
    true_end = (
        I(MM) ** 2 * I(decay) ** zero_steps * I(gnorm) * I(umax) / (1 - I(decay))
    ).hi
    alpha = F(1001, 1000)
    eta = F(bas["eta"])
    B = F(1, 10**100)
    K = (
        I(MM) * I(true_end) + I(MM) * I(gnorm) * I(B) / (I(alpha) - I(decay)) + I(B)
    ).hi
    assert q.hi < 0
    absq_lower = q.hi.copy_negate()
    S = (I(K) / (I(absq_lower) * (1 - I(alpha) / I(eta)))).hi
    assert S < Decimal(bas["sufficient_total_normalized_forcing_strict_upper"])
    out = {
        "claim": "finite exact-control endpoint lies strictly inside the quantitative escape neighborhood, with sufficient future forcing budget",
        "scope": "real-arithmetic mathematical MISG recursion; finite-prefix operational spectral stopping still checked separately",
        "center_distance_upper": str(cd),
        "transverse_infinity_upper": str(tinf),
        "transverse_adapted_norm_upper": str(astar),
        "amplitude_sign": "negative; normalized h_0,2 remains 1; growth preserves sign; bounds use |q|",
        "amplitude_absolute_lower": str(absq_lower),
        "true_matrix_decay_rate": str(decay),
        "true_scaled_power": m,
        "true_scaled_power_norm_upper": str(power),
        "true_matrix_prefactor_upper": str(MM),
        "true_endpoint_norm_upper": str(true_end),
        "future_envelope_B": str(B),
        "future_envelope_alpha": str(alpha),
        "output_envelope_K_upper": str(K),
        "total_normalized_future_forcing_upper": str(S),
        "allowed_forcing_budget": bas[
            "sufficient_total_normalized_forcing_strict_upper"
        ],
        "gaussian_event": "For every n>=0, |u_(4001+n)| and |v_(4001+n)| <= B*alpha^n. Independent Gaussian tails give strictly positive probability; no numerical probability estimate claimed.",
    }
    dest = ROOT / "basin_entry_verified.json"
    if dest.exists():
        raise SystemExit("Refusing overwrite")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    for key in [
        "center_distance_upper",
        "transverse_adapted_norm_upper",
        "true_endpoint_norm_upper",
        "total_normalized_future_forcing_upper",
        "allowed_forcing_budget",
    ]:
        print(key, out[key], flush=True)


if __name__ == "__main__":
    main()
