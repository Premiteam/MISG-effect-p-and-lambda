from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import normalized_escape as ne
import certify_transverse as ct

ROOT = Path(__file__).resolve().parent
CENTERS = [0, 1, 2, 3, 4, 6, 7]


def chart(w):
    rho, h, s = ne.unpack(w)
    z = np.sqrt(-rho[3])

    hstar = np.column_stack([np.zeros(9, dtype=w.dtype), z ** (-np.arange(9))])
    estar = (z**-4 + z**-6 + z**-8) * sum(z ** (-2 * j) for j in range(5))
    transverse = np.r_[
        rho[5] - z * z * (z - rho[1]),
        (h - hstar).ravel()[ne.MASK],
        s - estar / (z * z - 0.95),
    ]
    return np.r_[rho[CENTERS], transverse]


def invchart(v):
    rho = np.zeros(8, dtype=v.dtype)
    rho[CENTERS] = v[:7]
    z = np.sqrt(-rho[3])
    rho[5] = z * z * (z - rho[1]) + v[7]
    h = np.column_stack([np.zeros(9, dtype=v.dtype), z ** (-np.arange(9))])
    hf = h.ravel()
    hf[ne.MASK] += v[8:-1]
    estar = (z**-4 + z**-6 + z**-8) * sum(z ** (-2 * j) for j in range(5))
    return ne.pack(rho, hf.reshape(9, 2), estar / (z * z - 0.95) + v[-1])


def main():
    C, rhof = ct.exact_matrix()
    rho = np.array([float(x.v) for x in rhof])
    w = ne.equilibrium(rho, 509 / 500)
    v = chart(w)
    eps = 1e-25
    J = np.column_stack(
        [
            np.imag(
                chart(ne.step(invchart(v.astype(complex) + 1j * eps * np.eye(26)[i])))
            )
            / eps
            for i in range(26)
        ]
    )
    cf = np.array([[float(x) for x in row] for row in C])
    diff = float(np.max(np.abs(J[7:, 7:] - cf)))
    center_error = float(np.max(np.abs(J[:7, :7] - np.eye(7))))
    lower_left = float(np.max(np.abs(J[7:, :7])))
    assert diff < 1e-10 and center_error < 1e-10 and lower_left < 1e-10

    r = [x.v for x in rhof]
    f = [[F(0) for _ in range(6)] for _ in range(6)]
    f[0][1] = F(1)
    f[1][0] = r[0]
    f[1][1] = r[1]
    f[0][4] = r[2]
    f[0][5] = r[3]
    f[1][4] = r[4]
    f[1][5] = r[5]
    for i in range(4):
        f[i + 2][i] = F(1)
    a = [
        [
            (
                ct.rounded_fraction(x / F(6, 5), ct.ROUND_FLOOR),
                ct.rounded_fraction(x / F(6, 5), ct.ROUND_CEILING),
            )
            for x in row
        ]
        for row in f
    ]
    power = 1
    bounds = []
    while power <= 256:
        bound = ct.upper_norm(a)
        bounds.append({"power": power, "upper": str(bound)})
        if bound < 1:
            break
        a = ct.multiply(a, a)
        power *= 2
    assert bound < 1

    result = {
        "chart_roundtrip_max_error": float(max(abs(invchart(v) - w))),
        "transverse_exact_vs_complex_step_max_error": diff,
        "center_identity_max_error": center_error,
        "transverse_from_center_max_error": lower_left,
        "observer_scaled_matrix_divisor": "6/5",
        "observer_scaled_power_interval_bounds": bounds,
        "observer_spectral_radius_floating": float(
            max(abs(np.linalg.eigvals(np.array(f, dtype=float))))
        ),
        "scope": "Independent numerical derivative QA plus rigorous rho(Fhat)<1.2 matrix-power certificate at one equilibrium; no reachability proof.",
    }
    out = ROOT / "escape_certificate_crosscheck.json"
    if out.exists():
        raise SystemExit("Refusing to overwrite output.")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
