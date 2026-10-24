import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
D = 2
P = 5
LAM = 0.95
GAMMA = 1.0
DEPTH = D + P + 1
MASK = np.arange(2 * (DEPTH + 1)) != 1


def pack(rho, h, s):
    return np.r_[rho, h.ravel()[MASK], s]


def unpack(w):
    h = np.zeros(2 * (DEPTH + 1), dtype=w.dtype)
    h[1] = 1
    h[MASK] = w[8:-1]
    return w[:8], h.reshape(-1, 2), w[-1]


def step(w, return_growth=False):
    rho, h, s = unpack(w)
    phi = np.column_stack(
        [np.r_[h[j + 2], h[j + D + 1], h[j + D + 2], 0.0, 0.0] for j in range(P)]
    )
    energy = np.sum(phi * phi)
    new_r = LAM * s + energy
    rr = rho - GAMMA * phi @ (phi.T @ rho) / new_r
    new = np.array(
        [
            h[0, 1] + rr[2] * h[D, 0] + rr[3] * h[D, 1],
            rr[0] * h[0, 0] + rr[1] * h[0, 1] + rr[4] * h[D, 0] + rr[5] * h[D, 1],
        ]
    )
    z = new[1]
    out = pack(rr, np.vstack([new, h[:-1]]) / z, new_r / (z * z))
    return (out, z) if return_growth else out


def equilibrium(rho, z):
    rr = np.array(rho, dtype=float).copy()
    rr[3] = -(z**D)
    rr[5] = z**D * (z - rr[1])
    h = np.column_stack(
        [np.zeros(DEPTH + 1), z ** (-np.arange(DEPTH + 1, dtype=float))]
    )
    v = np.array([0.0, z**-2, 0.0, z ** (-D - 1), 0.0, z ** (-D - 2), 0.0, 0.0])
    e = float(v @ v) * sum(z ** (-2 * j) for j in range(P))
    return pack(rr, h, e / (z * z - LAM))


def jacobian(w):
    h = 1e-25
    return np.column_stack(
        [
            np.imag(step(w.astype(complex) + 1j * h * np.eye(len(w))[j])) / h
            for j in range(len(w))
        ]
    )


def main():
    d = json.loads((ROOT / "candidate_high_precision.json").read_text())[-1][
        "snapshots"
    ][-1]
    rho = np.array(list(map(float, d["rho"])))
    z = float(d["z_from_b12"])
    w = equilibrium(rho, z)
    J = jacobian(w)
    eigen = np.linalg.eigvals(J)
    neutral = np.abs(eigen - 1) < 1e-7
    transverse = eigen[~neutral]

    eps = 1e-5
    fd = np.column_stack(
        [
            (step(w + eps * np.eye(len(w))[j]) - step(w - eps * np.eye(len(w))[j]))
            / (2 * eps)
            for j in range(len(w))
        ]
    )
    rows = []
    rng = np.random.default_rng(630106)
    for radius in [1e-6, 1e-4, 1e-3, 1e-2]:
        for rep in range(10):
            dw = rng.normal(size=len(w))
            dw *= radius / np.linalg.norm(dw)
            v = w + dw
            mingrowth = 10.0
            maxnorm = 0.0
            for k in range(1000):
                v, g = step(v, True)
                mingrowth = min(mingrowth, float(g))
                maxnorm = max(maxnorm, float(np.linalg.norm(v - w)))
            rows.append(
                {
                    "perturbation_radius": radius,
                    "rep": rep,
                    "minimum_growth": mingrowth,
                    "max_distance_to_original_equilibrium": maxnorm,
                    "final_growth": float(g),
                    "final_fixed_point_residual": float(np.linalg.norm(step(v) - v)),
                }
            )
    data = {
        "scope": "local numerical diagnostic; eigenvalue estimates are not interval-certified",
        "dimension": len(w),
        "equilibrium_growth": z,
        "equilibrium_residual": float(np.linalg.norm(step(w) - w)),
        "complex_vs_finite_difference_max": float(np.max(abs(J - fd))),
        "neutral_eigenvalue_count": int(sum(neutral)),
        "max_transverse_modulus": float(max(abs(transverse))),
        "eigenvalues": [[float(a.real), float(a.imag)] for a in eigen],
        "perturbation_tests": rows,
    }
    dest = ROOT / "normalized_escape_diagnostics.json"
    if dest.exists():
        raise SystemExit("Output exists; do not overwrite.")
    dest.write_text(json.dumps(data, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in data.items()
            if k not in ["perturbation_tests", "eigenvalues"]
        }
    )
    print("min growth across tests", min(r["minimum_growth"] for r in rows))


if __name__ == "__main__":
    main()
