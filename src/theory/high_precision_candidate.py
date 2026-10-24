from decimal import Decimal, localcontext
import json, time
import numpy as np
import run_audit as audit


def run(precision, N=6000, original_j=None):
    fam = np.load(audit.ARCHIVE, allow_pickle=False)
    i = 319
    d = 2
    p = 5
    seed = (
        2026103790
        if original_j is None
        else np.random.SeedSequence([20260729, 1, int(fam["m"][i]), original_j])
    )
    draw_length = 100000 if original_j is None else 12000
    rg = np.random.default_rng(seed)
    uf = np.r_[0.0, rg.normal(size=draw_length)][: N + 1]
    vf = np.r_[0.0, rg.normal(scale=fam["sigma_v"][i], size=draw_length)][: N + 1]
    D = Decimal.from_float
    with localcontext() as ctx:
        ctx.prec = precision
        zero = Decimal(0)
        one = Decimal(1)
        u = [D(float(t)) for t in uf]
        v = [D(float(t)) for t in vf]
        true = [D(float(t)) for t in fam[f"rho_star_{i:06d}"]]

        rho = [Decimal("0.000001")] * 8
        lam = Decimal("0.95")
        r = one
        off = d + 4
        x = [[zero, zero] for _ in range(N + off + 3)]
        h = [[zero, zero] for _ in range(N + off + 3)]
        for j in range(off + 2):
            h[j] = [Decimal("0.000001")] * 2
        ph = [[zero] * 8 for _ in range(p)]
        y = [zero] * (N + 1)
        snap = []
        for k in range(1, N + 1):
            it = k + off
            xd = x[it - d]
            xh = h[it]
            hd = h[it - d]
            y[k] = x[it][0] + v[k]
            x[it + 1] = [
                x[it][1] + true[2] * xd[0] + true[3] * xd[1] + true[6] * u[k],
                true[0] * x[it][0]
                + true[1] * x[it][1]
                + true[4] * xd[0]
                + true[5] * xd[1]
                + true[7] * u[k],
            ]
            phi = (
                h[it - 2]
                + h[it - d - 1]
                + h[it - d - 2]
                + [u[k - 1] if k > 1 else zero, u[k - 2] if k > 2 else zero]
            )
            ph = [phi] + ph[:-1]
            energy = sum(t * t for col in ph for t in col)
            r = lam * r + energy
            errs = [
                (y[k - j] if k - j > 0 else zero) - sum(a * b for a, b in zip(col, rho))
                for j, col in enumerate(ph)
            ]
            new = [
                rho[b] + sum(ph[j][b] * errs[j] for j in range(p)) / r for b in range(8)
            ]
            increment = max(abs(a - b) for a, b in zip(new, rho))
            rho = new
            h[it + 1] = [
                xh[1] + rho[2] * hd[0] + rho[3] * hd[1] + rho[6] * u[k],
                rho[0] * xh[0]
                + rho[1] * xh[1]
                + rho[4] * hd[0]
                + rho[5] * hd[1]
                + rho[7] * u[k],
            ]
            if k in [1000, 2000, 3000, 4000, 5000, 6000, 9000, 12000]:
                z = (-rho[3]).sqrt() if rho[3] < 0 else zero
                constraint = rho[5] - z * z * (z - rho[1])
                snap.append(
                    {
                        "k": k,
                        "state_norm": str(sum(t * t for t in xh).sqrt()),
                        "state_first": str(xh[0]),
                        "state_second": str(xh[1]),
                        "rho": list(map(str, rho)),
                        "max_parameter_increment": str(increment),
                        "z_from_b12": str(z),
                        "hidden_mode_constraint_residual": str(constraint),
                    }
                )
        return {
            "precision": precision,
            "steps": N,
            "original_trajectory_index": original_j,
            "snapshots": snap,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--original-tests", action="store_true")
    args = parser.parse_args()
    dest = audit.ROOT / (
        "original_tests_high_precision.json"
        if args.original_tests
        else "candidate_high_precision.json"
    )
    if dest.exists():
        raise SystemExit("Refusing to overwrite previous check.")
    out = []
    cases = (
        [(digits, 12000, j) for j in [3, 4] for digits in [160, 240]]
        if args.original_tests
        else [(digits, 6000, None) for digits in [80, 120]]
    )
    for digits, N, j in cases:
        start = time.monotonic()
        out.append(run(digits, N, j))
        print(
            digits,
            "digits; trajectory",
            j,
            "completed in",
            round(time.monotonic() - start, 2),
            "seconds",
            flush=True,
        )
    dest.write_text(json.dumps(out, indent=2) + "\n")
    for res in out:
        print(
            res["precision"],
            [
                (
                    s["k"],
                    s["state_norm"][:15],
                    s["max_parameter_increment"][:18],
                    s["hidden_mode_constraint_residual"][:18],
                )
                for s in res["snapshots"]
            ],
        )
