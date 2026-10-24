from pathlib import Path
from decimal import Decimal
from fractions import Fraction as F
import argparse, json, time, hashlib
import numpy as np
import run_audit as audit
import interval_arithmetic as ia
import certify_transverse as ct

ROOT = Path(__file__).resolve().parent


def controls(cut):
    frozen = ROOT / "entry_controls.npz"
    if frozen.exists() and cut <= 2000:
        with np.load(frozen, allow_pickle=False) as data:
            return data["u"][: cut + 1].copy(), data["y"][: cut + 1].copy()
    fam = np.load(audit.ARCHIVE, allow_pickle=False)
    rho = fam["rho_star_000319"]
    d = 2
    rg = np.random.default_rng(2026103790)
    u = np.r_[0.0, rg.normal(size=100000)][: cut + 1]
    noise = np.r_[0.0, rg.normal(scale=fam["sigma_v"][319], size=100000)][: cut + 1]
    x = np.zeros((cut + 4, 2))
    y = np.zeros(cut + 1)
    for k in range(1, cut + 1):
        y[k] = x[k, 0] + noise[k]
        xd = x[k - d] if k >= d else np.zeros(2)
        x[k + 1] = [
            x[k, 1] + rho[2] * xd[0] + rho[3] * xd[1] + rho[6] * u[k],
            rho[0] * x[k, 0]
            + rho[1] * x[k, 1]
            + rho[4] * xd[0]
            + rho[5] * xd[1]
            + rho[7] * u[k],
        ]
    return u, y


def floating(cut, N):
    u0, y0 = controls(cut)
    u = np.zeros(N + 1)
    y = u.copy()
    u[: cut + 1] = u0
    y[: cut + 1] = y0
    rho = np.full(8, 1e-6)
    r = 1.0
    h = np.full((N + 20, 2), 1e-6)
    ph = np.zeros((5, 8))
    off = 10
    maxnorm = 0.0
    for k in range(1, N + 1):
        it = k + off
        phi = np.r_[
            h[it - 2], h[it - 3], h[it - 4], u[k - 1], u[k - 2] if k >= 2 else 0
        ]
        ph = np.vstack([phi, ph[:-1]])
        r = 0.95 * r + np.sum(ph**2)
        Y = np.array([y[k - j] if k > j else 0 for j in range(5)])
        rho += ph.T @ (Y - ph @ rho) / r
        maxnorm = max(maxnorm, float(np.linalg.norm(rho)))
        h[it + 1] = [
            h[it, 1] + rho[2] * h[it - 2, 0] + rho[3] * h[it - 2, 1] + rho[6] * u[k],
            rho[0] * h[it, 0]
            + rho[1] * h[it, 1]
            + rho[4] * h[it - 2, 0]
            + rho[5] * h[it - 2, 1]
            + rho[7] * u[k],
        ]
    q = h[N + off + 1, 1]
    z = np.sqrt(-rho[3]) if rho[3] < 0 else np.nan
    return {
        "cut": cut,
        "N": N,
        "q": float(q),
        "z": float(z),
        "rho": rho.tolist(),
        "max_parameter_norm": maxnorm,
        "constraint": float(rho[5] - z * z * (z - rho[1])),
    }


def spectral_reset_certificate(rho):
    from decimal import localcontext, ROUND_FLOOR, ROUND_CEILING

    a = []
    for x in rho:
        with localcontext() as ctx:
            ctx.prec = 70
            ctx.rounding = ROUND_FLOOR
            lo = +x.lo
        with localcontext() as ctx:
            ctx.prec = 70
            ctx.rounding = ROUND_CEILING
            hi = +x.hi
        a.append(ia.I(lo, hi))
    f = [[ia.I(0) for _ in range(6)] for _ in range(6)]
    f[0][1] = ia.I(1)
    f[1][0] = a[0]
    f[1][1] = a[1]
    f[0][4] = a[2]
    f[0][5] = a[3]
    f[1][4] = a[4]
    f[1][5] = a[5]
    for j in range(4):
        f[j + 2][j] = ia.I(1)
    mat = [
        [
            (
                ct.rounded_fraction(F(x.lo) / F(6, 5), ct.ROUND_FLOOR),
                ct.rounded_fraction(F(x.hi) / F(6, 5), ct.ROUND_CEILING),
            )
            for x in row
        ]
        for row in f
    ]
    power = 1
    while power <= 128:
        bound = ct.upper_norm(mat)
        if bound < 1:
            return {"power": power, "scaled_power_norm_upper": str(bound)}
        mat = ct.multiply(mat, mat)
        power *= 2
    return None


def certified(cut, N, digits, out, check_stops=False):
    ia.precision(digits)
    I = ia.I
    u0, y0 = controls(cut)
    u = [I(0)] * (N + 1)
    y = [I(0)] * (N + 1)
    for k in range(1, cut + 1):
        u[k] = I(Decimal.from_float(float(u0[k])))
        y[k] = I(Decimal.from_float(float(y0[k])))
    rho = [I("0.000001") for _ in range(8)]
    r = I(1)
    lam = I("0.95")
    h = [[I("0.000001"), I("0.000001")] for _ in range(9)]
    ph = [[I(0)] * 8 for _ in range(5)]
    start = time.monotonic()
    snap = []
    max_parameter_l1 = Decimal(0)
    resets = []
    last_reset = 0
    for k in range(1, N + 1):
        phi = h[2] + h[3] + h[4] + [u[k - 1], u[k - 2] if k >= 2 else I(0)]
        ph = [phi] + ph[:-1]
        r = lam * r + sum(a**2 for col in ph for a in col)
        errs = [
            (y[k - j] if k > j else I(0)) - sum(a * b for a, b in zip(col, rho))
            for j, col in enumerate(ph)
        ]
        inv = r.reciprocal()
        rho = [
            rho[b] + sum(ph[j][b] * errs[j] for j in range(5)) * inv for b in range(8)
        ]
        upper = Decimal(0)
        for a in rho:
            upper = ia.HI.add(upper, a.abs_upper())
        max_parameter_l1 = max(max_parameter_l1, upper)
        if check_stops:
            cert = spectral_reset_certificate(rho)
            if cert is not None:
                resets.append({"k": k, **cert})
                last_reset = k
            elif k - last_reset >= 50:
                raise ArithmeticError(
                    f"No certified spectral reset within 50 steps at k={k}"
                )
        nxt = [
            h[0][1] + rho[2] * h[2][0] + rho[3] * h[2][1] + rho[6] * u[k],
            rho[0] * h[0][0]
            + rho[1] * h[0][1]
            + rho[4] * h[2][0]
            + rho[5] * h[2][1]
            + rho[7] * u[k],
        ]
        h = [nxt] + h[:-1]
        if k % 250 == 0 or k == N:
            rel = (
                (h[0][1].width() / h[0][1].abs_upper())
                if h[0][1].abs_upper()
                else Decimal(0)
            )
            print(
                "k",
                k,
                "relative q enclosure width",
                f"{rel:.3E}",
                "elapsed",
                round(time.monotonic() - start, 1),
                flush=True,
            )
            snap.append(
                {
                    "k": k,
                    "q_mid_short": f"{h[0][1].mid():.12E}",
                    "q_width_relative_diagnostic": str(rel),
                }
            )
        if max(a.width() for a in rho) > Decimal("1e-6"):
            raise ArithmeticError(f"Wrapping too large at k={k}; increase precision.")
    q = h[0][1]
    if q.lo <= 0 <= q.hi:
        raise ArithmeticError("Nonzero amplitude not certified.")
    hn = [[a / q for a in row] for row in h]
    sn = r / q**2
    data = {
        "kind": "outward-rounded exact-control endpoint enclosure; does not itself certify basin entry",
        "steps": N,
        "zero_controls_after": cut,
        "precision": digits,
        "time_at_endpoint": "before update k=N+1",
        "rho": [a.serial() for a in rho],
        "h_normalized": [[a.serial() for a in row] for row in hn],
        "s_normalized": sn.serial(),
        "q": q.serial(),
        "snapshots": snap,
        "max_parameter_l1_upper": str(max_parameter_l1),
        "spectral_reset_certificates": resets,
        "last_reset_step": last_reset,
        "control_generation": "controls(cut) in this script; each nonzero float converted to its exact Decimal value",
        "controls_sha256": hashlib.sha256(
            np.stack([u0, y0]).astype("<f8").tobytes()
        ).hexdigest(),
        "not_certified": (
            "nonlinear basin membership"
            if check_stops
            else "original spectral stopping rule on the finite prefix; nonlinear basin membership"
        ),
    }
    out.write_text(json.dumps(data, indent=2) + "\n")
    print("Saved", str(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--precision", type=int, default=4000)
    ap.add_argument("--float-only", action="store_true")
    ap.add_argument("--check-stops", action="store_true")
    ap.add_argument("--output", default="entry_prefix_certificate.json")
    a = ap.parse_args()
    if a.float_only:
        print(json.dumps(floating(a.cut, a.steps), indent=2))
    else:
        out = ROOT / a.output
        if out.exists():
            raise SystemExit("Refusing to overwrite output.")
        certified(a.cut, a.steps, a.precision, out, a.check_stops)
