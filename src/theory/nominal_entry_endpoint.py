import json
from decimal import Decimal, localcontext
from pathlib import Path
from construct_entry_prefix import controls

ROOT = Path(__file__).resolve().parent


def main():
    cut = 2000
    N = 4000
    u0, y0 = controls(cut)
    with localcontext() as ctx:
        ctx.prec = 160
        Z = Decimal(0)
        u = [Z] * (N + 1)
        y = [Z] * (N + 1)
        for k in range(1, cut + 1):
            u[k] = Decimal.from_float(float(u0[k]))
            y[k] = Decimal.from_float(float(y0[k]))
        rho = [Decimal("1e-6")] * 8
        r = Decimal(1)
        h = [[Decimal("1e-6")] * 2 for _ in range(9)]
        ph = [[Z] * 8 for _ in range(5)]
        for k in range(1, N + 1):
            ph = [h[2] + h[3] + h[4] + [u[k - 1], u[k - 2] if k >= 2 else Z]] + ph[:-1]
            r = Decimal(".95") * r + sum(a * a for col in ph for a in col)
            errs = [
                (y[k - j] if k > j else Z) - sum(a * b for a, b in zip(col, rho))
                for j, col in enumerate(ph)
            ]
            rho = [
                rho[b] + sum(ph[j][b] * errs[j] for j in range(5)) / r for b in range(8)
            ]
            nxt = [
                h[0][1] + rho[2] * h[2][0] + rho[3] * h[2][1] + rho[6] * u[k],
                rho[0] * h[0][0]
                + rho[1] * h[0][1]
                + rho[4] * h[2][0]
                + rho[5] * h[2][1]
                + rho[7] * u[k],
            ]
            h = [nxt] + h[:-1]
        z = (-rho[3]).sqrt()
        q = h[0][1]

        center = [format(rho[i], ".65f") for i in [0, 1, 2, 4, 6, 7]]
        zr = format(z, ".65f")
        data = {
            "scope": "nominal reference-point selection; exact iterates not certified here",
            "centers": center,
            "z": zr,
            "q_nominal": str(q),
            "rho_nominal": list(map(str, rho)),
            "normalized_h_nominal": [[str(a / q) for a in row] for row in h],
            "normalized_s_nominal": str(r / q**2),
        }
    out = ROOT / "entry_reference_point.json"
    if out.exists():
        raise SystemExit("Refusing overwrite")
    out.write_text(json.dumps(data, indent=2) + "\n")
    print("z", zr, "q", format(q, ".8E"))


if __name__ == "__main__":
    main()
