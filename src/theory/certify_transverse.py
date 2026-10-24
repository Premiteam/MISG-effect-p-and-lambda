from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from math import isqrt
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DIM = 19


class Dual:
    def __init__(self, v, g=None):
        self.v = F(v)
        self.g = tuple(g) if g is not None else (F(0),) * DIM

    @staticmethod
    def cast(x):
        return x if isinstance(x, Dual) else Dual(x)

    def __add__(self, x):
        x = self.cast(x)
        return Dual(self.v + x.v, [a + b for a, b in zip(self.g, x.g)])

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.g])

    def __sub__(self, x):
        return self + -self.cast(x)

    def __rsub__(self, x):
        return self.cast(x) + -self

    def __mul__(self, x):
        x = self.cast(x)
        return Dual(self.v * x.v, [a * x.v + self.v * b for a, b in zip(self.g, x.g)])

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self.cast(x)
        return Dual(
            self.v / x.v,
            [(a * x.v - self.v * b) / (x.v * x.v) for a, b in zip(self.g, x.g)],
        )

    def __rtruediv__(self, x):
        return self.cast(x) / self

    def __pow__(self, n):
        if n < 0:
            return Dual(1) / (self ** (-n))
        return (
            Dual(self.v**n, [n * self.v ** (n - 1) * a for a in self.g])
            if n
            else Dual(1)
        )

    def sqrt(self):
        a = isqrt(self.v.numerator)
        b = isqrt(self.v.denominator)
        assert a * a == self.v.numerator and b * b == self.v.denominator
        s = F(a, b)
        return Dual(s, [a / (2 * s) for a in self.g])


def base_s(z):
    return (
        (z**-4 + z**-6 + z**-8)
        * sum(z ** (-2 * j) for j in range(5))
        / (z * z - F(19, 20))
    )


def exact_matrix(centers=None, zvalue=None):
    z = Dual(F(509, 500) if zvalue is None else F(zvalue))
    cc = (
        [
            F(2717, 5000),
            F(3051, 5000),
            F(-6513, 10000),
            F(323, 2500),
            F(-15091, 10000),
            F(4359, 2500),
        ]
        if centers is None
        else list(map(F, centers))
    )
    a1, a2, b11, b21, f1, f2 = map(Dual, cc)
    rho = [a1, a2, b11, -z * z, b21, z * z * (z - a2), f1, f2]
    variables = [Dual(0, [F(int(i == j)) for j in range(DIM)]) for i in range(DIM)]
    rho[5] = rho[5] + variables[0]
    h = []
    idx = 1
    for j in range(9):
        row = []
        for b in range(2):
            val = Dual(0) if b == 0 else z ** (-j)
            if not (j == 0 and b == 1):
                val += variables[idx]
                idx += 1
            row.append(val)
        h.append(row)
    assert idx == 18
    s = base_s(z) + variables[18]
    phi = [h[j + 2] + h[j + 3] + h[j + 4] + [Dual(0), Dual(0)] for j in range(5)]
    denom = F(19, 20) * s + sum(a * a for col in phi for a in col)
    predicted = [sum(a * b for a, b in zip(col, rho)) for col in phi]
    rr = [
        rho[b] - sum(col[b] * err for col, err in zip(phi, predicted)) / denom
        for b in range(8)
    ]
    nxt = [
        h[0][1] + rr[2] * h[2][0] + rr[3] * h[2][1],
        rr[0] * h[0][0] + rr[1] * h[0][1] + rr[4] * h[2][0] + rr[5] * h[2][1],
    ]
    growth = nxt[1]
    hh = [[a / growth for a in row] for row in [nxt] + h[:-1]]
    zz = (-rr[3]).sqrt()
    normal = [rr[5] - zz * zz * (zz - rr[1])]
    for j in range(9):
        for b in range(2):
            if not (j == 0 and b == 1):
                normal.append(hh[j][b] - (Dual(0) if b == 0 else zz ** (-j)))
    normal.append(denom / (growth * growth) - base_s(zz))
    assert len(normal) == DIM and all(x.v == 0 for x in normal)
    assert growth.v == z.v
    return [[v for v in row.g] for row in normal], rho


def rounded_fraction(a, rounding):
    with localcontext() as ctx:
        ctx.prec = 70
        ctx.rounding = rounding
        return Decimal(a.numerator) / Decimal(a.denominator)


def multiply(A, B):
    n = len(A)
    C = [[(Decimal(0), Decimal(0)) for _ in range(n)] for _ in range(n)]
    with localcontext() as lo:
        lo.prec = 70
        lo.rounding = ROUND_FLOOR
        for i in range(n):
            for j in range(n):
                lower = Decimal(0)
                for k in range(n):
                    a, b = A[i][k]
                    c, d = B[k][j]
                    lower += min(a * c, a * d, b * c, b * d)
                C[i][j] = (lower, Decimal(0))
    with localcontext() as hi:
        hi.prec = 70
        hi.rounding = ROUND_CEILING
        for i in range(n):
            for j in range(n):
                upper = Decimal(0)
                for k in range(n):
                    a, b = A[i][k]
                    c, d = B[k][j]
                    upper += max(a * c, a * d, b * c, b * d)
                C[i][j] = (C[i][j][0], upper)
    return C


def upper_norm(A):
    with localcontext() as ctx:
        ctx.prec = 70
        ctx.rounding = ROUND_CEILING

        return max(sum(max(a.copy_abs(), b.copy_abs()) for a, b in row) for row in A)


def main():
    C, rho = exact_matrix()
    cf = np.array([[float(x) for x in row] for row in C])
    interval = [
        [
            (rounded_fraction(x, ROUND_FLOOR), rounded_fraction(x, ROUND_CEILING))
            for x in row
        ]
        for row in C
    ]
    norms = []
    power = 1
    while power <= 256:
        bound = upper_norm(interval)
        norms.append({"power": power, "certified_infinity_norm_upper": str(bound)})
        print(power, str(bound), flush=True)
        if bound < 1:
            break
        interval = multiply(interval, interval)
        power *= 2
    assert bound < 1, "No contraction certified."

    result = {
        "claim": "transverse Jacobian at the stated rational equilibrium is strictly Schur stable",
        "not_claimed": "reachability from fixed initialization; nonlinear escape radius; global mean square instability",
        "dimension": DIM,
        "p": 5,
        "d": 2,
        "lambda": "19/20",
        "gamma": "1",
        "growth": "509/500",
        "parameter_rationals": [str(a.v) for a in rho],
        "exact_jacobian_rationals": [[str(x) for x in row] for row in C],
        "interval_precision_decimal_digits": 70,
        "certified_norms": norms,
        "floating_eigenvalue_modulus_for_context_only": float(
            max(abs(np.linalg.eigvals(cf)))
        ),
    }
    dest = ROOT / "transverse_interval_certificate.json"
    if dest.exists():
        raise SystemExit("Refusing to overwrite certificate.")
    dest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
