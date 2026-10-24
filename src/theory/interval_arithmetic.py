from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction


def precision(digits):
    global LO, HI, NEAR
    LO = Context(prec=digits, rounding=ROUND_FLOOR, Emax=999999999, Emin=-999999999)
    HI = Context(prec=digits, rounding=ROUND_CEILING, Emax=999999999, Emin=-999999999)
    NEAR = Context(
        prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999
    )


precision(80)


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if isinstance(lo, I):
            self.lo, self.hi = lo.lo, lo.hi
            return
        if isinstance(lo, Fraction):
            assert hi is None
            self.lo = LO.divide(Decimal(lo.numerator), Decimal(lo.denominator))
            self.hi = HI.divide(Decimal(lo.numerator), Decimal(lo.denominator))
            return
        self.lo = lo if isinstance(lo, Decimal) else Decimal(lo)
        self.hi = (
            self.lo if hi is None else (hi if isinstance(hi, Decimal) else Decimal(hi))
        )
        assert self.lo <= self.hi

    @staticmethod
    def cast(x):
        return x if isinstance(x, I) else I(x)

    def __add__(self, x):
        x = I.cast(x)
        return I(LO.add(self.lo, x.lo), HI.add(self.hi, x.hi))

    __radd__ = __add__

    def __neg__(self):
        return I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, x):
        return self + -I.cast(x)

    def __rsub__(self, x):
        return I.cast(x) + -self

    def __mul__(self, x):
        x = I.cast(x)
        if self.lo >= 0:
            if x.lo >= 0:
                return I(LO.multiply(self.lo, x.lo), HI.multiply(self.hi, x.hi))
            if x.hi <= 0:
                return I(LO.multiply(self.hi, x.lo), HI.multiply(self.lo, x.hi))
        elif self.hi <= 0:
            if x.lo >= 0:
                return I(LO.multiply(self.lo, x.hi), HI.multiply(self.hi, x.lo))
            if x.hi <= 0:
                return I(LO.multiply(self.hi, x.hi), HI.multiply(self.lo, x.lo))
        return I(
            min(LO.multiply(a, b) for a in (self.lo, self.hi) for b in (x.lo, x.hi)),
            max(HI.multiply(a, b) for a in (self.lo, self.hi) for b in (x.lo, x.hi)),
        )

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ArithmeticError("Interval denominator contains zero")
        return I(LO.divide(Decimal(1), self.hi), HI.divide(Decimal(1), self.lo))

    def __truediv__(self, x):
        return self * I.cast(x).reciprocal()

    def __rtruediv__(self, x):
        return I.cast(x) * self.reciprocal()

    def __pow__(self, n):
        if n < 0:
            return (self ** (-n)).reciprocal()
        if n == 0:
            return I(1)
        if n == 1:
            return self
        if n == 2:
            if self.lo <= 0 <= self.hi:
                return I(
                    0, max(HI.multiply(self.lo, self.lo), HI.multiply(self.hi, self.hi))
                )
            return I(
                min(LO.multiply(self.lo, self.lo), LO.multiply(self.hi, self.hi)),
                max(HI.multiply(self.lo, self.lo), HI.multiply(self.hi, self.hi)),
            )
        a = self ** (n // 2)
        b = a**2
        return b * self if n % 2 else b

    def sqrt(self):
        assert self.lo >= 0

        a = NEAR.sqrt(self.lo)
        b = NEAR.sqrt(self.hi)
        return I(LO.next_minus(a) if a else a, HI.next_plus(b))

    def abs_upper(self):
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def width(self):
        return HI.subtract(self.hi, self.lo)

    def mid(self):
        return NEAR.divide(NEAR.add(self.lo, self.hi), Decimal(2))

    def serial(self):
        return [str(self.lo), str(self.hi)]


class AD:
    __slots__ = ("v", "g")
    dim = 38

    def __init__(self, v, g=None):
        self.v = I.cast(v)
        self.g = g if g is not None else [I(0)] * self.dim

    @staticmethod
    def cast(x):
        return x if isinstance(x, AD) else AD(x)

    def __add__(self, x):
        x = AD.cast(x)
        return AD(self.v + x.v, [a + b for a, b in zip(self.g, x.g)])

    __radd__ = __add__

    def __neg__(self):
        return AD(-self.v, [-a for a in self.g])

    def __sub__(self, x):
        return self + -AD.cast(x)

    def __rsub__(self, x):
        return AD.cast(x) + -self

    def __mul__(self, x):
        x = AD.cast(x)
        return AD(self.v * x.v, [a * x.v + self.v * b for a, b in zip(self.g, x.g)])

    __rmul__ = __mul__

    def reciprocal(self):
        v = self.v.reciprocal()
        return AD(v, [-a * v**2 for a in self.g])

    def __truediv__(self, x):
        return self * AD.cast(x).reciprocal()

    def __rtruediv__(self, x):
        return AD.cast(x) * self.reciprocal()

    def __pow__(self, n):
        if n < 0:
            return (self ** (-n)).reciprocal()
        return (
            AD(self.v**n, [n * self.v ** (n - 1) * a for a in self.g]) if n else AD(1)
        )

    def sqrt(self):
        v = self.v.sqrt()
        return AD(v, [a / (2 * v) for a in self.g])
