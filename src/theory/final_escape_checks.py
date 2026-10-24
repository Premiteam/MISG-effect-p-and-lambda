from fractions import Fraction as F
from decimal import Decimal
from pathlib import Path
import json, random, hashlib
import interval_arithmetic as ia
from construct_entry_prefix import spectral_reset_certificate

ROOT = Path(__file__).resolve().parent


def inside(x, v):
    return F(x.lo) <= v <= F(x.hi)


def main():
    ia.precision(80)
    I = ia.I
    rng = random.Random(61006)
    for _ in range(200):
        a = F(rng.randint(-10000, 10000), rng.randint(1, 10000))
        b = F(rng.randint(1, 10000), rng.randint(1, 10000))
        aa = I(a)
        bb = I(b)
        for value, exact in [
            (aa + bb, a + b),
            (aa - bb, a - b),
            (aa * bb, a * b),
            (aa / bb, a / b),
            (aa**2, a * a),
            (bb**-3, b**-3),
        ]:
            assert inside(value, exact)
        sq = bb.sqrt()
        assert F(sq.lo) ** 2 <= b <= F(sq.hi) ** 2
    ep = json.loads((ROOT / "entry_prefix_certificate.json").read_text())
    ep2 = json.loads((ROOT / "entry_prefix_with_stops.json").read_text())
    assert (
        ep["controls_sha256"] == ep2["controls_sha256"]
        and ep["steps"] == ep2["steps"] == 4000
    )
    pairs = list(zip(ep["rho"], ep2["rho"])) + [
        (ep["q"], ep2["q"]),
        (ep["s_normalized"], ep2["s_normalized"]),
    ]
    pairs += list(
        zip(
            [a for row in ep["h_normalized"] for a in row],
            [a for row in ep2["h_normalized"] for a in row],
        )
    )
    assert all(
        Decimal(b[0]) <= Decimal(a[0]) <= Decimal(a[1]) <= Decimal(b[1])
        for a, b in pairs
    )
    resets = ep2["spectral_reset_certificates"]
    times = [0] + [a["k"] for a in resets] + [4000]
    assert max(b - a for a, b in zip(times, times[1:])) < 50
    assert Decimal(ep2["max_parameter_l1_upper"]) < 100

    bas = json.loads((ROOT / "quantitative_escape_basin.json").read_text())
    rad = Decimal(bas["coordinate_box_radius"])
    c = [I(F(a)) + I(-rad, rad) for a in bas["center_rationals"]]
    z = (-c[3]).sqrt()
    rho = [c[0], c[1], c[2], c[3], c[4], z * z * (z - c[1]) + I(-rad, rad), c[5], c[6]]
    spec = spectral_reset_certificate(rho)
    assert spec is not None
    total = Decimal(0)
    for a in rho:
        total = ia.HI.add(total, a.abs_upper())
    assert total < 100
    report = {
        "arithmetic_tests": "200 exact rational cases, six operations each, and square-root enclosure tests passed",
        "prefix_precision_nesting": "5000-digit enclosure lies inside independently recomputed 1200-digit enclosure",
        "finite_prefix_reset_count": len(resets),
        "maximum_spacing_between_certified_resets": max(
            b - a for a, b in zip(times, times[1:])
        ),
        "finite_prefix_parameter_l1_upper": ep2["max_parameter_l1_upper"],
        "local_box_spectral_certificate": spec,
        "local_box_parameter_l1_upper": str(total),
        "scope": "checks of a computer-assisted proof, not a proof-assistant formalization or external peer review",
        "artifact_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "entry_reference_point.json",
                "entry_prefix_certificate.json",
                "entry_prefix_with_stops.json",
                "quantitative_escape_basin.json",
                "basin_entry_verified.json",
            ]
        },
    }
    out = ROOT / "escape_final_checks.json"
    if out.exists():
        raise SystemExit("Refusing overwrite")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
