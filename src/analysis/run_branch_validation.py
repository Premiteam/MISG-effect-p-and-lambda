from pathlib import Path
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, argparse, hashlib, time, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "core"))
from config import Config, traj_seed
from systems import load_family, simulate, unpack_rho, build_F, pack_rho
from estimators import misg_run

N = 12000
CAP = 10.0
CONFIGS = [(1, 1.0, 2.5), (5, 1.0, 3.5), (1, 0.95, 1.0), (5, 0.95, 1.0)]
OUT = REPO / "data" / "branch_results"


def alternative(s):
    A, B, f = s["A"], s["B"], s["f"]
    h = f[0] / f[1]
    g = A[1, 1] + A[1, 0] * h
    ell = B[1, 1] - B[0, 0] - g * B[0, 1] + h * B[1, 0]
    altB = B + ell / (1 - h * g) * np.array([[1.0, -h], [g, -1.0]])
    return pack_rho(A, altB, f, s["d"])


def coefficients(rho, d):
    A, B, f = unpack_rho(rho, 2, d)
    return np.array(
        [
            A[1, 0],
            A[1, 1],
            f[0],
            f[1] - A[1, 1] * f[0],
            B[0, 1] * f[1] - B[1, 1] * f[0],
            B[0, 0] + B[1, 1],
            A[1, 1] * B[0, 0] - A[1, 0] * B[0, 1] - B[1, 0],
            np.linalg.det(B),
        ]
    )


def forward(rho, d, u):
    A, B, f = unpack_rho(rho, 2, d)
    x = np.zeros((len(u) + 1, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, len(u)):
            xd = x[k - d] if k - d > 0 else np.zeros(2)
            x[k + 1] = A @ x[k] + B @ xd + f * u[k]
            if not np.all(np.isfinite(x[k + 1])) or np.max(np.abs(x[k + 1])) > 1e100:
                return None
    return x[1 : len(u), 0]


def bounded(value):
    return CAP if value is None or not np.isfinite(value) else min(float(value), CAP)


def task(index):
    s = load_family(REPO / "data" / "systems.npz")["test"][index]
    truth = s["rho_star"]
    alt = alternative(s)
    coeff_diff = float(
        np.max(np.abs(coefficients(truth, s["d"]) - coefficients(alt, s["d"])))
    )
    np.testing.assert_allclose(
        coefficients(truth, s["d"]), coefficients(alt, s["d"]), rtol=1e-8, atol=1e-8
    )
    cfg = replace(Config(), N=N)
    rows = []
    max_output_difference = 0.0
    for j in (3, 4):
        trajectory = simulate(
            s, np.random.default_rng(traj_seed(cfg, int(s["m"]), j)), N
        )

        rng = np.random.default_rng(
            np.random.SeedSequence([20261013, 160, int(s["m"]), j])
        )
        u = np.r_[0.0, rng.standard_normal(N)]
        fresh_noise = rng.normal(0.0, float(s["sigma_v"]), N)
        y0 = forward(truth, s["d"], u)
        ya = forward(alt, s["d"], u)
        assert y0 is not None and ya is not None
        odiff = float(np.linalg.norm(ya - y0) / np.linalg.norm(y0))
        assert odiff < 1e-6, odiff
        max_output_difference = max(max_output_difference, odiff)
        for p, lam, gamma in CONFIGS:
            result = misg_run(
                s,
                trajectory,
                np.ones(p),
                gamma,
                "cumulative" if lam == 1 else "forgetting",
                "observer",
                cfg,
                lam_g=lam,
            )
            estimates = result["rho_hat_traj"]
            block = estimates[N // 2 : N]
            finite = result["status"] == "ok" and np.all(np.isfinite(block))
            den = float(truth @ truth)
            r = req = None
            branch = "stopped"
            final = estimates[-1]
            predicted = None
            spectral = None
            predraw = prednoise = None
            if finite:
                e0 = np.sum((block - truth) ** 2, axis=1) / den
                e1 = np.sum((block - alt) ** 2, axis=1) / den
                r = float(e0.mean())
                req = float(np.minimum(e0, e1).mean())
                d0 = float(np.sum((final - truth) ** 2) / den)
                d1 = float(np.sum((final - alt) ** 2) / den)
                branch = (
                    "tie"
                    if np.isclose(d0, d1, rtol=1e-10, atol=1e-12)
                    else ("generator" if d0 < d1 else "alternative")
                )
                A, B, _ = unpack_rho(final, 2, s["d"])
                spectral = float(
                    np.max(np.abs(np.linalg.eigvals(build_F(A, B, s["d"]))))
                )
                predicted = forward(final, s["d"], u)
                if predicted is not None:
                    predraw = float(np.sum((predicted - y0) ** 2) / np.sum(y0 * y0))
                    prednoise = float(
                        np.sum((predicted - y0 - fresh_noise) ** 2)
                        / np.sum((y0 + fresh_noise) ** 2)
                    )
            rows.append(
                dict(
                    system_id=int(s["m"]),
                    trajectory=j,
                    p=p,
                    lam=lam,
                    gamma=gamma,
                    R=bounded(r),
                    Req=bounded(req),
                    R_raw=r,
                    Req_raw=req,
                    branch=branch,
                    status=result["status"],
                    output_error=bounded(predraw),
                    output_error_raw=predraw,
                    noisy_output_error=bounded(prednoise),
                    frozen_spectral_radius=spectral,
                    output_overflow=finite and predicted is None,
                    final_estimate=final.tolist()
                    if np.all(np.isfinite(final))
                    else None,
                )
            )
    return dict(
        index=index,
        system_id=int(s["m"]),
        coefficient_difference=coeff_diff,
        equivalent_output_relative_difference=max_output_difference,
        rows=rows,
    )


def interval(a, b, confidence=0.95):
    values = np.log((a + 1e-12) / (b + 1e-12))
    rng = np.random.default_rng(20261013)
    res = np.median(values[rng.integers(0, len(values), (10000, len(values)))], axis=1)
    alpha = (1 - confidence) / 2
    return [
        float(np.exp(np.median(values))),
        *np.exp(np.quantile(res, [alpha, 1 - alpha])).tolist(),
    ]


def summarize(records):
    rows = [r for rec in records for r in rec["rows"]]
    ids = sorted({r["system_id"] for r in rows})
    arrays = {}
    summaries = []
    for p, lam, gamma in CONFIGS:
        rr = [r for r in rows if r["p"] == p and r["lam"] == lam]
        data = {
            metric: np.array(
                [np.mean([r[metric] for r in rr if r["system_id"] == i]) for i in ids]
            )
            for metric in ("R", "Req", "output_error", "noisy_output_error")
        }
        arrays[(p, lam)] = data
        summaries.append(
            dict(
                p=p,
                lam=lam,
                gamma=gamma,
                n_systems=len(ids),
                n_trajectories=len(rr),
                medians={k: float(np.median(v)) for k, v in data.items()},
                branches={
                    k: sum(r["branch"] == k for r in rr)
                    for k in ("generator", "alternative", "tie", "stopped")
                },
                frozen_unstable=sum(
                    r["frozen_spectral_radius"] is not None
                    and r["frozen_spectral_radius"] >= 1
                    for r in rr
                ),
                output_overflow=sum(r["output_overflow"] for r in rr),
                output_capped=sum(r["output_error"] >= CAP for r in rr),
                max_R_reduction=max(r["R"] - r["Req"] for r in rr),
            )
        )
    contrasts = []
    pairs = [
        ((1, 1.0), (5, 1.0), 0.95),
        ((1, 0.95), (5, 0.95), 0.90),
        ((1, 0.95), (1, 1.0), 0.95),
        ((5, 0.95), (5, 1.0), 0.95),
    ]
    for a, b, c in pairs:
        contrasts.append(
            dict(
                numerator=a,
                denominator=b,
                confidence=c,
                metrics={
                    metric: interval(arrays[a][metric], arrays[b][metric], c)
                    for metric in arrays[a]
                },
            )
        )
    summary = dict(
        protocol=dict(
            N=N,
            test_trajectories=[3, 4],
            systems=len(ids),
            cap=CAP,
            output_seed_namespace=[20261013, 160],
            bootstrap=10000,
            bootstrap_seed=20261013,
            output_definition="zero-history frozen-final-model free simulation, independent Gaussian input, noiseless plant output denominator; failures retained at cap 10",
            branch_definition="nearest equivalent parameter at final update, not convergence or basin membership",
        ),
        configs=summaries,
        contrasts=contrasts,
        maximum_coefficient_discrepancy=max(
            r["coefficient_difference"] for r in records
        ),
        maximum_equivalent_output_discrepancy=max(
            r["equivalent_output_relative_difference"] for r in records
        ),
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (OUT / "trajectory_metrics.csv").open("w") as f:
        fields = [k for k in rows[0] if k != "final_estimate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=201)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    inputs = [
        REPO / "data" / "systems.npz",
        *sorted((REPO / "src" / "core").glob("*.py")),
        Path(__file__),
    ]
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs
            },
            indent=2,
        )
        + "\n"
    )
    done = {int(p.stem) for p in (OUT / "checkpoints").glob("*.json")}
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, i): i for i in range(args.limit) if i not in done}
        for future in as_completed(futures):
            rec = future.result()
            (OUT / "checkpoints" / f"{rec['index']:03d}.json").write_text(
                json.dumps(rec, allow_nan=False) + "\n"
            )
            done.add(rec["index"])
            print(
                f"{len(done)}/{args.limit} systems completed; {time.time() - start:.0f}s",
                flush=True,
            )
    summarize(
        [
            json.loads((OUT / "checkpoints" / f"{i:03d}.json").read_text())
            for i in range(args.limit)
        ]
    )


if __name__ == "__main__":
    main()
