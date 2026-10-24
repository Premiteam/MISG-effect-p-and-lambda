from pathlib import Path
import argparse, ctypes, hashlib, json, time
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "systems.npz"
MODELS = [(1, 0.95, 1.0), (5, 0.95, 1.0), (1, 1.0, 2.5), (5, 1.0, 3.5)]


def reference(rho_star, d, p, lam, gamma, u, v):
    N = len(u) - 1
    off = d + 4
    x = np.zeros((N + off + 3, 2))
    h = np.zeros_like(x)
    h[: off + 2] = 1e-6
    rho = np.ones(8) * 1e-6
    phi = np.zeros((8, p))
    y = np.zeros(N + 1)
    r = 1.0
    ro = []
    ho = []
    xo = []
    rr = []
    A = np.array([[0, 1], rho_star[:2]])
    B = rho_star[2:6].reshape(2, 2)
    f = rho_star[6:]
    for k in range(1, N + 1):
        i = k + off
        y[k] = x[i, 0] + v[k]
        x[i + 1] = A @ x[i] + B @ x[i - d] + f * u[k]
        phi[:, 1:] = phi[:, :-1].copy()
        phi[:, 0] = np.r_[
            h[i - 2],
            h[i - d - 1],
            h[i - d - 2],
            u[k - 1] if k > 1 else 0,
            u[k - 2] if k > 2 else 0,
        ]
        Y = np.array([y[k - j] if k - j > 0 else 0.0 for j in range(p)])
        r = lam * r + np.linalg.norm(phi, "fro") ** 2
        rho = rho + gamma * phi @ (Y - phi.T @ rho) / r
        ro.append(rho.copy())
        ho.append(h[i].copy())
        xo.append(x[i].copy())
        rr.append(r)
        h[i + 1] = (
            np.array([[0, 1], rho[:2]]) @ h[i]
            + rho[2:6].reshape(2, 2) @ h[i - d]
            + rho[6:] * u[k]
        )
    return [np.array(a) for a in [ro, ho, xo, rr]]


def core(rho, d, p, lam, gamma, u, v):
    N = len(u) - 1
    arrays = [
        np.full((N, 8), np.nan),
        np.full((N, 2), np.nan),
        np.full((N, 2), np.nan),
        np.full(N, np.nan),
    ]
    fn = LIB.run_core
    vals = [np.ascontiguousarray(z, dtype=np.float64) for z in [rho, u, v]] + arrays
    stop = fn(
        N,
        int(d),
        p,
        lam,
        gamma,
        *[a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in vals],
    )
    return stop, arrays


def strict(obj):
    if isinstance(obj, dict):
        return {k: strict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strict(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--output", default="long_horizon_diagnostics.json")
    parser.add_argument("--system-index", type=int)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise SystemExit("Output exists; select a new name instead of overwriting.")
    fam = np.load(ARCHIVE, allow_pickle=False)
    ids = np.flatnonzero(fam["item_splits"] == "test")
    if args.system_index is not None:
        if args.system_index not in ids:
            raise SystemExit("Selected index is not in the test split.")
        ids = np.array([args.system_index])

    cross = []
    for i in ids[:3]:
        rho = fam[f"rho_star_{i:06d}"]
        d = int(fam["d"][i])
        rg = np.random.default_rng(991 + int(i))
        u = np.r_[0.0, rg.normal(size=300)]
        v = np.r_[0.0, rg.normal(scale=fam["sigma_v"][i], size=300)]
        for p, lam, gam in MODELS:
            stop, actual = core(rho, d, p, lam, gam, u, v)
            expected = reference(rho, d, p, lam, gam, u, v)
            errors = [
                float(np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b))))
                for a, b in zip(actual, expected)
            ]
            assert stop == 0 and max(errors) < 1e-8, (int(i), p, lam, errors)
            cross.append(
                {"system": int(i), "p": p, "lambda": lam, "relative_max_errors": errors}
            )
    results = []
    start = time.monotonic()
    for it, i in enumerate(ids):
        rho = fam[f"rho_star_{i:06d}"]
        d = int(fam["d"][i])
        sigma = float(fam["sigma_v"][i])
        for rep in range(args.replicates):
            seed = 2026100600 + int(i) * 10 + rep
            rg = np.random.default_rng(seed)
            u = np.r_[0.0, rg.normal(size=args.steps)]
            v = np.r_[0.0, rg.normal(scale=sigma, size=args.steps)]
            for p, lam, gam in MODELS:
                stop, (rh, h, x, r) = core(rho, d, p, lam, gam, u, v)
                valid = stop - 1 if stop else args.steps
                with np.errstate(over="ignore", invalid="ignore"):
                    e = np.sum((rh[:valid] - rho) ** 2, axis=1)
                    s = np.sum((h[:valid] - x[:valid]) ** 2, axis=1)
                checkpoints = []
                for t in [100, 1000, 10000, 100000, 1000000]:
                    if t <= valid:
                        checkpoints.append(
                            {
                                "time": t,
                                "parameter_mse_latter_half": np.mean(e[t // 2 : t]),
                                "state_mse_latter_half": np.mean(s[t // 2 : t]),
                                "rho_norm": np.linalg.norm(rh[t - 1]),
                                "state_norm": np.linalg.norm(h[t - 1]),
                                "normalizer": r[t - 1],
                            }
                        )
                results.append(
                    {
                        "system": int(i),
                        "replicate": rep,
                        "seed": seed,
                        "delay": d,
                        "p": p,
                        "lambda": lam,
                        "gamma": gam,
                        "numerical_failure_step": stop or None,
                        "max_parameter_error2": float(np.max(e)) if valid else None,
                        "max_state_error2": float(np.max(s)) if valid else None,
                        "max_rho_norm": float(
                            np.max(np.linalg.norm(rh[:valid], axis=1))
                        )
                        if valid
                        else None,
                        "true_rho_norm": float(np.linalg.norm(rho)),
                        "checkpoints": checkpoints,
                    }
                )
        if (it + 1) % 25 == 0:
            print(
                f"{it + 1}/{len(ids)} systems checked; elapsed {time.monotonic() - start:.1f}s",
                flush=True,
            )
    data = {
        "scope": "UNSTOPPED diagnostic extension, not manuscript risk and not a proof; no survivors silently excluded",
        "archive": str(ARCHIVE),
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "steps": args.steps,
        "replicates": args.replicates,
        "system_count": len(ids),
        "crosschecks": cross,
        "results": results,
    }
    output.write_text(json.dumps(strict(data), ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(
        "Numerical failures:",
        sum(a["numerical_failure_step"] is not None for a in results),
    )


if __name__ == "__main__":
    LIB = ctypes.CDLL(str(ROOT / "uncapped_core.dylib"))
    LIB.run_core.argtypes = (
        [ctypes.c_int] * 3
        + [ctypes.c_double] * 2
        + [ctypes.POINTER(ctypes.c_double)] * 7
    )
    LIB.run_core.restype = ctypes.c_int
    main()
