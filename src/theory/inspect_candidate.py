import ast, __future__, ctypes, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_audit as audit

SRC = Path(__file__).resolve().parents[1] / "core"


def original_functions():
    ns = {"np": np}
    for name in ["systems.py", "estimators.py"]:
        tree = ast.parse((SRC / name).read_text())
        funcs = [
            x
            for x in tree.body
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        code = compile(
            ast.Module(body=funcs, type_ignores=[]),
            str(SRC / name),
            "exec",
            flags=__future__.annotations.compiler_flag,
        )
        exec(code, ns)
    return ns


def main():
    audit.LIB = ctypes.CDLL(str(audit.ROOT / "uncapped_core.dylib"))
    audit.LIB.run_core.argtypes = (
        [ctypes.c_int] * 3
        + [ctypes.c_double] * 2
        + [ctypes.POINTER(ctypes.c_double)] * 7
    )
    audit.LIB.run_core.restype = ctypes.c_int
    fam = np.load(audit.ARCHIVE, allow_pickle=False)
    i = 319
    d = int(fam["d"][i])
    rho = fam[f"rho_star_{i:06d}"]
    N = 12000

    rg = np.random.default_rng(2026103790)
    u = np.r_[0.0, rg.normal(size=100000)][: N + 1].copy()
    v = np.r_[0.0, rg.normal(scale=fam["sigma_v"][i], size=100000)][: N + 1].copy()
    stop, (rh, h, x, r) = audit.core(rho, d, 5, 0.95, 1.0, u, v)
    ns = original_functions()
    cfg = SimpleNamespace(
        p0_init=1e6, div_rho_factor=100.0, div_spec_radius=1.2, div_spec_patience=50
    )
    sys = {"n": 2, "d": d, "rho_star": rho, "sigma_v": float(fam["sigma_v"][i])}
    traj = {"N": N, "u": u, "y": np.r_[0.0, x[:, 0] + v[1:]]}
    original = ns["misg_run"](
        sys, traj, np.ones(5), 1.0, "forgetting", "observer", cfg, lam_g=0.95
    )
    valid = original["diverged_at"] or N

    early = min(valid, 1000)
    err_r = float(np.max(np.abs(original["rho_hat_traj"][:early] - rh[:early])))
    err_x = float(np.max(np.abs(original["x_hat_traj"][:early] - h[:early])))
    assert err_r < 1e-8 and err_x < 1e-7, (err_r, err_x)
    snap = []
    for k in [1000, 3000, 5000, 10000, 12000]:
        if k > valid:
            continue
        rr = original["rho_hat_traj"][k - 1]
        A, B, f = ns["unpack_rho"](rr, 2, d)
        F = ns["build_F"](A, B, d)
        eig, vec = np.linalg.eig(F)
        idx = int(np.argmax(abs(eig)))
        snap.append(
            {
                "k": k,
                "rho": rr.tolist(),
                "estimated_spectral_radius": float(abs(eig[idx])),
                "dominant_eigenvalue_real": float(eig[idx].real),
                "dominant_eigenvalue_imag": float(eig[idx].imag),
                "dominant_output_component": float(
                    abs(vec[0, idx]) / np.linalg.norm(vec[:, idx])
                ),
                "state_norm_original": float(
                    np.linalg.norm(original["x_hat_traj"][k - 1])
                ),
                "state_norm_cpp": float(np.linalg.norm(h[k - 1])),
            }
        )
    output = {
        "system_archive_index": i,
        "test_split_ordinal_1_based": int(
            np.where(np.flatnonzero(fam["item_splits"] == "test") == i)[0][0]
        )
        + 1,
        "delay": d,
        "true_rho": rho.tolist(),
        "true_spectral_radius": float(fam["rho_F"][i]),
        "noise_sigma": float(fam["sigma_v"][i]),
        "original_status": original["status"],
        "original_diverged_at": original["diverged_at"],
        "original_divergence_reason": original["div_reason"],
        "early_1000_steps_max_rho_difference": err_r,
        "early_1000_steps_max_state_difference": err_x,
        "original_source_sha256": hashlib.sha256(
            (SRC / "estimators.py").read_bytes()
        ).hexdigest(),
        "snapshots": snap,
        "interpretation": "Numerical escape under original code is not by itself a proof of infinite-time mean-square divergence.",
    }
    dest = audit.ROOT / "candidate_original_source_check.json"
    if dest.exists():
        raise SystemExit("Refusing to overwrite previous check.")
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
