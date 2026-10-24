import json
from types import SimpleNamespace
import numpy as np
import run_audit as audit
from inspect_candidate import original_functions


def main():
    ns = original_functions()
    fam = np.load(audit.ARCHIVE, allow_pickle=False)
    i = 319
    system = {key: fam[f"{key}_{i:06d}"] for key in ["A", "B", "f", "rho_star"]}
    for key in ["d", "n", "m"]:
        system[key] = int(fam[key][i])
    system["sigma_v"] = float(fam["sigma_v"][i])
    cfg = SimpleNamespace(
        p0_init=1e6, div_rho_factor=100.0, div_spec_radius=1.2, div_spec_patience=50
    )
    rows = []
    for j in [3, 4]:
        seed = np.random.SeedSequence([20260729, 1, system["m"], j])
        traj = ns["simulate"](system, np.random.default_rng(seed), 12000)
        for p, lam, gamma in audit.MODELS:
            res = ns["misg_run"](
                system,
                traj,
                np.ones(p),
                gamma,
                "cumulative" if lam == 1 else "forgetting",
                "observer",
                cfg,
                lam_g=lam,
            )
            good = res["diverged_at"] - 1 if res["diverged_at"] else 12000
            rh = res["rho_hat_traj"][:good]
            h = res["x_hat_traj"][:good]
            x = traj["x"][1 : good + 1]
            A, B, f = ns["unpack_rho"](rh[-1], 2, system["d"])
            rows.append(
                {
                    "trajectory_index": j,
                    "p": p,
                    "lambda": lam,
                    "gamma": gamma,
                    "status": res["status"],
                    "diverged_at": res["diverged_at"],
                    "max_state_norm": float(np.max(np.linalg.norm(h, axis=1))),
                    "final_state_norm": float(np.linalg.norm(h[-1])),
                    "final_parameter_norm": float(np.linalg.norm(rh[-1])),
                    "final_estimated_spectral_radius": ns["spectral_radius"](
                        ns["build_F"](A, B, system["d"])
                    ),
                    "raw_latter_half_normalized_parameter_error": float(
                        np.mean(np.sum((rh[6000:] - system["rho_star"]) ** 2, axis=1))
                        / np.sum(system["rho_star"] ** 2)
                    )
                    if good == 12000
                    else None,
                    "latter_half_state_mse": float(
                        np.mean(np.sum((h[6000:] - x[6000:]) ** 2, axis=1))
                    )
                    if good == 12000
                    else None,
                }
            )
    result = {
        "archive_index": i,
        "system_m": system["m"],
        "N": 12000,
        "seed_rule": "SeedSequence([20260729,1,m,j])",
        "results": rows,
    }
    dest = audit.ROOT / "candidate_original_test_trajectories.json"
    if dest.exists():
        raise SystemExit("Refusing to overwrite.")
    result = audit.strict(result)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
