from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import chi2


SRC = Path(__file__).resolve().parents[1] / "src" / "core"
sys.path.insert(0, str(SRC))

from config import Config
import rng
import scenarios
from systems import (
    build_F,
    pack_rho,
    rho_dim,
    sample_system,
    spectral_radius,
    unpack_rho,
)


PINNED_NUMPY = "2.4.2"
DOMAIN = {
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (5, 3),
    (5, 4),
    (5, 5),
}


def make_system(cfg: Config, m: int = 0, d: int = 2, seed: int = 42) -> dict:
    system = sample_system(np.random.default_rng(seed), cfg, d)
    system.update(m=m, split="train")
    return system


def make_plan(cfg: Config, system: dict, *, retry: int = 0, seed: int = 43):
    second = make_system(cfg, m=system["m"], d=system["d"], seed=seed)
    return {
        system["m"]: scenarios.DriftPlanEntry(
            m=system["m"],
            second_system=second,
            n_retry=retry,
            source_rho_sha256=scenarios._rho_sha256(system),
        )
    }


def manual_time_varying_recurrence(result: dict, cfg: Config, d: int):
    x = np.zeros((cfg.N + 2, cfg.n), dtype=float)
    y = np.zeros(cfg.N + 1, dtype=float)
    for k in range(1, cfg.N + 1):
        y[k] = x[k, 0] + result["v"][k]
        A_k, B_k, f_k = unpack_rho(result["rho_star_traj"][k], cfg.n, d)
        delayed = x[k - d] if k - d > 0 else np.zeros(cfg.n, dtype=float)
        x[k + 1] = A_k @ x[k] + B_k @ delayed + f_k * result["u"][k]
    return x, y


def assert_state_equal(a, b, path="state"):
    try:
        if isinstance(a, dict) and isinstance(b, dict):
            assert set(a) == set(b)
            for key in a:
                assert_state_equal(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert np.array_equal(a, b)
        elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for index, (left, right) in enumerate(zip(a, b)):
                assert_state_equal(left, right, f"{path}[{index}]")
        else:
            assert type(a) is type(b) and a == b
    except (AssertionError, ValueError) as exc:
        raise AssertionError(path) from exc


def test_01_s1_s2_are_one_constructive_path(monkeypatch):
    assert np.__version__ == PINNED_NUMPY
    cfg = replace(Config(), N=30)
    system = make_system(cfg)
    states, entropies, calls = [], [], []
    real_simulate, real_traj_seed = scenarios.simulate, scenarios.traj_seed

    def record_simulate(*args, **kwargs):
        generator = args[1] if len(args) > 1 else kwargs["rng"]
        states.append(copy.deepcopy(generator.bit_generator.state))
        return real_simulate(*args, **kwargs)

    def record_traj_seed(*args, **kwargs):
        calls.append((args[1], args[2]))
        key = real_traj_seed(*args, **kwargs)
        entropies.append(key.entropy)
        return key

    monkeypatch.setattr(scenarios, "simulate", record_simulate, raising=True)
    monkeypatch.setattr(scenarios, "traj_seed", record_traj_seed, raising=True)
    s1 = scenarios.scenario_trajectory(system, 0, 0, 1, 0, cfg)
    s2 = scenarios.scenario_trajectory(system, 0, 0, 2, 0, cfg)
    for key in ("u", "v", "x", "y"):
        assert np.array_equal(s1[key], s2[key])
    assert len(states) == 2
    assert_state_equal(states[0], states[1])
    assert entropies[0] == entropies[1]
    assert calls == [(0, 0), (0, 0)]


def test_02_s4_zero_equals_s2_bitwise():
    cfg = replace(Config(), N=40)
    system = make_system(cfg)
    s2 = scenarios.scenario_trajectory(system, 0, 0, 2, 0, cfg)
    s4 = scenarios.scenario_trajectory(system, 0, 0, 5, 3, cfg)
    for key in ("u", "v", "x", "y"):
        assert np.array_equal(s2[key], s4[key])


def test_02b_s4_transform_is_deterministically_active():
    assert np.__version__ == PINNED_NUMPY
    cfg = Config()
    system = make_system(cfg)
    s2 = scenarios.scenario_trajectory(system, 0, 0, 2, 0, cfg)
    s4 = scenarios.scenario_trajectory(system, 0, 0, 5, 0, cfg)
    assert s4["provenance"]["outlier_index"].size == 150
    assert not np.array_equal(s2["v"], s4["v"])


def test_02c_s4_y_is_recomputed_from_x_and_transformed_v():
    cfg = replace(Config(), N=60)
    result = scenarios.scenario_trajectory(make_system(cfg), 0, 0, 5, 5, cfg)
    assert np.array_equal(
        result["y"][1:], result["x"][1 : cfg.N + 1, 0] + result["v"][1:]
    )


def test_03_s4_outlier_sets_are_nested():
    cfg = Config()
    system = make_system(cfg)
    i01 = scenarios.scenario_trajectory(system, 0, 0, 5, 4, cfg)["provenance"][
        "outlier_index"
    ]
    i05 = scenarios.scenario_trajectory(system, 0, 0, 5, 0, cfg)["provenance"][
        "outlier_index"
    ]
    i10 = scenarios.scenario_trajectory(system, 0, 0, 5, 5, cfg)["provenance"][
        "outlier_index"
    ]
    assert np.isin(i01, i05).all()
    assert np.isin(i05, i10).all()


def test_04_s4_variance_preserving_scaling_uses_registered_tolerance():
    cfg = Config()
    system = make_system(cfg)
    variances = []
    for condition in (3, 4, 0, 5):
        result = scenarios.scenario_trajectory(system, 0, 3, 5, condition, cfg)
        variances.append(float(np.var(result["v"][1:], ddof=1)))

    lower = (cfg.N - 1) / chi2.ppf(0.9995, cfg.N - 1)
    upper = (cfg.N - 1) / chi2.ppf(0.0005, cfg.N - 1)
    reference = variances[0]
    assert all(lower <= value / reference <= upper for value in variances[1:])


def test_04b_s4_uses_registered_kappa_ten_in_the_hand_formula():
    assert np.__version__ == PINNED_NUMPY
    cfg = Config()
    system = make_system(cfg)
    key = scenarios.traj_seed(cfg, 0, 0)
    generator = np.random.default_rng(key)
    base = scenarios.simulate(system, generator, cfg.N)
    branch_uniform = generator.random(cfg.N)
    epsilon = 0.05
    expected_multiplier = np.where(branch_uniform < epsilon, 10.0, 1.0)
    expected_v = np.asarray(base["v"]).copy()
    expected_v[1:] = (
        expected_v[1:] * expected_multiplier / np.sqrt(1.0 + epsilon * (10.0**2 - 1.0))
    )

    result = scenarios.scenario_trajectory(system, 0, 0, 5, 0, cfg)
    assert scenarios._S4_KAPPA == 10.0
    assert np.array_equal(result["v"], expected_v)


def test_05_s3_jump_path_and_magnitude(monkeypatch):
    cfg = Config()
    first = make_system(cfg, seed=11)
    second = make_system(cfg, seed=12)
    monkeypatch.setattr(scenarios, "_second_system", lambda *args: second)
    result = scenarios.scenario_trajectory(first, 0, 0, 3, 0, cfg)
    rho1, rho2 = scenarios._packed(first), scenarios._packed(second)
    assert np.array_equal(result["rho_star_traj"][1500], rho1)
    assert np.array_equal(result["rho_star_traj"][1501], rho2)
    expected = np.linalg.norm(rho2 - rho1) / np.linalg.norm(rho1)
    assert result["provenance"]["jump_magnitude"] == pytest.approx(expected, rel=1e-12)


def test_05b_jump_magnitude_never_filters_a_family(monkeypatch):
    cfg = Config()
    family = make_family(cfg)
    monkeypatch.setattr(
        scenarios,
        "_second_system",
        lambda m, retry, cfg, d: make_system(cfg, m, d, 10000 + m),
    )
    monkeypatch.setattr(scenarios, "_rho_path_is_stable", lambda *args: True)
    built = scenarios.build_s3_drift_family(family, cfg)
    for split in ("train", "val", "test"):
        audit = built["audit_by_split"][split]
        assert len(audit["surviving_system_ids"]) == len(family[split])
        assert set(audit["jump_magnitude_by_system"]) == set(
            audit["surviving_system_ids"]
        )


def test_06_retry_zero_candidate_is_shared_and_later_retry_is_not(monkeypatch):
    cfg = Config()
    first = make_system(cfg, seed=21)
    retry0 = make_system(cfg, seed=22)
    retry2 = make_system(cfg, seed=23)
    calls = []

    def second(m, retry, cfg_arg, d):
        calls.append((m, retry, cfg_arg, d))
        return retry0

    monkeypatch.setattr(scenarios, "_second_system", second)
    jump = scenarios.scenario_trajectory(first, 0, 0, 3, 0, cfg)
    assert calls == [(0, 0, cfg, int(first["d"]))]
    zero_plan = make_plan(cfg, first, retry=0, seed=22)
    drift0 = scenarios.scenario_trajectory(first, 0, 0, 4, 2, cfg, zero_plan)
    assert np.array_equal(drift0["rho_star_traj"][-1], jump["rho_star_traj"][-1])
    retry_plan = make_plan(cfg, first, retry=2, seed=23)
    drift2 = scenarios.scenario_trajectory(first, 0, 0, 4, 2, cfg, retry_plan)
    assert np.array_equal(drift2["rho_star_traj"][-1], scenarios._packed(retry2))
    assert not np.array_equal(drift2["rho_star_traj"][-1], jump["rho_star_traj"][-1])


def test_06b_jump_requires_the_registered_n_and_fixed_index(monkeypatch):
    cfg = replace(Config(), N=40)
    first = make_system(cfg, seed=24)
    second = make_system(cfg, seed=25)
    monkeypatch.setattr(scenarios, "_second_system", lambda *args: second)
    with pytest.raises(scenarios.ScenarioError, match=r"N=3000.+k_jump=1500"):
        scenarios.scenario_trajectory(first, 0, 0, 3, 0, cfg)


def test_06c_s3_jump_xy_follow_the_time_varying_truth(monkeypatch):
    cfg = Config()
    first = make_system(cfg, seed=26)
    second = make_system(cfg, seed=27)
    assert not np.array_equal(scenarios._packed(first), scenarios._packed(second))
    monkeypatch.setattr(scenarios, "_second_system", lambda *args: second)
    result = scenarios.scenario_trajectory(first, 0, 0, 3, 0, cfg)
    expected_x, expected_y = manual_time_varying_recurrence(
        result, cfg, int(first["d"])
    )
    static = scenarios.scenario_trajectory(first, 0, 0, 2, 0, cfg)
    assert np.array_equal(result["x"], expected_x)
    assert np.array_equal(result["y"], expected_y)
    assert not np.array_equal(result["x"], static["x"])


def test_07b_s3_drift_xy_follow_each_point_of_the_truth_path():
    cfg = replace(Config(), N=40)
    first = make_system(cfg, seed=33)
    plan = make_plan(cfg, first, seed=34)
    assert not np.array_equal(
        scenarios._packed(first), scenarios._packed(plan[0].second_system)
    )
    result = scenarios.scenario_trajectory(first, 0, 0, 4, 0, cfg, plan)
    expected_x, expected_y = manual_time_varying_recurrence(
        result, cfg, int(first["d"])
    )
    static = scenarios.scenario_trajectory(first, 0, 0, 2, 0, cfg)
    assert np.array_equal(result["x"], expected_x)
    assert np.array_equal(result["y"], expected_y)
    assert not np.array_equal(result["x"], static["x"])


def test_07_s3_drift_alpha_is_pointwise_registered():
    cfg = replace(Config(), N=40)
    first = make_system(cfg, seed=31)
    plan = make_plan(cfg, first, seed=32)
    rho1 = scenarios._packed(first)
    rho2 = scenarios._packed(plan[0].second_system)
    for condition, severity in ((0, 0.5), (1, 0.25), (2, 1.0)):
        result = scenarios.scenario_trajectory(first, 0, 0, 4, condition, cfg, plan)
        alpha = severity * np.arange(cfg.N + 1) / cfg.N
        expected = (1 - alpha[:, None]) * rho1 + alpha[:, None] * rho2
        assert np.allclose(result["rho_star_traj"], expected, rtol=1e-15, atol=0)
        assert alpha[-1] == severity


def test_08_each_drift_grid_is_checked_not_only_s_equal_one():
    cfg = replace(Config(), N=2)
    rho1 = np.array(
        [
            -0.26897604330625313,
            0.057105508472935274,
            0.13261994482509332,
            0.116257348984663,
            0.4260998232253619,
            0.4524664908504723,
            1.44272817654213,
            -0.9422404172575786,
        ]
    )
    rho2 = np.array(
        [
            -0.3358257561243354,
            -1.0271187426108033,
            -0.6650615519512633,
            -1.4932158280743153,
            0.24344040695368285,
            0.7046290733240603,
            -0.6068216807158181,
            0.16200740948697878,
        ]
    )
    radii = []
    for alpha in (0.0, 0.25, 0.5, 1.0):
        A, B, _ = unpack_rho((1 - alpha) * rho1 + alpha * rho2, 2, 2)
        radii.append(spectral_radius(build_F(A, B, 2)))
    assert radii[0] < cfg.rho_max <= radii[1]
    assert radii[2] < cfg.rho_max and radii[3] < cfg.rho_max
    assert scenarios._rho_path_is_stable(rho1, rho2, cfg, 2) is False


def make_family(cfg: Config) -> dict[str, list[dict]]:
    counts = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test": cfg.n_test,
        "pilot": cfg.n_pilot,
    }
    family, m = {}, 0
    for split, count in counts.items():
        rows = []
        for index in range(count):
            d = cfg.d_choices[index % len(cfg.d_choices)]
            local_m = index if split == "pilot" else m

            rows.append(make_system(cfg, local_m, d, seed=5000 + local_m))
            rows[-1]["split"] = split
            if split != "pilot":
                m += 1
        family[split] = rows
    return family


def test_09_retry_limit_records_twenty_and_excludes(monkeypatch):
    cfg = Config()
    family = make_family(cfg)
    target = family["test"][0]["m"]
    calls = []

    def second(m, retry, cfg, d):
        if m == target:
            calls.append(retry)
        return make_system(cfg, m, d, seed=9000 + m + retry)

    monkeypatch.setattr(scenarios, "_second_system", second)
    monkeypatch.setattr(
        scenarios,
        "_rho_path_is_stable",
        lambda r1, r2, cfg, d: not np.array_equal(
            r1, scenarios._packed(family["test"][0])
        ),
    )
    built = scenarios.build_s3_drift_family(family, cfg)
    audit = built["audit_by_split"]["test"]
    assert calls == list(range(21))
    assert audit["n_retry_by_system"][target] == 20
    assert target in audit["excluded_system_ids"]
    assert target not in audit["surviving_system_ids"]
    assert target not in built["drift_plan"]


def test_09b_retry_zero_path_calls_second_key_once(monkeypatch):
    cfg = Config()
    family = make_family(cfg)
    target = family["test"][0]["m"]
    calls = []

    def second(m, retry, cfg, d):
        if m == target:
            calls.append(retry)
        return make_system(cfg, m, d, seed=10000 + m)

    monkeypatch.setattr(scenarios, "_second_system", second)
    monkeypatch.setattr(scenarios, "_rho_path_is_stable", lambda *args: True)
    built = scenarios.build_s3_drift_family(family, cfg)
    assert calls == [0]
    assert built["audit_by_split"]["test"]["n_retry_by_system"][target] == 0


@pytest.mark.parametrize(
    ("excluded_count", "low_stratum", "raises"),
    [(11, False, True), (5, True, True), (10, False, False)],
)
def test_10_test_gate_has_both_registered_predicates(
    monkeypatch, excluded_count, low_stratum, raises
):
    cfg = Config()
    family = make_family(cfg)
    excluded = {row["m"] for row in family["test"][:excluded_count]}
    if low_stratum:
        d2 = [row["m"] for row in family["test"] if row["d"] == 2]
        excluded = set(d2[:8]) | {family["test"][-1]["m"]}

    monkeypatch.setattr(
        scenarios,
        "_second_system",
        lambda m, retry, cfg, d: make_system(cfg, m, d, 12000 + m + retry),
    )
    test_identity = {
        np.ascontiguousarray(scenarios._packed(row)).tobytes(): row["m"]
        for row in family["test"]
    }

    def stable(rho1, rho2, cfg, d):
        m = test_identity.get(np.ascontiguousarray(rho1).tobytes())
        return m is None or m not in excluded

    monkeypatch.setattr(scenarios, "_rho_path_is_stable", stable)
    if raises:
        with pytest.raises(scenarios.ScenarioInfeasible):
            scenarios.build_s3_drift_family(family, cfg)
    else:
        audit = scenarios.build_s3_drift_family(family, cfg)["audit_by_split"]["test"]
        assert audit["exclusion_ratio"] == pytest.approx(10 / 201)
        assert min(audit["per_d_remaining"].values()) >= 60


def test_11_train_exclusion_uses_dedicated_exception_with_audit(monkeypatch):
    cfg = Config()
    family = make_family(cfg)
    target_rho = scenarios._packed(family["train"][0])
    monkeypatch.setattr(
        scenarios,
        "_second_system",
        lambda m, retry, cfg, d: make_system(cfg, m, d, 14000 + m + retry),
    )
    monkeypatch.setattr(
        scenarios,
        "_rho_path_is_stable",
        lambda rho1, *args: not np.array_equal(rho1, target_rho),
    )
    with pytest.raises(scenarios.ScenarioGateUnspecified) as caught:
        scenarios.build_s3_drift_family(family, cfg)
    assert "D9" in str(caught.value) and "train" in str(caught.value)
    assert caught.value.audit["train"]["excluded_system_ids"]


def test_12_audit_is_reconstructible_global_and_json_serializable(monkeypatch):
    cfg = Config()
    family = make_family(cfg)
    monkeypatch.setattr(
        scenarios,
        "_second_system",
        lambda m, retry, cfg, d: make_system(cfg, m, d, 16000 + m),
    )
    monkeypatch.setattr(scenarios, "_rho_path_is_stable", lambda *args: True)
    result = scenarios.build_s3_drift_family(family, cfg)
    audit = result["audit_by_split"]
    val = audit["val"]
    assert len(val["surviving_system_ids"]) == len(val["d_labels"])
    assert val["surviving_system_ids"][0] == family["val"][0]["m"] != 0
    assert val["sha256_ids"] == scenarios._canonical_list_sha256(
        val["surviving_system_ids"]
    )
    assert val["sha256_labels"] == scenarios._canonical_list_sha256(val["d_labels"])
    assert set(val["jump_magnitude_by_system"]) == set(val["surviving_system_ids"])
    json.dumps(audit)


def test_12b_audit_hash_algorithms_have_independent_byte_oracles():
    cfg = Config()
    system = make_system(cfg, d=2, seed=16001)
    system["A"] = np.asfortranarray(np.asarray(system["A"], dtype=np.float32))
    system["B"] = np.asfortranarray(np.asarray(system["B"], dtype=np.float32))
    system["f"] = np.asarray(system["f"], dtype=">f4")
    manual_values = [
        *np.asarray(system["A"])[-1].tolist(),
        *np.asarray(system["B"])[0].tolist(),
        *np.asarray(system["B"])[1].tolist(),
        *np.asarray(system["f"]).tolist(),
    ]
    expected_rho = hashlib.sha256(
        np.asarray(manual_values, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert scenarios._rho_sha256(system) == expected_rho
    assert (
        scenarios._canonical_list_sha256([201, 202])
        == hashlib.sha256(b"[201,202]").hexdigest()
    )


@pytest.mark.parametrize("scenario_code", range(1, 6))
@pytest.mark.parametrize("condition_code", range(9))
def test_13_complete_scenario_condition_cartesian_domain(scenario_code, condition_code):
    assert set(scenarios.SCENARIO_CONDITION_DOMAIN) == DOMAIN
    assert len(scenarios.SCENARIO_CONDITION_DOMAIN) == 10
    cfg = Config() if scenario_code == 3 else replace(Config(), N=4)
    system = make_system(cfg)
    plan = make_plan(cfg, system)
    if (scenario_code, condition_code) in DOMAIN:
        scenarios.scenario_trajectory(
            system, 0, 0, scenario_code, condition_code, cfg, plan
        )
    else:
        with pytest.raises(scenarios.ScenarioError):
            scenarios.scenario_trajectory(
                system, 0, 0, scenario_code, condition_code, cfg, plan
            )


def test_13b_declared_domain_is_backed_by_signed_code_table():
    for scenario_code, condition_code in DOMAIN:
        assert any(
            rng.condition_tuple_is_legal(config, scenario_code, condition_code)
            for config in rng.CONFIG_CODES
        )


@pytest.mark.parametrize("condition_code", (6, 7, 8))
def test_13c_method_layer_conditions_give_executable_guidance(condition_code):
    cfg = replace(Config(), N=2)
    with pytest.raises(scenarios.ScenarioError, match=r"(W0|p).+condition_code=0"):
        scenarios.scenario_trajectory(make_system(cfg), 0, 0, 1, condition_code, cfg)


def test_14_source_has_no_global_rng_or_hash_calls():
    tree = ast.parse(Path(scenarios.__file__).read_text(encoding="utf-8"))
    banned = {"seed", "rand", "randn", "hash"}
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {node.id for node in calls if isinstance(node, ast.Name)}
    attrs = {node.attr for node in calls if isinstance(node, ast.Attribute)}
    assert not (banned & (names | attrs))


def test_14b_only_registered_business_seed_functions_create_sequences():
    source = Path(scenarios.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    seed_names = {
        node.id
        for node in calls
        if isinstance(node, ast.Name)
        and node.id in {"traj_seed", "s3_second_key", "sys_seed"}
    }
    assert seed_names == {"traj_seed", "s3_second_key"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "SeedSequence"
        for node in calls
    )


def test_15_drift_plan_missing_fails_before_stream_consumption(monkeypatch):
    cfg = replace(Config(), N=2)
    monkeypatch.setattr(
        scenarios,
        "traj_seed",
        lambda *args: (_ for _ in ()).throw(AssertionError("consumed")),
    )
    with pytest.raises(scenarios.ScenarioError, match="drift_plan"):
        scenarios.scenario_trajectory(make_system(cfg), 0, 0, 4, 0, cfg)


def test_16_drift_plan_m_missing():
    cfg = replace(Config(), N=2)
    with pytest.raises(scenarios.ScenarioError, match="no entry"):
        scenarios.scenario_trajectory(make_system(cfg), 0, 0, 4, 0, cfg, {})


@pytest.mark.parametrize("defect", ("m", "d", "sha"))
def test_17_drift_plan_identity_checks(defect):
    cfg = replace(Config(), N=3)
    system = make_system(cfg)
    plan = make_plan(cfg, system)
    if defect == "m":
        system["m"] = 1
    elif defect == "d":
        plan[0].second_system["d"] = 3
    else:
        system["A"] = np.asarray(system["A"]).copy()

        system["A"][-1, 0] += 0.01
    with pytest.raises(scenarios.ScenarioError):
        scenarios.scenario_trajectory(system, 0, 0, 4, 0, cfg, plan)


def test_18_consuming_plan_never_resamples(monkeypatch):
    cfg = replace(Config(), N=4)
    system = make_system(cfg)
    plan = make_plan(cfg, system)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("resampled"))
    monkeypatch.setattr(scenarios, "sample_system", boom, raising=True)
    monkeypatch.setattr(scenarios, "s3_second_key", boom, raising=True)
    scenarios.scenario_trajectory(system, 0, 0, 4, 0, cfg, plan)


def test_19_drift_provenance_and_endpoint_come_from_plan():
    cfg = replace(Config(), N=8)
    system = make_system(cfg)
    plan = make_plan(cfg, system, retry=7, seed=77)
    result = scenarios.scenario_trajectory(system, 0, 0, 4, 2, cfg, plan)
    assert result["provenance"]["n_retry"] == 7
    assert np.array_equal(
        result["rho_star_traj"][-1], scenarios._packed(plan[0].second_system)
    )


def test_20_namespace_assertion_is_called_and_not_vacuous(monkeypatch):
    cfg = replace(Config(), N=3)
    system = make_system(cfg)
    plan = make_plan(cfg, system)
    namespaces = []
    monkeypatch.setattr(
        scenarios,
        "assert_component_isolation",
        lambda key, *, namespace: namespaces.append(namespace),
        raising=True,
    )
    scenarios.scenario_trajectory(system, 0, 0, 4, 0, cfg, plan)
    assert namespaces and set(namespaces) == {"production"}
    from config import traj_seed

    with pytest.raises(rng.SeedKeyError):
        rng.assert_component_isolation(traj_seed(cfg, 0, 0).entropy, namespace="bench")


def test_20b_second_system_key_is_asserted_in_production_namespace(monkeypatch):
    cfg = replace(Config(), N=3)
    seed = rng.s3_second_key(7, 0)
    calls = []
    sampled = make_system(cfg, m=7, d=2, seed=700)

    monkeypatch.setattr(
        scenarios,
        "s3_second_key",
        lambda m, retry: calls.append(("key", m, retry)) or seed,
        raising=True,
    )
    monkeypatch.setattr(
        scenarios,
        "assert_component_isolation",
        lambda entropy, *, namespace: calls.append(("assert", entropy, namespace)),
        raising=True,
    )
    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: calls.append(("sample", generator, cfg_arg, d))
        or sampled,
        raising=True,
    )
    result = scenarios._second_system(7, 0, cfg, 2)
    assert result is sampled
    assert calls[0] == ("key", 7, 0)
    assert calls[1] == ("assert", seed.entropy, "production")
    assert calls[2][0] == "sample" and calls[2][2:] == (cfg, 2)


def unit_stream_factory(scenario_code: int, d: int, kind: str, index: int):
    return np.random.default_rng(
        np.random.SeedSequence(
            [777, scenario_code, d, 0 if kind == "system" else 1, index + 1]
        )
    )


@pytest.fixture(scope="module")
def unit_cost_result():
    return scenarios.scenario_unit_cost(Config(), 3, stream_factory=unit_stream_factory)


def test_21_unit_cost_has_all_cells_and_only_six_current_keys(unit_cost_result):
    expected = {
        "t_sys_call_s",
        "t_sys_with_outer_retry_s",
        "t_traj_s",
        "n_reject_mean",
        "n_outer_retry_mean",
        "outer_retry_status",
    }
    assert set(unit_cost_result) == {"1", "2", "3", "4", "5"}
    for by_d in unit_cost_result.values():
        assert set(by_d) == {"2", "3", "4"}
        assert all(set(cell) == expected for cell in by_d.values())


def test_21b_unit_cost_fields_have_distinct_predicates(unit_cost_result):
    for scenario_code, by_d in unit_cost_result.items():
        for cell in by_d.values():
            assert np.isfinite(cell["t_sys_call_s"]) and cell["t_sys_call_s"] > 0
            assert np.isfinite(cell["t_traj_s"]) and cell["t_traj_s"] > 0
            assert np.isfinite(cell["n_reject_mean"]) and cell["n_reject_mean"] >= 0
            assert cell["outer_retry_status"]
            if int(scenario_code) in {1, 2, 5}:
                assert cell["t_sys_with_outer_retry_s"] > 0
                assert cell["n_outer_retry_mean"] == 0.0
            else:
                assert cell["t_sys_with_outer_retry_s"] is None
                assert cell["n_outer_retry_mean"] is None
                assert "PENDING" in cell["outer_retry_status"]
                assert "s3_outer_retry_cost" in cell["outer_retry_status"]


def test_21c_unit_cost_is_json_roundtrippable(unit_cost_result):
    assert json.loads(json.dumps(unit_cost_result)) == unit_cost_result


def test_22_outer_retry_fields_reuse_or_remain_pending_merge(unit_cost_result):
    for scenario_code, by_d in unit_cost_result.items():
        for cell in by_d.values():
            if int(scenario_code) in {1, 2, 5}:
                assert cell["t_sys_with_outer_retry_s"] == cell["t_sys_call_s"]
                assert cell["n_outer_retry_mean"] == 0.0
            else:
                assert cell["t_sys_with_outer_retry_s"] is None
                assert cell["n_outer_retry_mean"] is None
                assert "PENDING" in cell["outer_retry_status"]


def test_22c_unit_cost_call_graph_has_no_seed_construction():
    tree = ast.parse(inspect.getsource(scenarios.scenario_unit_cost))
    banned = {"sys_seed", "traj_seed", "s3_second_key", "SeedSequence", "default_rng"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
    assert not (seen & banned)
    module = ast.parse(Path(scenarios.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "sys_seed" not in imported


def test_22d_injected_stream_objects_reach_real_samplers(monkeypatch):
    cfg = replace(Config(), N=4)
    registry, recorded_sys, recorded_traj, calls = [], [], [], []
    real_sys, real_traj = scenarios.sample_system, scenarios.simulate

    def factory(sc, d, kind, index):
        gen = np.random.default_rng(
            np.random.SeedSequence([808, sc, d, kind == "trajectory", index + 1])
        )
        registry.append((sc, d, kind, index, gen))
        calls.append((sc, d, kind, index))
        return gen

    def rec_sys(*args, **kwargs):
        gen = args[0] if args else kwargs["rng"]
        recorded_sys.append(gen)
        return real_sys(*args, **kwargs)

    def rec_traj(*args, **kwargs):
        gen = args[1] if len(args) > 1 else kwargs["rng"]
        recorded_traj.append(gen)
        return real_traj(*args, **kwargs)

    monkeypatch.setattr(scenarios, "sample_system", rec_sys, raising=True)
    monkeypatch.setattr(scenarios, "simulate", rec_traj, raising=True)
    scenarios.scenario_unit_cost(cfg, 1, stream_factory=factory)
    expected_sys = [row[4] for row in registry if row[2] == "system"]
    expected_traj = [row[4] for row in registry if row[2] == "trajectory"]
    assert len(registry) == 15 * 2 * 2
    assert len(recorded_sys) == len(recorded_traj) == 15 * 2
    assert all(got is expected for got, expected in zip(recorded_sys, expected_sys))
    assert all(got is expected for got, expected in zip(recorded_traj, expected_traj))
    assert {row[2] for row in registry} == {"system", "trajectory"}
    assert sum(index == -1 for *_, index in calls) == 15 * 2


def test_22e_inner_rejects_are_median_and_independent_of_outer_retry(unit_cost_result):
    assert np.__version__ == PINNED_NUMPY
    cell = unit_cost_result["1"]["2"]
    assert cell["n_reject_mean"] == 2.0
    assert cell["n_outer_retry_mean"] == 0.0


def test_22f_timing_windows_have_phase_ownership_and_exact_values(monkeypatch):
    cfg = replace(Config(), N=3)
    events, registry = [], {}
    real_sys, real_traj = scenarios.sample_system, scenarios.simulate
    clock_index = 0

    def fake_clock():
        nonlocal clock_index
        k = clock_index
        value = 1000.0 + k * (k + 1) / 2.0
        events.append(("clock", k, value))
        clock_index += 1
        return value

    def factory(sc, d, kind, index):
        gen = np.random.default_rng(
            np.random.SeedSequence([909, sc, d, kind == "trajectory", index + 1])
        )
        registry[(sc, d, kind, index)] = gen
        events.append(("stream", sc, d, kind, index, gen))
        return gen

    def rec_sys(*args, **kwargs):
        gen = args[0] if args else kwargs["rng"]
        events.append(("sys_action", args[2] if len(args) > 2 else kwargs["d"], gen))
        return real_sys(*args, **kwargs)

    def rec_traj(*args, **kwargs):
        gen = args[1] if len(args) > 1 else kwargs["rng"]
        events.append(("traj_action", args[2] if len(args) > 2 else kwargs["N"], gen))
        return real_traj(*args, **kwargs)

    monkeypatch.setattr(scenarios, "perf_counter", fake_clock, raising=True)
    monkeypatch.setattr(scenarios, "sample_system", rec_sys, raising=True)
    monkeypatch.setattr(scenarios, "simulate", rec_traj, raising=True)
    result = scenarios.scenario_unit_cost(cfg, 3, stream_factory=factory)
    assert len({id(gen) for gen in registry.values()}) == len(registry)
    actions = [event for event in events if event[0] != "stream"]
    assert len(actions) % 6 == 0
    blocks = [actions[i : i + 6] for i in range(0, len(actions), 6)]
    assert len(blocks) == 15 * 4
    expected_types = ("clock", "sys_action", "clock", "clock", "traj_action", "clock")
    owners, sys_widths, traj_widths = set(), {}, {}

    def owner(gen):
        matches = [key for key, value in registry.items() if value is gen]
        if len(matches) != 1:
            raise AssertionError("generator ownership is not unique")
        return matches[0]

    for block in blocks:
        assert tuple(event[0] for event in block) == expected_types
        clocks = [event for event in block if event[0] == "clock"]
        assert [event[1] for event in clocks] == list(
            range(clocks[0][1], clocks[0][1] + 4)
        )
        sys_owner, traj_owner = owner(block[1][2]), owner(block[4][2])
        assert sys_owner[2] == "system" and traj_owner[2] == "trajectory"
        assert (sys_owner[0], sys_owner[1], sys_owner[3]) == (
            traj_owner[0],
            traj_owner[1],
            traj_owner[3],
        )
        identity = (sys_owner[0], sys_owner[1], sys_owner[3])
        assert identity not in owners
        owners.add(identity)
        if identity[2] >= 0:
            sys_widths.setdefault(identity[:2], []).append(clocks[1][2] - clocks[0][2])
            traj_widths.setdefault(identity[:2], []).append(clocks[3][2] - clocks[2][2])
    assert owners == {
        (sc, d, i) for sc in range(1, 6) for d in (2, 3, 4) for i in (-1, 0, 1, 2)
    }
    for (sc, d), widths in sys_widths.items():
        assert result[str(sc)][str(d)]["t_sys_call_s"] == float(np.median(widths))
        assert result[str(sc)][str(d)]["t_traj_s"] == float(
            np.median(traj_widths[(sc, d)])
        )
        if sc in {1, 2, 5}:
            assert (
                result[str(sc)][str(d)]["t_sys_with_outer_retry_s"]
                == result[str(sc)][str(d)]["t_sys_call_s"]
            )


def test_22g_median_is_distinguishable_from_mean(monkeypatch):
    cfg = replace(Config(), N=2)
    clock_index = 0

    def fake_clock():
        nonlocal clock_index
        block, phase = divmod(clock_index, 4)
        sample_index = block % 4
        sys_duration = (99.0, 1.0, 1.0, 100.0)[sample_index]
        traj_duration = (199.0, 2.0, 2.0, 200.0)[sample_index]
        base = block * 1000.0
        values = (base, base + sys_duration, base + 500.0, base + 500.0 + traj_duration)
        clock_index += 1
        return values[phase]

    monkeypatch.setattr(scenarios, "perf_counter", fake_clock, raising=True)
    result = scenarios.scenario_unit_cost(cfg, 3, stream_factory=unit_stream_factory)
    for by_d in result.values():
        for cell in by_d.values():
            assert cell["t_sys_call_s"] == 1.0
            assert cell["t_traj_s"] == 2.0
            assert cell["t_sys_call_s"] != pytest.approx((1.0 + 1.0 + 100.0) / 3.0)
            assert cell["t_traj_s"] != pytest.approx((2.0 + 2.0 + 200.0) / 3.0)


def test_22h_unit_cost_executes_each_scenario_specific_trajectory_path(monkeypatch):
    cfg = replace(Config(), N=3)
    calls = {"rho": 0, "s4": 0}
    real_rho_path = scenarios._simulate_rho_path
    real_s4 = scenarios._apply_s4_transform

    def record_rho(*args, **kwargs):
        calls["rho"] += 1
        return real_rho_path(*args, **kwargs)

    def record_s4(*args, **kwargs):
        calls["s4"] += 1
        return real_s4(*args, **kwargs)

    monkeypatch.setattr(scenarios, "_simulate_rho_path", record_rho, raising=True)
    monkeypatch.setattr(scenarios, "_apply_s4_transform", record_s4, raising=True)
    scenarios.scenario_unit_cost(cfg, 1, stream_factory=unit_stream_factory)
    assert calls == {"rho": 2 * 3 * 2, "s4": 1 * 3 * 2}


def test_22i_registered_outer_retry_adapter_measures_both_s3_paths(monkeypatch):
    cfg = replace(Config(), N=3)
    systems_by_d = {d: make_system(cfg, d=d, seed=600 + d) for d in cfg.d_choices}
    system_key_calls, second_key_calls, owners = [], [], {}
    real_system_key = scenarios.bench_system_key
    real_second_key = scenarios.bench_s3_second_key
    stability_call = 0
    clock_call = 0

    def system_id_factory(sc, d, index):
        m = sc * 10_000 + d * 100 + index + 1
        owners[m] = (sc, d, index)
        return m

    def record_system_key(purpose_code, m):
        system_key_calls.append((purpose_code, m))
        return real_system_key(purpose_code, m)

    def record_second_key(purpose_code, m, retry):
        second_key_calls.append((purpose_code, m, retry))
        return real_second_key(purpose_code, m, retry)

    def stable_after_two_retries(*args):
        nonlocal stability_call
        retry = stability_call % 3
        stability_call += 1
        return retry == 2

    def fake_clock():
        nonlocal clock_call
        sample, phase = divmod(clock_call, 2)
        clock_call += 1
        return sample * 100.0 + (5.0 if phase else 0.0)

    monkeypatch.setattr(scenarios, "bench_system_key", record_system_key, raising=True)
    monkeypatch.setattr(
        scenarios, "bench_s3_second_key", record_second_key, raising=True
    )
    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: systems_by_d[d],
        raising=True,
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", stable_after_two_retries, raising=True
    )
    monkeypatch.setattr(scenarios, "perf_counter", fake_clock, raising=True)
    monkeypatch.setattr(
        scenarios,
        "s3_second_key",
        lambda *args, **kwargs: pytest.fail("production component 11 was used"),
        raising=True,
    )

    result = scenarios.s3_outer_retry_cost(
        cfg,
        1,
        purpose_code=0,
        system_id_factory=system_id_factory,
    )

    assert set(result) == {"3", "4"}
    assert len(owners) == 2 * 3 * 2
    assert len(system_key_calls) == 2 * 3 * 2
    for scenario_code, by_d in result.items():
        assert set(by_d) == {"2", "3", "4"}
        for cell in by_d.values():
            assert cell["t_sys_with_outer_retry_s"] == 5.0
            assert "bench_s3_second" in cell["outer_retry_status"]
            assert "exhausted_samples=0" in cell["outer_retry_status"]
            if scenario_code == "3":
                assert cell["n_outer_retry_mean"] == 0.0
            else:
                assert cell["n_outer_retry_mean"] == 2.0

    retries_by_m = {}
    for purpose_code, m, retry in second_key_calls:
        assert purpose_code == 0
        retries_by_m.setdefault(m, []).append(retry)
    for m, retries in retries_by_m.items():
        if owners[m][0] == 3:
            assert retries == [0]
        else:
            assert retries == [0, 1, 2]


def test_22j_outer_retry_adapter_exhausts_at_registered_retry_twenty(monkeypatch):
    cfg = replace(Config(), N=3)
    systems_by_d = {d: make_system(cfg, d=d, seed=700 + d) for d in cfg.d_choices}
    second_calls = []
    real_second_key = scenarios.bench_s3_second_key

    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: systems_by_d[d],
        raising=True,
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", lambda *args: False, raising=True
    )
    monkeypatch.setattr(scenarios, "perf_counter", lambda: 1.0, raising=True)

    def record_second_key(purpose_code, m, retry):
        second_calls.append((m, retry))
        return real_second_key(purpose_code, m, retry)

    monkeypatch.setattr(
        scenarios, "bench_s3_second_key", record_second_key, raising=True
    )
    result = scenarios.s3_outer_retry_cost(
        cfg,
        1,
        purpose_code=0,
        system_id_factory=lambda sc, d, index: sc * 10_000 + d * 100 + index + 1,
    )

    for cell in result["4"].values():
        assert cell["n_outer_retry_mean"] == 20.0
        assert "exhausted_samples=1" in cell["outer_retry_status"]
    assert max(retry for _, retry in second_calls) == 20
    assert all(retry <= 20 for _, retry in second_calls)


@pytest.mark.parametrize("bad_m", [True, -1, 1.5])
def test_22k_outer_retry_adapter_rejects_invalid_or_reused_system_ids(bad_m):
    cfg = replace(Config(), N=2)
    with pytest.raises(ValueError, match="nonnegative integers"):
        scenarios.s3_outer_retry_cost(
            cfg,
            1,
            purpose_code=0,
            system_id_factory=lambda sc, d, index: bad_m,
        )
    with pytest.raises(ValueError, match="distinct m"):
        scenarios.s3_outer_retry_cost(
            cfg,
            1,
            purpose_code=0,
            system_id_factory=lambda sc, d, index: 0,
        )


def test_22l_outer_retry_cost_median_is_distinguishable_from_mean(monkeypatch):
    cfg = replace(Config(), N=2)
    systems_by_d = {d: make_system(cfg, d=d, seed=800 + d) for d in cfg.d_choices}
    clock_call = 0
    sample_durations = (999.0, 1.0, 1.0, 100.0)

    def fake_clock():
        nonlocal clock_call
        sample, phase = divmod(clock_call, 2)
        clock_call += 1
        start = sample * 1_000.0
        duration = sample_durations[sample % len(sample_durations)]
        return start + (duration if phase else 0.0)

    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: systems_by_d[d],
        raising=True,
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", lambda *args: True, raising=True
    )
    monkeypatch.setattr(scenarios, "perf_counter", fake_clock, raising=True)

    result = scenarios.s3_outer_retry_cost(
        cfg,
        3,
        purpose_code=0,
        system_id_factory=lambda sc, d, index: sc * 10_000 + d * 100 + index + 1,
    )

    arithmetic_mean = (1.0 + 1.0 + 100.0) / 3.0
    for by_d in result.values():
        for cell in by_d.values():
            assert cell["t_sys_with_outer_retry_s"] == 1.0
            assert cell["t_sys_with_outer_retry_s"] != pytest.approx(arithmetic_mean)


def test_22m_outer_retry_timing_window_owns_first_system_sampling(monkeypatch):
    cfg = replace(Config(), N=2)
    systems_by_d = {d: make_system(cfg, d=d, seed=900 + d) for d in cfg.d_choices}
    events = []
    clock_call = 0
    real_second_key = scenarios.bench_s3_second_key

    def fake_clock():
        nonlocal clock_call
        value = float(clock_call)
        clock_call += 1
        events.append("clock")
        return value

    def record_sample(generator, cfg_arg, d):
        events.append("sample")
        return systems_by_d[d]

    def record_second_key(purpose_code, m, retry):
        events.append("second_key")
        return real_second_key(purpose_code, m, retry)

    monkeypatch.setattr(scenarios, "perf_counter", fake_clock, raising=True)
    monkeypatch.setattr(scenarios, "sample_system", record_sample, raising=True)
    monkeypatch.setattr(
        scenarios, "bench_s3_second_key", record_second_key, raising=True
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", lambda *args: True, raising=True
    )

    scenarios.s3_outer_retry_cost(
        cfg,
        1,
        purpose_code=0,
        system_id_factory=lambda sc, d, index: sc * 10_000 + d * 100 + index + 1,
    )

    sample_count = 2 * len(cfg.d_choices) * 2
    expected_block = ["clock", "sample", "second_key", "sample", "clock"]
    assert len(events) == sample_count * len(expected_block)
    for start in range(0, len(events), len(expected_block)):
        assert events[start : start + len(expected_block)] == expected_block


def test_22n_outer_retry_first_system_key_is_asserted_in_bench_namespace(monkeypatch):
    cfg = replace(Config(), N=2)
    systems_by_d = {d: make_system(cfg, d=d, seed=1_000 + d) for d in cfg.d_choices}
    isolation_calls = []
    real_assert = scenarios.assert_component_isolation

    def record_assert(entropy, *, namespace):
        isolation_calls.append((tuple(entropy), namespace))
        return real_assert(entropy, namespace=namespace)

    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: systems_by_d[d],
        raising=True,
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", lambda *args: True, raising=True
    )
    monkeypatch.setattr(scenarios, "perf_counter", lambda: 1.0, raising=True)
    monkeypatch.setattr(
        scenarios, "assert_component_isolation", record_assert, raising=True
    )

    scenarios.s3_outer_retry_cost(
        cfg,
        1,
        purpose_code=0,
        system_id_factory=lambda sc, d, index: sc * 10_000 + d * 100 + index + 1,
    )

    sample_count = 2 * len(cfg.d_choices) * 2
    first_system_calls = [call for call in isolation_calls if len(call[0]) == 5]
    assert len(first_system_calls) == sample_count
    assert all(namespace == "bench" for _, namespace in first_system_calls)


def test_22o_outer_retry_propagates_purpose_code_one_to_both_bench_keys(monkeypatch):
    cfg = replace(Config(), N=2)
    systems_by_d = {d: make_system(cfg, d=d, seed=1_100 + d) for d in cfg.d_choices}
    first_purpose_codes = []
    second_purpose_codes = []
    real_first_key = scenarios.bench_system_key
    real_second_key = scenarios.bench_s3_second_key

    def record_first_key(purpose_code, m):
        first_purpose_codes.append(purpose_code)
        return real_first_key(purpose_code, m)

    def record_second_key(purpose_code, m, retry):
        second_purpose_codes.append(purpose_code)
        return real_second_key(purpose_code, m, retry)

    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: systems_by_d[d],
        raising=True,
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", lambda *args: True, raising=True
    )
    monkeypatch.setattr(scenarios, "perf_counter", lambda: 1.0, raising=True)
    monkeypatch.setattr(scenarios, "bench_system_key", record_first_key, raising=True)
    monkeypatch.setattr(
        scenarios, "bench_s3_second_key", record_second_key, raising=True
    )

    scenarios.s3_outer_retry_cost(
        cfg,
        1,
        purpose_code=1,
        system_id_factory=lambda sc, d, index: sc * 10_000 + d * 100 + index + 1,
    )

    sample_count = 2 * len(cfg.d_choices) * 2
    assert len(first_purpose_codes) == sample_count
    assert set(first_purpose_codes) == {1}
    assert len(second_purpose_codes) == sample_count
    assert set(second_purpose_codes) == {1}


def test_22p_outer_retry_count_median_is_distinguishable_from_mean(monkeypatch):
    cfg = replace(Config(), N=2)
    systems_by_d = {d: make_system(cfg, d=d, seed=1_200 + d) for d in cfg.d_choices}
    owners = {}
    current_second = None
    real_second_key = scenarios.bench_s3_second_key

    def system_id_factory(sc, d, index):
        m = sc * 10_000 + d * 100 + index + 1
        owners[m] = (sc, d, index)
        return m

    def record_second_key(purpose_code, m, retry):
        nonlocal current_second
        current_second = (m, retry)
        return real_second_key(purpose_code, m, retry)

    def stable_at_target_retry(*args):
        assert current_second is not None
        m, retry = current_second
        scenario_code, _, index = owners[m]
        assert scenario_code == scenarios.SCENARIO_CODE["S3-drift"]
        target_retry = 6 if index == 2 else 0
        return retry == target_retry

    monkeypatch.setattr(
        scenarios,
        "sample_system",
        lambda generator, cfg_arg, d: systems_by_d[d],
        raising=True,
    )
    monkeypatch.setattr(
        scenarios, "bench_s3_second_key", record_second_key, raising=True
    )
    monkeypatch.setattr(
        scenarios, "_rho_path_is_stable", stable_at_target_retry, raising=True
    )
    monkeypatch.setattr(scenarios, "perf_counter", lambda: 1.0, raising=True)

    result = scenarios.s3_outer_retry_cost(
        cfg,
        3,
        purpose_code=0,
        system_id_factory=system_id_factory,
    )

    arithmetic_mean = (0.0 + 0.0 + 6.0) / 3.0
    for cell in result["4"].values():
        assert cell["n_outer_retry_mean"] == 0.0
        assert cell["n_outer_retry_mean"] != pytest.approx(arithmetic_mean)


def test_23_pilot_nonconstruction_is_visible(monkeypatch):
    cfg = Config()
    family = make_family(cfg)
    monkeypatch.setattr(
        scenarios,
        "_second_system",
        lambda m, retry, cfg, d: make_system(cfg, m, d, 17000 + m),
    )
    monkeypatch.setattr(scenarios, "_rho_path_is_stable", lambda *args: True)
    pilot = scenarios.build_s3_drift_family(family, cfg)["audit_by_split"]["pilot"]
    assert pilot["status"] == "not applicable"
    assert pilot["reason"] and "pilot" in pilot["reason"]


@pytest.mark.parametrize(
    ("scenario_code", "condition_code"),
    ((1, 0), (2, 0), (3, 0), (4, 0), (5, 3)),
)
def test_24_rho_truth_path_shape_initial_and_terminal(
    monkeypatch, scenario_code, condition_code
):
    cfg = replace(Config(), N=3000 if scenario_code == 3 else 20)
    system = make_system(cfg, m=0, d=2, seed=314)
    rho_1 = pack_rho(system["A"], system["B"], system["f"], int(system["d"]))
    kwargs = {}
    rho_2 = None
    if scenario_code == 3:
        second = make_system(cfg, m=0, d=2, seed=315)
        monkeypatch.setattr(scenarios, "_second_system", lambda *args: second)
        rho_2 = pack_rho(second["A"], second["B"], second["f"], 2)
    elif scenario_code == 4:
        plan = make_plan(cfg, system, seed=315)
        kwargs["drift_plan"] = plan
        rho_2 = pack_rho(
            plan[0].second_system["A"],
            plan[0].second_system["B"],
            plan[0].second_system["f"],
            2,
        )
    result = scenarios.scenario_trajectory(
        system, 0, 0, scenario_code, condition_code, cfg, **kwargs
    )
    rho = result["rho_star_traj"]
    assert rho.shape == (cfg.N + 1, rho_dim(cfg.n, int(system["d"])))
    assert np.array_equal(rho[0], rho_1)
    if scenario_code in {1, 2, 5}:
        assert np.array_equal(rho, np.broadcast_to(rho_1, rho.shape))
    elif scenario_code == 3:
        assert np.array_equal(rho[1500], rho_1)
        assert np.array_equal(rho[1501], rho_2)
    else:
        severity = scenarios._DRIFT_SEVERITY[condition_code]
        assert np.array_equal(rho[cfg.N], (1.0 - severity) * rho_1 + severity * rho_2)


@pytest.mark.parametrize("d", (2, 3, 4))
def test_25_rho_shape_is_neither_state_shape_nor_parameter_count(d):
    cfg = replace(Config(), N=8)
    system = make_system(cfg, m=0, d=d, seed=400 + d)
    result = scenarios.scenario_trajectory(system, 0, 0, 1, 0, cfg)
    rho = result["rho_star_traj"]
    assert rho.shape[0] != result["x"].shape[0]
    assert rho.shape[1] == 8
    assert rho.shape[1] != cfg.n


def test_26_business_seed_object_is_the_one_used_by_simulate(monkeypatch):
    cfg = replace(Config(), N=6)
    system = make_system(cfg, m=0, d=2, seed=501)
    real_default_rng = np.random.default_rng
    real_simulate = scenarios.simulate
    real_traj_seed = scenarios.traj_seed
    ss_registry, records, recorded_by_mj = {}, [], {}
    current = {"mj": None}

    def seed_stub(cfg_arg, m, j):
        seed = real_traj_seed(cfg_arg, m, j)
        ss_registry[(m, j)] = seed
        current["mj"] = (m, j)
        return seed

    def default_rng_probe(seed=None):
        gen = real_default_rng(seed)
        records.append({"i": len(records), "seed": seed, "gen": gen})
        return gen

    def rec_simulate(*args, **kwargs):
        gen = args[1] if len(args) > 1 else kwargs["rng"]
        mj = current["mj"]
        if mj is None or mj in recorded_by_mj:
            raise AssertionError(
                "simulate call cannot be uniquely attributed to a business seed"
            )
        recorded_by_mj[mj] = gen
        return real_simulate(*args, **kwargs)

    monkeypatch.setattr(scenarios, "traj_seed", seed_stub, raising=True)
    monkeypatch.setattr(np.random, "default_rng", default_rng_probe, raising=True)
    monkeypatch.setattr(scenarios, "simulate", rec_simulate, raising=True)
    for m, j in ((0, 0), (0, 1), (1, 0)):
        system_m = dict(system, m=m)
        scenarios.scenario_trajectory(system_m, m, j, 1, 0, cfg)

    selected = []
    for mj, seed in ss_registry.items():
        matches = [record for record in records if record["seed"] is seed]
        if len(matches) != 1:
            raise AssertionError(
                f"seed object for {mj} mapped to {len(matches)} generators"
            )
        selected.append(matches[0])
        assert recorded_by_mj[mj] is matches[0]["gen"]
    assert len(selected) == 3
    assert [record["i"] for record in selected] == sorted(
        record["i"] for record in selected
    )
    foreign = [
        record
        for record in records
        if all(record["seed"] is not seed for seed in ss_registry.values())
    ]
    assert all(
        all(record["seed"] is not seed for seed in ss_registry.values())
        for record in foreign
    )


def test_26c_provenance_minimum_keys_and_outlier_indices_are_bound_to_v(monkeypatch):
    assert np.__version__ == PINNED_NUMPY
    cfg = Config()
    system = make_system(cfg, m=0, d=2, seed=502)
    result = scenarios.scenario_trajectory(system, 0, 0, 5, 0, cfg)
    provenance = result["provenance"]
    required = {
        "seed_key",
        "n_retry",
        "jump_magnitude",
        "s",
        "epsilon",
        "outlier_index",
        "scenario_code",
        "condition_code",
    }
    assert required <= set(provenance)
    assert provenance["seed_key"] == scenarios.traj_seed(cfg, 0, 0).entropy

    generator = np.random.default_rng(scenarios.traj_seed(cfg, 0, 0))
    base = scenarios.simulate(system, generator, cfg.N)
    branch_uniform = generator.random(cfg.N)
    expected_index = np.flatnonzero(branch_uniform < 0.05) + 1
    assert np.array_equal(provenance["outlier_index"], expected_index)
    assert np.all((expected_index >= 1) & (expected_index <= cfg.N))
    assert np.array_equal(
        result["v"][expected_index],
        base["v"][expected_index] * 10.0 / np.sqrt(1.0 + 0.05 * (10.0**2 - 1.0)),
    )


def test_26b_business_seed_has_two_pinned_golden_prefixes(monkeypatch):
    assert np.__version__ == PINNED_NUMPY
    cfg = replace(Config(), N=6)
    system = make_system(cfg, m=0, d=2, seed=501)
    original = scenarios.scenario_trajectory(system, 0, 0, 1, 0, cfg)["y"][:4]
    real_seed = scenarios.traj_seed

    def replacement(cfg_arg, m, j):
        if (m, j) == (0, 0):
            return np.random.SeedSequence([cfg_arg.seed_base, 1, 0, 1])
        return real_seed(cfg_arg, m, j)

    monkeypatch.setattr(scenarios, "traj_seed", replacement, raising=True)
    changed = scenarios.scenario_trajectory(system, 0, 0, 1, 0, cfg)["y"][:4]
    assert np.array_equal(
        original,
        np.array([0.0, -0.3048579977044566, 1.390145435590712, 0.17581715150034002]),
    )
    assert np.array_equal(
        changed,
        np.array([0.0, 0.04822040817054322, 0.5803850099443779, 0.9836861122930944]),
    )
    assert not np.array_equal(original, changed)
