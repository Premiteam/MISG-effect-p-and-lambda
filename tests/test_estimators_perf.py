from __future__ import annotations

import hashlib
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "core"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config
from estimators import misg_run
from systems import pack_rho, simulate


_DEFAULT_DIGESTS = {
    (0, "true", 1): (
        "a6a2d463d83fa2c1912897d7e43ffe74da90a3437cc6c9443ac415b6e779da36",
        "ef925ffa7d1fd5ac63fb2a3592d7c12cc77fbbb6e0d5f5557578f4ca11da5919",
    ),
    (0, "true", 5): (
        "376f9fb48ac613af4dece56352f6b824e398f12aabd7500c7a83101f0d861c06",
        "ef925ffa7d1fd5ac63fb2a3592d7c12cc77fbbb6e0d5f5557578f4ca11da5919",
    ),
    (0, "observer", 1): (
        "a4a49fd2b2c513c93d77dc6d1f55f970ff8dd8dd3132cd3ab5d47904a8a8aff5",
        "ad55d1078a6bfc1a6144608992ed7914039f92c9b068d5c6e8e5d7b2fcebd674",
    ),
    (0, "observer", 5): (
        "fa048c0dc1719851a28a53a68242aa3ad559711af495917dd743cfaca28d8d2c",
        "5e6c1ab9adfa589e4f35c936d9d72dfcb3298f0a1f4b6a71d63ef9c18b16394e",
    ),
    (0, "kalman", 1): (
        "8a17047660d9bb512a5f8bb327d4ee20226320f69b209f7b720c8db84ac4ddd1",
        "d4c418c430ff6dcab7ae0df07c0c14c50ad6042d0174d5b403a9832ef6f0631e",
    ),
    (0, "kalman", 5): (
        "3c800d6bbbebfc4caec7754fe7c481b0a57067319665215196575c51628aaf72",
        "4b046e48a493691f7e9428e2e1b6d06379ba4e62a84ada88337b9dd0c8e5d83b",
    ),
    (1, "true", 1): (
        "e300754aa3071bc21f4f316069a0523ee9f3513adbb61448baf227e1b206c2a8",
        "798cd524cbd0ad68fe6b4f701bfea105944ecbd1b10881c9baa77a751f6ec6a6",
    ),
    (1, "true", 5): (
        "938433b31f48f68467b76f8952be9bb6a848287cd203ef6283bf0adb918c99be",
        "798cd524cbd0ad68fe6b4f701bfea105944ecbd1b10881c9baa77a751f6ec6a6",
    ),
    (1, "observer", 1): (
        "fb7a689c51f2eb756388ad9981dbfe0c4c80820a388dfee5120fa58c4644e01d",
        "c21f93b2571355f862d62f34e9d07f44012719ee40314b8e963579193b6779f8",
    ),
    (1, "observer", 5): (
        "6be7edd92a10f38a8316eacec0e4040262abc89c571b503fd1a82f438729daed",
        "7f49a5ce237f1e5f0e6965ddca92cdd6d1e537ec3a20517351b20f8bab4e6bff",
    ),
    (1, "kalman", 1): (
        "2f11a2f260c4dcb3fa6d001ab21bc50987f776885d525ef7cf853378e6284ae4",
        "73912066654fc578da0755abb0c204ff290003279d88d6295d418dbe646eb353",
    ),
    (1, "kalman", 5): (
        "c56a3d6ad650e8c0db7484b729fa0368de247161378106d293267633aa98e37d",
        "d618d787eb974ee0af52cdfd9bf9d389da5396b9ad3927b544984c1ed332179c",
    ),
    (2, "true", 1): (
        "45786c440c4e17c2d237a5624a932aabd2c9ecb1a71bbff0c1ec6ec21cab0efe",
        "a189893610a8278adcef6479509156ddc39dba603a6f79071162e9aa07d3817d",
    ),
    (2, "true", 5): (
        "d85f87e53935f313ce39d1511ea8d2233408c8797e1b8b6c6c5386ea889117d0",
        "a189893610a8278adcef6479509156ddc39dba603a6f79071162e9aa07d3817d",
    ),
    (2, "observer", 1): (
        "428fe499b91c70f196a84f0a171b6a1ddc29ee11cc6e90f1c6b473102dd1bc1a",
        "51a413a0e0693447de9d647267a4df6ec7b4a8d9c920ba6108faf6823f89089d",
    ),
    (2, "observer", 5): (
        "0a78050bbdbaf1aba712a6ca8c324b1cace39fc65cf3f0f60bfa445fbd622f88",
        "82e41acbf88bbd8014d7f1199b3251013fdad6f3b72b8f8b618aa6e6a35ee2b2",
    ),
    (2, "kalman", 1): (
        "4d9c83b44f52f7d54f1ba75ca8df9f09b2059553f803c5db85891350e43f0c8b",
        "dd2290a558c15abf3d32aed8596b68c4bd9c7233b21429fd64ad733c56b0d185",
    ),
    (2, "kalman", 5): (
        "284d49aa0cf375aa94d5aba1c981cbeec2e89c9ac76b2cb37dd5469ecf952852",
        "eefab7e14c749c6b2c52fb11eba6aeab3bcb9bf43c0a43a8528a17959b1b3eae",
    ),
    (3, "true", 1): (
        "cbeca88a12bf24214f0ff1ec71be171493dbfb96065788a456431a32ca172a50",
        "bbd269628911e90041607eabc7129fd4b3c33aec86af46a856582bf580015ca6",
    ),
    (3, "true", 5): (
        "c68137bdd07b69c75f6c8fe48f9a39dcc2e6e3cf7a0be8d97986faf38b7f4ffe",
        "bbd269628911e90041607eabc7129fd4b3c33aec86af46a856582bf580015ca6",
    ),
    (3, "observer", 1): (
        "7b66636f03056806a9e0bbdbde12e161e62e8b77c3028b7e4d6f6eb537de2e9f",
        "0bc196e8111017c6dcabd24642c1c4c2a062487fe4ba38fb9811c82e809458d7",
    ),
    (3, "observer", 5): (
        "c5349f07715d857de64d851cbe314602ecd147bd9e72084d516cda8707ec474e",
        "c44ea4413e347814b7071f516a54ce7114df6aa8dff70fbfc32268c8cbb9636a",
    ),
    (3, "kalman", 1): (
        "71d87ba2c44b658e0f7b531eb00c938d1bc613e2110e72ecd6de0f6c43570c18",
        "27ac0ae53cf5f420d2c437f56e92a312cc293a8640b27bf2b040b649fb8d54b3",
    ),
    (3, "kalman", 5): (
        "aa8fd260d83bbfc080e654f9d135cdcd83f7fde5b6358577593b04f365bb0766",
        "acd3bdcc465c3459e714b7a20fab69c284b9946ead56167cf4cda70b4cc4c8fe",
    ),
    (4, "true", 1): (
        "5223358f03ff43947a713228e97b8334c47f0d50d475851d276592c0d67eb6b3",
        "60ea3c1164d35fb8cb7d18abc8eee82cd9d9b1159d3b69769a6e79cf3befcbe0",
    ),
    (4, "true", 5): (
        "5cc828648f200c13ae77d13020a0da3be266cb9c5950be2567429ecf92277942",
        "60ea3c1164d35fb8cb7d18abc8eee82cd9d9b1159d3b69769a6e79cf3befcbe0",
    ),
    (4, "observer", 1): (
        "9f2ad833929eae40c5470e980b5bac581ecb3fff1b452a1453217f8090eddd5e",
        "d1773c439a0614ae25d3a56389f31cde010c147a3c6e91f0acf110f4d2fb0d1d",
    ),
    (4, "observer", 5): (
        "a5336ec8e55f71bc22db623225bbe76a656dcd641127e5b1a72822ead0178d68",
        "340b471dfbc1179aaf0a454486f3a950bfbe3231c7cec669d5be3f7e466c56b1",
    ),
    (4, "kalman", 1): (
        "cc4e1023ed6170d415770c3393836c80f093320581836e2d7e3f0af532faa79b",
        "236af8baeb3295456d72785cb3b092ac1ea50019f309f985749c67a0047d816d",
    ),
    (4, "kalman", 5): (
        "d18277b9e9953c042fae52240c4e80196d46a8a434215974f061b1395e81389c",
        "0c14556f1e98cfbd3ffe1c6b2610dcb04731367bc8da242e6c25e2b709ecc5bf",
    ),
    (5, "true", 1): (
        "f7ef940b713d2b34b21f9e01cb4b4f9d1c51dff5c8f60c57ba9ce4b40d7f3f54",
        "8ed36f74c3bf68eec7e14e02f21b47879bbd25d1b83f04345310aa02cf10a6b3",
    ),
    (5, "true", 5): (
        "3c4510a871792d56eed1b43abbc41c0f0e97ec2a97d34bfd5af6c9675b12880a",
        "8ed36f74c3bf68eec7e14e02f21b47879bbd25d1b83f04345310aa02cf10a6b3",
    ),
    (5, "observer", 1): (
        "c53de489e18a16b5d4cdf9306f77ddd54c93d76c8dedd42ad02d4bb58c90e9aa",
        "92ee59ae72a9bb54826c09cb3cc9f954e82b8d039b231283f0a9ff2cca703d9b",
    ),
    (5, "observer", 5): (
        "135c140aa2bc6b5dde8a93ae6b0396347506904a1dbd12229f787b9599d4eb39",
        "ce31b9aa1f5b126e256829c5b6acfc7da265596d1d8a48509b468fcd04eedf56",
    ),
    (5, "kalman", 1): (
        "cca4557114bd9246ec374c4f289977291be87805aaa3a065f1003a8c46d3d475",
        "3feae2238cbc640229aef9327bc32c7b2f37a763ce1bdf8a88f1f02827062606",
    ),
    (5, "kalman", 5): (
        "4f02c05b637bd5c2d5fee7764884ce78aa5d904cf2c727548f7537170b3ffa4f",
        "10bc9d4ff0e5882a3568d3e3f052ab9f2f0796ad98f999899b29890613c2f518",
    ),
}

_DIVERGED_RHO_DIGESTS = {
    "nan": "9245015a836af1cdc00bb505139477535e73f82171c804abb3b949031f7b4cbe",
    "norm": "4f43790a2a411d238a7fdc8246361442cca6df2f768892477ed9087d26e05e6d",
    "spec": "a41b28f50e21a9c6b11a013ebca4e2f378bfe2dc3bf3d17dd43d3a37d61b5d4e",
}
_EARLY_DIVERGENCE_X_DIGESTS = {
    "true": "01f0d5f915ddeaf17d41b382c421640e3e0af1fd5ca509ecff5a16abd3fd2417",
    "observer": "14d0e5d034ad4732eecc118458d32252da616337ac65b87191b1b2524da8d039",
    "kalman": "14d0e5d034ad4732eecc118458d32252da616337ac65b87191b1b2524da8d039",
}
_SPEC_X_DIGESTS = {
    "true": (
        "b98278a107e02908c8e4c285eaf7d025d5a6844ba514b84539c2bbe05ce6f522",
        "06475315ea6402c1284d6008b59aed2eb27e4ed6be688bfe162b9ad526b7404c",
        "15fa8821f394123cdf98274c132c3fb75295daf52617992bf51e3d8b3828ba7c",
        "efb7da89b1534041e6d6a54b4b93fd8b056ee144e18cdeb80feee0a43d1abd24",
        "c2ba06cd75c042993951ce20d1bd1136f843e06e65bb26589c85187ecf92f1a6",
        "001da785db7708e3718c7e9735fa7716d08ca24be7df9a00f54bd5e3fa28d212",
    ),
    "observer": ("a1bfa637db79556481859b5e6fa353006fad586a96a127105564fadb01e3e000",)
    * 6,
    "kalman": (
        "e40d2a9590d85dc142edbbccc10baa743c8293e7e671c2de07be4870d7037d34",
        "751eb0257ce83f00c4fec232f9bf875f375035b4b9d1b14f3f38b93161f65373",
        "8f91b30a2bc0fa64dfc66994cf30461bfe6864be6adce02d7928fd89fb7e3b81",
        "885a7d8e5fc5f4447c79ef4771effea7562681393dd3ba7e77aaf1c248c683e5",
        "b545639a9abcb45c3aa5e9dd83bf8e0e1b75fa123bab8fe79ac03dea551c603d",
        "7dfc5c8a66347a58b2cfb1ed9d30a3792d5c62a6ababe39a3daf3e5abe1f9f6f",
    ),
}


def _example_system() -> dict[str, Any]:
    A = np.array([[0.0, 1.0], [-0.01, -0.22]])
    B = np.array([[0.19, -0.08], [0.16, -0.12]])
    f = np.array([0.8, -0.8])
    return {
        "A": A,
        "B": B,
        "f": f,
        "d": 2,
        "n": 2,
        "sigma_v": 0.3,
        "rho_star": pack_rho(A, B, f, 2),
    }


def _array_digest(array: np.ndarray) -> str:
    canonical = np.array(array, dtype="<f8", order="C", copy=True)
    canonical[np.isnan(canonical)] = np.nan
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _rho_initialization(kind: str) -> tuple[np.ndarray | None, bool]:
    if kind == "default":
        return None, False
    if kind == "nan":
        return np.full(8, np.nan), True
    if kind == "norm":
        return np.full(8, 300.0), True
    if kind == "spec":
        return np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), True
    raise AssertionError(f"unknown initialization kind: {kind}")


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("state_mode", ("true", "observer", "kalman"))
@pytest.mark.parametrize("p", (1, 5))
@pytest.mark.parametrize("initialization", ("default", "nan", "norm", "spec"))
def test_output_is_bitwise_identical_to_pre_fix(
    seed: int, state_mode: str, p: int, initialization: str
) -> None:
    cfg = Config()
    system = _example_system()
    traj = simulate(system, np.random.default_rng(seed), 400)
    rho_hat0, freeze_rho = _rho_initialization(initialization)
    result = misg_run(
        system,
        traj,
        np.ones(p),
        1.0,
        "cumulative",
        state_mode,
        cfg,
        r_window="newest",
        rho_hat0=rho_hat0,
        freeze_rho=freeze_rho,
    )

    if initialization == "default":
        expected_rho, expected_x = _DEFAULT_DIGESTS[(seed, state_mode, p)]
        expected_status = "ok"
        expected_at = None
        expected_reason = None
    else:
        expected_rho = _DIVERGED_RHO_DIGESTS[initialization]
        if initialization == "spec":
            expected_x = _SPEC_X_DIGESTS[state_mode][seed]
            expected_at = 50
        else:
            expected_x = _EARLY_DIVERGENCE_X_DIGESTS[state_mode]
            expected_at = 1
        expected_status = "diverged"
        expected_reason = initialization

    assert _array_digest(result["rho_hat_traj"]) == expected_rho
    assert _array_digest(result["x_hat_traj"]) == expected_x
    assert result["status"] == expected_status
    assert result["diverged_at"] == expected_at
    assert result["div_reason"] == expected_reason


def _rollout_timings(N: int, repeats: int = 3) -> list[float]:
    cfg = Config()
    system = _example_system()
    traj = simulate(system, np.random.default_rng(2468), N)
    arguments = {
        "sys": system,
        "traj": traj,
        "w": np.ones(1),
        "gamma": 1.0,
        "r_mode": "cumulative",
        "state_mode": "true",
        "cfg": cfg,
        "r_window": "newest",
        "freeze_rho": True,
    }
    warmup = misg_run(**arguments)
    assert warmup["status"] == "ok"

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = misg_run(**arguments)
        timings.append(time.perf_counter() - start)
        assert result["status"] == "ok"
    return timings


def test_rollout_complexity_is_not_quadratic() -> None:
    short = min(_rollout_timings(400, repeats=5))
    long = min(_rollout_timings(1600, repeats=5))
    assert long < 8.0 * short, (short, long, long / short)


def test_kalman_p5_n3000_rollout_is_not_quadratically_slow() -> None:
    cfg = Config()
    system = _example_system()
    traj = simulate(system, np.random.default_rng(12345), 3000)

    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = misg_run(
            system,
            traj,
            np.ones(5),
            1.0,
            "cumulative",
            "kalman",
            cfg,
        )
        timings.append(time.perf_counter() - start)
        assert result["status"] == "ok"

    assert min(timings) < 3.0, timings
