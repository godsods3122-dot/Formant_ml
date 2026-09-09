"""치찰음 전용 경로의 성질 — docs/MEASUREMENTS.md §19.6, §27."""
import numpy as np
import pytest
import torch

from formant_ml.engine import sibilant as sb
from formant_ml.engine.control import INDEX


def test_wall_source_has_no_dipole_structurally():
    """`wall` 은 다이폴 몫이 **구조적으로 0** 이다 — 거리 함수로 작게가 아니다."""
    for name in ("f", "h", "whisper"):
        spec = sb.PRESETS[name]
        v = sb.dipole_share(spec, torch.tensor([[0.05]]), 3000.0)
        assert float(v.abs().max()) == 0.0


def test_wall_spec_rejects_obstacle_coupling():
    with pytest.raises(ValueError):
        sb.SibilantSpec("bad", 0.1, "wall", 0.1, 1.0, 0.15)


def test_voiced_sibilant_loses_dipole_through_the_glottis():
    """유성은 성문이 좁아 제트 속도가 베르누이 상한에서 내려온다 (§19.2).

    새 함수 없이 **성문 자세 하나로** 다이폴 몫이 준다.
    """
    s, z = sb.PRESETS["s"], sb.PRESETS["z"]
    v_uv = sb.jet_state(s, p_sub=8.0, a_g=0.20)["jet_v"]
    v_v = sb.jet_state(z, p_sub=8.0, a_g=0.04)["jet_v"]
    assert v_v < 0.6 * v_uv
    d_uv = float(sb.dipole_share(s, torch.tensor(s.a_target), v_uv))
    d_v = float(sb.dipole_share(z, torch.tensor(z.a_target), v_v))
    assert d_v < 0.5 * d_uv


def test_place_sets_the_downstream_cavity():
    """앞공동은 위치가 정한다. 성문 협착이면 성도 전체가 하류다."""
    assert sb.front_len_cm(sb.PRESETS["s"]) < 2.0
    assert sb.front_len_cm(sb.PRESETS["whisper"]) > 13.0
    # /s/ 의 앞공동 봉우리는 치찰음 대역에, 속삭임은 포먼트 대역에 있다
    assert sb.front_peak_hz(sb.PRESETS["s"]) > 5000.0
    assert sb.front_peak_hz(sb.PRESETS["whisper"]) < 1000.0


def test_glottal_constriction_routes_to_the_whole_tract():
    assert sb.PRESETS["whisper"].back_leak == 1.0
    assert sb.PRESETS["s"].back_leak <= 0.15


def test_control_values_feed_the_shared_engine_knobs():
    c = sb.control_values(sb.PRESETS["s"], p_sub=8.0, a_g=0.20)
    for k in ("a_c", "c_place", "front_len", "obstacle", "back_leak"):
        assert k in c
    assert c["obstacle"] > 0.0
    assert sb.control_values(sb.PRESETS["whisper"], 6.0, 0.06)["obstacle"] == 0.0


def test_accept_gate_catches_a_dead_dipole():
    """§27 의 고장 — 다이폴이 3.8 배 꺼졌는데 스펙트럼은 맞던 경우."""
    ok, why = sb.accept(dict(centroid_err=-78.0, peak_err=-94.0, obst_eff=0.00473),
                        baseline_obst_eff=0.01793)
    assert not ok and any("다이폴" in w for w in why)
    ok, _ = sb.accept(dict(centroid_err=-30.0, peak_err=0.0, obst_eff=0.01793),
                      baseline_obst_eff=0.01793)
    assert ok


def test_accept_gate_catches_the_collapsed_peak():
    """ss00 이 sigma 4.5 에서 봉우리 11344 -> 2484 Hz 로 무너진 경우."""
    ok, why = sb.accept(dict(centroid_err=-347.0, peak_err=-8859.0, obst_eff=0.01))
    assert not ok and any("봉우리" in w for w in why)


def test_obst_eff_from_track_matches_the_measured_regression():
    """§27.2 의 표를 재현한다 — 협착 바닥이 오르면 2.5 제곱으로 죽는다."""
    n = 60
    v = np.zeros((n, len(INDEX)))
    v[:, INDEX["obstacle"]] = 0.0989
    v[:, INDEX["a_c"]] = 0.30
    good = sb.obst_eff_from_track(v.copy(), INDEX)
    v2 = v.copy(); v2[:, INDEX["a_c"]] = 0.45
    worse = sb.obst_eff_from_track(v2, INDEX)
    assert worse < 0.4 * good


def test_place_comes_from_the_measured_peak():
    """위치를 상수로 박지 않고 프로파일 실측 봉우리에서 역산한다."""
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile()
    prof.sibilant["peak_hz"] = 8000.0
    s = sb.from_profile("s", prof)
    assert sb.front_peak_hz(s) == pytest.approx(8000.0, rel=0.02)
    prof2 = SpeakerProfile()
    prof2.sibilant["peak_hz"] = 6000.0                # 앞공동이 긴 화자
    assert sb.from_profile("s", prof2).place < s.place


def test_wall_sources_keep_their_anatomy():
    """`wall` 은 앞공동 봉우리가 정하는 양이 아니다 — 프로파일이 위치를 안 옮긴다."""
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile()
    prof.sibilant["peak_hz"] = 6000.0
    for nm in ("whisper", "h", "f"):
        assert sb.from_profile(nm, prof) is sb.PRESETS[nm]
