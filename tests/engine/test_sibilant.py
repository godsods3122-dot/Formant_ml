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


def _render(spec, prof, dur=0.400, t0=0.180):
    import torch
    from formant_ml.engine.control import track_from_keyframes
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    torch.set_num_threads(2)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female",
                                   residual=False), prof)
    kf = sb.gesture_keyframes(spec, prof, dur=dur, t0=t0)
    return eng.render(track_from_keyframes(kf, frame_ms=1.0))


def test_no_voicing_before_a_voiceless_sibilant():
    """개시 전에 성문이 열렸다 **다시 닫히면** 안 된다 — 그러면 목소리가 난다.

    실측으로 잡은 회귀: 옛 키프레임이 남아 adduction 이 0.06 -> 0.6 으로 되돌아갔고,
    선행 구간이 치찰음 고원보다 3.7 dB **더 컸다**.
    """
    from formant_ml.engine.control import INDEX, track_from_keyframes
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile.load("profiles/yang_female.json")
    spec = sb.from_profile("s", prof)
    trk = track_from_keyframes(sb.gesture_keyframes(spec, prof, dur=0.400, t0=0.180),
                               frame_ms=1.0)
    add = trk.values[100:170, INDEX["adduction"]]
    assert add.max() < 0.25, f"개시 전 성문이 다시 닫힌다 (adduction {add.max():.2f})"

    y = _render(spec, prof)
    fs = 48000
    pre = y[int(0.100 * fs):int(0.170 * fs)]
    plateau = y[int(0.267 * fs):int(0.510 * fs)]
    d = 20 * np.log10((np.sqrt((pre ** 2).mean()) + 1e-12)
                      / (np.sqrt((plateau ** 2).mean()) + 1e-12))
    assert d < -15.0, f"개시 전 소리가 고원보다 {d:+.1f} dB (−15 아래여야 한다)"


def test_fade_in_grows_with_duration():
    """페이드 인은 길이에 비례해야 한다 (§14.3). 절대값으로 박으면 안 된다."""
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile.load("profiles/yang_female.json")
    spec = sb.from_profile("s", prof)
    fs = 48000.0

    def t6(x, lo, hi):
        n = len(x)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / fs)
        X[(f < lo) | (f >= hi)] = 0
        e = np.abs(np.fft.irfft(X, n) + 1j * np.fft.irfft(-1j * X, n))
        k = int(0.005 * fs)
        e = np.convolve(e, np.ones(k) / k, "same")
        return int(np.argmax(e > e.max() * 10 ** (-6 / 20))) / fs * 1000

    lags = []
    for dur in (0.130, 0.400, 0.600):
        y = _render(spec, prof, dur=dur)
        hi = max(t6(y, 7000, 11000), t6(y, 11000, 16000))
        mid = min(t6(y, 2000, 4000), t6(y, 4000, 7000))
        lags.append(hi - mid)
    assert lags[0] < lags[1] < lags[2], f"길이에 안 비례한다: {lags}"
    assert lags[2] > 60.0, f"긴 /s/ 의 페이드 인이 {lags[2]:.0f} ms 로 짧다"


def test_plateau_is_a_sibilant():
    """고원부가 치찰음 지문을 낸다 — 프로파일 실측 봉우리 근처."""
    from formant_ml.engine import turbulence as tb
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile.load("profiles/yang_female.json")
    y = _render(sb.from_profile("s", prof), prof)
    plateau = y[int(0.267 * 48000):int(0.510 * 48000)]
    c = tb.compare(plateau, plateau, 48000.0)["target"]
    assert c["centroid"] > 8000.0, c["centroid"]
    assert abs(c["peak"] - prof.sibilant["peak_hz"]) < 1500.0, c["peak"]
    assert c["sibilance"] > 25.0, c["sibilance"]
