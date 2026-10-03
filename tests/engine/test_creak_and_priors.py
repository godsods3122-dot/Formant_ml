"""불규칙 발성(creak) 펄스·적합기 전역 스칼라 저장·내전 수준 사전 (MEASUREMENTS §52.439·441·442)."""
import importlib.util
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from formant_ml.engine import analyze as A
from formant_ml.engine import fit as F
from formant_ml.engine.profile import SpeakerProfile

FS = 48000
ROOT = Path(__file__).resolve().parents[2]


def _vowel_from_pulses(times, amps, n):
    x = np.zeros(n)
    for t, a in zip(times, amps):
        x[int(t * FS)] += a
    for fc in (150.0, 150.0):                                   # 성문 기울기 −12 dB/oct
        al = math.exp(-2 * math.pi * fc / FS)
        x = lfilter([1 - al], [1, -al], x)
    v = np.zeros(n)
    for fc, bw in ((700.0, 90.0), (1200.0, 110.0), (2600.0, 160.0), (3600.0, 200.0)):
        rr = math.exp(-math.pi * bw / FS)
        v += lfilter([1.0], [1, -2 * rr * math.cos(2 * math.pi * fc / FS), rr * rr], x)
    return v / np.abs(v).max() * 0.3


@pytest.fixture(scope="module")
def prof():
    return SpeakerProfile.load(str(ROOT / "profiles" / "yang_female.json"))


def _creak_signal(seed):
    rng = np.random.default_rng(seed)
    m1 = np.arange(0.05, 0.20, 1 / 330)
    tc = [m1[-1]]
    for d in (4.4, 3.0, 4.1, 3.6, 4.4, 4.5, 6.0, 3.8, 2.9, 4.8, 2.6, 4.9, 3.6, 4.3, 3.8, 6.5):   # C2 0.40–0.47 s 의 실측 간격
        tc.append(tc[-1] + d / 1000)
    cr = np.array(tc[1:])
    m2 = np.arange(cr[-1] + 1 / 300, cr[-1] + 0.15, 1 / 300)
    n = int(0.55 * FS)
    amps = np.concatenate([np.ones(m1.size),
                           [(1.0 if k % 2 == 0 else 0.45) * (1 + 0.2 * rng.standard_normal()) for k in range(cr.size)],
                           np.ones(m2.size)])
    v = _vowel_from_pulses(np.concatenate([m1, cr, m2]), amps, n) + rng.standard_normal(n) * 3e-4
    return v, m1, cr


@pytest.mark.parametrize("seed", [3, 4])
def test_creak_pulses_recover_irregular_closures(prof, seed):
    """Praat 이 놓치는 불규칙 폐쇄를 ±0.5 ms 안에서 대부분 되찾고 오경보가 거의 없다."""
    v, m1, cr = _creak_signal(seed)
    pp = A.glottal_pulses(v, FS, prof)
    in_gap = ((pp > cr[0] - 0.002) & (pp < cr[-1] + 0.002)).sum()
    assert in_gap <= 4, "시험 신호가 너무 쉽다 — Praat 이 creak 을 대부분 잡았다"
    ex, filled = A.creak_pulses(v, FS, pp)
    assert len(filled) == 1 and filled[0][3] == "creak"
    off = np.median([pp[np.argmin(np.abs(pp - t))] - t for t in m1[5:-5]])      # Praat 의 주기 안 기준점
    allp = np.sort(np.concatenate([pp, ex]))
    err = np.array([np.min(np.abs(allp - (t + off))) for t in cr])
    assert (err < 0.0005).sum() >= 13
    false = [e for e in ex if np.min(np.abs(cr + off - e)) > 0.0008]
    assert len(false) <= 1


def test_voiceless_stop_gap_is_not_filled(prof):
    """무성 파열(폐쇄 + 개방 버스트 + 기식)이 든 틈은 creak 으로 채우지 않는다 — 격음을 유성으로 만들면 안 된다."""
    rng = np.random.default_rng(7)
    n = int(0.5 * FS)
    m1 = np.arange(0.05, 0.20, 1 / 300)
    m2 = np.arange(0.27, 0.45, 1 / 300)
    v = _vowel_from_pulses(np.concatenate([m1, m2]), np.ones(m1.size + m2.size), n)
    r0 = int(0.235 * FS)
    v[r0:r0 + 240] += rng.standard_normal(240) * 0.15                         # 개방 버스트 5 ms
    from scipy.signal import butter, sosfiltfilt
    asp = sosfiltfilt(butter(4, [1500 / 24000, 8000 / 24000], "bandpass", output="sos"), rng.standard_normal(n)) * 0.04
    v[r0 + 240:int(0.27 * FS)] += asp[r0 + 240:int(0.27 * FS)]               # 기식
    v += rng.standard_normal(n) * 3e-4
    A.BURST_BACK_PCT = 90.0
    try:
        pp = A.glottal_pulses(v, FS, prof)
        ex, filled = A.creak_pulses(v, FS, pp)
    finally:
        A.BURST_BACK_PCT = 100.0
    assert not any(a < 0.235 < b for a, b, _, _m in filled)


def test_pulse_gate_uses_local_spacing_with_creak():
    """creak 을 켜면 게이트의 끊김 판정이 국소 간격으로 — 불규칙 6.5 ms 가 전역 중앙값의 1.8 배를 넘어도 끊기지 않는다."""
    p = np.concatenate([np.arange(0.05, 0.20, 0.0028), 0.2 + np.cumsum([4.4, 6.0, 3.8, 6.5, 4.3, 5.8]) / 1000,
                        0.234 + np.arange(0.0028, 0.15, 0.0028)])
    n = int(0.45 * FS)
    old = A.CREAK_PULSES
    try:
        A.CREAK_PULSES = False
        g_old = A.voicing_gate_from_pulses(p, n, FS)
        A.CREAK_PULSES = True
        g_new = A.voicing_gate_from_pulses(p, n, FS)
    finally:
        A.CREAK_PULSES = old
    mid = slice(int(0.205 * FS), int(0.23 * FS))
    assert g_old[mid].min() < 0.5          # 옛 판정은 creak 안에서 끊긴다
    assert g_new[mid].min() == 1.0


def _load_copyfit():
    spec = importlib.util.spec_from_file_location("copyfit_mod", ROOT / "scripts" / "copyfit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_scalars_restore_from_npz_and_legacy_log(tmp_path):
    from formant_ml.engine import pipeline as cf          # copyfit 에서 engine.pipeline 으로 옮겼다 (§52.470)
    fit = types.SimpleNamespace(vb_log_g=torch.zeros(1), rv_log_rt=torch.zeros(1), rv_g_db=torch.zeros(1))
    stem = str(tmp_path / "run")
    np.savez(stem + "_track.npz", fit_scalars=np.array('{"vb_log_g": 0.5, "rv_log_rt": -1.2, "rv_g_db": -1.4}'))
    with np.load(stem + "_track.npz") as z:
        msg = cf.restore_fit_scalars(fit, z, stem)
    assert "npz" in msg and float(fit.vb_log_g) == pytest.approx(0.5) and float(fit.rv_log_rt) == pytest.approx(-1.2)
    # 옛 판: npz 에 필드가 없으면 로그의 마지막 진행 줄에서 읽는다
    np.savez(stem + "_track.npz", values=np.zeros(1))
    Path(stem + ".log").write_text("    [ 100] 포락  90.00%  보이스바 +6.0dB  잔향 RT 330ms -11.5dB  손실 1\n", encoding="utf-8")
    with np.load(stem + "_track.npz") as z:
        msg = cf.restore_fit_scalars(fit, z, stem)
    assert "로그" in msg
    assert float(fit.vb_log_g) == pytest.approx(6.0 * math.log(10) / 20)
    assert float(torch.exp(fit.rv_log_rt)) == pytest.approx(0.33)
    assert float(fit.rv_g_db) == pytest.approx(-11.5 * math.log(10) / 20)


def test_add_prior_penalises_breathy_voiced_frames_only():
    """유성 틀에서 내전 0.13(숨 섞인)은 0.44(코퍼스 중앙)보다 비싸고, 무성 틀은 사전을 받지 않는다."""
    from formant_ml.engine.control import INDEX, PARAM_NAMES
    T = 20
    stub = types.SimpleNamespace(track=types.SimpleNamespace(voiced=np.r_[np.ones(10, bool), np.zeros(10, bool)]))

    def loss(add_voiced, add_unvoiced):
        stub._ap_mask = None
        c = torch.zeros(1, T, len(PARAM_NAMES), dtype=torch.float64)
        c[0, :10, INDEX["adduction"]] = add_voiced
        c[0, 10:, INDEX["adduction"]] = add_unvoiced
        return float(F.CopySynthFitter.add_prior_loss(stub, c))

    assert loss(0.44, 0.1) == pytest.approx(0.0, abs=1e-3)
    assert loss(0.13, 0.1) > 3.0
    assert loss(0.44, 0.02) == pytest.approx(loss(0.44, 0.9), abs=1e-9)      # 무성 틀은 무관
    assert loss(0.7, 0.1) < loss(0.13, 0.1)                                    # creak 쪽 꼬리는 숨 섞인 쪽보다 싸다


def test_voiced_closure_gets_pulses_voiceless_does_not(prof):
    """모음 사이 유성 폐쇄(보이스 바) 는 펄스를 되찾고, 같은 자리의 무성 폐쇄는 비워 둔다 (§52.443)."""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(11)
    n = int(0.5 * FS)
    T = 1 / 300
    v1 = np.arange(0.05, 0.20, T)
    cl = np.arange(v1[-1] + T, 0.24, T)                                        # 폐쇄 중에도 떤다
    v2 = np.arange(0.2415, 0.45, T)
    vow = _vowel_from_pulses(np.concatenate([v1, v2]), np.ones(v1.size + v2.size), n)
    xb = np.zeros(n)
    for t in cl:
        xb[int(t * FS)] = 1.0
    bar = sosfiltfilt(butter(2, [150 / 24000, 400 / 24000], "bandpass", output="sos"), xb)
    bar = bar / np.abs(bar).max() * 0.3 * 10 ** (-18 / 20)                   # 보이스 바: 모음보다 18 dB 작은 저역
    burst = np.zeros(n)
    r0 = int(0.2405 * FS)
    burst[r0:r0 + 200] = rng.standard_normal(200) * 0.08
    floor = rng.standard_normal(n) * 3e-5
    A.BURST_BACK_PCT = 90.0
    try:
        voiced = vow + bar + burst + floor
        pp = A.glottal_pulses(voiced, FS, prof)
        ex, filled = A.creak_pulses(voiced, FS, pp)
        allp = np.sort(np.concatenate([pp, ex]))
        in_cl = allp[(allp > 0.205) & (allp < 0.238)]
        assert in_cl.size >= 7, f"유성 폐쇄에 펄스가 {in_cl.size} 개뿐"
        assert np.all(np.diff(in_cl) > 0.7 * T)
        vless = vow + burst + floor                                            # 같은 자리, 폐쇄가 조용하다
        pp2 = A.glottal_pulses(vless, FS, prof)
        ex2, _ = A.creak_pulses(vless, FS, pp2)
        # Praat 자신은 모음 끝 울림에 점 한두 개를 찍는다 — 여기서 보는 것은 **더한** 펄스다
        assert ((ex2 > 0.205) & (ex2 < 0.238)).sum() == 0
    finally:
        A.BURST_BACK_PCT = 100.0


def test_sibilant_frames_and_prior():
    """치찰 틀 관측은 고역 잡음(/ㅅ/)만 잡고 기식(1–4 kHz)·모음은 안 잡는다. 사전은 치찰 틀의 넓은 협착만 비싸게 한다 (§52.446)."""
    from scipy.signal import butter, sosfiltfilt
    from formant_ml.engine.control import INDEX, PARAM_NAMES
    rng = np.random.default_rng(5)
    n = int(0.6 * FS)
    v = _vowel_from_pulses(np.arange(0.05, 0.20, 1 / 280), np.ones(len(np.arange(0.05, 0.20, 1 / 280))), n)
    s0, s1 = int(0.25 * FS), int(0.35 * FS)
    v[s0:s1] += sosfiltfilt(butter(4, [5000 / 24000, 11000 / 24000], "bandpass", output="sos"), rng.standard_normal(s1 - s0)) * 0.08
    h0, h1 = int(0.42 * FS), int(0.50 * FS)
    v[h0:h1] += sosfiltfilt(butter(4, [800 / 24000, 3500 / 24000], "bandpass", output="sos"), rng.standard_normal(h1 - h0)) * 0.05
    v += rng.standard_normal(n) * 3e-5
    nf = n // 48
    voi = np.zeros(nf, bool)
    voi[50:200] = True
    sib = A.sibilant_frames(v, FS, nf, 48, voi)
    assert sib[260:340].mean() > 0.9, "치찰을 못 잡았다"
    assert not sib[430:490].any(), "기식을 치찰로 잡았다"
    assert not sib[60:190].any(), "모음을 치찰로 잡았다"
    stub = types.SimpleNamespace(track=types.SimpleNamespace(sibilant=sib))

    def loss(ac):
        stub._sp_mask = None
        c = torch.zeros(1, nf, len(PARAM_NAMES), dtype=torch.float64)
        c[0, :, INDEX["a_c"]] = ac
        return float(F.CopySynthFitter.sib_prior_loss(stub, c))

    assert loss(0.1) == pytest.approx(0.0, abs=1e-6)
    assert loss(2.5) > 5.0 * loss(0.2)            # 모음처럼 연 협착(기식 지름길)은 크게 비싸다
