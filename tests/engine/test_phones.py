"""음소 제스처 + 프로파일 — 구조적 성질 (무성성, 레벨, 정점 주파수, 길이)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import VoiceEngine, EngineConfig, phones
from formant_ml.engine.profile import SpeakerProfile


@pytest.fixture(scope="module")
def prof():
    return SpeakerProfile.load("profiles/yang_female.json")


@pytest.fixture(scope="module")
def eng(prof):
    return VoiceEngine(EngineConfig(n_extra_formants=2), profile=prof)


def _bands(y, fs=48000):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / fs); tot = Y.sum()
    return (lambda lo, hi: 10 * np.log10(Y[(f >= lo) & (f < hi)].sum() / tot + 1e-20)), (f * Y).sum() / tot


def test_profile_roundtrip_and_front_length(prof):
    p2 = SpeakerProfile.from_json(prof.to_json())
    assert p2.vowels == prof.vowels and p2.name == "yang_female"
    assert abs(prof.sib_front_len_cm - 35000 / (4 * 9500)) < 1e-9


def test_sibilant_peak_follows_profile_and_is_voiceless(eng, prof):
    y = eng.render(phones.sa(prof))
    fs = 48000
    s = y[int(0.07 * fs):int(0.14 * fs)]
    band, cent = _bands(s)
    assert 6500 < cent < 13000                     # 프로파일 정점 9.5 kHz 근처
    assert band(0, 1000) < -25                     # 유성 성분 없음
    assert band(8000, 12000) > band(2000, 4000) + 10


def test_affricate_is_voiceless_and_has_abrupt_onset(eng, prof):
    y = eng.render(phones.ja(prof))
    fs = 48000
    e = np.array([20 * np.log10(np.sqrt((y[i:i + 240] ** 2).mean()) + 1e-9)
                  for i in range(0, len(y) - 240, 240)])
    # 폐쇄(0.05~0.11 s) 동안 조용하고, 해제 후 10 ms 안에 20 dB 이상 선다
    closure = e[int(0.07 * 200):int(0.10 * 200)].max()
    after = e[int(0.112 * 200):int(0.125 * 200)].max()
    assert after > closure + 15
    s = y[int(0.115 * fs):int(0.17 * fs)]
    band, cent = _bands(s)
    assert band(0, 1000) < -15 and cent > 3000


def test_nasal_murmur_level_and_spectrum(eng, prof):
    y = eng.render(phones.na(prof))
    fs = 48000
    m = y[int(0.06 * fs):int(0.10 * fs)]; v = y[int(0.20 * fs):int(0.30 * fs)]
    lvl = 20 * np.log10(np.sqrt((m ** 2).mean()) / np.sqrt((v ** 2).mean()))
    assert -12 < lvl < -2                          # 계측 −5..−10 dB
    band, _ = _bands(m)
    assert band(2000, 4000) < -25 and band(1000, 2000) > -45


def test_tap_dip_matches_profile_within_tolerance(eng, prof):
    y = eng.render(phones.ara(prof))
    e = np.array([20 * np.log10(np.sqrt((y[i:i + 480] ** 2).mean()) + 1e-9)
                  for i in range(0, len(y) - 480, 480)])
    dip = e[8:15].mean() - e[17:25].min()
    assert abs(dip - abs(prof.tap["dip_db"])) < 5.0


def test_phrase_has_expected_syllable_count_and_duration(prof):
    tr = phones.irinilssirirae(prof)
    assert 0.85 < tr.seconds() < 1.1               # 실측 0.83 s + 무음
    a_c = tr["a_c"]
    assert (a_c < 0.3).any() and (tr["lat_z1"] > 0).any() and (tr["velum"] > 0.5).any()


def test_voiced_affricate_turns_on_voicing_and_weakens_frication():
    """유성 ㅈ 은 **성문 자세 하나**로 갈려야 한다 — 따로 만든 소리가 아니다.

    내전만 0.07 -> 0.55 로 바꾸면 직렬 오리피스 모형이 나머지를 낸다:
    성문 면적이 작아 유량이 묶이고 협착부 속도가 떨어져 마찰이 약해지며,
    성대가 계속 울리므로 저역에 발성 바가 선다. 이 결합이 끊기면 유성 저해음의
    공기역학(Ohala)이 모형에서 사라진 것이다.
    """
    from formant_ml.engine.control import INDEX
    prof = SpeakerProfile.load("profiles/yang_female.json")
    eng = VoiceEngine(EngineConfig(seed=17), profile=prof)
    fs = 48000.0

    def fric_bands(fn):
        tr = fn(prof)
        with torch.no_grad():
            y = eng.render(tr)
        ac = tr.values[:, INDEX["a_c"]]
        hop = int(round(tr.frame_ms * fs / 1000))
        sel = (ac < 0.7) & (ac > 1e-3)
        i0 = int(np.argmax(sel))
        i1 = len(sel) - int(np.argmax(sel[::-1]))
        s = y[i0 * hop:i1 * hop]
        Y = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        f = np.fft.rfftfreq(len(s), 1 / fs)
        tot = Y.sum() + 1e-20
        return (10 * np.log10(Y[f < 300].sum() / tot + 1e-20),
                10 * np.log10(Y[(f >= 4000) & (f < 12000)].sum() / tot + 1e-20))

    lo_u, hi_u = fric_bands(phones.ja)
    lo_v, hi_v = fric_bands(phones.aja)
    assert lo_v - lo_u > 15.0        # 발성 바가 선다
    assert hi_u - hi_v > 8.0         # 마찰이 약해진다
