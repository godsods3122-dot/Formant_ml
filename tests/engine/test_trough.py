"""하인두 넓은 골 (`profile.trough_hz`, `tract.TRO_*`) — 7~8.5 kHz 를 메우던 원인 (MEASUREMENTS §52.231).

코퍼스 216 파일에서 **두 특징이 따로** 나온다: 추세를 뺀 국소 골 6117 Hz(깊이 9 dB, = 이상와 §52.76)와
생 스펙트럼의 넓은 골 8062 Hz(깊이 14 dB, −6 dB 너비 2145 Hz). 엔진에는 앞의 것만 있어 뒤엣것을 못 만들었다.
"""
import math

import numpy as np
import torch

from formant_ml.engine import tract as T
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.tviir import notch_coeffs
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000.0


def _resp(f0, bw, ratio, fr):
    b0, b1, b2, a1, a2 = (float(v) for v in notch_coeffs(
        torch.tensor(f0, dtype=torch.float64), bw, FS, ratio))
    z = np.exp(-1j * 2 * np.pi * fr / FS)
    return 20 * np.log10(np.abs((b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)) + 1e-30)


def test_the_constants_reproduce_the_measured_shape():
    """실측 중앙 (깊이 14 dB, −6 dB 너비 2145 Hz) 을 기본값이 내야 한다."""
    fr = np.arange(3000.0, 12001.0, 25.0)
    H = _resp(8062.0, T.TRO_BW, T.TRO_RATIO0, fr)
    j = int(np.argmin(H))
    depth = -H[j]
    lim = H[j] + 6.0
    a, b = j, j
    while a > 0 and H[a] < lim:
        a -= 1
    while b < len(H) - 1 and H[b] < lim:
        b += 1
    assert abs(depth - 14.0) < 2.0, depth
    assert abs((fr[b] - fr[a]) - 2145.0) < 400.0, fr[b] - fr[a]


def test_the_old_spec_could_not_make_it():
    """왜 새 구조가 필요한가: 예전 이상와 규격(bw 500)은 같은 깊이에서 너비가 1/2 도 안 된다."""
    fr = np.arange(3000.0, 12001.0, 25.0)
    H = _resp(8062.0, T.PIR_BW, T.PIR_RATIO_MAX, fr)
    j = int(np.argmin(H))
    lim = H[j] + 6.0
    a, b = j, j
    while a > 0 and H[a] < lim:
        a -= 1
    while b < len(H) - 1 and H[b] < lim:
        b += 1
    assert (fr[b] - fr[a]) < 1100.0, fr[b] - fr[a]


def test_profile_value_reaches_the_tract():
    prof = SpeakerProfile.load("profiles/yang_female.json")
    assert prof.trough_hz > 7000.0
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False,
                                   speaker="female"), prof)
    assert eng.tract.tro_f0 == prof.trough_hz
    assert eng.tract.pir_f0 == prof.piriform_hz        # 두 특징은 따로 남는다
    assert abs(eng.tract.tro_f0 - eng.tract.pir_f0) > 1000.0


def test_the_initial_depth_is_the_measured_median():
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False, speaker="female"),
                      SpeakerProfile.load("profiles/yang_female.json"))
    r = 1.0 + (T.TRO_RATIO_MAX - 1.0) * float(torch.sigmoid(eng.tract.tro_depth.detach()))
    assert abs(r - T.TRO_RATIO0) < 1e-4, r


def test_the_fit_range_covers_the_measured_quartiles():
    """적합 범위가 실측 25~75 % 분위(6955~9381 Hz)를 덮어야 한다 — §52.76 이 겪은 '범위 끝에 박힘' 을 막는다."""
    lo = 8062.0 * math.exp(-T.TRO_DF_LIM)
    hi = 8062.0 * math.exp(+T.TRO_DF_LIM)
    assert lo <= 6955.0 and hi >= 9381.0, (lo, hi)


def test_zero_profile_value_turns_it_off():
    prof = SpeakerProfile.load("profiles/yang_female.json")
    prof.trough_hz = 0.0
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False,
                                   speaker="female"), prof)
    assert eng.tract.tro_f0 == 0.0
