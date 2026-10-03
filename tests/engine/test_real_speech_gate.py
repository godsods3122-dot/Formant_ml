"""**실제 발화**로 초기 포락을 재는 문지기 (docs/MEASUREMENTS §52.102).

오늘 두 번, 합성 트랙 시험이 전부 통과하는데 실제 발화에서 렌더가 무너졌다.
 · Q 상자를 **무성 구간**에도 걸어 초기 포락이 27.5 % → −6.8 %
 · Rd 잔차에서 유도한 `tilt` 을 켜서 같은 붕괴
합성 트랙에는 무성 구간이 거의 없고 잔차 구조도 없어서 시험이 못 잡았다. 그래서 코퍼스의 실제
발화 하나로 **분석 → 초기 손실**까지 가고, 그 값이 기준 아래로 떨어지면 실패로 만든다.

기준값은 절대 성능이 아니라 **회귀 감지선**이다 — 지금 27.5 % 근처이므로 20 % 를 바닥으로 둔다.
"""
import os

import numpy as np
import pytest
import torch

WAV = "wavs/yang_00000101.wav"
FLOOR = 20.0
SEG = 68160


@pytest.mark.skipif(not os.path.exists(WAV), reason="코퍼스가 없는 환경")
def test_initial_envelope_on_real_speech_is_not_broken():
    import soundfile as sf

    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.fit import CopySynthFitter
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine

    prof = SpeakerProfile.load("profiles/yang_female.json")
    x, fs = sf.read(WAV)
    x = np.asarray(x, dtype=float)[:SEG]
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False,
                                   speaker="female"), prof)
    tr = analyze(x, fs, prof, 48, t0=0.0, full=x)
    f = CopySynthFitter(eng, x, 48000, tr)
    with torch.no_grad():
        _loss, _sc, env_sc, _per = f.loss()
    env = 100.0 * (1.0 - float(env_sc))
    assert env > FLOOR, (
        f"실제 발화의 초기 포락이 {env:.2f} % 로 바닥({FLOOR}) 아래다 — "
        "분석기나 렌더 경로에 회귀가 있다 (§52.102 의 두 사례를 보라)")


@pytest.mark.skipif(not os.path.exists(WAV), reason="코퍼스가 없는 환경")
def test_voiced_only_constraints_do_not_touch_unvoiced():
    """유성에서 잰 제약(`Q_BAND`)이 **무성 구간의 제어값을 바꾸지 않아야** 한다."""
    import soundfile as sf

    from formant_ml.engine import fit as F
    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.fit import CopySynthFitter
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine

    prof = SpeakerProfile.load("profiles/yang_female.json")
    x, fs = sf.read(WAV)
    x = np.asarray(x, dtype=float)[:SEG]
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False,
                                   speaker="female"), prof)
    tr = analyze(x, fs, prof, 48, t0=0.0, full=x)
    f = CopySynthFitter(eng, x, 48000, tr)
    if not F.Q_BAND:
        pytest.skip("Q_BAND 가 비어 있으면 볼 것이 없다")
    v = np.asarray(getattr(tr, "voiced", np.zeros(0, dtype=bool)), bool)
    with torch.no_grad():
        c_on = f.control()[0]
        keep, F.Q_BAND = F.Q_BAND, {}
        try:
            f._qb = None                      # 캐시를 비운다
            c_off = f.control()[0]
        finally:
            F.Q_BAND = keep
            f._qb = None
    T = c_on.shape[0]
    if v.size < T or v[:T].all():
        pytest.skip("무성 프레임이 없다")
    un = torch.as_tensor(~v[:T])
    for nm in ("bw1", "bw2", "bw3", "bw4"):
        if nm not in f.names:
            continue
        j = f.names.index(nm)
        a, b = c_on[un, j], c_off[un, j]
        rel = float(((a - b).abs() / b.abs().clamp_min(1e-6)).max())
        assert rel < 1e-4, (nm, rel)          # 무성 구간은 Q 상자가 건드리면 안 된다
