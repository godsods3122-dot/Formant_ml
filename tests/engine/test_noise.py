"""노이즈 생성기 — 레이놀즈 게이트, 스펙트럼, 템플릿 변조."""
import numpy as np
import pytest
import torch

from formant_ml.engine.noise import (FricationNoise, TransientTemplateBank, reynolds,
                                     series_flow, RE_CRIT)
from formant_ml.engine.control import PARAM_NAMES, INDEX, default_vector

FS, HOP = 44100, 44


def _ctrl(t, **kw):
    v = np.tile(default_vector(), (t, 1))
    for k, x in kw.items():
        v[:, INDEX[k]] = x
    ten = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    return {n: ten[..., INDEX[n]] for n in PARAM_NAMES}


def test_series_flow_reproduces_v1_pressure_split():
    ps = torch.tensor([8.0]); ag = torch.tensor([0.46]); ac = torch.tensor([0.10])
    u, dp_c = series_flow(ps, ag, ac)
    from formant_ml.engine.glottis import CMH2O
    po_over_ps = dp_c / (ps * CMH2O)
    expect = ag ** 2 / (ag ** 2 + ac ** 2)                # Po/Ps = Ag²/(Ag²+Ac²)
    assert abs(float(po_over_ps - expect)) < 1e-3
    re, _, _ = reynolds(u, ac)
    assert 5000 < float(re) < 12000                       # /s/ 급 협착


def test_vowel_posture_is_below_turbulence_threshold():
    u, _ = series_flow(torch.tensor([7.5]), torch.tensor([0.07]), torch.tensor([3.0]))
    re, _, _ = reynolds(u, torch.tensor([3.0]))
    assert float(re) < RE_CRIT


def test_frication_silent_open_and_loud_closed():
    n = FricationNoise(FS, HOP)
    N = 200 * HOP
    ph = torch.zeros(1, N); vo = torch.zeros(1, N)
    with torch.no_grad():
        o = n(_ctrl(200, p_sub=7.5, a_c=3.0), torch.full((1, 200), 0.07), ph, vo)   # 모달 성문 + 모음
        c = n(_ctrl(200, p_sub=7.5, a_c=0.10), torch.full((1, 200), 0.46), ph, vo)  # 벌린 성문 + /s/
    assert float(o["source"].abs().max()) < 1e-6
    assert float(c["source"].pow(2).mean()) > 0
    y = c["source"][0].numpy()
    f = np.fft.rfftfreq(len(y), 1 / FS); Y = np.abs(np.fft.rfft(y)) ** 2
    # 소스는 광대역이다(정점은 앞공동이 만든다). 저역이 죽지 않았는지, 그리고 위쪽 절벽이
    # 있는지만 본다 — 대역통과로 두었을 때 1~2 kHz 가 실측보다 15 dB 낮았다(ADR 0011).
    band = lambda a, b: Y[(f > a) & (f < b)].sum()
    assert band(1000, 2000) / band(4000, 8000) > 0.05
    assert band(18000, 24000) / band(4000, 8000) < 3.0


def test_template_bank_modulation_and_growth():
    bank = TransientTemplateBank(FS)
    n = FS // 10
    a = bank.render([dict(t=0.01, name="tongue_contact", amp=1.0)], n)
    b = bank.render([dict(t=0.02, name="tongue_contact", amp=1.0)], n)[:, 441:]
    a = a[:, :n - 441]
    assert float(a.abs().max()) > 0.5
    assert not torch.allclose(a, b)                        # 같은 소리가 두 번 나지 않는다
    i = bank.add("custom_click", np.hanning(200))
    c = bank.render([dict(t=0.02, id=i, rate=2.0)], n)
    assert float(c.abs().sum()) > 0 and bank.bank.shape[0] == 6


def test_butterworth_q_is_staggered_not_repeated():
    """2n 차 버터워스의 Q 는 절편마다 다르다. 같은 Q 반복은 무릎을 뭉갠다."""
    from formant_ml.engine.noise import butterworth_q
    assert butterworth_q(1) == [pytest.approx(0.7071, abs=1e-3)]
    q2 = butterworth_q(2)
    assert q2[0] == pytest.approx(0.5412, abs=1e-3)
    assert q2[1] == pytest.approx(1.3066, abs=1e-3)
    assert butterworth_q(4)[-1] == pytest.approx(2.5629, abs=1e-3)


def _sib_bands(front_bw_slope=None, front_len=1.46, n=300):
    """치찰음 자세를 렌더하고 2 kHz 대역 레벨(dB)을 돌려준다."""
    import numpy as np
    from formant_ml.engine.control import ControlTrack, default_vector
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    prof = SpeakerProfile()
    if front_bw_slope is not None:
        prof.sibilant = dict(prof.sibilant, front_bw_slope=front_bw_slope)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False), prof)
    v = np.tile(default_vector(), (n, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 8.0; tr["adduction"] = 0.05; tr["a_c"] = 0.06; tr["c_place"] = 0.95
    tr["front_len"] = front_len; tr["obstacle"] = 0.15; tr["fric_gain"] = 8.0
    tr["f1"] = 600; tr["f2"] = 1400; tr["f3"] = 2500; tr["f4"] = 3500
    tr["residual_mix"] = 0.0
    y = eng.render(tr.clamp())[9600:]
    f = np.fft.rfftfreq(len(y), 1 / 48000)
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return lambda lo, hi: 10 * np.log10(p[(f >= lo) & (f < hi)].mean() + 1e-20)


def test_frication_has_a_shoulder_then_a_slope_then_a_cliff():
    """마찰 스펙트럼의 모양: 정점 아래 어깨 → 완만한 내리막 → 극단 고역의 절벽.

    **내리막을 절벽으로 만들면 안 된다.** 소스만 보고 lp_ratio 를 맞췄더니(0.3.2)
    앞공동 극이 그 위에 −12 dB/oct 를 또 얹어 출력의 12~16 kHz 가 실측보다 17~27 dB
    어두워졌다. 통과대역의 기울기는 소스 절벽이 아니라 **앞공동 극의 대역폭**이 정한다.
    """
    band = _sib_bands()
    peak = band(5000, 8000)
    assert band(4000, 6000) > peak - 6.0            # 어깨 (실측 정점 −2.8 dB)
    assert peak - 20.0 < band(12000, 16000) < peak - 4.0     # 벽이 아니라 내리막
    # 계속 내려가야 한다. 고쳐야 했던 실패는 고역이 **다시 올라오는** 것이었다.
    assert band(18000, 22000) < band(12000, 16000)
    assert band(18000, 22000) < peak - 8.0


def test_front_cavity_bandwidth_controls_the_high_frequency_slope():
    """앞공동 극의 대역폭(프로파일)이 고역 기울기를 단조로 정한다.

    한 상수로 두면 두 화자가 반대로 잡아당긴다 — 남 /ㅅ/(평음)은 고역이 적고
    여 /ㅆ/(경음)은 많다. 실측 적합값은 남 0.08, 여 0.36 이다.
    """
    slopes = [0.08, 0.20, 0.36]
    drop = []
    for sl in slopes:
        band = _sib_bands(front_bw_slope=sl)
        drop.append(band(5000, 8000) - band(12000, 16000))
    assert drop[0] > drop[1] > drop[2], drop      # 넓을수록 고역이 덜 떨어진다
    assert drop[0] - drop[2] > 3.0                # 쓸모 있는 범위여야 한다


def test_profile_supplies_front_bandwidth_to_the_tract():
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    prof = SpeakerProfile()
    prof.sibilant = dict(prof.sibilant, front_bw_slope=0.33)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False), prof)
    assert abs(eng.tract.front_bw_slope - 0.33) < 1e-9
