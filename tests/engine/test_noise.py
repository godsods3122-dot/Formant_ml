"""노이즈 생성기 — 레이놀즈 게이트, 스펙트럼, 템플릿 변조."""
import numpy as np
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
    hi = Y[(f > 13000)].sum() / Y[(f > 2000) & (f < 6000)].sum()
    assert hi < 0.3                                        # 소스는 13 kHz 위에서 절벽


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
