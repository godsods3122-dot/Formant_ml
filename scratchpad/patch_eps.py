p="src/formant_ml/engine/voice_td.py"; s=open(p,encoding="utf-8").read()
a='''def vf_ns_eps(f0: torch.Tensor) -> torch.Tensor:
    """목표 f0 [Hz] → 늘어남 ε (모드 3 의 f0(ε) 표를 log f0 에 대해 조각 직선으로 뒤집는다, 양 끝 밖은 끝 조각으로 늘인다)."""
    lf = torch.log(torch.tensor(VF_NS_F0_TAB, dtype=f0.dtype, device=f0.device))
    ee = torch.tensor(VF_NS_EPS_TAB, dtype=f0.dtype, device=f0.device)
    x = torch.log(f0.clamp(60.0, 900.0))
    i = torch.searchsorted(lf, x.detach().contiguous()).clamp(1, lf.numel() - 1)
    x0, x1, y0, y1 = lf[i - 1], lf[i], ee[i - 1], ee[i]
    return (y0 + (x - x0) * (y1 - y0) / (x1 - x0)).clamp(-0.3, 0.7)'''
b='''def _pchip_slopes(x, y):
    """Fritsch–Carlson 단조 3 차 보간의 마디 기울기 (numpy)."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = np.diff(x)
    dl = np.diff(y) / h
    m = np.zeros_like(y)
    m[0], m[-1] = dl[0], dl[-1]
    for k in range(1, len(y) - 1):
        if dl[k - 1] * dl[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k])
    return m


def vf_ns_eps(f0: torch.Tensor) -> torch.Tensor:
    """목표 f0 [Hz] → 늘어남 ε — 모드 3 의 f0(ε) 표를 log f0 에 대해 **단조 3 차(PCHIP)** 로 뒤집는다 (C¹, §52.524; 예전 조각 직선은 마디에서
    꺾였다). 양 끝 밖은 끝 기울기로 곧게 늘인다."""
    import math as _m
    lfl = [_m.log(v) for v in VF_NS_F0_TAB]
    mk = _pchip_slopes(lfl, VF_NS_EPS_TAB)
    lf = torch.tensor(lfl, dtype=f0.dtype, device=f0.device)
    ee = torch.tensor(VF_NS_EPS_TAB, dtype=f0.dtype, device=f0.device)
    mm = torch.tensor(mk, dtype=f0.dtype, device=f0.device)
    x = torch.log(f0.clamp(60.0, 900.0))
    i = torch.searchsorted(lf, x.detach().contiguous()).clamp(1, lf.numel() - 1)
    x0, x1, y0, y1, m0, m1 = lf[i - 1], lf[i], ee[i - 1], ee[i], mm[i - 1], mm[i]
    hh = x1 - x0
    t = (x - x0) / hh
    inside = (t >= 0) & (t <= 1)
    tc = t.clamp(0.0, 1.0)
    h00, h10, h01, h11 = 2 * tc ** 3 - 3 * tc ** 2 + 1, tc ** 3 - 2 * tc ** 2 + tc, -2 * tc ** 3 + 3 * tc ** 2, tc ** 3 - tc ** 2
    yin = h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1
    ylo = ee[0] + mm[0] * (x - lf[0])
    yhi = ee[-1] + mm[-1] * (x - lf[-1])
    y = torch.where(inside, yin, torch.where(x < lf[0], ylo, yhi))
    return y.clamp(-0.3, 0.7)'''
assert s.count(a)==1; s=s.replace(a,b,1); open(p,"w",encoding="utf-8",newline="\n").write(s)
p="src/formant_ml/engine/fit.py"; s=open(p,encoding="utf-8").read()
a='''                       td_log_ag_max=_voice_mod.TD_TUBE and __import__("formant_ml.engine.voice_td", fromlist=["x"]).GLOTTIS != "vf",'''
b='''                       td_log_ag_max=_voice_mod.TD_TUBE and (__import__("formant_ml.engine.voice_td", fromlist=["x"]).GLOTTIS != "vf"
                                                             or __import__("formant_ml.engine.voice_td", fromlist=["x"]).VF_BODY_DRIVE),   # 몸체 진폭이 쓴다 (§52.524)'''
assert s.count(a)==1; s=s.replace(a,b,1); open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
