"""외부 성문 음원 갈아 끼우기 (`VoiceEngine.set_external_source`, docs/MEASUREMENTS §52.31)."""
import numpy as np
import pytest

from formant_ml.engine import EngineConfig, VoiceEngine, phones


def _render_capturing_du(eng, track):
    got = {}
    orig = eng.glottis.forward

    def wrap(*args, **kwargs):
        out = orig(*args, **kwargs)
        got["du"] = out["du"].detach().clone()
        return out

    eng.glottis.forward = wrap
    try:
        y = eng.render(track)
    finally:
        del eng.glottis.forward
    return y, got["du"]


@pytest.fixture(scope="module")
def setup():
    eng = VoiceEngine(EngineConfig(n_extra_formants=3))
    tr = phones.ara()
    tr["voice_gain"] = 0.0
    y0, du = _render_capturing_du(eng, tr)
    return eng, tr, y0, du


def test_none_is_bypass_and_own_lf_du_reproduces_lf_render(setup):
    eng, tr, y0, du = setup
    assert np.array_equal(eng.render(tr), y0)
    eng.set_external_source(du[0].numpy())
    try:
        y = eng.render(tr)
    finally:
        eng.set_external_source(None)
    assert np.allclose(y, y0, rtol=0.0, atol=1e-5 * np.abs(y0).max())
    assert np.array_equal(eng.render(tr), y0)


def test_voice_gain_multiplies_the_external_source(setup):
    eng, tr, _y0, du = setup
    base = du[0].numpy()
    try:
        eng.set_external_source(2.0 * base)
        y_double = eng.render(tr)
        tr2 = phones.ara()
        tr2["voice_gain"] = 20.0 * np.log10(2.0)
        eng.set_external_source(base)
        y_gain = eng.render(tr2)
    finally:
        eng.set_external_source(None)
    assert np.abs(y_double).max() > 0
    assert np.allclose(y_gain, y_double, rtol=0.0, atol=1e-4 * np.abs(y_double).max())


def test_too_short_or_bad_source_is_rejected(setup):
    eng, tr, _y0, du = setup
    with pytest.raises(ValueError):
        eng.set_external_source(np.array([[1.0, np.nan]]))
    eng.set_external_source(du[0].numpy()[:100])
    try:
        with pytest.raises(ValueError):
            eng.render(tr)
    finally:
        eng.set_external_source(None)


def _render_capturing(eng, track, keys=("du", "open_phase")):
    got = {}
    orig = eng.glottis.forward

    def wrap(*args, **kwargs):
        out = orig(*args, **kwargs)
        for k in keys:
            got[k] = out[k].detach().clone()
        return out

    eng.glottis.forward = wrap
    try:
        y = eng.render(track)
    finally:
        del eng.glottis.forward
    return y, got


def test_external_open_curve_drives_open_damp_like_lf(setup):
    # §52.38 — OPEN_DAMP 를 켜고 LF 의 du·개방 곡선을 외부로 되먹이면 LF 렌더와 같아야 한다 (개방 곡선이 제자리에 들어간다).
    from formant_ml.engine import tract as tract_mod
    eng, tr, _y0, _du = setup
    old = tract_mod.OPEN_DAMP
    tract_mod.OPEN_DAMP = True
    try:
        y_lf, got = _render_capturing(eng, tr)
        assert float(got["open_phase"].abs().max()) > 0
        du0 = got["du"][0].numpy()
        o0 = got["open_phase"][0].clamp(0, 1).numpy()
        eng.set_external_source(du0, g_open=o0)
        y_ext = eng.render(tr)
        eng.set_external_source(du0)                           # 개방 곡선이 없으면 성도는 LF 개방 곡선을 그대로 본다
        y_noopen = eng.render(tr)
        eng.set_external_source(du0, g_open=np.roll(o0, 96))   # 반 주기쯤 민 곡선 — 곡선이 실제로 쓰이면 결과가 달라야 한다
        y_shift = eng.render(tr)
    finally:
        eng.set_external_source(None)
        tract_mod.OPEN_DAMP = old
    tol = 1e-5 * np.abs(y_lf).max()
    assert np.allclose(y_ext, y_lf, rtol=0.0, atol=tol)
    assert np.allclose(y_noopen, y_lf, rtol=0.0, atol=tol)
    assert np.abs(y_shift - y_lf).max() > 1e-3 * np.abs(y_lf).max()


def test_external_open_curve_is_validated(setup):
    eng, _tr, _y0, du = setup
    base = du[0].numpy()
    with pytest.raises(ValueError):
        eng.set_external_source(base, g_open=np.zeros(len(base) - 1))
    with pytest.raises(ValueError):
        eng.set_external_source(base, g_open=np.full(len(base), 1.5))
    eng.set_external_source(None)


def test_external_amp_envelope_scales_with_voice_gain(setup):
    # §52.46 — 외부 음원이면 무성 벌점이 쓸 amp_ext (음원 포락 × voice_gain) 를 낸다. 없으면 키가 없다.
    eng, tr, _y0, du = setup
    base = du[0].numpy()
    try:
        eng.set_external_source(base)
        _y, p0 = eng.render(tr, return_parts=True)
        tr2 = phones.ara()
        tr2["voice_gain"] = 20.0 * np.log10(2.0)
        _y, p1 = eng.render(tr2, return_parts=True)
    finally:
        eng.set_external_source(None)
    assert "amp_ext" in p0
    a0, a1 = p0["amp_ext"], p1["amp_ext"]
    assert np.allclose(a1, 2.0 * a0, rtol=1e-5, atol=1e-7)
    live = a0 > 0.01 * a0.max()
    assert 0.5 < float(np.median(a0[live])) < 2.0
    _y, pn = eng.render(tr, return_parts=True)
    assert "amp_ext" not in pn


def _warp_track(f0):
    tr = phones.ara()
    tr["voice_gain"] = 0.0
    tr["f0_target"] = float(f0)
    return tr


def test_ext_warp_ratio_one_is_bitwise_identical(setup):
    # §52.59 — 음원 F0 = f0_target 이면 비율 1, 어긋남 0 → 늘이기 없는 경로와 비트 단위로 같다.
    eng, _tr, _y0, du = setup
    base = du[0].numpy()
    tr = _warp_track(250.0)
    try:
        eng.set_external_source(base)
        y_plain = eng.render(tr)
        nfr = tr.values.shape[0]
        eng.set_external_source(base, f0_src=np.full(nfr, 250.0))
        y_warp = eng.render(tr)
    finally:
        eng.set_external_source(None)
    assert np.array_equal(y_plain, y_warp)


def test_ext_warp_constant_ratio_reads_ahead_by_leaky_offset(setup):
    # 비율 1.03 이 이어지면 어긋남은 0.03/(1 − a) 표본으로 수렴한다 (a = exp(−1/(τ·fs))). 매끈한 음원이라 선형 보간 오차가 작다.
    from formant_ml.engine import voice as voice_mod
    eng, _tr, _y0, du = setup
    n = du.shape[-1]
    k = np.arange(n)
    base = (np.sin(2 * np.pi * 180.0 * k / 48000.0) * 0.1).astype(np.float32)
    tr = _warp_track(250.0 * 1.03)
    nfr = tr.values.shape[0]
    try:
        eng.set_external_source(base, f0_src=np.full(nfr, 250.0))
        _y, parts = eng.render(tr, return_parts=True)
    finally:
        eng.set_external_source(None)
    a = np.exp(-1.0 / (voice_mod.EXT_WARP_TAU * 48000.0))
    d_ss = 0.03 / (1.0 - a)
    j = np.arange(int(8 * voice_mod.EXT_WARP_TAU * 48000), n - 400)
    want = np.interp(j + d_ss, k, base)
    got = parts["du"][j]
    assert np.max(np.abs(got - want)) < 2e-3 * 0.1


def test_ext_warp_gradient_reaches_f0_target(setup):
    import torch
    from formant_ml.engine.control import INDEX
    eng, _tr, _y0, du = setup
    base = du[0].numpy()
    tr = _warp_track(250.0)
    nfr = tr.values.shape[0]
    try:
        eng.set_external_source(base, f0_src=np.full(nfr, 245.0))
        eng.reset()
        ctrl = torch.as_tensor(tr.values, dtype=torch.float32).unsqueeze(0).clone().requires_grad_(True)
        out = eng(ctrl, list(tr.events), 0.0)
        out["du"].pow(2).sum().backward()
    finally:
        eng.set_external_source(None)
    g = ctrl.grad[0, :, INDEX["f0_target"]]
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0.0
