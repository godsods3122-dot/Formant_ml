"""`--noise-v2` 경로의 유성 기식에 **대역별 결맞음 구조**를 건다 (noise.ASP_SPLIT_BANDS, §52.275).

`ASP_SPLIT` 이 켜지면 `forward` 가 먼저 반환해서 `_partial_coherence` 를 안 불렀다 — ρ(공통 몫)가
통째로 빠져 대역마다 같은 AM 을 받았다. 사용자가 말한 "산란되는 파동과 산란되지 않는 파동" 의 분리다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import noise as N

FS, HOP, NS = 48000, 48, 9600


@pytest.fixture
def _parts():
    t = np.arange(NS) / FS
    env = torch.as_tensor((0.5 + 0.5 * np.sin(2 * np.pi * 200 * t))[None, :], dtype=torch.float64)
    voiced = torch.ones((1, NS), dtype=torch.float64)
    return env, voiced


def _run(monkeypatch, bands_on, parts):
    env, voiced = parts
    m = N.AspirationNoise(FS, HOP).double()
    monkeypatch.setattr(N, "ASP_SPLIT", True)
    monkeypatch.setattr(N, "ASP_SPLIT_BANDS", bands_on)
    with torch.no_grad():
        out = m.forward(env, noise=N.NoiseBank(), sample0=0, state={}, voiced=voiced)
    return out["source"][0].detach().numpy()


def test_flag_changes_the_voiced_aspiration(monkeypatch, _parts):
    off = _run(monkeypatch, False, _parts)
    on = _run(monkeypatch, True, _parts)
    assert np.isfinite(on).all()
    assert np.abs(on - off).max() > 1e-9, "깃발이 아무것도 안 바꿨다"


def test_rho_now_has_authority(monkeypatch, _parts):
    """ρ 를 바꾸면 소리가 바뀌어야 한다 — 우회되던 동안에는 0.95 와 0.05 가 **같은 출력**이었다(실측)."""
    monkeypatch.setattr(N, "NOISE_RHO", ((1e9, 0.95),))
    hi = _run(monkeypatch, True, _parts)
    monkeypatch.setattr(N, "NOISE_RHO", ((1e9, 0.05),))
    lo = _run(monkeypatch, True, _parts)
    assert np.abs(hi - lo).max() > 1e-9, "ρ 가 여전히 안 걸린다"


def test_rho_has_no_authority_while_bypassed(monkeypatch, _parts):
    """되돌림 확인 — 깃발을 끄면 예전처럼 ρ 가 **아무 일도 안 한다**. 이것이 우리가 고친 병이다."""
    monkeypatch.setattr(N, "NOISE_RHO", ((1e9, 0.95),))
    hi = _run(monkeypatch, False, _parts)
    monkeypatch.setattr(N, "NOISE_RHO", ((1e9, 0.05),))
    lo = _run(monkeypatch, False, _parts)
    assert np.abs(hi - lo).max() == 0.0


def test_off_by_default():
    assert N.ASP_SPLIT_BANDS is False


def test_common_am_depth_is_per_band(monkeypatch, _parts):
    """**공통 AM 깊이**는 대역마다 달라야 한다 (§52.276) — ρ 와 달리 이것이 펄스 동기를 정한다."""
    monkeypatch.setattr(N, "NOISE_AM_BETA", ((1e9, 1.0),))
    a = _run(monkeypatch, True, _parts)
    monkeypatch.setattr(N, "NOISE_AM_BETA", ((9000.0, 0.2), (1e9, 1.0)))
    b = _run(monkeypatch, True, _parts)
    assert np.abs(a - b).max() > 1e-9, "β 가 안 걸린다"


def test_beta_one_everywhere_is_the_old_behaviour(monkeypatch, _parts):
    monkeypatch.setattr(N, "NOISE_AM_BETA", ((1e9, 1.0),))
    a = _run(monkeypatch, True, _parts)
    monkeypatch.setattr(N, "NOISE_AM_BETA", ((500.0, 1.0), (1e9, 1.0)))
    b = _run(monkeypatch, True, _parts)
    assert np.abs(a - b).max() == 0.0


def test_band_gain_changes_the_fill(monkeypatch, _parts):
    """**대역별 기식 세기** (§52.276) — 펄스 사이를 메우는 양을 대역마다 정한다."""
    monkeypatch.setattr(N, "NOISE_BAND_GAIN_DB", ((1e9, 0.0),))
    a = _run(monkeypatch, True, _parts)
    monkeypatch.setattr(N, "NOISE_BAND_GAIN_DB", ((9000.0, 0.0), (1e9, -12.0)))
    b = _run(monkeypatch, True, _parts)
    assert np.abs(a - b).max() > 1e-9
    assert np.sqrt(np.mean(b ** 2)) < np.sqrt(np.mean(a ** 2))     # 깎았으니 줄어야 한다


def test_zero_db_everywhere_is_the_old_behaviour(monkeypatch, _parts):
    monkeypatch.setattr(N, "NOISE_BAND_GAIN_DB", ((1e9, 0.0),))
    a = _run(monkeypatch, True, _parts)
    monkeypatch.setattr(N, "NOISE_BAND_GAIN_DB", ((500.0, 0.0), (1e9, 0.0)))
    b = _run(monkeypatch, True, _parts)
    assert np.abs(a - b).max() == 0.0


def test_hf_rolloff_attenuates_the_top(monkeypatch, _parts):
    """기식 **고역 감쇠** (§52.276) — 제트 난류 스펙트럼은 봉우리를 지나 굴러떨어진다.

    지금 색(고역 셸프 + 고역통과)에는 굴러떨어질 수단이 아예 없어 20 kHz 까지 평평하다.
    """
    monkeypatch.setattr(N, "ASP_LP_HZ", 0.0)
    flat = _run(monkeypatch, True, _parts)
    monkeypatch.setattr(N, "ASP_LP_HZ", 9000.0)
    monkeypatch.setattr(N, "ASP_LP_ORDER", 2)
    rolled = _run(monkeypatch, True, _parts)

    def band(x, lo, hi):
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1.0 / FS)
        return 10 * np.log10(X[(f >= lo) & (f < hi)].mean() + 1e-30)

    lo_change = band(rolled, 3000, 6000) - band(flat, 3000, 6000)
    hi_change = band(rolled, 14000, 20000) - band(flat, 14000, 20000)
    assert hi_change < -8.0, hi_change          # 위는 확실히 깎인다
    assert lo_change > -3.0, lo_change          # 아래는 거의 그대로


def test_rolloff_off_by_default_and_is_a_no_op(monkeypatch, _parts):
    assert N.ASP_LP_HZ == 0.0
    monkeypatch.setattr(N, "ASP_LP_HZ", 0.0)
    a = _run(monkeypatch, True, _parts)
    b = _run(monkeypatch, True, _parts)
    assert np.abs(a - b).max() == 0.0
