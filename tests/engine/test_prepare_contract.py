"""적합 준비 단계의 **계약 시험** (MEASUREMENTS §52.434).

위치를 아는 합성 신호로 `engine/prepare.py` 를 부른다 — `copyfit` 과 같은 경로다. 계약:

1. 파열은 개방 자리에서 잡힌다.
2. 폐쇄 관측은 폐쇄를 닫고 모음은 닫지 않는다.
3. **같은 소리를 파일의 다른 자리에 두어도 구간 원점의 관측이 같다** — `--from` 만큼 어긋났던 결함(§52.430·431)을 잡는 시험.
4. 구간 시작을 옮기면 관측도 정확히 그만큼 옮겨진다.
"""
import numpy as np
import pytest
from scipy.signal import lfilter

from formant_ml.engine import analyze as an
from formant_ml.engine import prepare as prep
from formant_ml.engine.profile import SpeakerProfile

FS = 48000
HOP = 48                                   # 1 ms 틀
T_CLOSE, T_REL = 0.35, 0.43                # 폐쇄 시작·개방 [s]
VOW_A, VOW_B = (0.15, 0.35), (0.45, 0.70)


def _vowel(n, f0=220.0, formants=(700, 1200, 2600), bws=(80, 90, 120), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    ph = np.cumsum(np.full(n, f0 * (1 + 0.003 * rng.standard_normal())) / FS)
    src = np.zeros(n)
    src[np.flatnonzero(np.diff(np.floor(ph)) > 0) + 1] = 1.0
    y = src
    for f, b in zip(formants, bws):
        r = np.exp(-np.pi * b / FS)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r], y)
    return y / (np.abs(y).max() + 1e-12) * np.hanning(n) ** 0.1


def utterance(seed=0):
    """무음 · 모음 A · 유성 폐쇄(보이스 바) · 파열 + 기식 · 모음 B · 무음 (0.9 s)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(int(0.9 * FS)) * 1e-4
    a0, a1 = (int(v * FS) for v in VOW_A)
    y[a0:a1] += 0.2 * _vowel(a1 - a0)
    c0, c1 = int(T_CLOSE * FS), int(T_REL * FS)
    vb = _vowel(c1 - c0, formants=(250,), bws=(120,), seed=1)
    y[c0:c1] += 0.004 * vb                                     # 보이스 바 (≈ −34 dB)
    b = rng.standard_normal(int(0.002 * FS)) * 0.08            # 파열 2 ms
    y[c1:c1 + b.size] += b
    asp = rng.standard_normal(int(0.02 * FS))
    asp = lfilter([1, -0.9], [1], asp) * 0.01 * np.linspace(1, 0.3, asp.size)
    y[c1 + b.size:c1 + b.size + asp.size] += asp               # 기식 20 ms
    v0, v1 = (int(v * FS) for v in VOW_B)
    y[v0:v1] += 0.2 * _vowel(v1 - v0, seed=2)
    return y


@pytest.fixture(scope="module")
def prof():
    return SpeakerProfile.load("profiles/yang_female.json")


@pytest.fixture(autouse=True)
def _restore_globals():
    saved = (an.STOP_CLOSE, an.FRONT_LEN_OBS)
    yield
    an.STOP_CLOSE, an.FRONT_LEN_OBS = saved


def _prepare(full, t0, t1, prof):
    seg = prep.Segment.cut(full, FS, t0, t1)
    tr = prep.observe(seg, prof, HOP, front_obs=True, log=lambda m: None)
    prep.burst_observations(tr, seg, HOP, burst_smooth_ms=12.0, stop_close=True, log=lambda m: None)
    return tr


@pytest.fixture(scope="module")
def base(prof):
    return _prepare(utterance(), 0.0, 0.9, prof)


def test_burst_is_found_at_the_release(base):
    b = np.asarray(base.bursts) / 1000.0
    assert b.size >= 1
    assert np.min(np.abs(b - T_REL)) < 0.006, f"파열 {b} — 개방은 {T_REL}"


def test_closure_closes_and_vowels_stay_open(base):
    oo = np.asarray(base["oral_open"])
    ms = lambda a, b: slice(int(a * 1000), int(b * 1000))
    assert oo[ms(T_CLOSE + 0.015, T_REL - 0.010)].max() < 0.1, "폐쇄 구간이 닫히지 않았다"   # 파열 검출 ±5 ms, 풀림 경사 5 ms
    assert oo[ms(0.20, 0.30)].min() > 0.5, "모음 A 가 닫혔다"
    assert oo[ms(0.50, 0.60)].min() > 0.5, "모음 B 가 닫혔다"


def test_observations_do_not_depend_on_where_the_segment_sits_in_the_file(base, prof):
    pad = 0.2
    y = utterance()
    full = np.concatenate([np.random.default_rng(9).standard_normal(int(pad * FS)) * 1e-4, y])
    tr = _prepare(full, pad, pad + 0.9, prof)
    n = min(tr.n_frames, base.n_frames)
    for nm in ("oral_open", "a_c", "front_len"):
        a, b = np.asarray(base[nm])[:n], np.asarray(tr[nm])[:n]
        bad = np.abs(np.log(np.maximum(a, 1e-3)) - np.log(np.maximum(b, 1e-3))) > 0.2
        assert bad.mean() < 0.02, f"{nm}: 파일 속 위치에 따라 {bad.mean() * 100:.1f} % 틀이 달라졌다"
    assert set(np.asarray(tr.bursts).tolist()) == set(np.asarray(base.bursts).tolist())
    va, vb = np.asarray(base.voiced)[:n], np.asarray(tr.voiced)[:n]
    assert (va != vb).mean() < 0.02


def test_moving_the_segment_start_shifts_observations_by_the_same_amount(base, prof):
    sh = 0.05                                   # 앞 무음 50 ms 를 잘라 낸다
    tr = _prepare(utterance(), sh, 0.9, prof)
    k = int(round(sh * 1000))
    b0 = np.asarray(base.bursts); b1 = np.asarray(tr.bursts)
    assert set((b1 + k).tolist()) == set(b0.tolist())
    oo0, oo1 = np.asarray(base["oral_open"]), np.asarray(tr["oral_open"])
    s = slice(int(T_CLOSE * 1000) + 15, int(T_REL * 1000) - 10)
    assert np.allclose(oo0[s], oo1[s.start - k:s.stop - k], atol=0.05)


def test_closure_is_aerodynamically_closed(base):
    """폐쇄 틀에서는 협착 제트의 레이놀즈 수가 난류 문턱 **밑**이어야 한다 — 가장 나쁜 조건(폐압 20, 성문 활짝)에서도 (§52.435)."""
    import torch
    from formant_ml.engine.noise import RE_CRIT, reynolds, series_flow
    ac = np.asarray(base["a_c"])
    s = slice(int(T_CLOSE * 1000) + 15, int(T_REL * 1000) - 10)
    a_c = torch.as_tensor(ac[s])
    ag = torch.full_like(a_c, 0.02 + 0.5 * (1 - 0.0) ** 2.5)          # 내전 0 = 가장 넓은 성문
    u, _ = series_flow(torch.full_like(a_c, 20.0), ag, a_c)
    re, _, _ = reynolds(u, a_c)
    assert float(re.max()) < RE_CRIT, f"폐쇄인데 Re {float(re.max()):.0f} ≥ {RE_CRIT:.0f} — 마찰 잡음을 낸다"


def test_sharp_release_opens_at_the_burst(prof):
    """`STOP_RELEASE_SHARP`: 폐쇄는 개방 2 ms 앞까지 닫혀 있고, 개방 3 ms 뒤에는 열려 있다 (§52.451). 예전 관측은 5 ms 경사로 일찍 열었다."""
    y = utterance()
    seg = prep.Segment.cut(y, FS, 0.0, None)
    old = an.STOP_RELEASE_SHARP
    try:
        an.STOP_RELEASE_SHARP = True
        tr = prep.observe(seg, prof, HOP, log=lambda m: None)
        prep.burst_observations(tr, seg, HOP, burst_smooth_ms=12, stop_close=True, log=lambda m: None)
    finally:
        an.STOP_RELEASE_SHARP = old
    ac = tr["a_c"]
    rel = int(T_REL * 1000)
    assert ac[rel - 8:rel - 2].max() < 0.01, "개방 앞이 이미 열렸다"
    assert ac[rel + 3] > 0.5, "개방 뒤가 아직 닫혀 있다"
