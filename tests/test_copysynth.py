"""복사합성의 최소 조건 — 되합성이 원본을 따라가는가.

    PYTHONPATH=src python3 tests/test_copysynth.py

합성한 신호를 **원본과 대조**하는 유일한 검사다. 손으로 작곡한 음절에는 정답이
없어서 "지표는 맞는데 사람 소리가 아닌" 상태를 못 잡는다. 여기서는 정답이 있다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from formant_ml.analysis.track import track_formants

FS = 24000


def _synthetic_vowel(f0=120.0, formants=(700, 1200, 2500, 3500),
                     bws=(80, 90, 120, 160), dur=0.4, sr=FS):
    """알려진 포먼트를 가진 합성 모음 — 추적기의 정답을 우리가 안다."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    src = np.zeros(n)
    for k in range(1, int(sr / 2 / f0)):
        src += np.cos(2 * np.pi * k * f0 * t) / k
    y = src
    for f, b in zip(formants, bws):
        r = np.exp(-np.pi * b / sr)
        th = 2 * np.pi * f / sr
        a1, a2 = -2 * r * np.cos(th), r * r
        out = np.zeros(n)
        g = 1 + a1 + a2
        for i in range(2, n):
            out[i] = g * y[i] - a1 * out[i - 1] - a2 * out[i - 2]
        y = out
    return y / np.abs(y).max()


def test_tracker_finds_known_formants():
    """알려진 포먼트를 넣으면 그대로 찾아야 한다."""
    y = _synthetic_vowel()
    F, _ = track_formants(y, FS, n=6)
    got = np.median(F[5:-5], axis=0)
    for i, want in enumerate((700, 1200, 2500, 3500)):
        assert abs(got[i] - want) / want < 0.08, f"F{i+1} {got[i]:.0f} vs {want}"


def test_tracker_never_emits_zero_formants():
    """못 찾은 슬롯을 0 으로 두면 안 된다.

    합성기가 f_min 으로 클램프해서 저역에 가짜 공명기를 만들고, 하나당
    -12 dB/oct 씩 감쇠가 붙는다. 실제로 빈 슬롯 4 개가 150 Hz 유령 극이 되어
    고역을 40 dB 죽였고, 되합성 오차가 33 dB 였다 (고친 뒤 5 dB).
    """
    y = _synthetic_vowel()
    F, B = track_formants(y, FS, n=12)      # 실제보다 훨씬 많이 요구한다
    assert float(F.min()) > 200.0, f"최소 포먼트 {F.min():.0f} Hz"
    assert float(B.min()) > 0.0
    # 남는 슬롯은 아주 높고 아주 넓어야 한다(무해)
    assert float(F[:, -1].min()) > 5000.0


def test_tracked_formants_are_ordered_and_continuous():
    """포먼트는 교차하지 않고, 프레임 사이에서 튀지 않아야 한다."""
    y = _synthetic_vowel()
    F, _ = track_formants(y, FS, n=6)
    assert bool((np.diff(F, axis=1) >= 0).all()), "포먼트 순서가 뒤집혔다"
    jump = np.abs(np.diff(F[5:-5, :4], axis=0)).max()
    assert jump < 400.0, f"프레임 간 최대 도약 {jump:.0f} Hz"




# --------------------------------------------------------------------------
# 아래는 **실제 녹음**을 쓰는 검사다. `reference/recordings/` 에 커밋되어 있고,
# 없으면 건너뛴다. 내부 상태가 아니라 **소리를 재는** 쪽이다 (HANDOFF §0.4).

REC = os.path.join(os.path.dirname(__file__), "..", "reference", "recordings",
                   "ko_liquid_ra-eulla-ara_male_44k.wav")
RA = (0.50, 1.02)                       # reference/README.md 의 '라' 구간
#: copysynth 의 기본값을 그대로 읽는다 (아래 테스트 안에서 import 한다).


def _bands_db(y, hop=240, n_fft=1024):
    """전체 대비 대역별 에너지 [dB]: 0.1~2.5k / 2.5~4k / 4~7k / 7~11k.

    **켑스트럼 포락선의 대역 평균으로 이걸 재지 마라.** 그건 dB 의 평균이라
    넓은 대역에서 변화를 눌러 버린다 — 기식을 75 배 올려도 7~11 kHz 가
    안 움직인다는 잘못된 결론을 한 번 냈다(실제로는 10 dB 움직였다).
    """
    y = np.asarray(y, dtype=np.float64)
    t = max(1, 1 + (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(t)[:, None]
    P = np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft), n_fft, axis=1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    tot = P.sum()
    return np.array([10 * np.log10(P[:, (f >= lo) & (f < hi)].sum() / tot + 1e-12)
                     for lo, hi in ((100, 2500), (2500, 4000),
                                    (4000, 7000), (7000, 11000))])


def test_resynthesis_matches_the_high_band_of_the_recording():
    """되합성의 2.5~7 kHz 가 원본과 맞아야 한다 — 소스 기울기가 맞는가.

    한때 `--tilt -12` 가 기본값이었다. 그 값은 `GlottalSource` 에서
    하모닉 k 에 10^(tilt*log2(k)/20) 을 곱하는데, **부호 규약을 반대로**
    알고 있었고(§4.3 의 "클수록 어두워진다" 는 `filters.one_pole_tilt` 이야기다)
    게다가 `(tilt*oct_).clamp(-40, 40)` 이 k≈10(약 1.1 kHz)에서 포화해
    **그 위 전부를 -40 dB 로 평평하게 눌렀다.** 결과: 2.5~4 kHz 와 4~7 kHz 가
    원본보다 35 dB 낮았다. 그런데도 당시 측정으로는 문제가 안 보였는데,
    `ltv_filter` 의 직사각 블록이 만든 가짜 에너지가 그 자리를 메우고 있었기
    때문이다(그 결함은 test_dsp 가 따로 잡는다).

    지금 값(`TILT`/`RD`)은 copysynth 의 기본값과 같아야 한다 — 기본값을 바꾸면
    여기도 바꿔라. 4~7 kHz 와 7~11 kHz 는 여기서 보지 않는다: 그 대역은
    추적기가 극을 못 찾아 남긴 결손이 지배해서, 소스 기울기로 판정할 수 없다
    (docs/HANDOFF_LIQUID.md §2.7).
    """
    if not os.path.exists(REC):
        return                                  # 기준 녹음이 없으면 건너뛴다
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import torch
    from formant_ml.analysis.acoustic import load
    from formant_ml.config import Config
    from formant_ml.models.synth import PhysicalVoiceSynth
    from copysynth import (DEFAULT_RD, DEFAULT_TILT, analyse,
                           build_controls, match_envelope)

    cfg = Config()
    y, _ = load(REC, FS)
    y = y[int(RA[0] * FS):int(RA[1] * FS)]
    y = y / max(np.abs(y).max(), 1e-9)

    a = analyse(y, cfg)
    torch.manual_seed(0)
    syn = PhysicalVoiceSynth(cfg, tract_mode="formant")
    c = build_controls(a, cfg, 0.002, 0.02, DEFAULT_TILT, DEFAULT_RD)
    with torch.no_grad():
        o = syn(c)["audio"][0].numpy().astype(np.float64)
    o = match_envelope(o, a["rms"], cfg.audio.hop_size)
    d = _bands_db(o / max(np.abs(o).max(), 1e-9)) - _bands_db(y)

    # 이 검사는 **기울기의 총체적 오류**를 잡는 것이지 값을 고정하는 것이 아니다.
    # 대역별 절대 오차로는 못 잡는다 — 추적기가 남긴 결손(§2.7)이 대역마다
    # 다르게 섞여 있어서, 발췌 구간에 따라 부호까지 뒤집힌다.
    # 반면 `tilt` 가 틀리면 오차가 **주파수에 대해 단조롭게** 기울어진다.
    assert abs(d[0]) < 6.0, (
        f"0.1~2.5 kHz 가 원본과 {d[0]:+.1f} dB 어긋난다 (소스 기울기를 의심하라)")
    slope = d[2] - d[0]                      # 저역 -> 4~7 kHz 오차 기울기
    # 실측: tilt=+6 -> +3.4, +2 -> -14.8, 0 -> -22.7, -6 -> -36.2, -12 -> -42.2.
    # 클램프는 |tilt| >= 40/log2(n_harmonics) = 5.06 부터 포화하므로,
    # -30 이면 포화 영역(±6 이상)만 걸러낸다.
    assert slope > -30.0, (
        f"소스가 주파수에 따라 {slope:+.1f} dB 무너진다 — tilt 부호나 "
        f"glottal.py 의 clamp 포화를 의심하라 (§2.2)")


def test_root_assignment_keeps_the_poles_lpc_found():
    """근을 찾아 놓고 버리지 마라 — 2.5~4 kHz −7 dB 의 정체가 이것이었다.

    옛 탐욕적 배정은 슬롯을 낮은 쪽부터 훑으며 포인터를 앞으로만 옮겨서,
    앞 슬롯이 근을 집으면 뒤 슬롯이 그 근을 영영 못 봤다. 실측 '라' 에서
    2.5~4 kHz 의 근 39 % 가 버려졌고, 그게 F3/F4 대역이라 명료도가 무너졌다.
    (docs/HANDOFF_LIQUID.md §2.7-b)

    **합성 모음으로는 이 결함이 안 잡힌다.** 포먼트가 안 움직이면 탐욕적
    배정도 다 맞힌다. 결함은 포먼트가 빠르게 움직일 때 — 즉 유음에서 — 난다.
    그래서 실제 녹음을 쓴다.
    """
    import pytest
    from scipy.signal import lfilter

    from formant_ml.analysis.acoustic import load
    from formant_ml.analysis.track import _roots

    wav = os.path.join(os.path.dirname(__file__), "..", "reference",
                       "recordings", "ko_liquid_ra-eulla-ara_male_44k.wav")
    if not os.path.exists(wav):
        pytest.skip("기준 녹음이 없다 (reference/README.md)")
    y, sr = load(wav, FS)
    y = y / max(np.abs(y).max(), 1e-9)

    hop, win, n = 240, 1024, 12
    F, _ = track_formants(y, sr, hop, win, n=n)

    order = int(2 + sr / 1000)
    x = lfilter([1.0, -0.97], [1.0], y)
    w = np.hanning(win)
    t = min(len(F), 1 + (len(x) - win) // hop)
    cand = [_roots(x[i * hop:i * hop + win] * w, sr, order,
                   120.0, 9000.0, 900.0)[0] for i in range(t)]
    # 무음 프레임의 근은 신호가 아니라 잡음 바닥에 맞춘 것이라 뜻이 없다.
    en = np.array([np.sqrt((y[i * hop:i * hop + win] ** 2).mean())
                   for i in range(t)])
    act = en > en.max() * 10 ** (-35.0 / 20.0)

    for lo, hi in ((2500.0, 4000.0), (4000.0, 7000.0)):
        raw = float(np.mean([((c >= lo) & (c < hi)).sum()
                             for c, a in zip(cand, act) if a]))
        kept = float(((F[:t] >= lo) & (F[:t] < hi)).sum(1)[act].mean())
        # 탐욕적 배정에서는 2.5~4 kHz 가 1.06 -> 0.65 (61 %) 였다.
        assert kept >= raw * 0.85, (
            f"{lo/1000:g}~{hi/1000:g} kHz: LPC 가 {raw:.2f} 개/프레임을 찾았는데 "
            f"배정 뒤 {kept:.2f} 개만 남았다 (§2.7-b)")


def test_empty_formant_slot_is_actually_neutral():
    """'극 없음' 은 응답이 1 이어야 한다. 20 kHz 는 +2.4 dB 셸프였다(§2.7-a)."""
    import torch

    from formant_ml.config import Config
    from formant_ml.dsp.filters import resonator_response

    cfg = Config()
    f = torch.full((1, 1, 1), cfg.audio.sample_rate * 0.45)
    b = torch.full((1, 1, 1), cfg.filt.bw_neutral)
    h = resonator_response(f, b, torch.ones_like(f),
                           float(cfg.audio.sample_rate), 129).abs()
    db = float((20 * torch.log10(h.clamp_min(1e-12))).abs().max())
    assert db < 0.1, f"빈 슬롯 하나가 최대 {db:.2f} dB 를 바꾼다 (§2.7-a)"


def test_higher_pole_correction_lifts_only_the_top():
    """고차극 보정은 나이퀴스트 쪽만 들어 올린다 — 저역을 건드리면 안 된다."""
    import torch

    from formant_ml.config import Config
    from formant_ml.dsp.filters import higher_pole_correction

    cfg = Config()
    nf = 513
    g = higher_pole_correction(float(cfg.audio.sample_rate), nf,
                               n_poles=cfg.filt.higher_poles)
    db = 20 * torch.log10(g.clamp_min(1e-12))
    fr = torch.linspace(0.0, cfg.audio.sample_rate / 2, nf)

    def at(hz):
        return float(db[int(torch.argmin((fr - hz).abs()))])

    assert at(1000.0) < 0.5, f"1 kHz 를 {at(1000.0):+.1f} dB 건드린다"
    assert 4.0 < at(8000.0) < 12.0, f"8 kHz 가 {at(8000.0):+.1f} dB (측정된 구멍 +7)"
    assert 12.0 < at(11000.0) < 24.0, f"11 kHz 가 {at(11000.0):+.1f} dB (구멍 +16)"
    assert bool((db[1:] >= db[:-1] - 1e-6).all()), "보정이 단조 증가가 아니다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
