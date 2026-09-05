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

    지금은 tilt=+4 / Rd=0.9 로 2.5~4k -1.7 dB, 4~7k -0.3 dB 다.
    7~11 kHz 는 아직 13 dB 부족하고(유성 기식이 할 일이다) 여기서 안 본다 —
    docs/HANDOFF_LIQUID.md §2.5.
    """
    if not os.path.exists(REC):
        return                                  # 기준 녹음이 없으면 건너뛴다
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import torch
    from formant_ml.analysis.acoustic import load
    from formant_ml.config import Config
    from formant_ml.models.synth import PhysicalVoiceSynth
    from copysynth import analyse, build_controls, match_envelope

    cfg = Config()
    y, _ = load(REC, FS)
    y = y[int(RA[0] * FS):int(RA[1] * FS)]
    y = y / max(np.abs(y).max(), 1e-9)

    a = analyse(y, cfg)
    torch.manual_seed(0)
    syn = PhysicalVoiceSynth(cfg, tract_mode="formant")
    c = build_controls(a, cfg, 0.002, 0.02, 4.0, 0.9)
    with torch.no_grad():
        o = syn(c)["audio"][0].numpy().astype(np.float64)
    o = match_envelope(o, a["rms"], cfg.audio.hop_size)
    d = _bands_db(o / max(np.abs(o).max(), 1e-9)) - _bands_db(y)

    for i, name in ((0, "0.1~2.5 kHz"), (1, "2.5~4 kHz"), (2, "4~7 kHz")):
        assert abs(d[i]) < 6.0, (
            f"{name} 가 원본과 {d[i]:+.1f} dB 어긋난다 (소스 기울기를 의심하라)")


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
