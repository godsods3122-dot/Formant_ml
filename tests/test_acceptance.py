"""수용 검사 — **출력이 음성인가**를 파형만 보고 판정한다.

    PYTHONPATH=src python3 tests/test_acceptance.py

왜 따로 있는가
--------------
이 레포의 다른 테스트는 전부 기계의 **내부 상태**를 본다 — 접촉 횟수, 간극,
반사계수, 노치 깊이. 그것들이 전부 통과하는 동안 합성음은 사람 소리가 아니었다.
내부가 설계대로 도는 것과 출력이 음성인 것은 다른 명제이고, 후자를 아무도
묻지 않았다. HANDOFF §0 이 경고한 underdetermined 측정이 정확히 이것이다.

여기 있는 것은 전부 **합성 내부에 접근하지 않는다.** 파형을 렌더하고, 실제
녹음에 쓰는 것과 같은 측정기(`analysis.acoustic`)로 재고, 목표값과 비교한다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.analysis.acoustic import formants, voicing
from formant_ml.config import Config, sections_for
from formant_ml.liquid import vowel_area
from formant_ml.models.synth import Controls, PhysicalVoiceSynth

FS = 24000
FEMALE_CM = 14.6

# Peterson & Barney 여성 기준. F3 는 아직 계통적으로 낮아 검사에서 뺀다
# (presets.VOWEL_AREA_20 주석의 알려진 한계 — 숨기지 말고 명시해 둔다).
TARGET = {"a": (850, 1220), "i": (310, 2790), "u": (370, 950)}


def _render_vowel(vowel: str, seconds: float = 0.5) -> np.ndarray:
    cfg = Config()
    cfg.filt.n_tract_sections = sections_for(FS, FEMALE_CM)
    n_sec = cfg.filt.n_tract_sections
    t = int(seconds * FS / cfg.audio.hop_size)
    area = vowel_area(n_sec, vowel).reshape(1, 1, -1).expand(1, t, n_sec)
    K, nb = cfg.filt.n_formants, cfg.noise.n_bands
    syn = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    c = Controls(
        f0=torch.full((1, t, 1), 200.0), harmonic_amp=torch.ones(1, t, 1),
        rd=torch.full((1, t, 1), 1.1),
        formant_freq=torch.linspace(500, 6000, K).reshape(1, 1, -1)
                          .expand(1, t, K).contiguous(),
        formant_bw=torch.full((1, t, K), 90.0),
        formant_gain=torch.ones(1, t, K),
        noise_bands=torch.full((1, t, nb), 1e-4),
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.zeros(1, t, 1), area=area.contiguous())
    with torch.no_grad():
        y = syn(c)["audio"][0]
    return y.numpy().astype(np.float64)


def _median_formants(y, sr=FS, skip=0.1):
    F = formants(y, sr)
    F = F[int(skip / 0.01):]
    return np.nanmedian(F, axis=0)


def test_rendered_vowels_hit_their_formant_targets():
    """합성한 모음을 **파형에서 재서** F1/F2 가 목표의 20 % 안에 있어야 한다.

    이게 없어서 /아/ 가 F2 1928 Hz(목표 1220)로 나가는 걸 못 잡았다.
    """
    for v, (f1, f2) in TARGET.items():
        got = _median_formants(_render_vowel(v))
        for i, (g, want) in enumerate(zip(got[:2], (f1, f2))):
            assert not np.isnan(g), f"/{v}/ F{i+1} 미검출"
            assert abs(g - want) / want < 0.20, \
                f"/{v}/ F{i+1} = {g:.0f} Hz, 목표 {want} Hz"


def test_vowels_are_actually_distinguishable():
    """모음이 서로 구별되어야 한다. 전부 슈와로 수렴하면 무의미하다."""
    a, i, u = (_median_formants(_render_vowel(v)) for v in ("a", "i", "u"))
    assert a[0] > i[0] * 1.8, f"/a/ F1 {a[0]:.0f} vs /i/ F1 {i[0]:.0f}"
    assert i[1] > a[1] * 1.4, f"/i/ F2 {i[1]:.0f} vs /a/ F2 {a[1]:.0f}"
    assert a[1] > u[1] * 1.15, f"/a/ F2 {a[1]:.0f} vs /u/ F2 {u[1]:.0f}"


def test_voiced_output_is_actually_periodic():
    """유성음이면 주기적이어야 한다. 잡음이나 무음이면 여기서 걸린다."""
    per, f0 = voicing(_render_vowel("a"), FS)
    assert float(np.median(per)) > 0.5, float(np.median(per))
    assert 150.0 < float(np.median(f0[per > 0.5])) < 260.0


def test_lateral_antiformants_do_not_leak_into_the_vowel():
    """반공명은 설측 구간에만 걸려야 한다.

    발화 전체에 걸었더니 모음의 F3 가 3050 -> 1890 Hz 로 눌렸다. 옆 통로가
    닫힌 구간에는 영점이 존재하지 않는다.
    """
    from formant_ml.liquid import lateral_antiformants
    t = 40
    mix = torch.zeros(t)                       # 전 구간 설측음 아님
    fz, bw = lateral_antiformants(t, mix=mix)
    from formant_ml.dsp.filters import antiresonator_response
    H = antiresonator_response(fz, bw, FS, 513)
    assert float((H.abs() - 1.0).abs().max()) < 0.05, "꺼져도 응답이 평탄하지 않다"


def _render_syllable(keyframes, h0, seconds, po, tip_overlay, lat=None):
    from formant_ml.liquid import liquid_syllable
    cfg = Config()
    cfg.filt.n_tract_sections = sections_for(FS, FEMALE_CM)
    area, _, _ = liquid_syllable(seconds, keyframes, h0, cfg, po=po,
                                 lateral_area_cm2=lat,
                                 tip_overlay=tip_overlay)
    t = area.shape[1]
    K, nb = cfg.filt.n_formants, cfg.noise.n_bands
    syn = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    c = Controls(
        f0=torch.full((1, t, 1), 200.0), harmonic_amp=torch.ones(1, t, 1),
        rd=torch.full((1, t, 1), 1.1),
        formant_freq=torch.linspace(500, 6000, K).reshape(1, 1, -1)
                          .expand(1, t, K).contiguous(),
        formant_bw=torch.full((1, t, K), 90.0),
        formant_gain=torch.ones(1, t, K),
        noise_bands=torch.full((1, t, nb), 1e-4),
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.zeros(1, t, 1), area=area)
    with torch.no_grad():
        return syn(c)["audio"][0].numpy().astype(np.float64)


# 사용자 녹음에서 잰 뒤 화자 정규화한 값 (presets.LIQUID_AREA_20 주석 참조)
LIQUID_TARGET = (328, 1457)          # 어두 설측음 F1/F2


def test_rendered_lateral_matches_the_measured_liquid():
    """렌더된 설측음 구간의 F1/F2 가 실측(정규화) 목표의 15 % 안이어야 한다.

    여기서 두 가지 이중계산이 걸린다. 자세 면적함수는 실측 포먼트에 맞춰 푼
    것이라 그 자세의 협착을 이미 담고 있는데,
      (1) 그 위에 혀끝 폐쇄를 또 곱하면 316/1412/2795 -> 222/676/1470,
      (2) 그 위에 이론 측지 영점을 또 걸면 316/1412/2795 -> 294/578/1227
    로 무너진다. 둘 다 실제로 겪었고, 이 검사가 그 재발을 막는다.
    """
    y = _render_syllable(
        [(0.0, "l_onset"), (0.55, "l_onset"), (0.75, "a"), (1.0, "a")],
        [(0.0, -0.05), (0.20, -0.05), (0.26, 0.30), (0.32, 0.30)],
        0.32, 1200.0, tip_overlay=False)
    F = formants(y[: int(0.16 * FS)], FS)
    got = np.nanmedian(F, axis=0)[:2]
    for i, (g, want) in enumerate(zip(got, LIQUID_TARGET)):
        assert not np.isnan(g), f"F{i+1} 미검출"
        assert abs(g - want) / want < 0.15, \
            f"설측음 F{i+1} = {g:.0f} Hz, 목표 {want} Hz"


def test_liquid_and_vowel_are_distinct_in_the_same_syllable():
    """한 음절 안에서 유음과 모음이 구별되어야 한다.

    자세 궤적(축 3)이 없으면 혀끝만 앞쪽 몇 단을 건드리므로 F1 이 거의 안
    움직인다 — 실측은 345 -> 739 Hz 로 두 배 넘게 뛴다.
    """
    y = _render_syllable(
        [(0.0, "l_onset"), (0.35, "l_onset"), (0.50, "a"), (1.0, "a")],
        [(0.0, -0.05), (0.16, -0.05), (0.22, 0.30), (0.45, 0.30)],
        0.45, 1200.0, tip_overlay=False)
    liq = np.nanmedian(formants(y[: int(0.13 * FS)], FS), axis=0)
    vow = np.nanmedian(formants(y[int(0.28 * FS):], FS), axis=0)
    assert vow[0] > liq[0] * 1.8, f"F1 유음 {liq[0]:.0f} -> 모음 {vow[0]:.0f}"


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
