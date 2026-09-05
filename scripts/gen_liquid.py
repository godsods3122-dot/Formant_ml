"""유음 합성 데모 — 혀끝 물리모델이 실제로 소리가 되는지 듣는다.

    PYTHONPATH=src python3 scripts/gen_liquid.py --out out

전부 방정식이다. 조음 방식을 고르는 스위치가 없고, 혀끝 목표 간극 h0(t) 와
구강압 Po 두 물리량만 바꾼다. 접촉 횟수는 방정식이 낸다.

목표 화자가 여성이므로 성도를 14.6 cm(20 단)로 잡고 F0 를 210 Hz 대로 둔다.
모음 면적함수는 아직 파라메트릭 근사다(presets.area_function) — 여성 실측
면적함수로 바꾸는 것은 RIEUL.md §4 의 남은 일이다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.config import Config, sections_for
from formant_ml.liquid import (lateral_antiformants, liquid_area,
                               liquid_syllable)
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.utils import ramp, save_wav

FEMALE_TRACT_CM = 14.6


def build(cfg: Config, t: int, area, f0_pts, lateral=None, level=None,
          lat_mix=None):
    """도파관 모드용 Controls. formant_* 는 이 경로에서 안 쓰이지만 필수 필드다."""
    K, nb = cfg.filt.n_formants, cfg.noise.n_bands
    amp = torch.ones(1, t, 1) if level is None else ramp(t, level)
    c = dict(
        f0=ramp(t, f0_pts),
        harmonic_amp=amp,
        rd=torch.full((1, t, 1), 1.1),
        formant_freq=torch.linspace(500, 6000, K).reshape(1, 1, -1)
                          .expand(1, t, K).contiguous(),
        formant_bw=torch.full((1, t, K), 90.0),
        formant_gain=torch.ones(1, t, K),
        noise_bands=torch.full((1, t, nb), 2e-4),
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.full((1, t, 1), 0.15),
        area=area,
    )
    if lateral is not None:
        # 반공명은 **옆 통로가 열려 있는 동안만** 존재한다. 발화 전체에 걸면
        # 모음의 F3 까지 눌린다(측정: 3050 -> 1890 Hz).
        fz, bw = lateral_antiformants(t, mix=lat_mix, **lateral)
        c["antiformant_freq"], c["antiformant_bw"] = fz, bw
    return Controls(**c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    cfg = Config()
    cfg.filt.n_tract_sections = sections_for(cfg.audio.sample_rate,
                                             FEMALE_TRACT_CM)
    sr = cfg.audio.sample_rate
    syn = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    OPEN = 0.30                      # 혀끝이 구개에서 떨어진 상태 [cm]

    def render(name, seconds, keyframes, h0_pts, po, lateral=None, f0=None,
               level=None, n_masses=5, tip_overlay=None):
        # 설측음은 자세가 이미 협착을 담고 있으므로 혀끝을 덧씌우지 않는다.
        if tip_overlay is None:
            tip_overlay = lateral is None
        area, contact, _ = liquid_syllable(
            seconds, keyframes, h0_pts, cfg, po=po, n_masses=n_masses,
            lateral_area_cm2=(lateral or {}).pop("area_cm2", None)
            if lateral else None, tip_overlay=tip_overlay)
        t = area.shape[1]
        f0 = f0 or [(0.0, 215.0), (1.0, 180.0)]
        mix = None
        # 반공명은 지금 끈다. 자세 면적함수를 **실측 포먼트에 맞춰** 풀었으므로
        # 측면 통로의 효과가 이미 극 배치에 들어가 있다(LPC 가 실제 음성의
        # 영점을 여분의 극으로 모델하는 것과 같다). 그 위에 이론 영점을 또 걸면
        # 방금 맞춘 F3 를 부순다 — 측정: 설측 자세 316/1412/2795 가 렌더 후
        # 294/578/1227 이 됐다. 측지를 따로 모델할 때 다시 켠다.
        if lateral is not None and lateral.get("explicit_zeros"):
            m = contact[:t].to(torch.float32).reshape(1, 1, -1)
            m = torch.nn.functional.avg_pool1d(
                torch.nn.functional.pad(m, (2, 2), mode="replicate"), 5, 1)
            mix = m.reshape(-1).clamp(0.0, 1.0)
        c = build(cfg, t, area, f0,
                  lateral if (lateral or {}).get("explicit_zeros") else None,
                  level, mix)
        with torch.no_grad():
            y = syn(c)["audio"]
        save_wav(os.path.join(args.out, name), y, sr)
        n_ev = int((contact[1:] - contact[:-1] == 1).sum()) + int(contact[0] > 0)
        print(f"  {name:32s} {seconds:.2f}s  접촉 {n_ev}회 "
              f"({float(contact.sum()) * cfg.audio.hop_size / sr * 1000:.0f} ms)")

    print("유음 합성 (여성 성도 14.6 cm / 20 단, F0 215->180 Hz)")
    print("  자세 궤적이 성도 전체를 움직이고, 그 위에 혀끝이 협착을 덧씌운다.")

    # 타이밍은 녹음에서 잰 값이다: 유음 유지 ~150 ms, 유음->모음 전이 ~60 ms.
    # 1) 라 — 어두 설측음. Lee(2015): 초성은 설측음 지각.
    render("31_ra.wav", 0.55,
           [(0.0, "l_onset"), (0.27, "l_onset"), (0.38, "a"), (1.0, "a")],
           [(0.0, -0.05), (0.15, -0.05), (0.21, OPEN), (0.55, OPEN)],
           po=1200.0, lateral=dict(area_cm2=0.20, supra_cm=3.0, inter_cm=2.2),
           level=[(0.0, 0.7), (0.35, 1.0), (1.0, 0.85)])

    # 2) 아라 — 모음 사이 탄음. 폐쇄가 짧아 압력이 안 쌓인다(Cathcart 2012).
    render("32_ara.wav", 0.70,
           [(0.0, "a"), (0.30, "a"), (0.42, "tap"), (0.52, "tap"),
            (0.63, "a"), (1.0, "a")],
           [(0.0, OPEN), (0.24, OPEN), (0.275, -0.04), (0.315, -0.04),
            (0.35, OPEN), (0.70, OPEN)], po=2000.0,
           level=[(0.0, 0.9), (0.4, 0.75), (0.6, 1.0), (1.0, 0.8)])

    # 3) 알 — 종성 설측음. Lee(2015): 종성의 혀끝 변위가 가장 크다.
    render("33_al.wav", 0.60,
           [(0.0, "a"), (0.40, "a"), (0.55, "l_onset"), (1.0, "l_onset")],
           [(0.0, OPEN), (0.26, OPEN), (0.34, -0.05), (0.60, -0.05)],
           po=1200.0, lateral=dict(area_cm2=0.20, supra_cm=3.2, inter_cm=2.4),
           level=[(0.0, 1.0), (0.6, 0.95), (1.0, 0.6)])

    # 4) 을라 — /으/ 뒤 설측음. 녹음의 두 번째 토큰.
    render("34_eulla.wav", 0.85,
           [(0.0, "eu"), (0.28, "eu"), (0.40, "l_onset"), (0.58, "l_onset"),
            (0.68, "a"), (1.0, "a")],
           [(0.0, OPEN), (0.30, OPEN), (0.36, -0.05), (0.50, -0.05),
            (0.56, OPEN), (0.85, OPEN)],
           po=1200.0, lateral=dict(area_cm2=0.20, supra_cm=3.0, inter_cm=2.2),
           level=[(0.0, 0.8), (0.35, 0.9), (0.7, 1.0), (1.0, 0.8)])

    # 5) 전동음 — 같은 계에 좁은 자세 + 압력. 접촉 횟수는 방정식이 낸다.
    render("35_trill_rr.wav", 0.75,
           [(0.0, "a"), (0.22, "tap"), (0.68, "tap"), (0.85, "a"), (1.0, "a")],
           [(0.0, OPEN), (0.14, OPEN), (0.20, 0.03), (0.50, 0.03),
            (0.56, OPEN), (0.75, OPEN)], po=8000.0)

    # 6) 접근음 — 닿지 않을 만큼만 좁힌다.
    render("36_approximant.wav", 0.65,
           [(0.0, "a"), (0.25, "tap"), (0.60, "tap"), (0.80, "a"), (1.0, "a")],
           [(0.0, OPEN), (0.18, OPEN), (0.25, 0.22), (0.42, 0.22),
            (0.49, OPEN), (0.65, OPEN)], po=3000.0)

    print(f"\nWAV 를 {args.out}/ 에 저장했다.")


if __name__ == "__main__":
    main()
