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
from formant_ml.liquid import lateral_antiformants, liquid_area
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

    def render(name, seconds, h0_pts, po, lateral=None, f0=None, level=None,
               n_masses=5):
        area, contact, _ = liquid_area(seconds, h0_pts, cfg, po=po,
                                       n_masses=n_masses, vowel="a",
                                       lateral_area_cm2=(
                                           lateral or {}).pop("area_cm2", None)
                                       if lateral else None)
        t = area.shape[1]
        f0 = f0 or [(0.0, 215.0), (1.0, 185.0)]
        # 설측 구간 = 혀끝이 닿아 있는 구간. 경계를 매끄럽게 3 프레임 번지게 한다.
        mix = None
        if lateral is not None:
            m = contact[:t].to(torch.float32).reshape(1, 1, -1)
            m = torch.nn.functional.avg_pool1d(
                torch.nn.functional.pad(m, (2, 2), mode="replicate"), 5, 1)
            mix = m.reshape(-1).clamp(0.0, 1.0)
        c = build(cfg, t, area, f0, lateral, level, mix)
        with torch.no_grad():
            y = syn(c)["audio"]
        save_wav(os.path.join(args.out, name), y, sr)
        n_ev = int((contact[1:] - contact[:-1] == 1).sum()) + int(contact[0] > 0)
        print(f"  {name:34s} {seconds:.2f}s  접촉 {n_ev}회 "
              f"({float(contact.sum()) * cfg.audio.hop_size / sr * 1000:.0f} ms)")

    print("유음 합성 (여성 성도 14.6 cm / 20 단, F0 215->185 Hz)")

    # 1) 아라 — 모음 사이 탄음. Lee(2015): 모음 사이가 탄음 지각.
    #    폐쇄가 짧아 구강압이 안 쌓인다 -> Po 낮음 -> 접촉 한 번 (Cathcart 2012).
    render("21_ara_tap.wav", 0.70,
           [(0.0, OPEN), (0.26, OPEN), (0.285, -0.04), (0.315, -0.04),
            (0.34, OPEN), (0.70, OPEN)], po=2000.0)

    # 2) 라 — 어두 설측음. Lee(2015): 초성은 설측음 지각이다.
    #    중앙이 막혀도 옆이 열려 있으므로 면적이 0 이 아니라 측면 통로에서 바닥을
    #    치고, 측지 반공명 2 개가 걸린다.
    render("22_ra_lateral.wav", 0.70,
           [(0.0, -0.05), (0.16, -0.05), (0.24, OPEN), (0.70, OPEN)],
           po=1200.0, lateral=dict(area_cm2=0.22, supra_cm=3.0, inter_cm=2.2),
           level=[(0.0, 0.75), (0.25, 1.0), (1.0, 0.9)])

    # 3) 알 — 종성 설측음. Lee(2015): 종성의 혀끝 변위가 가장 크다.
    render("23_al_lateral_coda.wav", 0.70,
           [(0.0, OPEN), (0.30, OPEN), (0.40, -0.05), (0.70, -0.05)],
           po=1200.0, lateral=dict(area_cm2=0.18, supra_cm=3.2, inter_cm=2.4),
           level=[(0.0, 1.0), (0.6, 0.95), (1.0, 0.7)])

    # 4) 전동음 — 이탈리아어 rr. 같은 계인데 좁게 유지 + 압력을 준다.
    #    접촉 횟수를 우리가 안 정한다.
    render("24_trill_rr.wav", 0.75,
           [(0.0, OPEN), (0.18, OPEN), (0.24, 0.03), (0.52, 0.03),
            (0.58, OPEN), (0.75, OPEN)], po=8000.0)

    # 5) 접근음 — 닿지 않을 만큼만 좁힌다.
    render("25_approximant.wav", 0.70,
           [(0.0, OPEN), (0.20, OPEN), (0.27, 0.22), (0.45, 0.22),
            (0.52, OPEN), (0.70, OPEN)], po=3000.0)

    # 6) 같은 제스처, 구강압만 바꾼다: 탄음 -> 전동음.
    #    조음 방식이 파라미터가 아니라 결과라는 것을 한 파일에서 듣는다.
    # 자세를 처음부터 잡고 시작한다. 열린 데서 급히 좁히면 혀끝이 도착하면서
    # 제 고유진동수로 울려(접촉이 생겨) 압력의 효과와 섞인다 — 측정에서 확인.
    hold = [(0.0, 0.03), (0.45, 0.03), (0.52, OPEN), (0.70, OPEN)]
    for po in (500.0, 2500.0, 8000.0):
        render(f"26_pressure_{int(po):04d}.wav", 0.70, hold, po=po)

    print(f"\nWAV 를 {args.out}/ 에 저장했다.")


if __name__ == "__main__":
    main()
