"""복사합성 — 녹음의 노이즈를 지우고, 스펙트럼이 맞을 때까지 물리 파라미터를 추적한다.

    OMP_NUM_THREADS=2 python scripts/copyfit.py data/voices/yang_00000034.wav \
        --profile profiles/yang_female.json --from 0.510 --to 0.665 --out out/fit

무엇을 하는가
    1. 잡음 프로파일 추정 -> 위너 스펙트럼 차감 (engine/denoise.py)
    2. 1 ms 프레임마다 F0·유성도·포먼트·세기·마찰 추정 -> 제어열 초기값 (engine/analyze.py)
    3. 미분가능 엔진으로 제어열을 역추정 (engine/fit.py) — 전역 스칼라 -> 성김/촘촘함 프레임별
    4. 원본·복원·차이를 wav 로, 제어열을 npz 로, 일치율 표를 표준출력으로

일치율을 어떻게 읽는가
    **포락** (멜 dB) — 조음이 맞는가. 사람이 듣는 음색에 대응한다.
    **정밀** (선형 다해상도 STFT) — F0 궤적과 성문 펄스 위치까지 맞는가.
    **보정 정밀** — 실현 잡음을 뺀 값. **치찰음이 든 구간에서는 이것만 읽는다.**

`정밀` 은 난류에서 34.5 % 가 원리적 상한이다 — 같은 스펙트럼의 두 독립 실현조차
33.3 % 다 (레일리 크기, √((4−π)/2)). 완벽한 물리 모형도 그 이상 못 받으므로, 마찰이
든 구간에서 그 값을 모형 품질로 읽으면 안 된다. 같이 찍히는 `상한` 이 그 구간에서
`정밀` 이 받을 수 있는 최대값이고, `보정 정밀` 은 완벽한 모형에서 97 % 로 수렴한다.
자세한 것은 engine/turbulence.py 머리말과 docs/MEASUREMENTS.md §9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine.analyze import analyze
from formant_ml.engine.control import PARAM_NAMES
from formant_ml.engine.denoise import denoise, noise_profile, snr_report
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000),
         (5000, 8000), (8000, 12000), (12000, 24000)]


def band_db(x: np.ndarray, sr: int) -> list[float]:
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    tot = p.sum() + 1e-20
    return [10 * np.log10(p[(f >= lo) & (f < hi)].sum() / tot + 1e-20) for lo, hi in BANDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--profile", default=None, help="화자 프로파일 json")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--out", default="out/fit")
    ap.add_argument("--frame-ms", type=float, default=1.0)
    ap.add_argument("--global-iters", type=int, default=200)
    ap.add_argument("--stage-iters", type=int, default=100)
    ap.add_argument("--phase-iters", type=int, default=200,
                    help="마지막 위상 단계. 0 이면 끔 (크기만 맞춘다)")
    ap.add_argument("--lr-global", type=float, default=None,
                    help="생략하면 짧은 탐침으로 자동 선택 (구간마다 맞는 값이 다르다)")
    ap.add_argument("--lr-frame", type=float, default=0.04)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--phase", type=float, default=0.0, help="복소 STFT 항 가중(2 단계용)")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    y, sr = sf.read(a.wav)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    if not a.no_denoise:
        noise = noise_profile(y, sr)
        yd = denoise(y, sr, noise)
        r = snr_report(y, yd, sr, noise)
        print(f"잡음 제거: 바닥 {r['noise_db_in']:.1f} -> {r['noise_db_out']:.1f} dB "
              f"(−{r['removed_db']:.1f} dB), 정점 {r['peak_db_in']:.1f} -> "
              f"{r['peak_db_out']:.1f} dB")
        y = yd

    t1 = a.t1 if a.t1 is not None else len(y) / sr
    seg = y[int(a.t0 * sr):int(t1 * sr)]
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    hop = max(1, int(round(a.frame_ms * sr / 1000.0)))
    track = analyze(seg, sr, prof, hop, t0=a.t0, full=y)
    print(f"구간 {a.t0:.3f}~{t1:.3f} s, {track.n_frames} 프레임 × {track.frame_ms} ms")

    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=a.frame_ms,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False), prof)
    fit = CopySynthFitter(eng, seg, sr, track, phase_weight=a.phase)
    kw = {} if a.lr_global is None else {"lr_global": a.lr_global}
    rep = fit.fit_staged(global_iters=a.global_iters, stage_iters=a.stage_iters,
                         lr_frame=a.lr_frame, phase_iters=a.phase_iters, **kw)
    print(rep)

    # **치찰음이 든 구간에서는 위의 `정밀` 을 모형 품질로 읽으면 안 된다.** 그 자는
    # 난류에서 34.5 % 가 원리적 상한이다 (engine/turbulence.py 머리말). 실현 잡음을
    # 뺀 값을 같이 낸다 — 완벽한 모형이면 여기서 97 % 가 나온다.
    fid = fit.fidelity()
    print(f"실현 잡음을 빼면: 보정 정밀 {fid['fine_corr']:5.2f} %   "
          f"(기존 자의 이 구간 상한 {fid['floor']:.1f} %,  "
          f"시간평균 스펙트럼 {fid['spectrum_match']:.1f} %)")

    out = fit.render()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + "_target.wav", seg, sr)
    sf.write(a.out + "_fit.wav", out, 48000)
    res = fit.result_track()
    np.savez(a.out + "_track.npz", values=res.values, frame_ms=res.frame_ms,
             names=np.array(PARAM_NAMES))
    with open(a.out + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"env": rep.env, "fine": rep.fine, "per_size": rep.per_size,
                   "fine_corr": fid["fine_corr"], "floor": fid["floor"],
                   "spectrum_match": fid["spectrum_match"],
                   "loss": rep.loss, "gain_db": fit.gain_db(),
                   "moved": {n: [x, z] for n, x, z in fit.moved()}},
                  f, ensure_ascii=False, indent=1)

    print("\n대역 에너지 (총합 대비 dB)")
    print("           " + " ".join(f"{lo // 1000}-{hi // 1000}k".rjust(6) for lo, hi in BANDS))
    print("목표      " + " ".join(f"{v:6.1f}" for v in band_db(seg, sr)))
    print("합성      " + " ".join(f"{v:6.1f}" for v in band_db(out, 48000)))
    # 치찰음 지문 — 조음 위치가 맞는가 (Jongman et al. 2000). 대역 에너지표가 맞아도
    # 무게중심이 1 kHz 어긋나면 /s/ 가 /ʃ/ 로 들린다.
    from formant_ml.engine import turbulence as tb
    c = tb.compare(fit.target[0].numpy(), out, 48000.0)
    print("\n치찰음 지문 (목표 -> 합성)")
    print(f"  무게중심 {c['target']['centroid']:7.0f} -> {c['synth']['centroid']:7.0f} Hz "
          f"({c['centroid_err']:+.0f})   봉우리 {c['target']['peak']:6.0f} -> "
          f"{c['synth']['peak']:6.0f} Hz ({c['peak_err']:+.0f})")
    print(f"  치찰도   {c['target']['sibilance']:7.1f} -> {c['synth']['sibilance']:7.1f} dB "
          f"({c['sibilance_err']:+.1f})   대역 MAE {c['band_mae_db']:.2f} dB   "
          f"변조 MAE {c['mod_mae_pct']:.1f} %p")

    print("\n움직인 파라미터 (초기 중앙값 -> 적합 중앙값)")
    for n, x, z in fit.moved():
        if abs(z - x) > 0.05 * max(abs(x), 1e-6):
            print(f"  {n:12s} {x:10.3f} -> {z:10.3f}")
    print(f"\n결과: {a.out}_target.wav / {a.out}_fit.wav / {a.out}_track.npz")


if __name__ == "__main__":
    main()
