"""복사합성 — 녹음의 노이즈를 지우고, 스펙트럼이 맞을 때까지 물리 파라미터를 추적한다.

    OMP_NUM_THREADS=2 python scripts/copyfit.py data/ref/female_yang_ilin-ilsil.wav \
        --profile profiles/yang_female.json --from 2.86 --to 3.06 --out out/fit

무엇을 하는가
    1. 잡음 프로파일 추정 -> 위너 스펙트럼 차감 (engine/denoise.py)
    2. 1 ms 프레임마다 F0·유성도·포먼트·세기·마찰 추정 -> 제어열 초기값 (engine/analyze.py)
    3. 미분가능 엔진으로 제어열을 역추정 (engine/fit.py) — 전역 스칼라 -> 성김/촘촘함 프레임별
    4. 원본·복원·차이를 wav 로, 제어열을 npz 로, 일치율 표를 표준출력으로

일치율은 두 가지다. **포락**(멜 dB, 조음이 맞는가)과 **정밀**(선형 다해상도 STFT,
F0 궤적과 성문 펄스 위치까지 맞는가). 자세한 정의는 engine/fit.py 머리말에 있다.
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

    out = fit.render()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + "_target.wav", seg, sr)
    sf.write(a.out + "_fit.wav", out, 48000)
    res = fit.result_track()
    np.savez(a.out + "_track.npz", values=res.values, frame_ms=res.frame_ms,
             names=np.array(PARAM_NAMES))
    with open(a.out + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"env": rep.env, "fine": rep.fine, "per_size": rep.per_size,
                   "loss": rep.loss, "gain_db": fit.gain_db(),
                   "moved": {n: [x, z] for n, x, z in fit.moved()}},
                  f, ensure_ascii=False, indent=1)

    print("\n대역 에너지 (총합 대비 dB)")
    print("           " + " ".join(f"{lo // 1000}-{hi // 1000}k".rjust(6) for lo, hi in BANDS))
    print("목표      " + " ".join(f"{v:6.1f}" for v in band_db(seg, sr)))
    print("합성      " + " ".join(f"{v:6.1f}" for v in band_db(out, 48000)))
    print("\n움직인 파라미터 (초기 중앙값 -> 적합 중앙값)")
    for n, x, z in fit.moved():
        if abs(z - x) > 0.05 * max(abs(x), 1e-6):
            print(f"  {n:12s} {x:10.3f} -> {z:10.3f}")
    print(f"\n결과: {a.out}_target.wav / {a.out}_fit.wav / {a.out}_track.npz")


if __name__ == "__main__":
    main()
