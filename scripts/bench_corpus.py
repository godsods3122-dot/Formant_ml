"""복사합성 적합기의 회귀 벤치 — **코퍼스**의 고정 구간으로.

왜 새로 만드는가
----------------
`bench_copyfit.py` 는 `data/ref/female_yang_ilin-ilsil.wav` 의 네 구간을 쓴다. 그 파일은
레포에 없다 (`.gitignore` 의 `*.wav`). 그리고 직전 세션이 남긴 최대 약점이 바로
**"표본이 파일 하나다"** 였다 (docs/HANDOFF.md §5). 여기 구간은 orphan 브랜치의
코퍼스 205 개(45 분, 같은 화자)에서 자동 선별로 고른 것이다.

무엇을 재는가
-------------
기존 벤치의 네 지표(위상·조화 SNR·궤적 속도·비조화)에 **난류를 정직하게 재는 자**를
더했다:

* `보정정밀` — 실현 잡음을 뺀 정밀 일치율. 완벽한 모형에서 100 % 로 수렴한다.
  기존 `정밀` 은 치찰음에서 34 % 가 상한이라 애초에 95 % 를 말할 수 없다.
* `상한` — 그 구간에서 기존 자가 원리적으로 넘을 수 없는 값. `정밀` 이 이 근처면
  **모형이 나쁜 게 아니라 자가 바닥에 닿은 것**이다.
* `무게중심` / `대역MAE` — 치찰음의 조음 위치가 맞는가 (Jongman et al. 2000).

구간은 어떻게 골랐나
--------------------
`engine/segment.py` 로 전 코퍼스를 훑고, 마찰 중 **치찰도 > 8 dB 이고 무게중심
> 6 kHz** 인 것만 남겼다. 이 거르기 전에는 무게중심 중앙값이 5175 Hz 로 프로파일
실측(8100 Hz)과 3 kHz 어긋났다 — 호흡과 기식이 섞여 있었기 때문이다. 거른 뒤에는
7997 Hz 로 실측과 맞는다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine import turbulence as tb
from formant_ml.engine.analyze import analyze
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine
from formant_ml.engine.waveform import decompose, phase_error, seg_snr

FS = 48000.0
HOP = 48
GATE_DB = -25.0

# (이름, 파일, 시작 s, 끝 s, 유형)
SEGMENTS = [
    ("치찰 ㅅ (34)",   "data/voices/yang_00000034.wav", 0.510, 0.665, "sib"),
    ("치찰 ㅆ (00)",   "data/voices/yang_00000000.wav", 5.090, 5.235, "sib"),
    ("치찰+모음 (11)", "data/voices/yang_00000011.wav", 14.495, 14.725, "sib"),
    ("모음 (00)",      "data/voices/yang_00000000.wav", 16.500, 16.700, "vowel"),
]


def nonharmonic(sig, har, voi, nf):
    v = []
    for i in range(nf):
        a, b = i * HOP, (i + 1) * HOP
        if b > len(sig) or not voi[i]:
            continue
        eh = (har[a:b] ** 2).sum()
        er = ((sig[a:b] - har[a:b]) ** 2).sum()
        if eh + er > 0:
            v.append(er / (eh + er))
    return 100 * float(np.median(v)) if v else float("nan")


def bench(path, t0, t1, prof, verbose=False, iters=(200, 150, 800)):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    y = denoise(y, sr, noise_profile(y, sr))
    seg = y[int(t0 * sr):int(t1 * sr)]
    tr = analyze(seg, sr, prof, HOP, t0=t0, full=y)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0,
                                   speaker="female", residual=False), prof)
    f = CopySynthFitter(eng, seg, sr, tr)
    rep = f.fit_staged(global_iters=iters[0], stage_iters=iters[1],
                       phase_iters=iters[2], verbose=verbose, log_every=10 ** 9)
    out = f.render()
    tgt = f.target[0].numpy()
    n = min(len(tgt), len(out))
    tgt, out = tgt[:n], out[:n]
    f0 = np.asarray(tr["f0_target"])
    voi = np.asarray(tr.voiced).astype(bool)
    at, ht, _ = decompose(tgt, FS, f0, HOP)
    as_, hs, _ = decompose(out, FS, f0, HOP)
    nf = min(len(at), len(as_), len(voi))
    rms = np.array([np.sqrt((tgt[i * HOP:(i + 1) * HOP] ** 2).mean() + 1e-20)
                    for i in range(nf)])
    db = 20 * np.log10(rms / rms.max() + 1e-12)
    ph = [phase_error(at[i], as_[i])[0] for i in range(nf)
          if voi[i] and db[i] > GATE_DB and np.any(at[i])]
    ph = [x for x in ph if np.isfinite(x)]
    ft = f.result_track()
    fid = f.fidelity()
    cmp = tb.compare(tgt, out, FS)
    return dict(
        env=rep.env, fine=rep.fine, **fid,
        phase=float(np.median(ph)) if ph else float("nan"),
        snr=seg_snr(ht, hs, FS),
        v1=float(np.median(np.abs(np.diff(np.asarray(ft["f1"]))))),
        v2=float(np.median(np.abs(np.diff(np.asarray(ft["f2"]))))),
        nh_synth=nonharmonic(out, hs, voi, nf),
        nh_target=nonharmonic(tgt, ht, voi, nf),
        centroid_err=cmp["centroid_err"], band_mae=cmp["band_mae_db"],
        sib_err=cmp["sibilance_err"], mod_mae=cmp["mod_mae_pct"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quick", action="store_true", help="예산을 1/4 로 (개발용)")
    ap.add_argument("--only", default=None, help="이름에 이 문자열이 든 구간만")
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile)
    torch.set_num_threads(2)
    it = (60, 40, 250) if a.quick else (200, 150, 800)
    print(f"{'구간':>16s} {'포락':>6s} {'정밀':>6s} {'보정정밀':>8s} {'상한':>6s} "
          f"{'위상':>7s} {'조화SNR':>8s} {'무게중심':>9s} {'대역MAE':>8s} "
          f"{'변조MAE':>8s} {'비조화 합/목':>13s}")
    for name, path, t0, t1, kind in SEGMENTS:
        if a.only and a.only not in name:
            continue
        if not os.path.exists(path):
            print(f"{name:>16s}  (파일 없음: {path})")
            continue
        t = time.time()
        r = bench(path, t0, t1, prof, a.verbose, it)
        print(f"{name:>16s} {r['env']:6.2f} {r['fine']:6.2f} {r['fine_corr']:8.2f} "
              f"{r['floor']:6.2f} {r['phase']:6.1f}° {r['snr']:8.2f} "
              f"{r['centroid_err']:8.0f}  {r['band_mae']:8.2f} {r['mod_mae']:8.2f} "
              f"{r['nh_synth']:6.1f}% /{r['nh_target']:5.1f}%  ({time.time()-t:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
