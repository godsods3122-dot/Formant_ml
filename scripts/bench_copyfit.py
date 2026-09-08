"""복사합성 적합기의 회귀 벤치 — 같은 구간을 늘 같은 방식으로 잰다.

왜 스크립트로 두는가
--------------------
적합기를 손댈 때마다 "좋아졌나" 를 감으로 말하지 않으려면 고정된 구간과 고정된 자가
있어야 한다. 여기 구간은 `docs/MEASUREMENTS.md` §7.5 이후로 계속 쓰는 것들이라
과거 값과 직접 비교된다.

무엇을 재는가
-------------
* 위상 모양 오차 — 하모닉 차수 비례 성분(= 순수 시간 이동)을 뺀 나머지. 곧 필터
  위상 응답의 오차다. 목표 20° 이하.
* 조화 SNR — 조화 성분만 파형으로 비교. 난류는 원리적으로 재현 불가라 뺀다.
* 포먼트 이동 속도 — 적합된 궤적이 물리적인가. 실측 중앙과 대조한다.
* 비조화 비율 — 목표와 합성 각각. 벌어지면 합성이 덜 주기적이라는 뜻이다.

프레임은 유성이고 봉우리 대비 −25 dB 보다 큰 것만 센다. 진폭이 0 에 가까운 꼬리에서
잰 위상은 잡음이라 한때 이걸 빠뜨려 "설측 마지막 5분의 1이 151.5°" 라고 잘못 적었다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine.analyze import analyze
from formant_ml.engine.denoise import denoise, noise_profile
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine
from formant_ml.engine.waveform import decompose, phase_error, seg_snr

FS = 48000.0
HOP = 48
GATE_DB = -25.0

# (이름, 파일, 시작 s, 끝 s, 실측 F1·F2 속도 중앙 Hz/ms)
#
# 속도는 이 창들에서 직접 잰 값이다 (Praat 1 ms, 유성 프레임만, 인접차의 중앙값).
# 전역 통계가 아니라 **이 구간의** 값이라는 게 중요하다 — 지속 모음은 1.6/1.9 인데
# 설측은 F2 가 26.5 다. 하나의 전역 상한으로는 이 차이를 표현할 수 없다.
SEGMENTS = [
    ("여 긴 /아/",     "data/ref/female_yang_ilin-ilsil.wav", 2.86, 3.06, 1.6, 1.9),
    ("여 ㅆ+이 (씨)",  "data/ref/female_yang_ilin-ilsil.wav", 1.12, 1.26, 1.9, 15.6),
    ("여 설측 (닐)",   "data/ref/female_yang_ilin-ilsil.wav", 1.04, 1.16, 4.1, 26.5),
    ("여 탄음 (이리)", "data/ref/female_yang_ilin-ilsil.wav", 0.75, 0.83, 2.8, 23.0),
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


def bench(name, path, t0, t1, prof, verbose=False):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    y = denoise(y, sr, noise_profile(y, sr))
    seg = y[int(t0 * sr):int(t1 * sr)]
    tr = analyze(seg, sr, prof, HOP, t0=t0, full=y)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0,
                                   speaker="female", residual=False), prof)
    f = CopySynthFitter(eng, seg, sr, tr)
    f.fit_staged(verbose=verbose)
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
    return dict(
        phase=float(np.median(ph)) if ph else float("nan"),
        snr=seg_snr(ht, hs, FS),
        v1=float(np.median(np.abs(np.diff(np.asarray(ft["f1"]))))),
        v2=float(np.median(np.abs(np.diff(np.asarray(ft["f2"]))))),
        nh_synth=nonharmonic(out, hs, voi, nf),
        nh_target=nonharmonic(tgt, ht, voi, nf),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile)
    torch.set_num_threads(2)
    print(f"{'구간':>16s} {'위상':>8s} {'조화SNR':>9s} "
          f"{'F1속도 (실측)':>16s} {'F2속도 (실측)':>16s} {'비조화 합/목':>14s}")
    for name, path, t0, t1, r1, r2 in SEGMENTS:
        r = bench(name, path, t0, t1, prof, a.verbose)
        print(f"{name:>16s} {r['phase']:7.1f}° {r['snr']:8.2f} "
              f"{r['v1']:8.1f} ({r1:4.1f}) {r['v2']:8.1f} ({r2:4.1f}) "
              f"{r['nh_synth']:6.1f}% /{r['nh_target']:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
