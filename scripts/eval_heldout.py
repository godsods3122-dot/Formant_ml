#!/usr/bin/env python3
"""적합 결과를 **검증 실현**에서 채점한다 (docs/MEASUREMENTS.md §24).

    python scripts/eval_heldout.py out/bench3/s34 out/bench3/s34_dw ...

`copyfit` 은 시드 0 으로 적합하고 시드 0 으로 채점한다. 그 점수는 **표본 내**다 —
적합기가 그 시드의 난류 실현을 외운 만큼 부풀어 있다(실측 격차 19.8 %p). 여기서는
같은 트랙을 다른 시드로 다시 렌더해 검증 점수를 낸다.

  학습   : 시드 0 (copyfit 이 쓴 것)
  검증   : 시드 1~N, 평균과 표준편차
  격차   : 학습 − 검증. 클수록 외운 것이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from formant_ml.engine import noise as N                        # noqa: E402
from formant_ml.engine import turbulence as tb                  # noqa: E402
from formant_ml.engine.control import INDEX, ControlTrack       # noqa: E402
from formant_ml.engine.profile import SpeakerProfile            # noqa: E402
from formant_ml.engine.voice import EngineConfig, VoiceEngine   # noqa: E402
from sizzle_ruler import ruler                                  # noqa: E402

FS = 48000.0


def env_db_rate(v: np.ndarray) -> float:
    """마찰 소스 포락의 dB/ms 중앙값 (제어열만 보므로 시드 무관)."""
    g = lambda k: torch.tensor(v[:, INDEX[k]], dtype=torch.float64)[None]
    ps, ac, fg = g("p_sub"), g("a_c"), g("fric_gain")
    u, _ = N.series_flow(ps, torch.full_like(ps, 0.2), ac)
    re, _, _ = N.reynolds(u, ac)
    env = ((((re ** 2 - N.RE_CRIT ** 2).clamp_min(0.0) / N.RE_REF ** 2) ** 1.5
            * (0.1 / ac.clamp_min(0.02))) * fg)[0].numpy()
    return float(np.median(np.abs(np.diff(20 * np.log10(np.maximum(env, 1e-12))))))


def band_excess(stem: str, lo: float = 80.0, hi: float = 150.0) -> float:
    """저장된 합성의 80~150 Hz 가 목표보다 몇 dB 많은가 (총합 대비 비교)."""
    t, sr = sf.read(stem + "_target.wav")
    s, _ = sf.read(stem + "_fit.wav")
    t, s = np.asarray(t, float), np.asarray(s, float)
    if sr != FS:
        t = np.interp(np.arange(0, len(t) / sr, 1 / FS), np.arange(len(t)) / sr, t)
    n = min(len(t), len(s))
    def band(x):
        P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / FS)
        return 10 * np.log10(P[(f >= lo) & (f < hi)].sum() / P.sum() + 1e-30)
    return band(s[:n]) - band(t[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--seeds", type=int, default=8, help="검증 시드 수 (1..N)")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    prof = SpeakerProfile.load(a.profile)

    print(f"검증 시드 1~{a.seeds}.  학습 시드는 0 (copyfit 이 쓴 것).")
    print(f"{'조건':<22}{'스펙 학습':>10}{'스펙 검증':>10}{'±':>7}{'격차':>7}"
          f"{'첨두/s 검증':>12}{'±':>6}{'목표':>6}{'비':>6}{'dB/ms':>8}"
          f"{'80-150 초과':>12}")
    for stem in a.stems:
        d = np.load(stem + "_track.npz", allow_pickle=True)
        trk = ControlTrack(values=d["values"], frame_ms=float(d["frame_ms"]))
        tgt, sr = sf.read(stem + "_target.wav")
        tgt = np.asarray(tgt, float)
        if sr != FS:
            tgt = np.interp(np.arange(0, len(tgt) / sr, 1 / FS),
                            np.arange(len(tgt)) / sr, tgt)
        sm, pk = [], []
        for s in range(0, a.seeds + 1):
            eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0,
                                           speaker="female", residual=False, seed=s), prof)
            y = eng.render(trk)
            n = min(len(y), len(tgt))
            yy = y[:n] * (np.sqrt((tgt[:n] ** 2).mean())
                          / (np.sqrt((y[:n] ** 2).mean()) + 1e-20))
            sm.append(tb.spectrum_match(tgt[:n], yy, FS))
            pk.append(ruler(yy)["peaks_per_s"])
        sm, pk = np.array(sm), np.array(pk)
        tp = ruler(tgt)["peaks_per_s"]                      # 목표의 첨두 밀도
        print(f"{os.path.basename(stem):<22}{sm[0]:10.1f}{sm[1:].mean():10.1f}"
              f"{sm[1:].std():7.1f}{sm[0] - sm[1:].mean():7.1f}"
              f"{pk[1:].mean():12.0f}{pk[1:].std():6.0f}{tp:6.0f}"
              f"{pk[1:].mean() / max(tp, 1e-9):6.2f}"
              f"{env_db_rate(d['values']):8.2f}{band_excess(stem):12.1f}")


if __name__ == "__main__":
    main()
