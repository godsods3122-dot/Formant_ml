#!/usr/bin/env python3
"""지글거림을 **가우시안 기준에 대어** 재는 자 + 소스/성도 귀속.

    python scripts/sizzle_ruler.py out/bench3/s34 [out/bench3/s34_r0.01 ...]
    python scripts/sizzle_ruler.py out/bench3/s34 --parts     # 소스/성도로 갈라 본다

왜 이 자인가
------------
`copyfit` 의 지글거림 표는 목표 대비 **비율**만 낸다. 그러면 목표가 얼마나 맥동하는
것이 정상인지를 모른 채 비교하게 된다. 대역제한 가우시안 잡음의 힐베르트 포락은
변조 지수가 sqrt(4/π − 1) = 0.523, 첨도가 3.3 으로 **이론값이 있다.** 그 기준에
대면 "목표도 원래 맥동한다" 와 "우리만 튄다" 가 갈린다.

실측 (yang_34 치찰음, 포락 92.08 % 로 게이트를 통과한 구간인데 귀에 지글거린다):

    포락 변조 지수   가우시안 0.52 / 목표 0.96 / 합성 1.32
    포락 첨도        가우시안 3.3  / 목표 5.3  / 합성 8.7
    첨두(국소평균 3 배 초과) 밀도   목표 97/s / 합성 323/s

즉 문제는 "변조가 깊다" 가 아니라 **드문 첨두가 3 배 많다** — 임펄스성이다.
`--parts` 로 갈라 보면 그 첨두는 성도가 아니라 **마찰 소스에서 태어난다**.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FS = 48000.0
GAUSS_MI = float(np.sqrt(4.0 / np.pi - 1.0))       # 0.5227


def band_envelope(x: np.ndarray, fs: float = FS,
                  lo: float = 4000.0, hi: float = 12000.0) -> np.ndarray:
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    X[(f < lo) | (f >= hi)] = 0.0
    return np.abs(np.fft.irfft(X, n) + 1j * np.fft.irfft(-1j * X, n))


def ruler(x: np.ndarray, fs: float = FS, local_ms: float = 10.0,
          thr: float = 3.0) -> dict:
    """첨두 밀도 / 변조 지수 / 첨도. 첨두는 **국소 평균**으로 정규화한 뒤 센다 —
    전체 진폭 포락(개시 램프)이 첨두로 세어지지 않게."""
    if np.abs(x).max() < 1e-12:
        return dict(peaks_per_s=float("nan"), mi=float("nan"), kurt=float("nan"))
    e = band_envelope(x, fs)
    w = max(1, int(local_ms * fs / 1000))
    loc = np.convolve(e, np.ones(w) / w, "same")
    z = e / np.maximum(loc, 1e-20)
    pk = (z[1:-1] > z[:-2]) & (z[1:-1] > z[2:]) & (z[1:-1] > thr)
    return dict(peaks_per_s=float(pk.sum() / (len(x) / fs)),
                mi=float(e.std() / (e.mean() + 1e-20)),
                kurt=float(((e - e.mean()) ** 4).mean() / (e.var() ** 2 + 1e-30)))


def _load(stem: str) -> tuple[np.ndarray, np.ndarray]:
    t, sr_t = sf.read(stem + "_target.wav")
    s, _ = sf.read(stem + "_fit.wav")
    t, s = np.asarray(t, float), np.asarray(s, float)
    if sr_t != FS:
        t = np.interp(np.arange(0, len(t) / sr_t, 1 / FS),
                      np.arange(len(t)) / sr_t, t)
    n = min(len(t), len(s))
    return t[:n], s[:n]


def parts_table(stem: str) -> None:
    """적합된 트랙을 **copyfit 과 같은 설정**으로 다시 렌더해 신호 경로별로 잰다.

    재렌더 A/B 의 함정(docs/MEASUREMENTS.md §17)을 피하려고, 조건을 바꾸지 않고
    **한 렌더의 내부 신호만** 본다. 저장본과의 상관을 같이 찍어 같은 신호임을 보인다.
    """
    import torch
    from formant_ml.engine.control import ControlTrack
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    torch.set_num_threads(2)

    d = np.load(stem + "_track.npz", allow_pickle=True)
    trk = ControlTrack(values=d["values"], frame_ms=float(d["frame_ms"]))
    prof = SpeakerProfile.load("profiles/yang_female.json")
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female",
                                   residual=False), prof)
    y, parts = eng.render(trk, return_parts=True)
    tgt, fit = _load(stem)
    n = min(len(y), len(fit))
    a = y[:n] * (np.sqrt((fit[:n] ** 2).mean()) / (np.sqrt((y[:n] ** 2).mean()) + 1e-20))
    print(f"  재렌더 검증: 저장본과의 상관 {np.corrcoef(a, fit[:n])[0, 1]:.4f}")
    print(f"  {'신호':<26}{'첨두/s':>9}{'변조지수':>10}{'첨도':>9}")
    rows = [("목표 (녹음)", tgt[:n])]
    rows += [(lab, parts[k][:n]) for k, lab in
             [("fric", "마찰 소스 (성도 전)"), ("asp", "기식 소스 (성도 전)"),
              ("du", "성문 소스 du"), ("front_path", "앞공동 경로 출력"),
              ("glottal_path", "성문 경로 출력"), ("audio", "최종 출력")] if k in parts]
    for lab, v in rows:
        r = ruler(v)
        print(f"  {lab:<26}{r['peaks_per_s']:9.0f}{r['mi']:10.2f}{r['kurt']:9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--parts", action="store_true", help="첫 stem 을 신호 경로별로 가른다")
    a = ap.parse_args()

    g = np.random.default_rng(0).standard_normal(48000)
    gr = ruler(g)
    print(f"가우시안 기준 (이론 변조 지수 {GAUSS_MI:.3f}): "
          f"첨두 {gr['peaks_per_s']:.0f}/s  변조지수 {gr['mi']:.2f}  첨도 {gr['kurt']:.2f}\n")
    print(f"{'구간':<26}{'첨두/s 목표':>12}{'합성':>8}{'배수':>7}"
          f"{'첨도 목표':>11}{'합성':>8}")
    for stem in a.stems:
        tgt, fit = _load(stem)
        rt, rf = ruler(tgt), ruler(fit)
        print(f"{os.path.basename(stem):<26}{rt['peaks_per_s']:12.0f}{rf['peaks_per_s']:8.0f}"
              f"{rf['peaks_per_s'] / max(rt['peaks_per_s'], 1e-9):7.2f}"
              f"{rt['kurt']:11.2f}{rf['kurt']:8.2f}")
    if a.parts:
        print(f"\n신호 경로별 — {a.stems[0]}")
        parts_table(a.stems[0])


if __name__ == "__main__":
    main()
