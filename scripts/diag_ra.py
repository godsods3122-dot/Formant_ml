"""'라' 진단 — 노이즈 / 고역 조각남 / 타격 지점의 주파수별 위상차.

    PYTHONPATH=src python3 scripts/diag_ra.py

**측정만 한다. 합성 코드를 건드리지 않는다.** 렌더 파라미터는
`scripts/gen_liquid.py` 의 `31_ra.wav` 와 같은 값을 그대로 복사한 것이고,
여기서 다시 렌더하는 이유는 h_harm(성도 응답)을 프레임별로 꺼내
**위상**을 봐야 하기 때문이다 — wav 만 봐서는 필터의 위상을 못 잰다.

측정 넷:
  1. 구간별 대역 레벨      — 실측 녹음과 나란히. 고역이 얼마나 뜨는가.
  2. 빗살 깊이             — 대역 안 파워평균 − 로그평균. 크면 성긴 하모닉,
                             작으면 노이즈 안개. 사람 목소리의 고역은 안개다.
  3. 블록 OLA 인공물       — dsp.core.ltv_filter(창 없는 블록 OLA)를 같은 응답의
                             50 % 교차창 OLA 와 비교한 차이 신호.
  4. 타격 지점의 위상      — h_harm 의 프레임 간 위상차 Δφ(f) 와 그것이 함의하는
                             시간 이동, 그리고 대역별 해제 상승 시각.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, stft

from formant_ml.config import Config, sections_for
from formant_ml.dsp.core import fft_convolve, ltv_filter, response_to_ir
from formant_ml.liquid import contact_dynamics, liquid_syllable
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.utils import ramp

FS = 24000
BANDS = [(200, 1000), (1000, 3000), (3000, 5000), (5000, 8000),
         (8000, 11500), (11500, 11990)]
PHASE_BANDS = [(300, 800), (800, 1500), (1500, 2500), (2500, 4000),
               (4000, 6000), (6000, 9000), (9000, 11500)]
REC = "reference/recordings/ko_liquid_ra-eulla-ara_male_44k.wav"
REC_RA = (0.50, 1.02)          # reference/README.md 의 구간표


# ------------------------------------------------------------------ 렌더/로드
def render_ra():
    """gen_liquid.py 의 31_ra.wav 와 같은 렌더. 내부까지 돌려준다."""
    cfg = Config()
    cfg.filt.n_tract_sections = sections_for(FS, 14.6)
    syn = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    area, contact, pgain, (zf, zb) = liquid_syllable(
        0.55, [(0.0, "l_onset"), (0.27, "l_onset"), (0.38, "a"), (1.0, "a")],
        [(0.0, -0.05), (0.15, -0.05), (0.21, 0.30), (0.55, 0.30)],
        cfg, po=1200.0, n_masses=5, lateral_area_cm2=0.20, tip_overlay=False)
    t = area.shape[1]
    lvl, rho = contact_dynamics(contact[:t], cfg.audio.hop_size, FS,
                                level_drop_db=5.5)
    K, nb = cfg.filt.n_formants, cfg.noise.n_bands
    c = Controls(
        f0=ramp(t, [(0.0, 215.0), (1.0, 180.0)]), harmonic_amp=lvl * pgain,
        rd=torch.full((1, t, 1), 1.1),
        formant_freq=torch.linspace(500, 6000, K).reshape(1, 1, -1)
                          .expand(1, t, K).contiguous(),
        formant_bw=torch.full((1, t, K), 90.0), formant_gain=torch.ones(1, t, K),
        noise_bands=torch.full((1, t, nb), 2e-4),
        noise_entry=torch.zeros(1, t, 1), noise_am=torch.full((1, t, 1), 0.15),
        tilt=torch.full((1, t, 1), 7.0), area=area, tract_rho=rho)
    c.antiformant_freq, c.antiformant_bw = zf, zb
    with torch.no_grad():
        out = syn(c)
    return cfg, out, area, lvl * pgain, rho


def load_ref():
    y, sr = sf.read(REC, always_2d=True)
    y = y.mean(1)[int(REC_RA[0] * sr):int(REC_RA[1] * sr)]
    g = np.gcd(int(sr), FS)
    return resample_poly(y, FS // g, sr // g)


def band_db(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10((X[m] ** 2).mean() + 1e-20)


# ------------------------------------------------------------------ 1) 대역 레벨
def levels(y, ref):
    print("\n=== 1. 구간별 대역 레벨 (dB, 두 신호를 같은 최대진폭으로) ===")
    ref = ref / np.abs(ref).max() * np.abs(y).max()
    print("  " + " " * 26 + "  ".join(f"{lo // 1000}-{hi // 1000}k"
                                      for lo, hi in BANDS))
    for name, t0, t1 in [("폐쇄(유지)  0.02-0.14", 0.02, 0.14),
                         ("타격/해제  0.14-0.21", 0.14, 0.21),
                         ("모음        0.25-0.45", 0.25, 0.45)]:
        for lab, x in [("합성", y), ("실측", ref)]:
            s = x[int(t0 * FS):int(t1 * FS)]
            print(f"  {name:24s}{lab}  " +
                  " ".join(f"{band_db(s, lo, hi):7.1f}" for lo, hi in BANDS))
        print()


# ------------------------------------------------------------------ 2) 빗살 깊이
def comb(y, ref):
    print("=== 2. 빗살 깊이 (dB) — 파워평균 − 로그평균. 크면 성긴 하모닉 빗살 ===")
    print("  " + " " * 26 + "  ".join(f"{lo // 1000}-{hi // 1000}k"
                                      for lo, hi in BANDS[:-1]))
    ref = ref / np.abs(ref).max() * np.abs(y).max()
    for name, t0, t1 in [("폐쇄(유지)  0.02-0.14", 0.02, 0.14),
                         ("모음        0.25-0.45", 0.25, 0.45)]:
        for lab, x in [("합성", y), ("실측", ref)]:
            s = x[int(t0 * FS):int(t1 * FS)]
            f, _, Z = stft(s, FS, nperseg=512, noverlap=448, window="hann")
            P = (np.abs(Z) ** 2).mean(1) + 1e-20
            row = []
            for lo, hi in BANDS[:-1]:
                m = (f >= lo) & (f < hi)
                row.append(f"{10 * np.log10(P[m].mean()) - (10 * np.log10(P[m])).mean():7.1f}")
            print(f"  {name:24s}{lab}  " + " ".join(row))
        print()


# ------------------------------------------------------------------ 3) OLA 인공물
def ltv_crossfaded(x, H, hop, ir_size):
    """같은 응답을 50 % 겹침 Hann 교차창으로 OLA. 비교 기준일 뿐이다."""
    b, n = x.shape
    t = H.shape[1]
    L = 2 * hop
    IR = response_to_ir(H, ir_size)
    w = torch.hann_window(L).reshape(1, 1, L)
    xp = torch.nn.functional.pad(x[:, :t * hop], (hop // 2, hop))
    fr = torch.stack([xp[:, i * hop:i * hop + L] for i in range(t)], 1) * w
    wet = fft_convolve(fr, IR)
    o = torch.zeros(b, t * hop + L + ir_size)
    for i in range(t):
        o[:, i * hop:i * hop + wet.shape[-1]] += wet[:, i]
    d = ir_size // 2 + hop // 2
    return o[:, d:d + n]


def ola_artifact(cfg, out):
    hop, ir = cfg.audio.hop_size, cfg.filt.ir_size
    a = ltv_filter(out["source"], out["h_harm"], hop, ir)[0].numpy()
    b = ltv_crossfaded(out["source"], out["h_harm"], hop, ir)[0].numpy()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - ((a @ b) / (b @ b + 1e-20)) * b
    print("=== 3. 창 없는 블록 OLA 가 만드는 인공물 (교차창 OLA 대비 차이) ===")
    for lo, hi in BANDS[:-1]:
        print(f"   {lo // 1000}-{hi // 1000}k: 인공물 {band_db(d, lo, hi):7.1f}"
              f"  신호 {band_db(a, lo, hi):7.1f}"
              f"  차 {band_db(d, lo, hi) - band_db(a, lo, hi):+6.1f} dB")
    print(f"   전체 {10 * np.log10((d ** 2).mean() / (a ** 2).mean()):.1f} dB")
    T = len(d) // hop
    per = (d[:T * hop] ** 2).reshape(T, hop)
    prof = per.mean(0) / per.mean()
    print("   프레임 안 위치별 인공물 분포(240 샘플 -> 8 구간): " +
          " ".join(f"{v:.2f}" for v in prof.reshape(8, -1).mean(1)))
    fe = 10 * np.log10(per.mean(1) + 1e-20)
    top = sorted(np.argsort(fe)[-6:])
    print("   가장 큰 프레임: " +
          " ".join(f"{i * hop / FS:.3f}s({fe[i]:.0f}dB)" for i in top))
    return d


# ------------------------------------------------------------------ 4) 위상
def phase_at_strike(cfg, out):
    hop = cfg.audio.hop_size
    H = out["h_harm"][0]
    f = np.linspace(0, FS / 2, H.shape[1])
    dph = np.angle(np.exp(1j * np.diff(np.angle(H.numpy()), axis=0)))
    print("\n=== 4a. 타격 지점의 프레임 간 위상차 Δφ(f) [rad] 와 시간 이동 [샘플] ===")
    print("   fr   t(s)  " + " ".join(f"{a // 100 / 10}-{b // 100 / 10}k"
                                      for a, b in PHASE_BANDS))
    for i in range(13, 22):
        r1, r2 = [], []
        for a, b in PHASE_BANDS:
            m = (f >= a) & (f < b)
            r1.append(f"{np.median(np.abs(dph[i, m])):8.2f}")
            r2.append(f"{np.median(dph[i, m] / (2 * np.pi * np.maximum(f[m], 1e-6))) * FS:8.2f}")
        print(f"  {i:3d} {i * hop / FS:5.3f} rad " + " ".join(r1))
        print(f"      {'':5s} smp " + " ".join(r2))


def onset_times(y, t0, t1, label):
    print(f"\n[{label}] 해제에서 대역별 상승 시각 (ms)")
    print("   대역           10%      90%    상승폭   첫 대역 대비")
    ref = None
    for lo, hi in PHASE_BANDS:
        sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos")
        e = np.abs(hilbert(sosfiltfilt(sos, y)))
        k = int(0.004 * FS)
        e = np.convolve(e, np.ones(k) / k, "same")[int(t0 * FS):int(t1 * FS)]
        a, b = e.min(), e.max()
        if b <= a * 1.05:
            print(f"   {lo:5d}-{hi:<5d} 상승 없음")
            continue
        th = lambda p: (t0 + np.argmax(e >= a + p * (b - a)) / FS) * 1000
        if ref is None:
            ref = th(0.1)
        print(f"   {lo:5d}-{hi:<5d} {th(0.1):7.1f} {th(0.9):8.1f}"
              f" {th(0.9) - th(0.1):8.1f}   {ref - th(0.1):+7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="인공물 신호를 wav 로 저장할 폴더")
    args = ap.parse_args()

    cfg, out, area, lvl, rho = render_ra()
    y = out["audio"][0].numpy()
    ref = load_ref()

    levels(y, ref)
    comb(y, ref)
    d = ola_artifact(cfg, out)
    phase_at_strike(cfg, out)
    onset_times(y / np.abs(y).max(), 0.13, 0.26, "합성 31_ra")
    onset_times(ref / np.abs(ref).max(), 0.13, 0.28, "실측 라")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        sf.write(os.path.join(args.out, "ra_ola_artifact.wav"),
                 d / (np.abs(d).max() + 1e-12) * 0.9, FS)
        print(f"\n인공물 신호를 {args.out}/ra_ola_artifact.wav 에 썼다.")


if __name__ == "__main__":
    main()
