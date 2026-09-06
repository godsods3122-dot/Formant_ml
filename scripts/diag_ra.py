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
        tilt=torch.zeros(1, t, 1), area=area, tract_rho=rho)
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
def ltv_rect_block(x, H, hop, ir_size):
    """되돌리기 전의 구현 — 창도 보간도 없는 직사각 블록 OLA. 비교 기준이다."""
    b, n = x.shape
    t = H.shape[1]
    wet = fft_convolve(x[:, :t * hop].reshape(b, t, hop), response_to_ir(H, ir_size))
    o = torch.zeros(b, t * hop + ir_size)
    for i in range(t):
        o[:, i * hop:i * hop + wet.shape[-1]] += wet[:, i]
    return o[:, ir_size // 2:ir_size // 2 + n]


def ltv_exact(x, H, hop, ir_size):
    """엄밀한 시변 컨볼루션 y[n] = sum_k h_{n-k}[k] x[n-k] (무차별 계산).

    h_m 은 프레임 임펄스응답을 샘플마다 선형보간한 것이다. 느리지만 정확해서
    블록 구현이 무엇을 더하고 있는지 재는 잣대가 된다.
    """
    ir = response_to_ir(H, ir_size)[0].numpy()
    t = H.shape[1]
    n = t * hop
    xs = x[0, :n].numpy()
    u = ((np.arange(n) % hop) + 0.5) / hop
    i = np.arange(n) // hop
    j = np.minimum(i + 1, t - 1)
    y = np.zeros(n + ir_size)
    for k in range(ir_size):
        y[k:k + n] += ((1 - u) * ir[i, k] + u * ir[j, k]) * xs
    return y[ir_size // 2:ir_size // 2 + n]


def ola_artifact(cfg, out):
    hop, ir = cfg.audio.hop_size, cfg.filt.ir_size
    x, H = out["source"], out["h_harm"]
    n = H.shape[1] * hop
    ref = ltv_exact(x, H, hop, ir)
    now = ltv_filter(x, H, hop, ir)[0, :n].numpy()
    old = ltv_rect_block(x, H, hop, ir)[0, :n].numpy()
    print("=== 3. 시변 필터가 만드는 인공물 (엄밀한 시변 컨볼루션 대비) ===")
    print("   대역        옛 블록 OLA    현재 구현")
    for lo, hi in BANDS[:-1]:
        r = band_db(ref, lo, hi)
        print(f"   {lo // 1000:2d}-{hi // 1000:<2d}k  "
              f"{band_db(old - ref, lo, hi) - r:+10.1f} dB {band_db(now - ref, lo, hi) - r:+11.1f} dB")
    for lab, y in [("옛 블록 OLA", old), ("현재 구현", now)]:
        e = y - ref
        tot = 10 * np.log10((e ** 2).mean() / (ref ** 2).mean())
        per = (e[:len(e) // hop * hop] ** 2).reshape(-1, hop)
        print(f"   {lab}: 전체 {tot:+.1f} dB, 프레임 안 분포 " +
              " ".join(f"{v:.2f}" for v in (per.mean(0) / per.mean()).reshape(8, -1).mean(1)))
    return old - ref


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
