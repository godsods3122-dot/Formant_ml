#!/usr/bin/env python3
"""지터가 150 Hz 아래를 만드는 통로 (docs/MEASUREMENTS.md §18).

HANDOFF §10.3-3 이 "지터·시머의 실현이 저역을 ±10 dB 흔든다. 통로가 무엇인지는
아직 모른다" 로 남겨 둔 항목의 답이다. 성문 소스만 떼어 재므로 성도·난류·적합기가
섞이지 않는다 — 트랙 재렌더 A/B 의 함정(§17)이 원리적으로 없다.

    python scripts/probe_jitter_lowband.py

세 표를 낸다.
  A. 지터 세기 -> 저역. J² 로 자라고 위상 편차는 시간에 안 자란다 (랜덤워크가 아니다).
  B. 변조원 z 의 전력 분포. 지금 25 % 가 150 Hz 위에 있다.
  C. 주기율 구성으로 바꾼 처방. **국소 지터(RAP)를 맞춘 채** 저역만 내린다.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from formant_ml.engine.control import PARAMS                       # noqa: E402
from formant_ml.engine.glottis import GlottalSource                # noqa: E402
from formant_ml.engine.rng import NoiseBank                        # noqa: E402

FS, HOP, T, F0 = 48000, 48, 2000, 257.0      # 2 s, yang_female 의 F0 중앙
BANDS = [(0, 80), (80, 150), (150, 300)]


def const_track(**kw) -> dict:
    return {n: torch.full((1, T), float(kw.get(n, p.default))) for n, p in PARAMS.items()}


def band_db(y: torch.Tensor) -> list[float]:
    """총합 대비 대역 전력 [dB]. 창은 해닝 — 누설이 저역을 만들면 안 된다."""
    x = y[0].detach().numpy().astype(np.float64)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return [10 * np.log10(p[(f >= lo) & (f < hi)].sum() / p.sum() + 1e-30) for lo, hi in BANDS]


def mod_power(z: torch.Tensor) -> list[float]:
    x = z[0].numpy().astype(np.float64)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1e-3)                     # 프레임률 1 kHz
    return [100 * p[(f >= lo) & (f < hi)].sum() / p.sum()
            for lo, hi in [(0, 20), (20, 60), (60, 150), (150, 260), (260, 500)]]


def rap(pert: np.ndarray) -> float:
    """국소 지터 RAP [%] — 이웃 세 주기 평균에 대한 편차.

    **주기 섭동열에서 직접 낸다.** 렌더된 f0 에서 위상 교차로 주기를 세면 48 kHz
    표본 격자가 ±1 샘플(= 0.5 %)로 양자화해 자의 바닥이 0.22 % 로 깔린다 — 지터
    0 에서도 0.224 % 가 나온다. 그 바닥 위에서는 조건 간 비교가 안 된다.
    """
    per = 1.0 / (1.0 + pert)
    m = np.convolve(per, np.ones(3) / 3, "valid")
    return 100 * np.abs(per[1:-1] - m).mean() / per.mean()


def at_cycles(v: torch.Tensor, scale: float) -> np.ndarray:
    """변조원을 성문 주기 시각에서 뽑는다 -> 상대 주기 섭동열."""
    x = v[0].numpy().astype(np.float64) * scale
    return x[np.minimum((np.arange(int(T / (1000.0 / F0))) * (1000.0 / F0)).astype(int), len(x) - 1)]


def phase_dev(f0: torch.Tensor) -> list[float]:
    """위상 편차 [rad] 를 앞 1/8·1/4·1/2·전체 구간에서. 랜덤워크면 sqrt(t) 로 자란다."""
    ph = np.cumsum((f0[0].detach().numpy() - F0) / FS * 2 * math.pi)
    return [float(ph[:k].std()) for k in (len(ph) // 8, len(ph) // 4, len(ph) // 2, len(ph))]


def render(track: dict, seed: int = 0) -> dict:
    return GlottalSource(FS, HOP, speaker="female").forward(track, noise=NoiseBank(seed=seed))


def one_pole(z: torch.Tensor, a: float, poles: int = 1) -> torch.Tensor:
    """엔진(`glottis.forward`)과 같은 1 극 저역통과를 poles 번."""
    g = math.sqrt((1 - a) / (1 + a)) * 2.0
    out = z.clone()
    for _ in range(poles):
        y, prev = torch.zeros_like(out), torch.zeros(1)
        for i in range(out.shape[1]):
            prev = g * (1 - a) * out[:, i] + a * prev
            y[:, i] = prev
        out = y
    return out


def drive(track_z: torch.Tensor, scale: float) -> dict:
    """지터를 f0_target 트랙으로 직접 준다 (엔진 내부 지터는 끈 채)."""
    c = const_track(p_sub=8.0, adduction=0.6, tension=0.5, shimmer=0.0, jitter=0.0)
    c["f0_target"] = F0 * (1.0 + scale * track_z)
    return c


def main() -> None:
    base = dict(p_sub=8.0, adduction=0.6, tension=0.5, f0_target=F0, shimmer=0.0)

    print("A. 지터 세기 -> 저역 (성문 du, 총합 대비 dB)")
    print(f"{'jitter':<10}{'0-80':>9}{'80-150':>9}{'150-300':>9}"
          "   위상 편차 rad (1/8, 1/4, 1/2, 전체)")
    for j in (0.0, 0.002, 0.004, 0.010, 0.020):
        o = render(const_track(jitter=j, **base))
        b = band_db(o["du"])
        print(f"{j:<10.3f}{b[0]:9.1f}{b[1]:9.1f}{b[2]:9.1f}   "
              + " ".join(f"{v:5.2f}" for v in phase_dev(o["f0"])))

    z0 = NoiseBank(seed=0).white("jitter", 0, T, 1, torch.float32, torch.device("cpu"))
    per_ms = 1000.0 / F0
    tt = np.arange(T) / per_ms
    n_cyc = int(T / per_ms) + 2
    seq = z0[0, :n_cyc].numpy()
    variants = {
        "현재: 프레임률 1 극 50 Hz": one_pole(z0, 0.6),
        "주기율 + 영차유지": torch.tensor(
            seq[np.minimum(tt.astype(int), n_cyc - 1)], dtype=torch.float32)[None],
        "주기율 + 선형보간": torch.tensor(
            np.interp(tt, np.arange(n_cyc), seq), dtype=torch.float32)[None],
        "느린 변조 (25 Hz 3 극)": one_pole(z0, 0.85, 3),
    }

    print("\nB. 변조원 z 의 전력 분포 [%]")
    print(f"{'':<26}{'0-20':>8}{'20-60':>8}{'60-150':>8}{'150-260':>8}{'260-500':>8}")
    for k, v in variants.items():
        print(f"{k:<26}" + "".join(f"{x:8.2f}" for x in mod_power(v)))

    print("\nC. **국소 지터 RAP 를 0.40 % 로 맞춘 채** 잰 저역")
    print(f"{'':<26}{'RAP %':>8}{'0-80':>9}{'80-150':>9}{'주기 CV %':>11}")
    for k, v in variants.items():
        sc = 0.004 / float(v.std())
        for _ in range(8):                                  # RAP 를 목표에 맞춰 반복 조정
            r = rap(at_cycles(v, sc))
            sc *= 0.40 / max(r, 1e-9)
        r = rap(at_cycles(v, sc))
        o = render(drive(v, sc))
        b = band_db(o["du"])
        cvp = 100 * float((sc * v).std())
        print(f"{k:<26}{r:8.3f}{b[0]:9.1f}{b[1]:9.1f}{cvp:11.2f}")


if __name__ == "__main__":
    main()
