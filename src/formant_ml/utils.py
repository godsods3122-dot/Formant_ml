"""오디오 입출력/제어신호 만들기 보조."""
from __future__ import annotations

import math
import os

import torch


def save_wav(path: str, x: torch.Tensor, sample_rate: int = 24000,
             normalize: bool = True) -> None:
    import soundfile as sf
    y = x.detach().squeeze().to(torch.float32).cpu()
    if normalize:
        y = y / y.abs().max().clamp_min(1e-6) * 0.9
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, y.numpy(), sample_rate)


def load_wav(path: str, sample_rate: int = 24000) -> torch.Tensor:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = torch.from_numpy(y).mean(dim=1)
    if sr != sample_rate:
        raise ValueError(f"{path}: {sr} Hz != {sample_rate} Hz (리샘플 필요)")
    return y


def n_frames(seconds: float, sample_rate: int = 24000, hop: int = 240) -> int:
    return int(seconds * sample_rate / hop)


def ramp(t: int, values, device=None) -> torch.Tensor:
    """구간별 목표값을 선형보간해 (1, T, 1) 제어신호로 만든다.

    values: [(위치0~1, 값), ...]
    """
    pos = torch.tensor([p for p, _ in values], dtype=torch.float32, device=device)
    val = torch.tensor([v for _, v in values], dtype=torch.float32, device=device)
    x = torch.linspace(0, 1, t, device=device)
    idx = torch.searchsorted(pos, x.clamp(pos[0], pos[-1])).clamp(1, len(pos) - 1)
    p0, p1 = pos[idx - 1], pos[idx]
    v0, v1 = val[idx - 1], val[idx]
    w = ((x - p0) / (p1 - p0).clamp_min(1e-6)).clamp(0, 1)
    return (v0 + (v1 - v0) * w).reshape(1, t, 1)


def band_bump(n_bands: int, center_hz: float, bw_hz: float, gain: float,
              sample_rate: int = 24000, floor: float = 1e-4) -> torch.Tensor:
    """가우시안 형태의 노이즈 대역 게인 (n_bands,)."""
    f = torch.linspace(0, sample_rate / 2, n_bands)
    return gain * torch.exp(-0.5 * ((f - center_hz) / (bw_hz / 2)) ** 2) + floor


def vibrato(t: int, f0: float, rate_hz: float = 5.0, depth_cents: float = 30.0,
            frame_rate: float = 100.0) -> torch.Tensor:
    n = torch.arange(t, dtype=torch.float32) / frame_rate
    return (f0 * 2 ** (depth_cents / 1200 * torch.sin(2 * math.pi * rate_hz * n))
            ).reshape(1, t, 1)


def band_shelf(n_bands: int, cutoff_hz: float, gain: float = 1.0,
               sample_rate: int = 24000, slope_oct: float = 1.0,
               floor: float = 1e-4) -> torch.Tensor:
    """넓은 대역 난류 소스 (n_bands,). 협착부 제트 자체는 광대역이다.

    마찰음의 스펙트럼 모양은 소스가 아니라 *앞공동 공진*(치찰음 필터)이 만든다.
    소스 대역게인으로 모양까지 만들면 두 손잡이가 서로 싸워서, 치찰음 파라미터를
    돌려도 소리가 안 바뀐다.
    """
    f = torch.linspace(0, sample_rate / 2, n_bands).clamp_min(20.0)
    o = torch.log2(f / max(cutoff_hz, 20.0))
    return gain * torch.sigmoid(o * 3.0 * slope_oct) + floor
