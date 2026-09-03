"""난류(turbulence) 노이즈 소스: 치찰음/기식음/파열음의 물리적 근원.

두 가지를 분리해서 다룬다.
1) 마찰 노이즈: 협착부에서 생기는 정상 난류. 성문 위상과 무관.
2) 기식 노이즈: 성문 개방기에 동기화되어 진폭변조된 난류.
   이 펄스동기 변조가 없으면 합성음이 '두 개의 층(하모닉 + 쉬익 소리)'처럼 들린다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .core import TWO_PI, ltv_filter, upsample
from .filters import bands_to_response


class NoiseSource(nn.Module):
    def __init__(self, sample_rate: int, hop_size: int, n_freq: int = 513,
                 ir_size: int = 256):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_freq = n_freq
        self.ir_size = ir_size

    def forward(self, band_gains: torch.Tensor, am_depth: torch.Tensor | None = None,
                glottal_phase: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """band_gains: (B, T, n_bands), am_depth: (B, T, 1), glottal_phase: (B, N)."""
        b, t, _ = band_gains.shape
        n = t * self.hop_size
        w = torch.randn(b, n, device=band_gains.device, dtype=band_gains.dtype,
                        generator=generator)

        if am_depth is not None and glottal_phase is not None:
            frac = torch.frac(glottal_phase / TWO_PI)             # 0..1 주기 내 위치
            # 개방기(0..0.6)에 에너지가 몰리는 부드러운 창
            env = torch.sin(torch.pi * frac.clamp(0.0, 0.6) / 0.6) ** 2
            d = upsample(am_depth, self.hop_size).squeeze(-1).clamp(0.0, 1.0)
            w = w * ((1.0 - d) + d * 2.0 * env)

        H = bands_to_response(band_gains, self.n_freq, min_phase=True)
        return ltv_filter(w, H, self.hop_size, self.ir_size)
