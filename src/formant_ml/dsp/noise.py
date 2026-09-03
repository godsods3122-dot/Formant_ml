"""난류(turbulence) 소스 — **학습되는 노이즈**.

기존 구현은 `white noise x 대역게인` 이었다. 그건 정상(stationary) 백색잡음이라
실제 마찰음/기식음과 두 가지가 다르다.

1. **진짜 난류는 정상적이지 않다.** 협착부 제트는 스스로 요동해서 진폭이 계속
   흔들린다. 그 변조 스펙트럼은 대략 1/f^β (β≈1) 로, 백색이 아니다. 이 변조가
   없으면 합성음이 '테이프 히스' 처럼 죽은 소리가 되고, 반대로 모델이 크기
   스펙트럼만 맞추다 보면 프레임마다 같은 벡터를 반복해 **미세하게 주기적인**
   텍스처가 생긴다.
2. **소스의 색(color)은 화자/조음에 공통인 부분이 있다.** 그 공통 부분을 매
   프레임 다시 예측하게 하면 낭비이고 불안정하다.

그래서 이 모듈은 두 종류의 파라미터를 가진다.

* **학습 파라미터**(nn.Parameter, 화자/코퍼스 전체에서 하나):
  난류 소스의 스펙트럼 사전 `log_prior`, 변조 스펙트럼의 기울기 `beta` 와
  꺾임 주파수 `knee`. => "노이즈 자체를 학습한다".
* **제어 파라미터**(프레임별, 인코더/스크립트가 준다):
  대역게인, 성문동기 AM 깊이, 난류 거칠기 `roughness`.

색칠(대역게인 x 치찰음 필터 x 성도)은 여기서 하지 않고 합성기에서 한 번에
LTV 필터로 적용한다 (IR 절단 오차를 두 번 겪지 않기 위해서).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import TWO_PI, upsample


class TurbulenceSource(nn.Module):
    """백색잡음 -> (학습된 변조 x 성문동기 AM) -> 난류 소스 파형.

    출력은 색이 입혀지지 않은 '소스'다. 분산은 대략 1 로 유지된다.
    """

    def __init__(self, sample_rate: int, hop_size: int, n_bands: int = 40,
                 init_beta: float = 1.0, init_knee_hz: float = 120.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_bands = n_bands
        # 난류 소스의 스펙트럼 사전 (평균 0 으로 초기화 = 백색에서 시작)
        self.log_prior = nn.Parameter(torch.zeros(n_bands))
        # 변조 스펙트럼: |M(f)| = (1 + (f/knee)^2)^(-beta/2)
        self.raw_beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.raw_knee = nn.Parameter(torch.tensor(float(init_knee_hz)).log())

    # ------------------------------------------------------------------ 사전
    def spectral_prior(self) -> torch.Tensor:
        """학습된 난류 색 (n_bands,). 기하평균 1 로 정규화해 게인과 안 싸우게 한다."""
        lp = self.log_prior - self.log_prior.mean()
        return torch.exp(lp.clamp(-4.0, 4.0))

    def modulation_envelope(self, b: int, n: int, device, dtype,
                            generator: torch.Generator | None = None) -> torch.Tensor:
        """1/f^beta 변조 포락선 (B, N), 평균 1 / 표준편차 1 로 정규화."""
        m = torch.randn(b, n, device=device, dtype=dtype, generator=generator)
        M = torch.fft.rfft(m)
        f = torch.linspace(0, self.sample_rate / 2, M.shape[-1],
                           device=device, dtype=dtype)
        beta = F.softplus(self.raw_beta).clamp(0.05, 4.0)
        knee = self.raw_knee.exp().clamp(10.0, 4000.0)
        w = (1.0 + (f / knee) ** 2) ** (-beta / 2.0)
        e = torch.fft.irfft(M * w.to(M.dtype), n)
        e = e - e.mean(-1, keepdim=True)
        return e / e.std(-1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------ 합성
    def forward(self, n_frames: int, batch: int = 1, device=None, dtype=torch.float32,
                am_depth: torch.Tensor | None = None,
                glottal_phase: torch.Tensor | None = None,
                roughness: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """반환: 난류 소스 (B, N), N = n_frames * hop_size."""
        n = n_frames * self.hop_size
        w = torch.randn(batch, n, device=device, dtype=dtype, generator=generator)

        if roughness is not None:
            env = self.modulation_envelope(batch, n, device, dtype, generator)
            r = upsample(roughness, self.hop_size).squeeze(-1).clamp(0.0, 1.0)
            r = r[..., :n]
            # 1 + r*env 를 아래에서 잘라내면 평균이 올라가므로 다시 정규화한다.
            g = (1.0 + r * env).clamp_min(0.0)
            g = g / g.pow(2).mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
            w = w * g

        if am_depth is not None and glottal_phase is not None:
            frac = torch.frac(glottal_phase[..., :n] / TWO_PI)
            # 성문 개방기(0..0.6)에 에너지가 몰리는 부드러운 창
            env = torch.sin(torch.pi * frac.clamp(0.0, 0.6) / 0.6) ** 2
            d = upsample(am_depth, self.hop_size).squeeze(-1).clamp(0.0, 1.0)[..., :n]
            w = w * ((1.0 - d) + d * 2.0 * env)
        return w


class NoiseSource(TurbulenceSource):
    """이전 API 호환용 얇은 래퍼 (색칠까지 한 번에 한다)."""

    def __init__(self, sample_rate: int, hop_size: int, n_freq: int = 513,
                 ir_size: int = 256, n_bands: int = 40):
        super().__init__(sample_rate, hop_size, n_bands)
        self.n_freq = n_freq
        self.ir_size = ir_size

    def forward(self, band_gains: torch.Tensor, am_depth=None, glottal_phase=None,
                roughness=None, generator=None) -> torch.Tensor:
        from .core import ltv_filter
        from .filters import bands_to_response
        b, t, _ = band_gains.shape
        w = super().forward(t, b, band_gains.device, band_gains.dtype,
                            am_depth, glottal_phase, roughness, generator)
        H = bands_to_response(band_gains * self.spectral_prior(), self.n_freq,
                              min_phase=True)
        return ltv_filter(w, H, self.hop_size, self.ir_size)
