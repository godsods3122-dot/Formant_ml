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


def analytic_envelope(x: torch.Tensor) -> torch.Tensor:
    """힐베르트 해석신호의 크기 = 순시 포락선. (B, N)"""
    n = x.shape[-1]
    X = torch.fft.rfft(x)
    A = torch.zeros(*x.shape[:-1], n, dtype=torch.complex64, device=x.device)
    A[..., : X.shape[-1]] = X * 2.0
    A[..., 0] = X[..., 0]
    return torch.fft.ifft(A, dim=-1).abs().to(x.dtype)


def low_noise_noise(shape, flatten: float = 1.0, iters: int = 8,
                    device=None, dtype=torch.float32,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """포락선 요동이 작은 잡음 (Pumplin 1985 의 low-noise noise).

    **가우시안 백색잡음은 주어진 스펙트럼에서 가장 '거친' 잡음이다.** 포락선이
    레일리 분포라 변동계수가 0.523 이나 되고, 그 요동의 상당 부분이 20~150 Hz —
    청각 거칠기(roughness)가 가장 예민한 대역 — 에 들어간다. 그래서 스펙트럼이
    아무리 맞아도 '지글거리는' 소리가 난다.

    Pumplin 의 방법: 목표 스펙트럼과 '포락선이 평평할 것'을 번갈아 강제한다.
        (1) 순시 포락선으로 나눈다      -> 포락선이 평평해짐 (스펙트럼이 틀어짐)
        (2) 크기 스펙트럼을 다시 씌운다  -> 스펙트럼 복원 (포락선이 조금 틀어짐)
    몇 번 반복하면 둘 다 거의 만족하는 신호로 수렴한다.

    `flatten` 으로 가우시안(0)과 완전 평탄(1) 사이를 연속적으로 오간다.
    주의: 뒤에서 대역제한을 하면 요동이 일부 되살아난다(Kohlrausch et al.).
    그래도 측정 가능한 만큼 줄어든다.
    """
    x = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    if flatten <= 0.0:
        return x
    mag = torch.fft.rfft(x).abs()                    # 목표 크기 스펙트럼(백색)
    for _ in range(iters):
        e = analytic_envelope(x).clamp_min(1e-6)
        x = x / e.pow(flatten)                       # (1) 포락선 평탄화
        X = torch.fft.rfft(x)
        X = mag * torch.exp(1j * torch.angle(X))     # (2) 스펙트럼 복원
        x = torch.fft.irfft(X, x.shape[-1])
    return x / x.std(-1, keepdim=True).clamp_min(1e-6)


def flatten_fast_envelope(x: torch.Tensor, sample_rate: int, strength: float = 1.0,
                          slow_hz: float = 15.0) -> torch.Tensor:
    """빠른 포락선 요동만 눌러 잡음을 매끄럽게 만든다. 느린 포락선은 보존한다.

    소스에서 low-noise noise 를 만들어도 **뒤에서 대역제한을 하면 요동이 되살아난다**
    (Kohlrausch et al.). 그래서 색을 다 입힌 *다음* 한 번 더 평탄화한다.

    `slow_hz` 아래(마찰음의 시작/끝 램프, 조음이 만드는 변화)는 건드리지 않고,
    그 위(청각 거칠기 대역 20~150 Hz)만 눌러서 나눈다. 그래서 음절 포락선이
    뭉개지지 않는다.
    """
    if strength <= 0.0:
        return x
    n = x.shape[-1]
    e = analytic_envelope(x).clamp_min(1e-6)
    E = torch.fft.rfft(e)
    f = torch.linspace(0, sample_rate / 2, E.shape[-1], device=x.device)
    lp = torch.exp(-0.5 * (f / slow_hz) ** 2)            # 가우시안 저역통과
    slow = torch.fft.irfft(E * lp.to(E.dtype), n).clamp_min(1e-6)
    ratio = (e / slow).clamp(0.2, 5.0)                   # 빠른 요동 성분만
    y = x / ratio.pow(strength)

    # 포락선을 평탄화하면 스펙트럼이 희어진다(측정: 9.2 dB rms). 치찰음의 정체가
    # 그 스펙트럼이므로 그냥 두면 안 된다. 넓게 평활한 크기 비로 되돌린다
    # (미세구조는 건드리지 않으므로 포락선 평탄화는 대부분 살아남는다:
    #  거칠기 0.150 -> 0.034, 스펙트럼 오차 9.2 -> 0.4 dB).
    w = 301
    def _smooth_mag(z):
        p = torch.fft.rfft(z, dim=-1).abs()
        k = torch.ones(1, 1, w, device=z.device, dtype=z.dtype) / w
        p = F.pad(p.reshape(-1, 1, p.shape[-1]), (w // 2, w // 2), mode="replicate")
        return F.conv1d(p, k).reshape(*z.shape[:-1], -1)
    corr = _smooth_mag(x) / _smooth_mag(y).clamp_min(1e-9)
    Y = torch.fft.rfft(y, dim=-1)
    return torch.fft.irfft(Y * corr.to(Y.dtype), n, dim=-1)


class TurbulenceSource(nn.Module):
    """백색잡음 -> (학습된 변조 x 성문동기 AM) -> 난류 소스 파형.

    출력은 색이 입혀지지 않은 '소스'다. 분산은 대략 1 로 유지된다.
    """

    def __init__(self, sample_rate: int, hop_size: int, n_bands: int = 40,
                 init_beta: float = 1.0, init_knee_hz: float = 6.0):
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
        # knee 는 이제 **수 Hz** 다. 20~150 Hz 변조는 청각 거칠기 대역이라
        # 절대 넣으면 안 되고, 난류의 '살아 있음'은 조음 속도(<10 Hz)에서 온다.
        w = (1.0 + (f / knee) ** 2) ** (-beta / 2.0)
        e = torch.fft.irfft(M * w.to(M.dtype), n)
        e = e - e.mean(-1, keepdim=True)
        return e / e.std(-1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------ 합성
    def forward(self, n_frames: int, batch: int = 1, device=None, dtype=torch.float32,
                am_depth: torch.Tensor | None = None,
                glottal_phase: torch.Tensor | None = None,
                roughness: torch.Tensor | None = None,
                voicing: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """반환: 난류 소스 (B, N), N = n_frames * hop_size.

        `roughness` 는 이제 **잡음 자체의 거칠기**다: 0 이면 포락선이 평평한
        low-noise noise, 1 이면 보통의 가우시안 백색잡음. 예전처럼 가우시안 위에
        변조를 *더하는* 게 아니라, 가우시안이 상한이 된다.
        """
        n = n_frames * self.hop_size
        r_mean = 0.0 if roughness is None else float(roughness.detach().mean())
        w = low_noise_noise((batch, n), flatten=1.0 - min(max(r_mean, 0.0), 1.0),
                            device=device, dtype=dtype, generator=generator)

        if roughness is not None:
            # 느린(<10 Hz) 진폭 흔들림만 더한다. 학습 파라미터(beta, knee)가
            # 그 모양을 정한다. 거칠기 대역에는 아무것도 넣지 않는다.
            env = self.modulation_envelope(batch, n, device, dtype, generator)
            r = upsample(roughness, self.hop_size).squeeze(-1).clamp(0.0, 1.0)[..., :n]
            g = (1.0 + 0.6 * r * env).clamp_min(0.05)
            w = w * g / g.pow(2).mean(-1, keepdim=True).clamp_min(1e-6).sqrt()

        if am_depth is not None and glottal_phase is not None:
            frac = torch.frac(glottal_phase[..., :n] / TWO_PI)
            # 성문 개방기(0..0.6)에 에너지가 몰리는 부드러운 창
            env = torch.sin(torch.pi * frac.clamp(0.0, 0.6) / 0.6) ** 2
            d = upsample(am_depth, self.hop_size).squeeze(-1).clamp(0.0, 1.0)[..., :n]
            # **성대가 실제로 진동할 때만** 성문동기 변조가 있다. 무성 마찰음에
            # 이걸 걸면 F0 주기의 진폭변조가 잡음에 얹혀 그대로 거칠기가 된다
            # (실측: /s/ 의 변조 스펙트럼 최대가 정확히 F0=121 Hz 에 있었다).
            if voicing is not None:
                d = d * upsample(voicing, self.hop_size).squeeze(-1
                                                                ).clamp(0.0, 1.0)[..., :n]
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
