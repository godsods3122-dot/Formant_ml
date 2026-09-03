"""성문 소스: LF(Liljencrants-Fant) 파형 사전 + 대역제한 가산합성 오실레이터.

- Rd(=긴장/기식 정도) 그리드로 LF 유량미분 파형을 미리 계산해 사전(bank)으로 둔다.
- 학습 시에는 연속적인 Rd 값에 대해 인접 두 테이블의 *스펙트럼*을 선형보간하므로
  Rd 에 대해 미분가능하다 (GOLF, Yu & Fazekas 2023 와 같은 전략).
- 합성은 하모닉 가산합성으로 하여 나이퀴스트 위 성분을 원천적으로 제거한다
  (웨이브테이블 직접 재생 시 발생하는 에일리어싱 = '디지털 잡음'을 없앤다).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core import TWO_PI, upsample


# ----------------------------------------------------------------------------- LF
def _rd_to_timing(rd: float):
    """Fant 의 Rd -> (tp, te, ta, tc) 정규화 타이밍(T0 = 1)."""
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    denom = 0.11 * rd - ra * (0.5 + 1.2 * rk)
    rg = (rk / 4.0) * (0.5 + 1.2 * rk) / max(denom, 1e-4)
    ra = min(max(ra, 1e-4), 0.3)
    tp = 1.0 / (2.0 * max(rg, 0.2))
    te = min(tp * (1.0 + rk), 0.99)
    return tp, te, ra, 1.0


def _lf_waveform(rd: float, n: int) -> torch.Tensor:
    """정규화 주기 [0,1) 위의 LF 유량미분 파형 (n,)."""
    tp, te, ta, tc = _rd_to_timing(rd)
    wg = math.pi / tp

    # 1) epsilon: eps*ta = 1 - exp(-eps*(tc-te))
    eps = 1.0 / max(ta, 1e-4)
    for _ in range(80):
        eps = (1.0 - math.exp(-eps * (tc - te))) / max(ta, 1e-6)

    # 2) alpha: 한 주기의 면적이 0 (유량이 다시 0으로 닫힘)
    def area(alpha: float) -> float:
        e0 = -1.0 / (math.exp(alpha * te) * math.sin(wg * te))
        a_open = e0 * (
            math.exp(alpha * te) * (alpha * math.sin(wg * te) - wg * math.cos(wg * te))
            + wg
        ) / (alpha * alpha + wg * wg)
        d = tc - te
        a_ret = -(1.0 / (eps * ta)) * ((1.0 - math.exp(-eps * d)) / eps
                                       - d * math.exp(-eps * d))
        return a_open + a_ret

    lo, hi = -50.0, 200.0
    f_lo = area(lo)
    for _ in range(120):                       # 이분법 (안정적)
        mid = 0.5 * (lo + hi)
        f_mid = area(mid)
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)

    t = torch.linspace(0.0, 1.0, n + 1)[:-1]
    e0 = -1.0 / (math.exp(alpha * te) * math.sin(wg * te))
    open_phase = e0 * torch.exp(alpha * t) * torch.sin(wg * t)
    d = t - te
    ret_phase = -(1.0 / (eps * ta)) * (torch.exp(-eps * d)
                                       - math.exp(-eps * (tc - te)))
    wave = torch.where(t <= te, open_phase, ret_phase)
    wave = torch.where(t > tc, torch.zeros_like(wave), wave)
    wave = wave - wave.mean()
    return wave / wave.abs().max().clamp_min(1e-6)


class LFTableBank(nn.Module):
    """Rd 그리드 위의 LF 파형 스펙트럼 사전."""

    def __init__(self, n_tables: int = 16, table_size: int = 2048,
                 rd_min: float = 0.3, rd_max: float = 2.7, n_harmonics: int = 180):
        super().__init__()
        rds = torch.linspace(rd_min, rd_max, n_tables)
        specs = []
        for rd in rds.tolist():
            w = _lf_waveform(rd, table_size)
            c = torch.fft.rfft(w) / table_size          # 복소 푸리에 계수
            specs.append(c[1 : n_harmonics + 1])
        self.register_buffer("rd_grid", rds)
        self.register_buffer("spectra", torch.stack(specs))   # (n_tables, K)
        self.n_harmonics = n_harmonics

    def interpolate(self, rd: torch.Tensor) -> torch.Tensor:
        """rd: (B, T, 1) -> 하모닉 복소계수 (B, T, K). rd 에 대해 미분가능."""
        g = self.rd_grid
        pos = (rd.squeeze(-1) - g[0]) / (g[-1] - g[0]) * (len(g) - 1)
        pos = pos.clamp(0.0, len(g) - 1 - 1e-4)
        i0 = pos.floor().long()
        frac = (pos - i0).unsqueeze(-1)
        s0 = self.spectra[i0]                    # (B, T, K)
        s1 = self.spectra[(i0 + 1).clamp(max=len(g) - 1)]
        return s0 * (1 - frac) + s1 * frac


# ---------------------------------------------------------------------- 오실레이터
class GlottalSource(nn.Module):
    """대역제한 가산합성으로 성문 유량미분 신호를 만든다."""

    def __init__(self, sample_rate: int, hop_size: int, n_harmonics: int = 180,
                 n_tables: int = 16, table_size: int = 2048,
                 rd_min: float = 0.3, rd_max: float = 2.7, chunk: int = 12000):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.chunk = chunk
        self.bank = LFTableBank(n_tables, table_size, rd_min, rd_max, n_harmonics)
        self.register_buffer("k", torch.arange(1, n_harmonics + 1).float())

    def forward(self, f0: torch.Tensor, rd: torch.Tensor, amp: torch.Tensor,
                phase0: torch.Tensor | None = None):
        """f0/rd/amp: (B, T, 1) 프레임률. 반환: (source (B,N), phase (B,N))."""
        b, t, _ = f0.shape
        n = t * self.hop_size
        coef = self.bank.interpolate(rd)                       # (B, T, K) complex
        cr = upsample(coef.real, self.hop_size)                # (B, N, K)
        ci = upsample(coef.imag, self.hop_size)
        f0_s = upsample(f0, self.hop_size).squeeze(-1)          # (B, N)
        amp_s = upsample(amp, self.hop_size).squeeze(-1)

        # 순시위상 (샘플 단위 누적) — 프레임 경계에서 위상 불연속이 없다.
        omega = TWO_PI * f0_s / self.sample_rate
        phase = torch.cumsum(omega, dim=1)
        if phase0 is not None:
            phase = phase + phase0

        nyq = self.sample_rate * 0.5
        out = torch.zeros(b, n, device=f0.device, dtype=f0.dtype)
        for s in range(0, n, self.chunk):                       # 메모리 상한 유지
            e = min(s + self.chunk, n)
            ph = phase[:, s:e, None] * self.k                   # (B, C, K)
            mag = torch.sqrt(cr[:, s:e] ** 2 + ci[:, s:e] ** 2 + 1e-20)
            ang = torch.atan2(ci[:, s:e], cr[:, s:e])
            # 나이퀴스트에서 '정확히 0' 이 되는 smoothstep 마스크.
            # 시그모이드로 하면 나이퀴스트 바로 위 하모닉이 7% 정도 남아 접힌다.
            fk = f0_s[:, s:e, None] * self.k
            u = ((nyq - fk) / (nyq * 0.08)).clamp(0.0, 1.0)
            mask = u * u * (3.0 - 2.0 * u)
            out[:, s:e] = 2.0 * (mag * mask * torch.cos(ph + ang)).sum(-1)
        return out * amp_s, phase
