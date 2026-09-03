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
from .phase import allpass_phase


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


def open_quotient(rd: float) -> float:
    """Rd -> 개방지수 OQ (한 주기 중 성문이 열려 있는 비율).

    LF 타이밍에서 직접 얻는다 (te + 복귀상 ta). 문헌의 회귀식을 베끼지 않고
    우리 사전에서 계산하므로 분석/합성이 어긋나지 않는다.
    """
    tp, te, ta, tc = _rd_to_timing(rd)
    return min(te + ta, 0.99)


def glottal_f1_damping(rd: torch.Tensor, max_hz: float = 130.0) -> torch.Tensor:
    """성문 개방에 의한 F1 대역폭 증가분 [Hz]. (B, T, 1) -> (B, T, 1)

    **소스-성도 상호작용의 1차 효과.** 성문이 열려 있는 동안 성문하 계통이
    성도에 연결되어 F1 이 손실을 본다. 주기 평균하면 F1 대역폭이 OQ 에 비례해
    넓어진다(폐쇄기 대비 개방기에 100~200 Hz 증가, 평균 50~100 Hz).

    완전한 상호작용(성도 입력 임피던스가 성대 진동에 되먹임)은 시간영역
    연립을 요구하지만, **들리는 것의 대부분은 이 F1 감쇠**다. 그래서 느린 ODE
    없이도 1차 효과를 정확히 같은 자리에 넣을 수 있다.

    부수 효과: 여성처럼 F0 가 높아 하모닉이 성길 때 F1 봉우리를 하모닉이 못 짚어
    'F0 만 튀어나오는' 문제가 완화된다 — 넓어진 F1 이 더 많은 하모닉을 덮는다.
    """
    grid = torch.linspace(0.3, 2.7, 25, device=rd.device, dtype=rd.dtype)
    oq = torch.tensor([open_quotient(float(v)) for v in grid],
                      device=rd.device, dtype=rd.dtype)
    pos = ((rd - grid[0]) / (grid[-1] - grid[0]) * (len(grid) - 1)).clamp(0, len(grid) - 1.001)
    i = pos.floor().long()
    frac = pos - i
    q = oq[i] * (1 - frac) + oq[(i + 1).clamp(max=len(grid) - 1)] * frac
    return max_hz * q


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
def _interp_frames(c: torch.Tensor, t0: int, t1: int, hop: int) -> torch.Tensor:
    """프레임률 (B, T, K) 의 [t0, t1) 구간만 샘플률로 선형보간 -> (B, (t1-t0)*hop, K).

    전체를 한 번에 upsample 하면 (B, N, K) 텐서가 통째로 메모리에 올라간다
    (24 kHz, 1.5 초, K=240 이면 배치당 ~35 MB). 블록 단위로 하면 상한이 고정되고,
    경계에서는 *진짜 다음 프레임* 을 써서 전역 보간과 정확히 같은 값이 나온다.
    """
    t = c.shape[1]
    nxt = c[:, min(t1, t - 1): min(t1, t - 1) + 1]
    seg = torch.cat([c[:, t0:t1], nxt], dim=1)                 # (B, tb+1, K)
    tb = t1 - t0
    pos = torch.arange(tb * hop, device=c.device, dtype=c.dtype) / hop
    i0 = pos.floor().long().clamp(max=tb)
    frac = (pos - i0).view(1, -1, 1)
    return seg[:, i0] * (1 - frac) + seg[:, (i0 + 1).clamp(max=tb)] * frac


def _smooth_noise(shape, hop_frames: int, device, dtype, generator=None,
                  smooth: int = 3) -> torch.Tensor:
    """저역통과된 단위분산 난수 (지터/시머의 느린 요동)."""
    z = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    if smooth > 1:
        k = torch.ones(1, 1, smooth, device=device, dtype=dtype) / smooth
        z = torch.nn.functional.conv1d(
            z.transpose(1, 2), k, padding=smooth // 2).transpose(1, 2)[:, :shape[1]]
        z = z / z.std().clamp_min(1e-6)
    return z


class GlottalSource(nn.Module):
    """대역제한 가산합성으로 성문 유량미분 신호를 만든다.

    Rd 외에 세 가지 손잡이가 더 있다.

    * `tilt`      : 옥타브당 dB 기울기. LF 의 Rd 만으로는 고역 감쇠 모양이 고정이라
                    6~12 kHz 를 원하는 만큼 살릴 수 없다(이전 샘플의 '고역 부족').
    * `disp_*`    : **위상차 파라미터**. 하모닉의 상대 위상만 바꾸고 크기응답은
                    건드리지 않는다(올패스). dsp/phase.py 참고.
    * `jitter/shimmer` : 주기/진폭의 미세 요동. 없으면 '부저 같은' 완전 주기성이 남는다.
    """

    def __init__(self, sample_rate: int, hop_size: int, n_harmonics: int = 240,
                 n_tables: int = 16, table_size: int = 2048,
                 rd_min: float = 0.3, rd_max: float = 2.7, chunk: int = 12000):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.chunk = chunk
        self.bank = LFTableBank(n_tables, table_size, rd_min, rd_max, n_harmonics)
        self.register_buffer("k", torch.arange(1, n_harmonics + 1).float())

    def forward(self, f0: torch.Tensor, rd: torch.Tensor, amp: torch.Tensor,
                phase0: torch.Tensor | None = None,
                tilt: torch.Tensor | None = None,
                disp_freq: torch.Tensor | None = None,
                disp_radius: torch.Tensor | None = None,
                jitter: torch.Tensor | None = None,
                shimmer: torch.Tensor | None = None,
                generator: torch.Generator | None = None):
        """f0/rd/amp/tilt/jitter/shimmer: (B, T, 1), disp_*: (B, T, S) 프레임률.

        반환: (source (B, N), glottal_phase (B, N)).
        """
        b, t, _ = f0.shape
        n = t * self.hop_size
        dev, dt = f0.device, f0.dtype

        if jitter is not None:
            z = _smooth_noise((b, t, 1), self.hop_size, dev, dt, generator)
            f0 = f0 * (1.0 + jitter.clamp(0.0, 0.2) * z)
        if shimmer is not None:
            z = _smooth_noise((b, t, 1), self.hop_size, dev, dt, generator)
            amp = amp * (1.0 + shimmer.clamp(0.0, 0.9) * z).clamp_min(0.0)

        coef = self.bank.interpolate(rd)                       # (B, T, K) complex
        fk = f0 * self.k.view(1, 1, -1)                        # (B, T, K) 하모닉 주파수

        # 스펙트럼 기울기: 1 kHz 를 축으로 옥타브당 tilt dB
        if tilt is not None:
            oct_ = torch.log2(fk.clamp_min(20.0) / 1000.0)
            g = 10.0 ** ((tilt * oct_).clamp(-40.0, 40.0) / 20.0)
            coef = coef * g.to(coef.dtype)

        # 위상차(올패스) — 크기는 그대로, 하모닉 간 상대위상만 바뀐다
        if disp_freq is not None and disp_radius is not None:
            ph = allpass_phase(fk, disp_freq, disp_radius, self.sample_rate)
            coef = coef * torch.polar(torch.ones_like(ph), ph).to(coef.dtype)

        f0_s = upsample(f0, self.hop_size).squeeze(-1)          # (B, N)
        amp_s = upsample(amp, self.hop_size).squeeze(-1)

        # 순시위상 (샘플 단위 누적) — 프레임 경계에서 위상 불연속이 없다.
        omega = TWO_PI * f0_s / self.sample_rate
        phase = torch.cumsum(omega, dim=1)
        if phase0 is not None:
            phase = phase + phase0

        nyq = self.sample_rate * 0.5
        out = torch.zeros(b, n, device=dev, dtype=dt)
        frames_per_chunk = max(1, self.chunk // self.hop_size)
        for t0 in range(0, t, frames_per_chunk):
            t1 = min(t0 + frames_per_chunk, t)
            s, e = t0 * self.hop_size, t1 * self.hop_size
            cr = _interp_frames(coef.real, t0, t1, self.hop_size)   # (B, C, K)
            ci = _interp_frames(coef.imag, t0, t1, self.hop_size)
            ph = phase[:, s:e, None] * self.k                       # (B, C, K)
            mag = torch.sqrt(cr ** 2 + ci ** 2 + 1e-20)
            ang = torch.atan2(ci, cr)
            # 나이퀴스트에서 '정확히 0' 이 되는 smoothstep 마스크.
            # 시그모이드로 하면 나이퀴스트 바로 위 하모닉이 7% 정도 남아 접힌다.
            fks = f0_s[:, s:e, None] * self.k
            u = ((nyq - fks) / (nyq * 0.08)).clamp(0.0, 1.0)
            mask = u * u * (3.0 - 2.0 * u)
            out[:, s:e] = 2.0 * (mag * mask * torch.cos(ph + ang)).sum(-1)
        return out * amp_s, phase
