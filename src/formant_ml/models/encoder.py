"""제어 파라미터 추정기: 멜스펙트로그램 + F0 -> 물리 파라미터.

신경망은 '파형'을 만들지 않는다. 오직 물리모델의 손잡이(포먼트, 대역폭, Rd,
노이즈 대역, 협착 위치...)만 예측한다. 파형은 전적으로 방정식이 만든다.
=> 신경망이 만들어낼 수 있는 아티팩트의 상한이 구조적으로 제한된다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config, DEFAULT
from ..dsp.core import exp_sigmoid, scale_sigmoid
from ..dsp.sibilant import SibilantParams
from .synth import Controls


class ConvGRUBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_conv: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_conv):
            layers += [nn.Conv1d(d, hidden, 5, padding=2), nn.GroupNorm(8, hidden),
                       nn.GELU()]
            d = hidden
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out_dim = hidden * 2

    def forward(self, x):                      # (B, T, C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return h


class ControlEncoder(nn.Module):
    def __init__(self, cfg: Config = DEFAULT, hidden: int = 256,
                 tract_mode: str = "formant"):
        super().__init__()
        self.cfg = cfg
        self.tract_mode = tract_mode
        K, Ka = cfg.filt.n_formants, cfg.filt.n_antiformants
        Na, Nb = cfg.filt.n_allpass, cfg.noise.n_bands
        Nd = cfg.filt.n_dispersion
        Ns = cfg.filt.n_tract_sections

        # 입력: log-mel + log-f0 + voicing
        self.backbone = ConvGRUBackbone(cfg.audio.n_mels + 2, hidden)
        d = self.backbone.out_dim

        # amp, rd, f0 보정, tilt, jitter, shimmer
        self.head_source = nn.Linear(d, 6)
        self.head_formant = nn.Linear(d, 3 * K)        # dfreq, bw, gain
        self.head_anti = nn.Linear(d, 2 * Ka)
        # 대역 + entry + am + roughness + 대역폭 배율 + 구강 결합
        self.head_noise = nn.Linear(d, Nb + 5)
        self.head_allpass = nn.Linear(d, 2 * Na)       # 성도 군지연
        self.head_disp = nn.Linear(d, 2 * Nd)          # 하모닉 위상차(위상차 파라미터)
        self.head_sib = nn.Linear(d, 8)                # 치찰음 극-영점 + 스커트
        self.head_area = nn.Linear(d, Ns)
        self.K, self.Ka, self.Na, self.Nb, self.Ns, self.Nd = K, Ka, Na, Nb, Ns, Nd

    def forward(self, mel: torch.Tensor, f0: torch.Tensor,
                voicing: torch.Tensor) -> Controls:
        """mel: (B, T, n_mels), f0: (B, T), voicing: (B, T)."""
        cfg = self.cfg
        lf0 = torch.log(f0.clamp_min(cfg.source.f0_min))[..., None] / 10.0
        h = self.backbone(torch.cat([mel, lf0, voicing[..., None]], dim=-1))

        s = self.head_source(h)
        harmonic_amp = exp_sigmoid(s[..., :1]) * voicing[..., None]
        rd = scale_sigmoid(s[..., 1:2], cfg.source.rd_min, cfg.source.rd_max)
        # F0 는 추정치를 신뢰하고 ±1 반음 이내 미세보정만 학습
        f0_out = f0[..., None] * torch.exp(0.06 * torch.tanh(s[..., 2:3]))
        # 소스 스펙트럼 기울기: 고역(6~12 kHz)을 살리거나 죽이는 직접 손잡이
        tilt = scale_sigmoid(s[..., 3:4], -12.0, 12.0)
        jitter = torch.sigmoid(s[..., 4:5]) * 0.03      # 최대 3% 주기 요동
        shimmer = torch.sigmoid(s[..., 5:6]) * 0.30

        # 포먼트: 누적합으로 F1 < F2 < ... 를 구조적으로 보장 (순서 뒤집힘 불가)
        fo = self.head_formant(h)
        step = F.softplus(fo[..., : self.K]) + 60.0
        freq = cfg.filt.f_min + torch.cumsum(step, dim=-1)
        freq = freq.clamp(cfg.filt.f_min, cfg.filt.f_max)
        bw = scale_sigmoid(fo[..., self.K: 2 * self.K],
                           cfg.filt.bw_min, cfg.filt.bw_max)
        gain = exp_sigmoid(fo[..., 2 * self.K:], max_value=4.0)

        an = self.head_anti(h)
        af = scale_sigmoid(an[..., : self.Ka], 200.0, 6000.0)
        ab = scale_sigmoid(an[..., self.Ka:], 60.0, 1500.0)

        nz = self.head_noise(h)
        bands = exp_sigmoid(nz[..., : self.Nb], max_value=1.0)
        # 범위를 K 가 아니라 K+6 까지 연다. 게이트가 w_i = sigmoid((i-c)/0.7) 라
        # entry=K 여도 마지막 포먼트가 19%, K+3 에서도 잔물결이 35% 남는다.
        # /s/ 처럼 노이즈가 성도를 사실상 통과하지 않는 소리를 표현하려면
        # 완전히 우회할 수 있어야 한다.
        entry = torch.sigmoid(nz[..., self.Nb: self.Nb + 1]) * (self.K + 6.0)
        am = torch.sigmoid(nz[..., self.Nb + 1: self.Nb + 2])
        rough = torch.sigmoid(nz[..., self.Nb + 2: self.Nb + 3])
        # 노이즈 경로 대역폭 배율 1~6. 데이터가 "이 화자의 마찰음은 얼마나
        # 뭉개져 있는가" 를 직접 알려 준다.
        bw_scale = 1.0 + 5.0 * torch.sigmoid(nz[..., self.Nb + 3: self.Nb + 4])
        # 구강 결합(협착 뒤로 새는 비율). 데이터가 화자/음소마다 정해 준다.
        back_leak = 0.6 * torch.sigmoid(nz[..., self.Nb + 4: self.Nb + 5])

        # 치찰음 필터: 앞공동 극 > 뒤공동 영점 이 되도록 구조적으로 강제한다
        # (포먼트 순서 보장과 같은 이유 — 학습 중 극/영점이 뒤바뀌면 회복이 안 된다).
        sb = self.head_sib(h)
        sp_f = scale_sigmoid(sb[..., 0:1], 1500.0, 11000.0)
        sp_bw = scale_sigmoid(sb[..., 1:2], 150.0, 4000.0)
        sz_f = sp_f * (0.12 + 0.76 * torch.sigmoid(sb[..., 2:3]))
        sz_bw = scale_sigmoid(sb[..., 3:4], 150.0, 4000.0)
        s_tilt = scale_sigmoid(sb[..., 4:5], -6.0, 6.0)
        s_mix = torch.sigmoid(sb[..., 5:6])
        # 봉우리 양옆 스커트 기울기 — 뾰족한 삼각형이냐 둥근 돔이냐를 정한다
        s_lo = 45.0 * torch.sigmoid(sb[..., 6:7])
        s_hi = -20.0 * torch.sigmoid(sb[..., 7:8])
        sib = SibilantParams(sp_f, sp_bw, sz_f, sz_bw, s_tilt, s_mix, rough,
                             s_lo, s_hi)

        ap = self.head_allpass(h)
        apf = scale_sigmoid(ap[..., : self.Na], 100.0, cfg.audio.sample_rate * 0.45)
        apr = torch.sigmoid(ap[..., self.Na:]) * 0.9

        # 하모닉 위상차: 크기응답을 건드리지 않는 올패스이므로 자유롭게 학습해도
        # phasiness 가 돌아오지 않는다 (dsp/phase.py 의 논거).
        dp = self.head_disp(h)
        dpf = scale_sigmoid(dp[..., : self.Nd], 200.0, cfg.audio.sample_rate * 0.45)
        dpr = torch.sigmoid(dp[..., self.Nd:]) * 0.95

        area = None
        if self.tract_mode == "waveguide":
            area = exp_sigmoid(self.head_area(h), max_value=8.0) + 0.05

        return Controls(
            f0=f0_out, harmonic_amp=harmonic_amp, rd=rd,
            formant_freq=freq, formant_bw=bw, formant_gain=gain,
            noise_bands=bands, noise_entry=entry, noise_am=am, noise_rough=rough,
            noise_bw_scale=bw_scale, noise_back_leak=back_leak,
            tilt=tilt, jitter=jitter, shimmer=shimmer,
            disp_freq=dpf, disp_radius=dpr,
            antiformant_freq=af, antiformant_bw=ab,
            allpass_freq=apf, allpass_radius=apr, sib=sib, area=area,
        )
