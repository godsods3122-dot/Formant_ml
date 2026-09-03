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
        Ns = cfg.filt.n_tract_sections

        # 입력: log-mel + log-f0 + voicing
        self.backbone = ConvGRUBackbone(cfg.audio.n_mels + 2, hidden)
        d = self.backbone.out_dim

        self.head_source = nn.Linear(d, 3)             # harmonic_amp, rd, f0 보정
        self.head_formant = nn.Linear(d, 3 * K)        # dfreq, bw, gain
        self.head_anti = nn.Linear(d, 2 * Ka)
        self.head_noise = nn.Linear(d, Nb + 2)         # 대역 + entry + am
        self.head_allpass = nn.Linear(d, 2 * Na)
        self.head_area = nn.Linear(d, Ns)
        self.K, self.Ka, self.Na, self.Nb, self.Ns = K, Ka, Na, Nb, Ns

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
        entry = torch.sigmoid(nz[..., self.Nb: self.Nb + 1]) * self.K
        am = torch.sigmoid(nz[..., self.Nb + 1:])

        ap = self.head_allpass(h)
        apf = scale_sigmoid(ap[..., : self.Na], 100.0, cfg.audio.sample_rate * 0.45)
        apr = torch.sigmoid(ap[..., self.Na:]) * 0.9

        area = None
        if self.tract_mode == "waveguide":
            area = exp_sigmoid(self.head_area(h), max_value=8.0) + 0.05

        return Controls(
            f0=f0_out, harmonic_amp=harmonic_amp, rd=rd,
            formant_freq=freq, formant_bw=bw, formant_gain=gain,
            noise_bands=bands, noise_entry=entry, noise_am=am,
            antiformant_freq=af, antiformant_bw=ab,
            allpass_freq=apf, allpass_radius=apr, area=area,
        )
