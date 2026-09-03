"""물리 기반 음성 합성기: 소스(성문) + 난류 노이즈 + 성도 공명 + 위상 정형.

신호 흐름
---------
    f0, Rd, amp ─▶ GlottalSource ─┐
                                  ├─▶  H_harm = Π(포먼트) · 반공명 · 올패스 ─┐
    노이즈 대역/AM ─▶ NoiseSource ─┴─▶  H_noise = Π(협착 하류 포먼트만) · … ─┴─▶ + ─▶ 음성

핵심: 노이즈는 성도 '전체'가 아니라 협착 지점 하류(front cavity)만 통과한다.
/s/ 가 고역에 몰리고 /ʃ/ 가 그보다 낮은 이유가 바로 이 앞공동 길이 차이이며,
이걸 구조에 박아 두면 모델이 데이터에서 다시 배울 필요가 없다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..config import Config, DEFAULT
from ..dsp.core import ltv_filter, upsample
from ..dsp.filters import (allpass_response, antiresonator_response,
                           lip_radiation_response, resonator_stage_responses)
from ..dsp.glottal import GlottalSource
from ..dsp.noise import NoiseSource
from ..dsp.tract import tract_response


@dataclass
class Controls:
    """모두 프레임률 (B, T, ·) 텐서. 물리 단위(Hz, dB 아님, 선형 게인)."""
    f0: torch.Tensor                      # (B, T, 1) Hz
    harmonic_amp: torch.Tensor            # (B, T, 1) 유성 성분 세기
    rd: torch.Tensor                      # (B, T, 1) LF 형상 (0.3 pressed ~ 2.7 breathy)
    formant_freq: torch.Tensor            # (B, T, K) Hz
    formant_bw: torch.Tensor              # (B, T, K) Hz
    formant_gain: torch.Tensor            # (B, T, K)
    noise_bands: torch.Tensor             # (B, T, n_bands) 난류 소스 스펙트럼
    noise_entry: torch.Tensor             # (B, T, 1) 협착 위치 (0=성문, K=입술)
    noise_am: torch.Tensor                # (B, T, 1) 성문동기 변조 깊이 (기식성)
    antiformant_freq: torch.Tensor | None = None   # (B, T, Ka)
    antiformant_bw: torch.Tensor | None = None
    allpass_freq: torch.Tensor | None = None       # (B, T, Na)
    allpass_radius: torch.Tensor | None = None
    area: torch.Tensor | None = None      # (B, T, N) 도파관 모드에서의 단면적


class PhysicalVoiceSynth(nn.Module):
    def __init__(self, cfg: Config = DEFAULT, tract_mode: str = "formant"):
        super().__init__()
        assert tract_mode in ("formant", "waveguide")
        self.cfg = cfg
        self.tract_mode = tract_mode
        a, s, f = cfg.audio, cfg.source, cfg.filt
        self.n_freq = a.n_fft // 2 + 1
        self.source = GlottalSource(a.sample_rate, a.hop_size, s.n_harmonics,
                                    s.n_rd_tables, s.table_size, s.rd_min, s.rd_max)
        self.noise = NoiseSource(a.sample_rate, a.hop_size, self.n_freq, ir_size=256)
        self.register_buffer(
            "radiation", lip_radiation_response(a.sample_rate, self.n_freq, 0.0))

    # ---------------------------------------------------------------- 성도 응답
    def _formant_paths(self, c: Controls):
        """(H_harm, H_noise) 복소응답. 노이즈는 협착 하류 단만 통과한다."""
        fs, nf = self.cfg.audio.sample_rate, self.n_freq
        stages = resonator_stage_responses(c.formant_freq, c.formant_bw,
                                           c.formant_gain, fs, nf)  # (B,T,K,F)
        h_harm = stages.prod(dim=2)

        k = stages.shape[2]
        idx = torch.arange(k, device=stages.device, dtype=stages.real.dtype)
        # w_i = 1 이면 그 단이 노이즈에도 적용됨(= 협착 하류)
        w = torch.sigmoid((idx.view(1, 1, k) - c.noise_entry) / 0.7)
        w = w.unsqueeze(-1).to(stages.dtype)
        h_noise = (1.0 - w + w * stages).prod(dim=2)
        return h_harm, h_noise

    def _waveguide_paths(self, c: Controls):
        fs, nf = self.cfg.audio.sample_rate, self.n_freq
        area = c.area
        h_harm = tract_response(area, fs, nf)
        # 협착(=면적 최소) 지점 하류만 노이즈 경로. 위치는 면적함수에서 직접 얻는다.
        pos = area.mean(dim=1).argmin(dim=-1).clamp(max=area.shape[-1] - 3)
        cut = int(pos.float().mean().item())
        h_noise = tract_response(area[..., cut:], fs, nf)
        return h_harm, h_noise

    def _shared_response(self, c: Controls, ref: torch.Tensor):
        """반공명·올패스·방사 등 두 경로가 공유하는 응답."""
        fs, nf = self.cfg.audio.sample_rate, self.n_freq
        h = torch.ones_like(ref)
        if c.antiformant_freq is not None:
            h = h * antiresonator_response(c.antiformant_freq, c.antiformant_bw, fs, nf)
        if c.allpass_freq is not None:
            h = h * allpass_response(c.allpass_freq, c.allpass_radius, fs, nf)
        return h

    # -------------------------------------------------------------------- 합성
    def forward(self, c: Controls, generator: torch.Generator | None = None) -> dict:
        cfg = self.cfg
        hop, ir = cfg.audio.hop_size, cfg.filt.ir_size

        src, phase = self.source(c.f0, c.rd, c.harmonic_amp)
        noise = self.noise(c.noise_bands, c.noise_am, phase, generator=generator)

        if self.tract_mode == "formant":
            h_harm, h_noise = self._formant_paths(c)
        else:
            h_harm, h_noise = self._waveguide_paths(c)
        shared = self._shared_response(c, h_harm)
        h_harm = h_harm * shared
        h_noise = h_noise * shared

        voiced = ltv_filter(src, h_harm, hop, ir)
        unvoiced = ltv_filter(noise, h_noise, hop, ir)
        audio = voiced + unvoiced
        return {
            "audio": audio,
            "voiced": voiced,
            "unvoiced": unvoiced,
            "source": src,
            "noise": noise,
            "glottal_phase": phase,
            "h_harm": h_harm,
            "h_noise": h_noise,
        }
