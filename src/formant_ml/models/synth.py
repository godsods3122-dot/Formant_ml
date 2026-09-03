"""물리 기반 음성 합성기: 소스(성문) + 난류 노이즈 + 성도 공명 + 위상 정형.

신호 흐름
---------
    f0, Rd, amp, tilt, 위상차, 지터/시머 ─▶ GlottalSource ─┐
                                                          ├▶ H_harm = Π(포먼트)·반공명·올패스 ─┐
    노이즈 대역/AM/거칠기 ─▶ TurbulenceSource ────────────┴▶ H_noise = 치찰음필터·(협착하류) ─┴▶ + ▶ 음성

설계 규칙
---------
1. 신경망은 파형을 만들지 않는다. 아래 `Controls` 의 값만 만든다.
2. 크기(magnitude)를 바꾸는 경로와 위상을 바꾸는 경로를 분리한다.
   위상 손잡이는 전부 올패스라 스펙트럼 포락선을 절대 건드리지 않는다.
3. 노이즈는 협착 하류(front cavity)만 통과한다. /s/ 가 고역인 이유가 그것이다.
4. 모든 손잡이는 스크립트에서 그대로 지정할 수 있다(학습 없이도 완전 제어).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from ..config import Config, DEFAULT
from ..dsp.core import ltv_filter, upsample
from ..dsp.filters import (allpass_response, antiresonator_response,
                           bands_to_response, gated_cascade_response,
                           lip_radiation_response, resonator_stage_responses)
from ..dsp.glottal import GlottalSource
from ..dsp.nasal import f1_bandwidth_factor, nasal_response
from ..dsp.noise import TurbulenceSource
from ..dsp.sibilant import SibilantParams, sibilant_response
from ..dsp.tract import tract_response


@dataclass
class Controls:
    """모두 프레임률 (B, T, ·) 텐서. 물리 단위(Hz, dB 아님, 선형 게인).

    필수 항목은 앞쪽 8개뿐이고 나머지는 전부 선택(없으면 그 효과가 꺼진다).
    """
    # --- 성문 소스 ---------------------------------------------------------
    f0: torch.Tensor                      # (B, T, 1) Hz
    harmonic_amp: torch.Tensor            # (B, T, 1) 유성 성분 세기
    rd: torch.Tensor                      # (B, T, 1) LF 형상 (0.3 pressed ~ 2.7 breathy)
    # --- 성도 -------------------------------------------------------------
    formant_freq: torch.Tensor            # (B, T, K) Hz
    formant_bw: torch.Tensor              # (B, T, K) Hz
    formant_gain: torch.Tensor            # (B, T, K)
    # --- 난류 노이즈 -------------------------------------------------------
    noise_bands: torch.Tensor             # (B, T, n_bands) 난류 소스 스펙트럼
    noise_entry: torch.Tensor             # (B, T, 1) 협착 위치
    #   0 = 성문(모든 포먼트 통과) … K = 입술 … K+6 = 성도 완전 우회.
    #   게이트가 부드러워서 K 에서 마지막 단이 19%, K+3 에서도 35% 잔물결이 남는다.
    noise_am: torch.Tensor                # (B, T, 1) 성문동기 변조 깊이 (기식성)
    #   기식(aspiration) 노이즈: **성문**에서 나 성도 '전체'를 통과하는 난류.
    #   마찰 노이즈(협착부에서 나 앞공동만 통과)와 물리적으로 다른 소스다.
    #   협착이 열리는 순간 압력이 성문으로 옮겨가며 나는 소리이고, 무성자음과
    #   뒤따르는 모음을 하나의 성도로 묶어주는 것이 바로 이 성분이다.
    aspiration: torch.Tensor | None = None       # (B, T, 1)
    noise_rough: torch.Tensor | None = None      # (B, T, 1) 난류 시간변조(비정상성)
    # --- 소스 스펙트럼/미세요동 --------------------------------------------
    tilt: torch.Tensor | None = None      # (B, T, 1) dB/oct @1 kHz — 고역 조절
    jitter: torch.Tensor | None = None    # (B, T, 1) 주기 요동 비율 (0.005 = 0.5%)
    shimmer: torch.Tensor | None = None   # (B, T, 1) 진폭 요동 비율
    # --- 위상 -------------------------------------------------------------
    disp_freq: torch.Tensor | None = None       # (B, T, S) 하모닉 위상차 올패스
    disp_radius: torch.Tensor | None = None
    allpass_freq: torch.Tensor | None = None    # (B, T, Na) 성도 군지연 정형
    allpass_radius: torch.Tensor | None = None
    # --- 그 밖 ------------------------------------------------------------
    #   연구개 개도 0~1. 0 이면 비강 극-영점이 정확히 상쇄되어 아무 일도 없다.
    velum_open: torch.Tensor | None = None       # (B, T, 1)
    antiformant_freq: torch.Tensor | None = None   # (B, T, Ka)
    antiformant_bw: torch.Tensor | None = None
    sib: SibilantParams | None = None     # 치찰음 극-영점 필터(화자 지문)
    area: torch.Tensor | None = None      # (B, T, N) 도파관 모드에서의 단면적

    # ------------------------------------------------------------------ 편의
    def to(self, device) -> "Controls":
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v):
                v = v.to(device)
            elif isinstance(v, SibilantParams):
                v = v.to(device)
            out[f.name] = v
        return Controls(**out)

    @property
    def n_frames(self) -> int:
        return self.f0.shape[1]

    def slice(self, t0: int, t1: int) -> "Controls":
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v) and v.dim() == 3:
                v = v[:, t0:t1]
            elif isinstance(v, SibilantParams):
                v = SibilantParams(**{k: (x[:, t0:t1] if torch.is_tensor(x) else x)
                                      for k, x in v.__dict__.items()})
            out[f.name] = v
        return Controls(**out)


class PhysicalVoiceSynth(nn.Module):
    def __init__(self, cfg: Config = DEFAULT, tract_mode: str = "formant"):
        super().__init__()
        assert tract_mode in ("formant", "waveguide")
        self.cfg = cfg
        self.tract_mode = tract_mode
        a, s = cfg.audio, cfg.source
        self.n_freq = a.n_fft // 2 + 1
        self.source = GlottalSource(a.sample_rate, a.hop_size, s.n_harmonics,
                                    s.n_rd_tables, s.table_size, s.rd_min, s.rd_max)
        self.noise = TurbulenceSource(a.sample_rate, a.hop_size, cfg.noise.n_bands)
        self.register_buffer(
            "radiation", lip_radiation_response(a.sample_rate, self.n_freq, 0.0))
        # 기식 소스의 고정 스펙트럼 모양(광대역, 저역이 약간 약하다).
        from ..utils import band_shelf
        self.register_buffer("asp_shape", bands_to_response(
            band_shelf(cfg.noise.n_bands, 400.0, 1.0, a.sample_rate,
                       slope_oct=0.5).reshape(1, 1, -1), self.n_freq,
            min_phase=True))

    # ---------------------------------------------------------------- 성도 응답
    def _formant_paths(self, c: Controls):
        """(H_harm, H_noise) 복소응답. 노이즈는 협착 하류 단만 통과한다."""
        fs, nf = self.cfg.audio.sample_rate, self.n_freq
        bw = c.formant_bw
        if c.velum_open is not None:
            # 곁가지로 에너지가 새면서 F1 이 넓어지고 약해진다 (비음화의 핵심 단서)
            bw = torch.cat([bw[..., :1] * f1_bandwidth_factor(c.velum_open),
                            bw[..., 1:]], dim=-1)
        stages = resonator_stage_responses(c.formant_freq, bw,
                                           c.formant_gain, fs, nf)  # (B,T,K,F)
        h_harm = stages.prod(dim=2)

        k = stages.shape[2]
        idx = torch.arange(k, device=stages.device, dtype=stages.real.dtype)
        # w_i = 1 이면 그 단이 노이즈에도 적용됨(= 협착 하류)
        w = torch.sigmoid((idx.view(1, 1, k) - c.noise_entry) / 0.7)
        # 로그 영역 보간 (filters.gated_cascade_response 의 주석 참고).
        h_noise = gated_cascade_response(c.formant_freq, bw,
                                         c.formant_gain, w, fs, nf)
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
        """반공명·올패스 등 두 경로가 공유하는 응답(성도 하류에 해당)."""
        fs, nf = self.cfg.audio.sample_rate, self.n_freq
        h = torch.ones_like(ref)
        if c.velum_open is not None:
            h = h * nasal_response(c.velum_open, fs, nf)
        if c.antiformant_freq is not None:
            h = h * antiresonator_response(c.antiformant_freq, c.antiformant_bw, fs, nf)
        if c.allpass_freq is not None:
            h = h * allpass_response(c.allpass_freq, c.allpass_radius, fs, nf)
        return h

    # -------------------------------------------------------------------- 합성
    def forward(self, c: Controls, generator: torch.Generator | None = None) -> dict:
        cfg = self.cfg
        hop, ir = cfg.audio.hop_size, cfg.filt.ir_size
        fs, nf = cfg.audio.sample_rate, self.n_freq
        b, t, _ = c.f0.shape

        src, phase = self.source(
            c.f0, c.rd, c.harmonic_amp, tilt=c.tilt,
            disp_freq=c.disp_freq, disp_radius=c.disp_radius,
            jitter=c.jitter, shimmer=c.shimmer, generator=generator)
        raw_noise = self.noise(t, b, c.f0.device, c.f0.dtype,
                               am_depth=c.noise_am, glottal_phase=phase,
                               roughness=c.noise_rough, generator=generator)

        if self.tract_mode == "formant":
            h_harm, h_noise = self._formant_paths(c)
        else:
            h_harm, h_noise = self._waveguide_paths(c)
        shared = self._shared_response(c, h_harm)
        h_harm = h_harm * shared

        # 난류 색칠: 학습된 소스 사전 x 프레임별 대역게인 x 치찰음 필터 x 성도
        h_noise = h_noise * shared
        h_noise = h_noise * bands_to_response(
            c.noise_bands * self.noise.spectral_prior(), nf, min_phase=True)
        if c.sib is not None:
            h_noise = h_noise * sibilant_response(c.sib, fs, nf)

        voiced = ltv_filter(src, h_harm, hop, ir)
        unvoiced = ltv_filter(raw_noise, h_noise, hop, ir)

        # 기식 노이즈: 성문에서 나 성도 전체를 통과한다. 마찰 노이즈와 별개의
        # 난수를 쓴다(같은 신호를 두 경로로 보내면 콤필터가 생긴다).
        aspirated = torch.zeros_like(voiced)
        if c.aspiration is not None and float(c.aspiration.abs().max()) > 0:
            asp_src = self.noise(t, b, c.f0.device, c.f0.dtype,
                                 am_depth=c.noise_am, glottal_phase=phase,
                                 roughness=c.noise_rough, generator=generator)
            g = upsample(c.aspiration.clamp_min(0.0), hop).squeeze(-1)
            n = min(g.shape[1], asp_src.shape[1])
            aspirated = ltv_filter(asp_src[:, :n] * g[:, :n],
                                   h_harm * self.asp_shape, hop, ir)

        audio = voiced + unvoiced + aspirated
        return {
            "audio": audio,
            "aspirated": aspirated,
            "voiced": voiced,
            "unvoiced": unvoiced,
            "source": src,
            "noise": raw_noise,
            "glottal_phase": phase,
            "h_harm": h_harm,
            "h_noise": h_noise,
        }
