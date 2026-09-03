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
from ..dsp.core import ltv_filter
from ..dsp.filters import (allpass_response, antiresonator_response,
                           bands_to_response, gated_cascade_response,
                           lip_radiation_response, resonator_stage_responses)
from ..dsp.glottal import GlottalSource
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
    noise_rough: torch.Tensor | None = None      # (B, T, 1) 난류 시간변조(비정상성)
    # (B, T, 1) 구강 결합. 협착은 음향적으로 완전한 벽이 아니다 — 난류음의 일부는
    # 협착을 통해/돌아 뒤공동까지 닿았다가 성도 전체를 거쳐 나온다.
    # 0 이면 앞공동(치찰음 필터)만 통과해서, 마찰음이 입 안에서 난 소리가 아니라
    # 위에 얹은 히스처럼 들린다(저·중역이 통째로 비고 고역만 평평하게 남는다).
    # 이 값이 있어야 /사/ 와 /시/ 의 마찰음이 서로 달라진다(동시조음).
    noise_back_leak: torch.Tensor | None = None
    # (B, T, 1) 노이즈 경로에서만 포먼트 대역폭에 곱하는 배율(>=1).
    # 성문 펄스가 아니라 난류가 성도를 울릴 때는 감쇠가 훨씬 크다: 여기소가 한 점이
    # 아니라 협착 하류에 퍼져 있어 공진이 뭉개지고, 마찰 구간에는 성문 폐쇄에 의한
    # 주기적 재여기도 없다. 유성음의 Q(F2 에서 25 까지)를 그대로 쓰면 잡음이
    # 공진에서 울려 속삭임·마찰음에 음조가 얹힌다.
    noise_bw_scale: torch.Tensor | None = None
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


def cat_controls(parts: list[Controls]) -> Controls:
    """프레임축으로 Controls 를 이어붙인다 (스트리밍/스크립트 합성용)."""
    if len(parts) == 1:
        return parts[0]
    out = {}
    for f in fields(Controls):
        vals = [getattr(p, f.name) for p in parts]
        if torch.is_tensor(vals[0]):
            out[f.name] = torch.cat(vals, dim=1)
        elif isinstance(vals[0], SibilantParams):
            out[f.name] = SibilantParams(**{
                k: torch.cat([getattr(v, k) for v in vals], dim=1)
                for k in vals[0].__dict__})
        else:
            out[f.name] = vals[0]
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
        # entry=0 이 '모든 포먼트 통과' 가 되도록 3 단 offset 을 준다.
        # offset 이 없으면 sigmoid((0-0)/0.7)=0.5 라 entry=0 인데도 F1 이 절반만
        # 걸린다(속삭임처럼 성문에서 주입하는 소리가 실제보다 밝아진다).
        w = torch.sigmoid((idx.view(1, 1, k) - c.noise_entry + 3.0) / 0.7)
        bw_n = c.formant_bw
        if c.noise_bw_scale is not None:
            bw_n = bw_n * c.noise_bw_scale.clamp_min(1.0)
        # 로그 영역 보간 (filters.gated_cascade_response 의 주석 참고).
        h_noise = gated_cascade_response(c.formant_freq, bw_n,
                                         c.formant_gain, w, fs, nf)
        return h_harm, h_noise

    def _oral_leak(self, c: Controls, h_front: torch.Tensor) -> torch.Tensor:
        """앞공동 경로 + 구강 전체 경로의 **병렬 합**.

        두 경로가 같은 난류원에 걸려 있으므로 신호를 섞는 것과 응답을 더하는 것이
        같다(선형). 뒤로 새는 경로는 치찰음 필터(=앞공동)를 거치지 않고 성도
        전체를 지난다.
        """
        if c.noise_back_leak is None:
            return h_front
        fs, nf = self.cfg.audio.sample_rate, self.n_freq
        bw_n = c.formant_bw
        if c.noise_bw_scale is not None:
            bw_n = bw_n * c.noise_bw_scale.clamp_min(1.0)
        k = c.formant_freq.shape[-1]
        full = torch.ones(1, 1, k, device=c.formant_freq.device,
                          dtype=c.formant_freq.dtype)
        h_full = gated_cascade_response(c.formant_freq, bw_n, c.formant_gain,
                                        full.expand_as(c.formant_freq), fs, nf)
        g = c.noise_back_leak.clamp(0.0, 1.0).to(h_front.dtype)  # (B,T,1) 로 방송된다
        return (1.0 - g) * h_front + g * h_full

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
        if c.antiformant_freq is not None:
            h = h * antiresonator_response(c.antiformant_freq, c.antiformant_bw, fs, nf)
        if c.allpass_freq is not None:
            h = h * allpass_response(c.allpass_freq, c.allpass_radius, fs, nf)
        return h

    # -------------------------------------------------------------------- 합성
    def forward(self, c: Controls, generator: torch.Generator | None = None,
                state: dict | None = None, emit_frames: int | None = None) -> dict:
        """`state` 를 주면 청크 단위 스트리밍이 된다.

        이어지는 청크 사이에서 (a) 성문 순시위상과 (b) LTV 필터의 OLA 꼬리를
        넘겨받는다. 반환 dict 의 "state" 를 다음 호출에 그대로 넣으면 된다.

        `emit_frames=T` 는 T+1 프레임의 제어를 받아 **T 프레임만** 낸다. 마지막
        프레임은 보간 기준점으로만 쓴다. 이 1 프레임(10 ms) 선행이 없으면 청크의
        마지막 hop 에서 F0 보간이 다음 프레임을 못 봐서 위상 오차가 누적된다.
        """
        cfg = self.cfg
        hop, ir = cfg.audio.hop_size, cfg.filt.ir_size
        fs, nf = cfg.audio.sample_rate, self.n_freq
        b, t_all, _ = c.f0.shape
        t = t_all if emit_frames is None else int(emit_frames)

        st = state or {}
        src, phase = self.source(
            c.f0, c.rd, c.harmonic_amp, phase0=st.get("phase"), tilt=c.tilt,
            disp_freq=c.disp_freq, disp_radius=c.disp_radius,
            jitter=c.jitter, shimmer=c.shimmer, generator=generator)
        # 성문동기 AM 은 성대가 실제로 떨 때만 존재한다. 무성음(속삭임, 마찰음)에
        # 이걸 걸면 F0 로 노이즈를 써는 셈이라 없는 주기성이 생긴다.
        voiced_gate = c.harmonic_amp / (c.harmonic_amp.abs() + 0.02)
        if t != t_all:                       # 선행 프레임은 보간 기준점일 뿐이다
            src, phase_full = src[:, : t * hop], phase
            phase = phase[:, : t * hop]
            c = c.slice(0, t)
            voiced_gate = voiced_gate[:, :t]
        else:
            phase_full = phase
        raw_noise = self.noise(t, b, c.f0.device, c.f0.dtype,
                               am_depth=c.noise_am * voiced_gate,
                               glottal_phase=phase,
                               roughness=c.noise_rough, generator=generator)

        if self.tract_mode == "formant":
            h_harm, h_noise = self._formant_paths(c)
        else:
            h_harm, h_noise = self._waveguide_paths(c)
        shared = self._shared_response(c, h_harm)
        h_harm = h_harm * shared

        # 난류 색칠: 학습된 소스 사전 x 프레임별 대역게인 x 치찰음 필터 x 성도
        h_noise = h_noise * shared
        if c.sib is not None:
            h_noise = h_noise * sibilant_response(c.sib, fs, nf)
        # 협착을 지나 뒤공동까지 닿는 경로를 병렬로 더한다(구강 결합)
        h_noise = self._oral_leak(c, h_noise)
        h_noise = h_noise * bands_to_response(
            c.noise_bands * self.noise.spectral_prior(), nf, min_phase=True)

        if state is None:
            voiced = ltv_filter(src, h_harm, hop, ir)
            unvoiced = ltv_filter(raw_noise, h_noise, hop, ir)
            new_state = None
        else:
            voiced, tv = ltv_filter(src, h_harm, hop, ir, st.get("tail_h"), True)
            unvoiced, tn = ltv_filter(raw_noise, h_noise, hop, ir,
                                      st.get("tail_n"), True)
            new_state = {"phase": phase[:, -1:], "tail_h": tv, "tail_n": tn}
            del phase_full
        audio = voiced + unvoiced
        return {
            "state": new_state,
            "audio": audio,
            "voiced": voiced,
            "unvoiced": unvoiced,
            "source": src,
            "noise": raw_noise,
            "glottal_phase": phase,
            "h_harm": h_harm,
            "h_noise": h_noise,
        }
