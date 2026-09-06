"""잔차 학습 단계 — "잔차학습에 의한 음원 합성".

물리 엔진의 출력에서 **설명되지 않은 나머지**만 신경망(shifted softplus)이 맡는다.
붕괴(신경망이 파형을 통째로 만들어 위상 아티팩트가 돌아오는 것)를 구조로 막는다:

1. 보정은 **최소위상 시변 피킹 EQ** 뿐이다(각 ±max_db). 시간영역 2차 필터이므로
   위상은 크기에서 유일하게 정해지고 프리에코가 없다. v1 의 "±6 dB 최소위상 LTV
   필터" 를 같은 취지로, 재귀 필터로 옮긴 것이다.
2. 추가 소스는 **템플릿 뱅크의 혼합**(과도음/노이즈 템플릿 + 변조)뿐이다.
   파형을 자유 예측하지 않는다.
3. 두 헤드 모두 0 으로 초기화 -> 학습 전에는 정확히 항등이다.

입력은 프레임률 제어열(물리 손잡이)과, 있으면 목표 특징(복사합성 학습 시).
인과 TCN 이라 스트리밍에서도 그대로 쓴다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .control import PARAM_NAMES, frames_to_samples
from .nn import SSPTCN
from .tviir import peaking_eq_coeffs, tv_biquad


class ResidualCorrector(nn.Module):
    def __init__(self, fs: float, hop: int, n_eq: int = 12, max_db: float = 6.0,
                 hidden: int = 96, n_templates: int = 5, cond_dim: int = 0):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.max_db = max_db
        self.n_eq, self.n_templates = n_eq, n_templates
        # EQ 중심: 로그 등간격 150 Hz ~ 0.4 fs
        f = torch.logspace(math.log10(150.0), math.log10(0.4 * fs), n_eq)
        self.register_buffer("eq_f", f)
        self.eq_q = 1.4
        in_dim = len(PARAM_NAMES) + cond_dim
        self.body = SSPTCN(in_dim, hidden, n_eq + n_templates + 1)
        from .control import PARAMS
        self.register_buffer("scale", torch.tensor([max(abs(PARAMS[n].hi), 1.0) for n in PARAM_NAMES]))

    def forward(self, ctrl_frames: torch.Tensor, cond: torch.Tensor | None = None,
                state: dict | None = None, emit: int | None = None) -> dict:
        """ctrl_frames: (B,T,P) 정규화 안 된 제어열. 반환 프레임률 헤드 (emit 프레임).

        state["hist"] 에 수용영역만큼의 과거 입력을 이어 붙여 청크 경계에서도 오프라인과 같다.
        """
        x = ctrl_frames / self.scale                       # 고정 스케일 (청크 최대값이면 스트리밍이 깨진다)
        if cond is not None:
            x = torch.cat([x, cond], -1)
        t = x.shape[1] if emit is None else int(emit)
        hist = None if state is None else state.get("hist")
        R = self.body.receptive_field - 1
        xin = x if hist is None else torch.cat([hist, x], 1)
        h = self.body(xin)[:, -x.shape[1]:][:, :t]
        if state is not None:
            keep = torch.cat([xin[:, : xin.shape[1] - (x.shape[1] - t)]], 1)
            state["hist"] = keep[:, -R:] if R > 0 else keep[:, :0]
        return dict(eq_db=torch.tanh(h[..., : self.n_eq]) * self.max_db,
                    template_mix=torch.sigmoid(h[..., self.n_eq: self.n_eq + self.n_templates]) * 0.5,
                    template_gate=torch.sigmoid(h[..., -1:]))

    def apply(self, audio: torch.Tensor, heads: dict, mix: torch.Tensor,
              extra: torch.Tensor | None = None, state: dict | None = None) -> dict:
        """audio (B,N) -> 보정된 (B,N). mix: (B,N) 적용 비율(residual_mix)."""
        state = {} if state is None else state
        y = audio
        db = frames_to_samples(heads["eq_db"], self.hop) * mix.unsqueeze(-1)
        for i in range(self.n_eq):
            f = self.eq_f[i].expand_as(y)
            y, state[f"eq{i}"] = tv_biquad(y, *peaking_eq_coeffs(f, self.eq_q, db[..., i], self.fs),
                                           zi=state.get(f"eq{i}"))
        if extra is not None:
            g = frames_to_samples(heads["template_gate"], self.hop)[..., 0] * mix
            y = y + g * extra
        return dict(audio=y, state=state)
