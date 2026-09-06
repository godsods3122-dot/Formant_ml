"""성도: 포먼트 + 위상차(올패스) 필터 -> 비강 필터 -> 방사.

전부 `tviir.tv_biquad` 위에 있다. 계수는 샘플마다 바뀌고 상태는 이어진다.
그래서 자음처럼 경계조건이 급변하는 곳에서 **과도응답과 위상은 방정식이 낸다**
— 우리가 프레임을 이어 붙이지 않는다.

신호 경로 (선형이므로 순서를 바꿔도 정상 상태는 같다; 과도 상태의 차이는 무시할 만하다)

    du (성문 유량미분) + 방사(기식) + back_leak·방사(마찰)  ──> 포먼트 캐스케이드 (K + 고차 보정) ─┐
    마찰·과도음 소스 ──> 방사 ──> 앞공동 극 + 뒤공동 영점 ─────────────────────────────────────┤
                                                                                              ▼
                                                          올패스 위상차 체인 -> 비강 극영점 -> 측지 영점 -> tract_gain

고차 극 보정(higher-pole correction)이 **필수**다. DC 정규화 공명기 K 개만 곱하면
K 번째 포먼트 위가 실제(무손실관 |H| ≥ 1)보다 수십 dB 어둡다 — v1 이 "모음의 9~13 kHz
가 34 dB 모자란다"(HANDOFF §6.8) 고 적은 것이 정확히 이것이다. 균일관 위치
F_n = (2n−1)·c/4L 의 고차 극을 나이퀴스트까지 넓은 대역폭으로 고정 배치한다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .control import N_ALLPASS, N_FORMANTS, frames_to_samples
from .noise import C_SOUND
from .tviir import (allpass_coeffs, antiresonator_coeffs, first_difference_coeffs,
                    notch_coeffs, resonator_coeffs, tv_biquad)


class VocalTract(nn.Module):
    def __init__(self, fs: float, hop: int, length_cm: float = 14.6,
                 bw_floor: float = 40.0, bw_slope: float = 0.05,
                 n_formants: int = N_FORMANTS, n_extra: int | None = None):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.length_cm = length_cm
        self.bw_floor, self.bw_slope = bw_floor, bw_slope
        self.K = n_formants
        f1 = C_SOUND / (4.0 * length_cm)
        # 나이퀴스트의 70 % 까지만. 이산 공명기는 θ→π 에서 두 극이 서로 붙어 이득이
        # 제곱으로 뛴다(19.8 kHz 극이 +58 dB: 측정). 연속계에는 없는 현상이다.
        fixed = [(2 * n - 1) * f1 for n in range(n_formants + 1, 64)
                 if (2 * n - 1) * f1 < 0.70 * fs / 2]
        if n_extra is not None:
            fixed = fixed[:n_extra]
        self.register_buffer("uniform_formants",
                             torch.tensor([(2 * n - 1) * f1 for n in range(1, n_formants + 1)]))
        self.register_buffer("extra_formants", torch.tensor(fixed, dtype=torch.float32))
        # 학습 파라미터: 고차 극 보정의 대역폭 배율(화자 고역 손실), 앞공동 대역폭 배율
        self.log_extra_bw = nn.Parameter(torch.tensor(0.0))
        self.log_front_bw = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------ 계수
    def default_bw(self, f):
        return self.bw_floor + self.bw_slope * f

    def _up(self, v):
        return frames_to_samples(v.unsqueeze(-1), self.hop)[..., 0][:, :self._n_emit]

    def _formant_tracks(self, c: dict):
        """(f_k, bw_k) 샘플률 리스트. 0 이면 균일관 기본값 / 물리 기본 대역폭.

        **기본값 대치는 프레임률에서** 한 뒤 샘플률로 보간한다. 샘플률에서 하면
        0 ↔ 값 사이를 선형 보간하며 F1 이 10 Hz 를 지나간다(첫 렌더의 폭주 원인 2).
        """
        out = []
        for k in range(1, self.K + 1):
            f = c[f"f{k}"]
            f = torch.where(f > 0, f, self.uniform_formants[k - 1].expand_as(f))
            bw = c[f"bw{k}"]
            bw = torch.where(bw >= 20.0, bw, self.default_bw(f))   # 20 Hz 미만 = 기본
            if k == 1:                                   # 비강 결합은 F1 을 넓힌다
                bw = bw + 120.0 * c["velum"]
            out.append((self._up(f), self._up(bw)))
        return out

    # ------------------------------------------------------------ 단계
    def _cascade(self, x, tracks, state, prefix):
        for i, (f, bw) in enumerate(tracks):
            key = f"{prefix}{i}"
            x, state[key] = tv_biquad(x, *resonator_coeffs(f, bw, self.fs), zi=state.get(key))
        return x

    def _extra_cascade(self, x, state):
        bw_scale = torch.exp(self.log_extra_bw)
        for i, f in enumerate(self.extra_formants):
            key = f"x{i}"
            fk = f.expand_as(x)
            x, state[key] = tv_biquad(x, *resonator_coeffs(fk, self.default_bw(fk) * bw_scale,
                                                           self.fs), zi=state.get(key))
        return x

    def _front_cavity(self, x, c, state):
        """협착 하류 앞공동 극 + 뒤공동 영점. 치찰음 지문이 여기 있다."""
        L = self.length_cm
        fl = c["front_len"]
        l_front = self._up(torch.where(fl > 0, fl, (1.0 - c["c_place"]) * L).clamp(0.3, L))
        f_p = (C_SOUND / (4.0 * l_front)).clamp(500.0, 0.45 * self.fs)     # 1/4 파장
        bw_p = (300.0 + 0.04 * f_p) * torch.exp(self.log_front_bw)
        l_back = (L - l_front).clamp_min(1.0)
        f_z = (C_SOUND / (2.0 * l_back)).clamp(300.0, 0.45 * self.fs)      # 뒤공동 반공진
        bw_z = 250.0 + 0.1 * f_z
        x, state["fp"] = tv_biquad(x, *resonator_coeffs(f_p, bw_p, self.fs), zi=state.get("fp"))
        # 영점은 반드시 극-영점 노치로: DC 정규화 영점쌍은 1.3 kHz 영점이 고역을 +40 dB
        # 들어올린다(측정: 앞공동 경로 13~20 kHz 가 지배적이었던 원인).
        x, state["fz"] = tv_biquad(x, *notch_coeffs(f_z, bw_z, self.fs, 3.0), zi=state.get("fz"))
        return x

    def _allpass(self, x, c, state):
        for k in range(1, N_ALLPASS + 1):
            f, r = c[f"ap{k}_f"], c[f"ap{k}_r"]
            if float(f.detach().abs().max()) == 0.0:
                continue
            r = torch.where(f > 0, r, torch.zeros_like(r)).clamp(0.0, 0.98)
            f = torch.where(f > 0, f, torch.full_like(f, 1000.0))
            f, r = self._up(f), self._up(r)
            x, state[f"ap{k}"] = tv_biquad(x, *allpass_coeffs(f, r, self.fs), zi=state.get(f"ap{k}"))
        return x

    def _nasal(self, x, c, state):
        """연구개 개방 velum ∈ [0,1]: 극(nasal_f)·영점(nasal_z) 의 대역폭을 1/velum 로.

        velum→0 이면 r→0 이라 두 응답 모두 정확히 1 (끊김 없이 사라진다).
        """
        if float(c["velum"].detach().max()) <= 1e-3:
            return x
        v = self._up(c["velum"]).clamp(1e-3, 1.0)
        f_p, f_z = self._up(c["nasal_f"]), self._up(c["nasal_z"])
        x, state["np"] = tv_biquad(x, *resonator_coeffs(f_p, 100.0 / v, self.fs), zi=state.get("np"))
        x, state["nz"] = tv_biquad(x, *notch_coeffs(f_z, 150.0 / v, self.fs), zi=state.get("nz"))
        return x * (1.0 - 0.2 * v)                      # 비강 벽 손실

    def _lateral(self, x, c, state):
        for k in (1, 2):
            fz = c[f"lat_z{k}"]
            if float(fz.detach().max()) <= 0.0:
                continue
            bw = c["lat_bw"]
            # 영점이 없는 구간(0)은 대역폭을 매우 넓게 -> 노치 깊이 0 -> 응답 1
            bw = torch.where(fz > 0, bw, torch.full_like(bw, 2.0e5))
            fz = torch.where(fz > 0, fz, torch.full_like(fz, 3000.0))
            x, state[f"lz{k}"] = tv_biquad(x, *notch_coeffs(self._up(fz), self._up(bw), self.fs),
                                           zi=state.get(f"lz{k}"))
        return x

    # ------------------------------------------------------------ 전체
    def forward(self, du: torch.Tensor, fric: torch.Tensor, asp: torch.Tensor,
                transient: torch.Tensor, c: dict, state: dict | None = None) -> dict:
        """소스 (B,N) 들과 프레임률 제어 c (B,T_all). N ≤ T_all·hop 이면 선행 프레임이 있는 것이다."""
        state = {} if state is None else state
        self._n_emit = du.shape[1]
        up = self._up
        # 난류·과도음은 압력원이므로 입술 방사(미분)를 받는다. du 는 이미 미분이다.
        asp_r, state["ra"] = tv_biquad(asp, *first_difference_coeffs(1.0, asp), zi=state.get("ra"))
        fr = fric + transient
        fr_r, state["rf"] = tv_biquad(fr, *first_difference_coeffs(1.0, fr), zi=state.get("rf"))
        leak = up(c["back_leak"])
        x_g = du + asp_r + leak * fr_r
        tracks = self._formant_tracks(c)
        y_g = self._extra_cascade(self._cascade(x_g, tracks, state, "f"), state)
        y_f = self._front_cavity((1.0 - leak) * fr_r, c, state)
        y = y_g + y_f
        y = self._allpass(y, c, state)
        y = self._nasal(y, c, state)
        y = self._lateral(y, c, state)
        y = y * up(c["tract_gain"])
        return dict(audio=y, glottal_path=y_g, front_path=y_f, state=state)
