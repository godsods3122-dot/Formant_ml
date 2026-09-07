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

import numpy as np
import torch
import torch.nn as nn

from .control import N_ALLPASS, N_FORMANTS, frames_to_samples
from .noise import C_SOUND
from .tviir import (allpass_coeffs, antiresonator_coeffs, first_difference_coeffs,
                    lowpass_coeffs, notch_coeffs, peak_coeffs, resonator_coeffs, tv_biquad)


def higher_pole_correction_fir(fs: float, length_cm: float, n_explicit: int, bw_floor: float,
                               bw_slope: float, extra_bw_floor: float, n_taps: int = 1024,
                               n_tail: int = 400, tail_bw_slope: float = 0.15,
                               cap_db: float = 40.0) -> torch.Tensor:
    """Fant 의 고차 극 보정을 **최소위상 FIR** 로 (n_explicit 번째 위의 극 전부).

    무손실관 |H| = 1/|cos(πf/2F₁)| 는 어디서나 ≥ 1 인데, DC 정규화 공명기 N 개의 곱은 그 위에서
    급히 떨어진다 — L=17 cm, N=11 이면 6 kHz 에서 −27 dB, 8 kHz 에서 −50 dB (계산). 모음의 4~8 kHz 가
    실측보다 35~60 dB 어두웠던 원인이다. 꼬리 극 F_n=(2n−1)c/4L (n>N) 을 감쇠 공명기로 두고 그 곱의
    크기를 켑스트럼으로 최소위상화한다. 매끈한 곡선이라 1024 탭이면 충분하다.
    """
    f1 = C_SOUND / (4.0 * length_cm)
    n_fft = 8 * n_taps
    f = np.linspace(0.0, fs / 2, n_fft // 2 + 1)
    w = 2 * np.pi * f / fs
    z1 = np.exp(-1j * w)
    logmag = np.zeros_like(f)
    for n in range(n_explicit + 1, n_explicit + 1 + n_tail):
        fn = (2 * n - 1) * f1
        # 꼬리 극의 손실: 벽·점성·방사가 주파수에 따라 커진다 → 대역폭이 가파르게 는다
        bw = max(bw_floor + tail_bw_slope * fn, extra_bw_floor)
        if fn > fs / 2:                                  # 나이퀴스트 위 극: 연속계 근사 (봉우리 없음)
            logmag += -np.log(np.abs(1.0 - (f / fn) ** 2 + 1j * (f / fn) * (bw / fn)))
            continue
        r = np.exp(-np.pi * bw / fs); th = 2 * np.pi * fn / fs
        a1, a2 = -2 * r * np.cos(th), r * r
        D = 1.0 + a1 * z1 + a2 * z1 * z1
        logmag += np.log(abs(1 + a1 + a2)) - np.log(np.abs(D))
    logmag = np.minimum(logmag, np.log(10.0 ** (cap_db / 20)))  # 상한 (실측 8~13 kHz 는 정점 −40~−50 dB)
    # 최소위상: 실켑스트럼 접기
    mag = np.concatenate([logmag, logmag[-2:0:-1]])
    cep = np.fft.ifft(mag).real
    n = len(cep); fold = np.zeros(n); fold[0] = cep[0]; fold[1:n // 2] = 2 * cep[1:n // 2]; fold[n // 2] = cep[n // 2]
    h = np.fft.ifft(np.exp(np.fft.fft(fold))).real[:n_taps]
    h *= np.hanning(2 * n_taps)[n_taps:]                      # 꼬리만 창
    return torch.tensor(h, dtype=torch.float32)


class VocalTract(nn.Module):
    def __init__(self, fs: float, hop: int, length_cm: float = 14.6,
                 bw_floor: float = 40.0, bw_slope: float = 0.05,
                 n_formants: int = N_FORMANTS, n_extra: int | None = None):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.length_cm = length_cm
        self.bw_floor, self.bw_slope = bw_floor, bw_slope
        self.K = n_formants
        self.extra_cap = 0.60
        f1 = C_SOUND / (4.0 * length_cm)
        # 고차 극 보정. **나이퀴스트까지 채우면 안 된다.** 이산 공명기는 θ→π 에서 극쌍이
        # z=−1 의 이중 실극으로 붙고, DC 정규화된 이득이 (1+r)²/(1−r)² 로 뛴다. 실제로
        # 0.98·fs/2 까지 12 개를 채웠더니 이 종속이 10 kHz 에서 +97 dB, 22 kHz 에서
        # **+151 dB** 였다(측정). 그게 복사합성에서 7 kHz 위 오차 +14~+22 dB 의 정체다.
        # Fant 의 고차 극 보정은 원래 "완만히 올라갔다가 내려오는" 몇 dB~10 dB 짜리다.
        # 그 모양이 나오는 지점: 0.60·fs/2 까지, 대역폭 Q≈1 (고역의 벽·점성·방사·횡모드
        # 손실). 그러면 DC 0 dB, 4 kHz +4, 10 kHz +13, 22 kHz −0.4 dB 가 된다.
        fixed = [(2 * n - 1) * f1 for n in range(n_formants + 1, 128)
                 if (2 * n - 1) * f1 < self.extra_cap * fs / 2]
        if n_extra is not None:
            fixed = fixed[:n_extra]
        self.register_buffer("uniform_formants",
                             torch.tensor([(2 * n - 1) * f1 for n in range(1, n_formants + 1)]))
        self.register_buffer("extra_formants", torch.tensor(fixed, dtype=torch.float32))
        # 학습 파라미터: 고차 극 보정의 대역폭 배율(화자 고역 손실), 앞공동 대역폭 배율
        self.log_extra_bw = nn.Parameter(torch.tensor(0.0))
        self.log_front_bw = nn.Parameter(torch.tensor(0.0))
        self.extra_bw_floor = 800.0
        self.front_bw_slope = 0.12      # 앞공동 극의 대역폭 = 500 + 이 값 × f (실측 적합)
        self.nasal_zfrac = 0.80         # 측지 영점의 대역폭 = fz × 이 값 (깊이·폭)
        self.nasal_gain = 1.60          # 머머가 모음보다 −7 dB (실측 M −6.7) 이 되는 값
        # 비강도 관이다 — 극 3 개로 자르면 2 kHz 위가 −60 dB 로 무너진다(측정). 인두+비강
        # 20 cm 의 c/2L ≈ 875 Hz 간격으로 나이퀴스트까지 채운다 (ADR 0009 와 같은 이유).
        nsp = C_SOUND / (2.0 * 20.0)
        self.register_buffer("nasal_extra", torch.tensor(
            [2400.0 + k * nsp for k in range(1, 64) if 2400.0 + k * nsp < 0.70 * fs / 2],
            dtype=torch.float32))
        self.extra_bw_slope = 1.00        # 고차 극 손실 (벽·점성·방사·횡모드). Q≈1
        self.use_hpc = False
        self.register_buffer("hpc", torch.zeros(1))

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
            f = torch.where(f > 0, f, self.uniform_formants[k - 1].to(f.dtype).expand_as(f))
            bw = c[f"bw{k}"]
            bw = torch.where(bw >= 20.0, bw, self.default_bw(f))   # 20 Hz 미만 = 기본
            if k == 1:      # 연구개가 열리면 F1 이 넓어진다 (에너지가 비강으로 샌다)
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
            fk = f.to(x.dtype).expand_as(x)
            bw = torch.clamp(self.bw_floor + self.extra_bw_slope * fk, min=self.extra_bw_floor) * bw_scale
            x, state[key] = tv_biquad(x, *resonator_coeffs(fk, bw, self.fs), zi=state.get(key))
        return x

    def _hpc(self, x, state):
        """고차 극 보정 FIR (LTI, 최소위상). 스트리밍은 마지막 n_taps−1 샘플을 이어 붙인다."""
        k = self.hpc.numel()
        hist = state.get("hpc")
        if hist is None:
            hist = torch.zeros(x.shape[0], k - 1, dtype=x.dtype, device=x.device)
        xin = torch.cat([hist, x], dim=1)
        y = torch.nn.functional.conv1d(xin.unsqueeze(1),
                                       self.hpc.to(x.dtype).flip(0).view(1, 1, -1)).squeeze(1)
        state["hpc"] = xin[:, -(k - 1):]
        return y

    def _front_cavity(self, x, c, state):
        """협착 하류 앞공동 극 + 뒤공동 영점. 치찰음 지문이 여기 있다."""
        L = self.length_cm
        fl = c["front_len"]
        l_front = self._up(torch.where(fl > 0, fl, (1.0 - c["c_place"]) * L).clamp(0.3, L))
        f_p = (C_SOUND / (4.0 * l_front)).clamp(500.0, 0.45 * self.fs)     # 1/4 파장
        bw_p = (500.0 + self.front_bw_slope * f_p) * torch.exp(self.log_front_bw)
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

    def _nasal_branch(self, x, c, state):
        """비강 분기 — 성문 소스에서 **갈라져 나와 콧구멍으로 방사**된다 (병렬 경로).

        v1·v2 초판은 비강을 구강 캐스케이드 **뒤에 얹은 필터**로 두었다. 그러면 구강이
        닫힐 때(tract_gain·a_c) 저역까지 같이 죽어 모음↔비음 경계가 계단이 된다.
        실측(같은 화자, 5 ms): 80~300 Hz 는 모음과 머머 사이에서 **±2 dB 로 연속**이고
        0.8~2.5 kHz 만 8~11 dB 떨어진다. 병렬 분기라야 그 성질이 나온다.

        구조 (Fant 1960 / Fujimura 1962 의 표준):
            인두 + 비강의 극 3 개  ×  닫힌 구강이 만드는 **측지 영점** 1 개
        영점 주파수는 폐쇄 위치가 정한다 — 구강 측지가 길수록(ㅁ) 낮고 짧을수록(ㅇ) 높다.
        비강은 점막 손실이 커서 대역폭이 넓다(`nasal_damp`).

        모음의 **비음화**는 따로 만들지 않는다. 연구개가 열린 채 구강도 열려 있으면 두
        분기가 그냥 더해지고, 그 간섭이 F1 부근의 극-영점 쌍으로 나타난다.
        """
        if float(c["velum"].detach().max()) <= 1e-4:
            return torch.zeros_like(x)
        d = self._up(c["nasal_damp"])
        y = x
        for i, (key, bw) in enumerate((("nasal_f", 130.0), ("nasal_f2", 300.0),
                                       ("nasal_f3", 500.0))):
            f = self._up(c[key])
            y, state[f"nb{i}"] = tv_biquad(y, *resonator_coeffs(f, bw * d, self.fs),
                                           zi=state.get(f"nb{i}"))
        for j, f in enumerate(self.nasal_extra):
            fk = f.to(y.dtype).expand_as(y)
            y, state[f"nx{j}"] = tv_biquad(
                y, *resonator_coeffs(fk, (300.0 + 0.25 * fk) * d, self.fs), zi=state.get(f"nx{j}"))
        fz = self._up(c["nasal_z"])
        y, state["nbz"] = tv_biquad(y, *antiresonator_coeffs(fz, fz * self.nasal_zfrac * d,
                                                             self.fs), zi=state.get("nbz"))
        return y * self.nasal_gain * self._up(c["nasal_gain"])

    def _lateral(self, x, c, state):
        """설측음의 측지 영점 2 개. 깊이는 `lat_mix` 로 **연속으로** 켜고 끈다.

        mix→0 이면 대역폭이 발산해 노치가 사라진다(응답이 정확히 1). 주파수를 0 으로 껐다
        켜는 방식은 로그 파라미터의 영차 유지 때문에 한 프레임에 도약하고, 그 계수 도약이
        필터 상태와 만나 클릭이 된다.
        """
        mix = self._up(c["lat_mix"]).clamp(1e-3, 1.0)
        if float(c["lat_mix"].detach().max()) <= 1e-3:
            return x
        for k in (1, 2):
            fz = c[f"lat_z{k}"]
            if float(fz.detach().max()) <= 0.0:
                continue
            fz = torch.where(fz > 0, fz, torch.full_like(fz, 3000.0))
            bw = self._up(c["lat_bw"]) / mix
            x, state[f"lz{k}"] = tv_biquad(x, *notch_coeffs(self._up(fz), bw, self.fs),
                                           zi=state.get(f"lz{k}"))
        return x

    # ------------------------------------------------------------ 전체
    def forward(self, du: torch.Tensor, fric: torch.Tensor, asp: torch.Tensor,
                transient: torch.Tensor, c: dict, state: dict | None = None) -> dict:
        """소스 (B,N) 들과 프레임률 제어 c (B,T_all). N ≤ T_all·hop 이면 선행 프레임이 있는 것이다."""
        state = {} if state is None else state
        self._n_emit = du.shape[1]
        up = self._up
        # 성도 전체를 float64 로. 캐스케이드가 10 kHz 에서 −85 dB 까지 내려갔다가 고차 극 보정이
        # 그만큼 되돌리므로, 단 사이의 float32 반올림(6e-8)이 −50 dB 잡음으로 올라온다.
        dt_in = du.dtype
        du, fric, asp, transient = du.double(), fric.double(), asp.double(), transient.double()
        c = {k: v.double() for k, v in c.items()}
        # 난류·과도음은 압력원이므로 입술 방사(미분)를 받는다. du 는 이미 미분이다.
        asp_r, state["ra"] = tv_biquad(asp, *first_difference_coeffs(1.0, asp), zi=state.get("ra"))
        fr = fric + transient
        fr_r, state["rf"] = tv_biquad(fr, *first_difference_coeffs(1.0, fr), zi=state.get("rf"))
        leak = up(c["back_leak"])
        x_g = du + asp_r + leak * fr_r
        # 구강 게이트는 **입력에** 건다. 출력에 걸면 폐쇄 동안에도 캐스케이드가 소스를 그대로
        # 받아 울리고, 해제 순간 그 저장된 에너지가 50 배로 풀려 12 배짜리 클릭이 났다(측정).
        # 입력을 막으면 공진기는 제 대역폭으로 1~2 ms 만에 잦아들고, 해제에서는 다시 울려
        # 오르는 데 시간이 걸린다 — 실측의 "중역이 45 ms 에 걸쳐 올라온다" 가 그것이다.
        # 두 분기는 **유량을 나눠 갖는다**. 연구개 포트와 구강이 동시에 열려 있으면(비음화
        # 모음) 각자 절반씩이고, 구강이 닫히면 전부 비강으로 간다(머머). 나누지 않으면
        # 비음화 구간에서 두 경로가 그대로 더해져 12 배짜리 봉우리가 났다(측정).
        vel = up(c["velum"]).clamp(0.0, 1.0)
        opn = up(c["oral_open"]).clamp(0.0, 1.0)
        tot = (vel + opn).clamp_min(1e-3)
        oral, nas = opn / tot, vel / tot
        tracks = self._formant_tracks(c)
        y_g = self._extra_cascade(self._cascade(x_g * oral, tracks, state, "f"), state)
        if self.use_hpc:
            y_g = self._hpc(y_g, state)
        y_f = self._front_cavity((1.0 - leak) * fr_r * oral, c, state)
        # 비강도 같은 이유로 **입력에** 건다. 출력에 걸면 연구개가 닫힌 동안에도 분기가
        # 소스를 받아 울리고, 열리는 순간 그 에너지가 통째로 튀어나온다(측정: 12 배 클릭).
        y_n = self._nasal_branch(x_g * nas, c, state)
        y = y_g + y_f
        y = self._allpass(y, c, state)
        y = y + y_n
        y = self._lateral(y, c, state)
        y = y * up(c["tract_gain"])
        return dict(audio=y.to(dt_in), glottal_path=y_g.to(dt_in), front_path=y_f.to(dt_in),
                    nasal_path=y_n.to(dt_in), state=state)
