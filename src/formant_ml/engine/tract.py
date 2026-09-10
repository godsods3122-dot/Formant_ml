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

# 고차 극 보정을 최소위상 FIR 로 되돌리는 경로가 여기 있었다. 12 kHz 에서 +160 dB 가
# 필요해 수치적으로 성립하지 않아 기각했고(ADR 0009·0012), 대역폭 법칙을 하나로
# 합치면서 그 함수가 참조하던 `extra_bw_floor`·`extra_bw_slope` 도 사라졌다.
# 기록은 ADR 에 있다 — 죽은 코드로 남겨 두면 낡은 파라미터로 되살아난다.


class VocalTract(nn.Module):
    def __init__(self, fs: float, hop: int, length_cm: float = 14.6,
                 bw_floor: float = 40.0, bw_slope: float = 0.05,
                 n_formants: int = N_FORMANTS, n_extra: int | None = None,
                 front_bw_slope: float = 0.20):
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
        # 고차 극의 손실 (벽·점성·방사·횡모드). Q≈1 — ADR 0012.
        self.extra_bw_floor = 800.0
        self.extra_bw_slope = 1.00
        # 앞공동 극의 대역폭 = 500 + 이 값 × f_p. **화자 프로파일이 준다** — 한 상수로
        # 두면 두 화자가 반대로 잡아당긴다. 실측 적합: 남 /ㅅ/ 0.08, 여 /ㅆ/ 0.36.
        self.front_bw_slope = float(front_bw_slope)
        self.nasal_zfrac = 0.80         # 측지 영점의 대역폭 = fz × 이 값 (깊이·폭)
        self.nasal_gain = 1.60          # 머머가 모음보다 −7 dB (실측 M −6.7) 이 되는 값
        # 비강도 관이다 — 극 3 개로 자르면 2 kHz 위가 −60 dB 로 무너진다(측정). 인두+비강
        # 20 cm 의 c/2L ≈ 875 Hz 간격으로 나이퀴스트까지 채운다 (ADR 0009 와 같은 이유).
        nsp = C_SOUND / (2.0 * 20.0)
        self.register_buffer("nasal_extra", torch.tensor(
            [2400.0 + k * nsp for k in range(1, 64) if 2400.0 + k * nsp < 0.70 * fs / 2],
            dtype=torch.float32))


    # ------------------------------------------------------------ 계수
    def default_bw(self, f):
        """극의 물리 기본 대역폭 (Klatt: B1 50~80, B4 175~280).

        고차 극은 이 법칙을 쓰지 않는다 — `_extra_cascade` 가 훨씬 센 감쇠를 쓴다.
        그 경계에서 Q 가 18 -> 0.9 로 급변하고, 거기에 −11.5 dB 골이 생긴다(실측:
        남성 8.5 kHz. 그 주파수는 (2K−1)·c/4L 이라 화자마다 다르다 — 코드 상수 K 가
        만든 인공물이다). **두 법칙을 하나의 매끈한 식으로 합치는 것을 시도했다가
        되돌렸다**: 고차 극이 날카로워지면서 종속 이득이 8~10 kHz 에서 +33.8 dB 가
        되고(남성), 복사합성이 그 리프트에 이득을 맞추느라 230 Hz 가 −27 dB 로
        무너졌다 (남 /사/ 포락 68.6 % -> 11.5 %). 무릎을 첫 고차 극에 묶어 Q≈1 로
        낮춰도 이번엔 9~13 kHz 를 메우는 힘이 6 dB -> 3.6 dB 로 약해졌다.
        골은 남은 문제로 MEASUREMENTS §7.6 에 적어 둔다.
        """
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
            bw = torch.clamp(self.bw_floor + self.extra_bw_slope * fk,
                             min=self.extra_bw_floor) * bw_scale
            x, state[key] = tv_biquad(x, *resonator_coeffs(fk, bw, self.fs), zi=state.get(key))
        return x

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
        # **극쌍 보정이 있는 노치를 쓴다.** `antiresonator_coeffs` 는 영점쌍만 DC 에서
        # 정규화하므로 먼 대역이 통째로 들린다 — 실측(fz 1400 Hz, bw 200~1400):
        #   4 kHz +15~17 dB,  12 kHz +33~35 dB,  20 kHz +39~41 dB
        # `notch_coeffs` 는 같은 각도에 4 배 넓은 극쌍을 두어 노치 바깥을 1 로 되돌린다
        # (같은 값에서 12 kHz +0.6~+11.6 dB). 그 함수의 주석이 경고하던 바로 그 경우인데
        # 비강 분기만 옛 형태로 남아 있었다 (MEASUREMENTS §34).
        fz = self._up(c["nasal_z"])
        y, state["nbz"] = tv_biquad(y, *notch_coeffs(fz, fz * self.nasal_zfrac * d,
                                                     self.fs), zi=state.get("nbz"))
        return y * self.nasal_gain * self._up(c["nasal_gain"])

    def _lateral(self, x, c, state):
        """설측음의 측지 영점 2 개. 깊이는 `lat_mix` 로 **연속으로** 켜고 끈다.

        **깊이는 젖음/마름 섞기로 준다** — `y = x + mix·(notch(x) − x)`.
        mix=0 이면 항등이고 mix=1 이면 노치 그대로다. 주파수를 0 으로 껐다 켜는 방식은
        로그 파라미터의 영차 유지 때문에 한 프레임에 도약하고, 그 계수 도약이 필터
        상태와 만나 클릭이 되므로 쓰지 않는다. 섞기는 mix 에 대해 연속이라 그 문제가 없다.

        **예전에는 대역폭을 `lat_bw / mix` 로 넓혀서 껐다. 그건 틀렸다.**
        주석에 "대역폭이 발산하면 응답이 정확히 1" 이라고 적어 두었는데 아니다.
        `notch_coeffs` 는 영점쌍(반지름 rz)과 4 배 넓은 극쌍(rp)을 두고 **DC 에서**
        정규화한다. 대역폭을 키우면 둘 다 원점으로 오그라드는데 극이 4 배 빨리
        오그라들므로, DC 는 1 이어도 **고역이 들린다.** 실측(적합값 lat_z 3300 Hz,
        lat_mix 0.065 → 실효 대역폭 4554 Hz):

            250 Hz −0.02 | 1 k −0.26 | 4 k −0.07 | 8 k **+7.4** | 12 k **+11.0** | 16 k **+12.6** dB

        영점이 둘이라 12~16 kHz 에서 +20 dB 를 넘는다. 적합기는 이것을 **공짜 고역
        셸프**로 썼다 — 파찰음 /ㅊ/ 구간에서 설측을 끄면 에너지가 9.5 dB 줄고 파형
        첨도가 14.15 → 9.45 로 내려간다(목표 3.49). 사용자가 "누가 봐도 튀는 파형"
        이라고 지적한 것이 이것이다 (MEASUREMENTS §34).
        """
        if float(c["lat_mix"].detach().max()) <= 1e-3:
            return x
        mix = self._up(c["lat_mix"]).clamp(0.0, 1.0)
        y = x
        for k in (1, 2):
            fz = c[f"lat_z{k}"]
            if float(fz.detach().max()) <= 0.0:
                continue
            fz = torch.where(fz > 0, fz, torch.full_like(fz, 3000.0))
            y, state[f"lz{k}"] = tv_biquad(y, *notch_coeffs(self._up(fz),
                                                            self._up(c["lat_bw"]), self.fs),
                                           zi=state.get(f"lz{k}"))
        return x + mix * (y - x)

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
