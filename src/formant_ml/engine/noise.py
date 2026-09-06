"""노이즈 생성기(여러 개) — 마찰(협착 공기역학), 기식(성문), 과도음(템플릿 + 변조).

세 생성기의 역할이 다르다는 점이 설계의 핵심이다.

* **마찰 노이즈는 물리로 직접 만든다.** 치찰음은 학습 단에서 쉽게 나오지 않는
  요소이므로 협착 면적·압력에서 레이놀즈 수를 계산해 세기·스펙트럼을 낸다.
  (Stevens 1971/1998, Shadle 1985: 소스 세기 ∝ (Re² − Re_c²)⁺, 스펙트럼 정점
  Strouhal ≈ 0.2, 장애물(앞니) 다이폴은 +6 dB/oct.)
* **기식 노이즈는 성문 난류**다. `glottis.py` 가 준 포락선(개방기 동기 AM 포함)
  에 색만 입힌다. 성도 **전체**를 통과한다.
* **과도음은 템플릿 + 변조**다. 혀가 입천장에 닿는 소리, 혀를 튕긴 뒤 입 바닥에
  닿는 소리, 침 소리, 입술 소리는 짧은 결정론적 사건이라 통계 학습보다
  **템플릿 뱅크**가 맞다. 템플릿은 학습 가능 텐서이고, 변조(속도/세기/기울기/
  미세 난수)로 매번 다르게 낸다.

모든 생성기는 "소스" 를 낸다. 성도 통과(앞공동 공명, 뒤로 새는 경로, 방사)는
`tract.py` 가 맡는다 — 소스와 필터를 섞지 않는다(v1 §5.2 의 실수).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .control import frames_to_samples
from .glottis import CMH2O, RHO
from .rng import NoiseBank
from .tviir import first_difference_coeffs, lowpass_coeffs, tv_biquad

NU = 0.15                # 공기 동점성 cm²/s
RE_CRIT = 1800.0         # 난류 개시 레이놀즈 수 (Stevens 1998, 1700~1800)
RE_REF = 8000.0          # /s/ 급 협착의 전형값 — 세기 정규화 기준
C_SOUND = 35000.0        # cm/s


def series_flow(p_sub_cmh2o, a_g, a_c):
    """성문(Ag)·구강 협착(Ac) 직렬 오리피스의 정상 유량 U [cm³/s] 와 협착 강하압.

    U = sqrt(2Ps/ρ) / sqrt(1/Ag² + 1/Ac²).  Po/Ps = Ag²/(Ag²+Ac²) (v1 §5.3 검증).
    """
    ps = p_sub_cmh2o.clamp_min(0.0) * CMH2O
    inv = 1.0 / a_g.clamp_min(1e-4) ** 2 + 1.0 / a_c.clamp_min(1e-4) ** 2
    u = torch.sqrt(2.0 * ps / RHO) / torch.sqrt(inv)
    dp_c = 0.5 * RHO * (u / a_c.clamp_min(1e-4)) ** 2
    return u, dp_c


def reynolds(u, a_c):
    """협착 제트의 레이놀즈 수. d = 등가 직경 sqrt(4A/π)."""
    a = a_c.clamp_min(1e-4)
    v = u / a
    d = torch.sqrt(4.0 * a / math.pi)
    return v * d / NU, v, d


def slow_modulation(white_frames: torch.Tensor, fs_frame: float, knee_hz, zi=None):
    """느린 1/f² 변조 포락선 (프레임률, 단위분산 근사). v1 §5 의 난류 비정상성(제트 사행).

    2 극 저역통과(RBJ, 꺾임 knee_hz)를 **상태를 이어가며** 적용한다 — FFT 로 만들면
    청크 경계에 따라 달라져 스트리밍 = 오프라인이 깨진다. 이득은 백색 입력의 출력
    분산이 1 이 되도록 근사 정규화한다.
    """
    k = float(knee_hz)
    co = lowpass_coeffs(k, 0.707, fs_frame)
    y, zf = tv_biquad(white_frames, *co, zi=zi)
    gain = math.sqrt(fs_frame / (2.0 * math.pi * k))          # 등가 잡음 대역폭 보정
    return y * gain, zf


class FricationNoise(nn.Module):
    """협착 공기역학 -> 마찰 소스 (샘플률). 색은 여기서 **소스 스펙트럼**까지만."""

    def __init__(self, fs: float, hop: int, mod_depth: float = 0.25,
                 amp_ref: float = 0.062, lp_ratio: float = 1.4,
                 source_hf_shelf_db: float = -9.0):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.mod_depth = mod_depth
        # 제트 난류 스펙트럼의 고역 절벽: Strouhal 정점의 lp_ratio 배 위에서 2차 저역통과.
        # 다이폴(+6)·방사(+6) 를 합쳐도 그 위에서 −6 dB/oct 로 떨어지게 한다
        # (실측 /s/ 는 13~14 kHz 위에서 절벽처럼 떨어진다 — v1 HANDOFF §6.10).
        self.log_lp_ratio = nn.Parameter(torch.tensor(math.log(lp_ratio)))
        # 실측 적합 (남성 ㅅ ×3, 0.6~20 kHz 7 대역, rms 3.2 dB): 셸프 −9 dB, lp_ratio 1.4,
        # back_leak 0.10, 앞공동 대역폭 500+0.12f, 장애물 다이폴 0 (ADR 0011).
        self.source_hf_shelf_db = source_hf_shelf_db
        # 학습 파라미터: 변조 스펙트럼 기울기/꺾임, 소스 세기 보정
        self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_knee = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_amp = nn.Parameter(torch.tensor(math.log(amp_ref)))

    def forward(self, c: dict, ag_dc: torch.Tensor, glottal_phase: torch.Tensor,
                voiced: torch.Tensor, noise=None, frame0: int = 0,
                state: dict | None = None, emit: int | None = None) -> dict:
        """c: 프레임률 (B,T) dict (p_sub, a_c, obstacle, fric_gain). ag_dc: **프레임률** (B,T) 성문 면적.

        noise: NoiseBank (위치 기반), frame0: 청크의 첫 프레임, state: 필터 상태(스트리밍).
        반환: source (B,N) 마찰 소스 파형, env (B,N), f_peak (B,N), reynolds, flow, state
        """
        fs, hop = self.fs, self.hop
        b, t_all = c["p_sub"].shape
        t = t_all if emit is None else int(emit)
        n = t * hop
        noise = noise or NoiseBank()
        state = {} if state is None else state
        up = lambda v: frames_to_samples(v.unsqueeze(-1), hop)[..., 0][:, :n]
        a_c = up(c["a_c"]); ps = up(c["p_sub"])
        agf = ag_dc                                      # 프레임률 (B, T_all)
        ag_s = up(agf)
        # 연구개가 열리면 기류는 코로 빠진다. 구강 협착을 지나는 유량과 구강압 축적은 (1−velum) 배.
        # (비음의 구강 폐쇄 a_c≈0.02 에서 Re=3700 짜리 마찰 + 방전 버스트가 나던 것 — 측정)
        oral = (1.0 - c["velum"]).clamp(0.0, 1.0)
        u, _ = series_flow(ps, ag_s, a_c)
        u = u * up(oral)
        # 구강압 저장 → 해제 시 방전 (파열음·파찰음의 버스트). 프레임률 1 차 계.
        #   닫힘(a_c < ~0.05): Po → Ps (τ 15 ms). 열림: Po → 정상값 (τ 6 ms).
        #   버스트 유량 = 저장된 초과압의 방전. ㅈ/ㅊ 개시가 10 ms 안에 −70→−30 dB 로 서는 것(계측).
        closed = torch.clamp((0.06 - c["a_c"]) / 0.04, 0.0, 1.0)
        closed = closed * closed * (3 - 2 * closed)
        po_ss = c["p_sub"] * agf ** 2 / (agf ** 2 + c["a_c"].clamp_min(1e-3) ** 2)   # 정상 구강압
        dt = hop / fs
        po = state.get("po", po_ss[:, 0] * 0.0)
        burst, po_hist = [], []
        for i in range(t_all):
            tgt = (closed[:, i] * c["p_sub"][:, i] + (1 - closed[:, i]) * po_ss[:, i]) * oral[:, i]
            tau = 0.015 * closed[:, i] + 0.006 * (1 - closed[:, i])
            po = po + dt * (tgt - po) / tau
            po_hist.append(po)
            burst.append((po - po_ss[:, i]).clamp_min(0.0) * (1 - closed[:, i]))
        burst = torch.stack(burst, 1)
        state["po"] = po_hist[t - 1]                     # 상태는 내보내는 마지막 프레임 기준
        burst_env = up(torch.sqrt(burst.clamp_min(0.0) / CMH2O)) * 3.0     # 초과압 → 속도 배율
        re, v, d = reynolds(u, a_c)
        # Stevens: 소스 압력 ∝ ρ·v³·√A. Re = v·d/ν ∝ v·√A 이므로 같은 Re 에서 v³√A ∝ Re³/A —
        # 넓은 통로(모음 자세)는 같은 Re 라도 조용하다. (Re²−Re_c²)^1.5/(A/A_ref), A_ref=0.1 cm².
        drive = (((re ** 2 - RE_CRIT ** 2).clamp_min(0.0) / RE_REF ** 2) ** 1.5
                 * (0.1 / a_c.clamp_min(0.02)))
        drive = drive + burst_env ** 3 * (0.1 / a_c.clamp_min(0.02))   # 버스트: 방전 속도의 세제곱
        env = drive * torch.exp(self.log_amp) * up(c["fric_gain"])
        # 장애물(앞니) 소스의 혹은 자유 제트의 Strouhal 정점보다 높고 넓다(Shadle). 같은 화자 A/B 에서
        # St=0.2 그대로 두면 1~3 kHz 가 10~12 dB 과했다. 정점 ×2.5, Q 0.5.
        f_peak = (0.5 * v / d.clamp_min(1e-3)).clamp(800.0, 0.45 * fs)
        # 느린 1/f^β 변조 (제트 사행) + 유성 구간에서는 성문 개방기 AM
        mod, zmod = slow_modulation(
            noise.white("mod", frame0, t_all, b, ps.dtype, ps.device), fs / hop,
            torch.exp(self.log_knee), state.get("mod"))
        if t != t_all:                       # 상태는 내보내는 마지막 프레임 기준
            _, zmod = slow_modulation(noise.white("mod", frame0, t, b, ps.dtype, ps.device),
                                      fs / hop, torch.exp(self.log_knee), state.get("mod"))
        state["mod"] = zmod
        mod = 1.0 + self.mod_depth * up(mod)
        frac = glottal_phase[:, :n] / (2 * math.pi)
        am = 1.0 - 0.5 * voiced[:, :n] * (frac < 0.35).float()   # 폐쇄기에 유량 감소
        white = noise.white("fric", frame0 * hop, n, b, ps.dtype, ps.device)
        # 난류 소스의 스펙트럼.
        #
        # 초판은 Strouhal 정점에 2 차 대역통과(Q 0.5)를 걸었다. 그러면 정점 아래가 −6 dB/oct
        # 로 떨어져 1~2 kHz 가 실측보다 15 dB 낮았다(같은 화자 A/B). 실측 /s/ 는 정점 아래가
        # **완만한 어깨**다: 5~8 kHz 정점 대비 1~2 kHz 가 −17 dB, 2~3 kHz 가 −22 dB.
        #
        # 물리적으로도 그쪽이 맞다. 마찰음의 스펙트럼 정점을 만드는 것은 소스가 아니라
        # **앞공동 공진**이고(Stevens: 소스는 광대역), 소스 자신은 에디 크기가 정하는
        # 모서리 위에서만 떨어진다. 그래서 여기서는
        #   백색 → 기울기(tilt, 학습) → 정점의 lp_ratio 배에서 2 차 저역통과
        # 로 두고, 봉우리는 tract 의 앞공동 극이 만들게 한다.
        # 소스의 고역 셸프 (1 kHz 모서리). 실측 적합값 −9 dB: 광대역 백색보다 조금 어둡다.
        shelf = self.source_hf_shelf_db
        src = white
        if abs(float(shelf)) > 1e-6:
            a = math.exp(-2 * math.pi * 1000.0 / fs)
            g_hi = 10.0 ** (shelf / 20.0)
            hp, state["tl"] = tv_biquad(src, 0.5 * (1 + a), -0.5 * (1 + a), 0.0, -a, 0.0,
                                        zi=state.get("tl"))
            src = src + (g_hi - 1.0) * hp
        # 장애물(앞니) 다이폴: +6 dB/oct 성분을 obstacle 비율로 섞는다 (Curle)
        obst = up(c["obstacle"])
        dip, state["dip"] = tv_biquad(src, *first_difference_coeffs(1.0, src), zi=state.get("dip"))
        src = (1.0 - obst) * src + obst * dip * 4.0
        return dict(source=src * env * mod * am, env=env, f_peak=f_peak,
                    reynolds=re, flow=u, state=state)


class AspirationNoise(nn.Module):
    """성문 기식: 포락선은 glottis 가 물리로 준다. 여기서는 색(완만한 고역 셸프)만."""

    def __init__(self, fs: float, hop: int, corner_hz: float = 3000.0, floor: float = 0.3,
                 amp_ref: float = 0.3):      # 실측 남성 /아/ HNR 22 dB, 4~8 kHz 대역에 적합 (2026-09-06)
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.corner, self.floor = corner_hz, floor
        self.log_amp = nn.Parameter(torch.tensor(math.log(amp_ref)))

    def forward(self, asp_env: torch.Tensor, noise=None, sample0: int = 0,
                state: dict | None = None) -> dict:
        b, n = asp_env.shape
        noise = noise or NoiseBank()
        state = {} if state is None else state
        white = noise.white("asp", sample0, n, b, asp_env.dtype, asp_env.device)
        # 1 차 고역 셸프: floor + (1−floor)·HP(corner)
        r = math.exp(-2 * math.pi * self.corner / self.fs)
        hp, state["hp"] = tv_biquad(white, 0.5 * (1 + r), -0.5 * (1 + r), 0.0, -r, 0.0,
                                    zi=state.get("hp"))
        colored = self.floor * white + (1.0 - self.floor) * hp
        return dict(source=colored * asp_env * torch.exp(self.log_amp), state=state)


# ------------------------------------------------------------ 과도음 템플릿
def _damped_burst(fs, f_hz, tau_ms, ms, noise=0.0, seed=0):
    n = int(fs * ms / 1000.0)
    t = np.arange(n) / fs
    g = np.random.default_rng(seed)
    x = np.exp(-t / (tau_ms / 1000.0)) * (np.sin(2 * np.pi * f_hz * t)
                                         + noise * g.standard_normal(n))
    return x / (np.abs(x).max() + 1e-12)


def default_templates(fs: float) -> dict[str, np.ndarray]:
    """물리적으로 그럴듯한 초기 템플릿. 학습/녹음으로 교체·추가한다."""
    g = np.random.default_rng(1)
    saliva = np.zeros(int(fs * 0.06))
    for k in g.integers(0, len(saliva) - 200, 7):
        burst = _damped_burst(fs, g.uniform(3500, 7000), 0.4, 3, 0.5, int(k))
        saliva[k:k + len(burst)] += burst * g.uniform(0.3, 1.0)
    return {
        "tongue_contact": _damped_burst(fs, 4200.0, 0.6, 4, 0.8, 2),   # 혀끝-구개 접촉 클릭
        "tongue_release": _damped_burst(fs, 2800.0, 1.2, 8, 1.5, 3),   # 접촉 해제(짧은 마찰 버스트)
        "tongue_floor":   _damped_burst(fs, 1600.0, 1.5, 8, 1.0, 4),   # 튕긴 혀가 입 바닥에 닿음
        "lip_smack":      _damped_burst(fs, 900.0, 2.5, 12, 0.6, 5),
        "saliva":         saliva / (np.abs(saliva).max() + 1e-12),
    }


class TransientTemplateBank(nn.Module):
    """템플릿 뱅크 + 변조. 이벤트 {t, id|name, amp, rate, tilt} -> 파형 (B,N)."""

    def __init__(self, fs: float, templates: dict[str, np.ndarray] | None = None,
                 max_len_ms: float = 80.0):
        super().__init__()
        self.fs = float(fs)
        tpl = templates or default_templates(fs)
        L = int(fs * max_len_ms / 1000.0)
        self.names = list(tpl.keys())
        bank = torch.zeros(len(tpl), L)
        for i, k in enumerate(self.names):
            v = torch.as_tensor(tpl[k], dtype=torch.float32)[:L]
            bank[i, :len(v)] = v
        self.bank = nn.Parameter(bank)                 # 학습 가능
        self.index = {k: i for i, k in enumerate(self.names)}

    def add(self, name: str, wave: np.ndarray) -> int:
        """새 템플릿(예: 녹음에서 오려낸 것). 학습 중 동적으로 늘어날 수 있다."""
        L = self.bank.shape[1]
        v = torch.zeros(L); w = torch.as_tensor(wave, dtype=torch.float32)[:L]
        v[:len(w)] = w
        with torch.no_grad():
            self.bank = nn.Parameter(torch.cat([self.bank.data, v.unsqueeze(0)], 0))
        self.names.append(name); self.index[name] = len(self.names) - 1
        return self.index[name]

    def render(self, events: list[dict], n: int, b: int = 1, noise=None,
               device=None, sample0: int = 0) -> torch.Tensor:
        """events: [{t: 초(청크 기준), name|id, amp(1.0), rate(1.0), tilt(0.0)}]. 반환 (B,N).

        변조 난수는 이벤트의 **절대 샘플 위치**로 시드되므로 청크 경계와 무관하다.
        """
        out = torch.zeros(b, n, device=device, dtype=self.bank.dtype)
        noise = noise or NoiseBank()
        for ev in events:
            i = self.index[ev["name"]] if "name" in ev else int(ev.get("id", 0))
            i = min(max(i, 0), self.bank.shape[0] - 1)
            tpl = self.bank[i]
            s_abs = sample0 + int(round(float(ev["t"]) * self.fs))
            # 변조: 속도(피치) 재표본화 + 미세 난수(같은 소리가 두 번 나지 않게)
            rate = float(ev.get("rate", 1.0)) * (1.0 + 0.05 * noise.scalar("transient", 2 * s_abs))
            amp = float(ev.get("amp", 1.0)) * 10 ** (0.1 * noise.scalar("transient", 2 * s_abs + 1) / 20)
            L = tpl.shape[0]
            m = max(8, int(L / rate))
            src = torch.arange(m, device=tpl.device, dtype=tpl.dtype) * rate
            i0 = src.floor().long().clamp(0, L - 2)
            w = src - i0
            y = tpl[i0] * (1 - w) + tpl[i0 + 1] * w
            tilt = float(ev.get("tilt", 0.0))
            if tilt != 0.0:                            # +: 밝게(미분), −: 어둡게(적분 근사)
                d = torch.cat([y[:1], y[1:] - y[:-1]])
                y = y + tilt * d * (4.0 if tilt > 0 else 0.5)
            s = int(round(float(ev["t"]) * self.fs))
            if s >= n or s + m <= 0:
                continue
            a, bnd = max(s, 0), min(n, s + m)
            out[:, a:bnd] = out[:, a:bnd] + amp * y[a - s: bnd - s]
        return out
