"""압력 구동 성문 소스 — "성대와 주변부의 압력을 통하여 원음을 생성".

입력은 근육·호흡 좌표다: 성문하압 `p_sub`, 내전 `adduction`, 긴장 `tension`.
F0·진폭·Rd(음질)·기식은 **결과**다. 스크립트는 F0 를 직접 줄 수도 있다(`f0_target`).

물리 (Titze 1988 소진폭 이론의 요약):

    발성 역치압  Pth = (1.5 + 0.7·(f0/f0n)²) · (0.6 + 2.4·(1−add)²)      [cmH2O]
    진폭 목표    A*  = sqrt((Ps − Pth)⁺ / Pth) · gate(add)                 (Ps ≫ Pth 에서 포화)
    진폭 기동    dA/dt = σ·A·(1 − A/A*),  σ = k·f0·(Ps−Pth)/Pth            (로지스틱: 볼록 S 자)
    소멸         dA/dt = −(f0/3)·A                                          (역치 아래: 서너 주기 안에 죽음)
    F0           f0 = f0_base(tension)·(1 + 0.04·(Ps−Pth)⁺)·f0_scale·(1+J·z)
    Rd           Rd = 0.3 + 2.4·(1−add)^1.5 + rd_offset                    (압착 0.3 ~ 기식 2.7)
    기식 포락선  asp = (1−add)²·sqrt(Ps)·(1 + d·open_phase)                  (Klatt & Klatt 1990 AM)

파형은 LF 모델(Fant 1995, Rd 한 개)의 **하모닉 가산합성**이다. 위상은 샘플 단위로
누적되고, 나이퀴스트 근방은 소프트 마스크로 에일리어싱을 막는다. 하모닉 상대위상
(RPS, 화자 고유량)은 올패스 체인의 위상을 k·f0 에서 평가해 오프셋으로 넣는다.

v1 에서 검증된 사실을 그대로 가져왔다(docs/HANDOFF.md §6.0): 1 차 지연으로는
발성 개시 모양이 반대다(오목). 로지스틱이어야 한다(볼록 S). 그리고 씨앗이 남아
있지 않으면 되살아나지 않으므로 역치 아래에서는 씨앗까지 내려간다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .control import frames_to_samples
from .rng import NoiseBank
from .tviir import tv_biquad

RHO = 1.14e-3            # 공기 밀도 g/cm³
CMH2O = 980.665          # 1 cmH2O = 980.665 dyn/cm²


# ------------------------------------------------------------------ LF 모델
#: 협착부(혀끝-치경 간극)의 길이 [cm]. /s/ 의 혀끝 협착은 1 cm 안팎이다.
CONSTRICTION_LEN = 1.0
#: 협착 앞쪽 공동(앞니까지)의 길이 [cm]. /s/ 는 1.5 cm 안팎 — 치찰음 앞공동 극과 같은 기하.
FRONT_CAVITY_LEN = 1.5
#: 협착이 없는 중립 성도의 단면적 [cm²]. 협착의 '좁음' 을 재는 기준이다.
NEUTRAL_TRACT_AREA = 3.0


def lf_pulse(rd: float, n: int = 4096) -> np.ndarray:
    """LF 유량미분 E(t) 한 주기 (Ee = 1 로 정규화). Fant(1995) 의 Rd 파라미터화."""
    rd = float(np.clip(rd, 0.3, 2.7))
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    rg = (rk / 4.0) * (0.5 + 1.2 * rk) / (0.11 * rd - ra * (0.5 + 1.2 * rk))
    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    ta = ra
    tc = 1.0
    # (i) ε·ta = 1 − exp(−ε(tc−te))  → 고정점
    eps = 1.0 / ta
    for _ in range(100):
        eps = (1.0 - math.exp(-eps * (tc - te))) / ta
    # (ii) ∫E = 0 → α 이분법
    wg = math.pi / tp
    t = (np.arange(n) + 0.5) / n

    def wave(alpha):
        e = np.zeros(n)
        m = t <= te
        e[m] = -np.exp(alpha * (t[m] - te)) * np.sin(wg * t[m]) / math.sin(wg * te)
        r = ~m
        e[r] = -(1.0 / (eps * ta)) * (np.exp(-eps * (t[r] - te)) - math.exp(-eps * (tc - te)))
        return e

    # α 가 클수록 개방기 후반(음의 부분)이 커져 ∫E 가 줄어든다 → ∫E > 0 이면 α 를 키운다.
    lo, hi = -50.0, 400.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if wave(mid).sum() > 0:
            lo = mid
        else:
            hi = mid
    e = wave(0.5 * (lo + hi))
    return e / (np.abs(e).max() + 1e-12)


# **성문 개방기에 동기한 기식 AM 의 깊이** (Klatt & Klatt 1990).
#
# `asp_env = asp·(1 + d·voiced·(open_phase − 0.5))` 이므로 d=0.7 이면 기식 잡음이 매
# 성문 주기마다 ±35 % 변조된다. 그 변조는 F0 와 그 배음에 그대로 실린다.
#
# 실측 (yang_00000101, 고역 4~12 kHz 포락의 변조 스펙트럼 **절대** 에너지, 목표 대비):
# F0~2.5F0 대역이 2.37 배, 60~150 Hz 가 4.01 배 과다하다. 사람은 그 대역의 진폭 변조를
# 거칠기(roughness)로 듣는다 — 사용자가 "지지직거린다" 고 한 성분이다.
#
# 값은 실측으로 정해야 하므로 상수로 빼 둔다. 하드코딩된 채로는 A/B 를 못 돌린다.
ASP_AM_DEPTH = 0.7


def lf_table(n_rd: int = 24, rd_min: float = 0.3, rd_max: float = 2.7,
             n_harm: int = 400, n: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Rd 격자 -> 하모닉 복소계수 (n_rd, n_harm). 격자 사이는 선형보간(미분가능)."""
    rds = torch.linspace(rd_min, rd_max, n_rd)
    coef = np.stack([np.fft.rfft(lf_pulse(float(r), n))[1:n_harm + 1] / n
                     for r in rds])
    return rds, torch.tensor(coef, dtype=torch.complex64)


# ------------------------------------------------------------------ 소스
class GlottalSource(nn.Module):
    """근육·압력 좌표 -> (성문 유량미분, 순시위상, 기식 포락선, 성문 면적)."""

    def __init__(self, fs: float, hop: int, speaker: str = "female",
                 n_rd: int = 24, n_harm: int | None = None, k_growth: float = 0.25,
                 cycles_decay: float = 3.0, f0_min: float = 50.0,
                 f0_range: tuple[float, float, float] | None = None,
                 noise_modulation: str = "legacy"):
        super().__init__()
        if noise_modulation not in ("legacy", "lf"):
            raise ValueError("noise_modulation must be 'legacy' or 'lf'")
        self.noise_modulation = noise_modulation
        self.fs, self.hop = float(fs), int(hop)
        self.k_growth, self.cycles_decay = k_growth, cycles_decay
        if f0_range is not None:
            self.f0_lo, self.f0_hi, self.f0_nom = map(float, f0_range)
        elif speaker == "female":
            self.f0_lo, self.f0_hi, self.f0_nom = 110.0, 440.0, 220.0
        else:
            self.f0_lo, self.f0_hi, self.f0_nom = 65.0, 260.0, 120.0
        n_harm = n_harm or int(fs / 2 / f0_min) + 1
        rds, coef = lf_table(n_rd, n_harm=n_harm)
        self.register_buffer("rd_grid", rds)
        self.register_buffer("lf_coef", coef)                      # (n_rd, K)
        self.register_buffer("k_idx", torch.arange(1, n_harm + 1, dtype=torch.float32))
        if noise_modulation == "lf":
            # Integrate the SAME LF derivative used for the harmonic source, from
            # opening (phase zero) to closure. No independent open-quotient fit.
            flows = []
            for r in rds:
                e = lf_pulse(float(r))
                flow = np.concatenate(([0.0], np.cumsum(e)[:-1]))
                flow /= flow.max()
                flows.append(flow - flow.mean())
            # Derived table, not checkpoint state: legacy checkpoints still load.
            self.register_buffer("lf_flow", torch.tensor(np.stack(flows), dtype=torch.float32),
                                 persistent=False)

    # ---------------------------------------------------------- 생리 상태
    def threshold(self, f0, adduction):
        return (1.5 + 0.7 * (f0 / self.f0_nom) ** 2) * (0.6 + 2.4 * (1.0 - adduction) ** 2)

    def f0_base(self, tension):
        return self.f0_lo * (self.f0_hi / self.f0_lo) ** tension.clamp(0.0, 1.0)

    def physiology(self, c: dict, amp0: torch.Tensor | None = None) -> dict:
        """프레임률 제어 (B,T) dict -> 프레임률 상태 dict. 로지스틱 기동은 프레임 루프.

        amp0: 이전 청크 끝의 (씨앗 포함) 진폭. 없으면 씨앗에서 시작."""
        ps, add, ten = c["p_sub"], c["adduction"].clamp(0, 1), c["tension"]
        direct = c["f0_target"] > 0
        f0 = torch.where(direct, c["f0_target"], self.f0_base(ten))
        pth = self.threshold(f0, add)
        over = (ps - pth).clamp_min(0.0)
        # 폐압-F0 결합은 **긴장으로 F0 를 정할 때만** 건다. `f0_target` 은 "이 주파수로
        # 울려라" 는 직접 지정이므로 그 위에 다시 곱하면 지정한 값이 안 나온다. 실제로
        # p_sub 6.7 · Pth 2.2 에서 ×1.18 이 걸려 238 Hz 지정이 281.6 Hz 로 났고(측정),
        # 복사합성에서 성문 펄스가 주기마다 0.18 주기씩 밀렸다.
        f0 = torch.where(direct, f0, f0 * (1.0 + 0.04 * over)) * c["f0_scale"]
        gate = torch.clamp((add - 0.10) / 0.15, 0.0, 1.0)
        gate = gate * gate * (3 - 2 * gate)                        # 벌린 성문(add≤0.10)은 안 떤다
        a_star = torch.sqrt(over / pth + 1e-12) * gate   # +eps: sqrt(0) 의 기울기가 무한대다
        a_star = a_star / (1.0 + 0.5 * a_star)                     # 포화
        dt = self.hop / self.fs
        seed = 0.02
        amp = []
        a = torch.full_like(ps[:, 0], seed) if amp0 is None else amp0.clone()
        for i in range(ps.shape[1]):
            tgt = a_star[:, i]
            grow = self.k_growth * f0[:, i] * over[:, i] / pth[:, i]
            up = a + dt * grow * a * (1.0 - a / tgt.clamp_min(seed))
            down = a * torch.exp(-dt * f0[:, i] / self.cycles_decay)
            a = torch.where(tgt > seed, up, down.clamp_min(seed))
            amp.append(a)
        amp_raw = torch.stack(amp, 1)
        amp = (amp_raw - seed).clamp_min(0.0) / (1.0 - seed)
        rd = (0.3 + 2.4 * (1.0 - add) ** 1.5 + c["rd_offset"]).clamp(0.3, 2.7)
        ag_dc = 0.02 + 0.5 * (1.0 - add) ** 2.5                    # 정적 성문 면적 cm² (모달 ≈0.07 → U≈250 cm³/s)
        # 성문 난류는 **성문 양단의 압력 강하**로 난다. 구강 협착이 있으면 압력의 대부분이
        # 협착에서 떨어지고(Po/Ps = Ag²/(Ag²+Ac²), v1 §5.3) 성문 제트는 느려진다 — /s/ 동안
        # 성문 기식이 1~6 kHz 를 채우던 원인(측정: 앞공동 경로와 같은 크기).
        a_c = c["a_c"].clamp_min(1e-3)
        frac = a_c ** 2 / (a_c ** 2 + ag_dc ** 2)                  # ΔPg/Ps
        # **그리고 난 잡음이 협착을 지나 나와야 한다.** 위 `frac` 은 성문에서 난류가
        # 얼마나 **생기는가** 이고, 이건 그게 얼마나 **나오는가** 다 — 서로 다른 물리라
        # 둘 다 곱한다 (v1 `aeroacoustic.constriction_transmission`).
        #
        # 저역에서(파장 ≫ 길이) 협착과 앞공동은 둘 다 관성 임피던스 Z ∝ L/A 이므로,
        # 협착이 없을 때(전부 A0) 대비 전달비는 임피던스 분배로
        #
        #     T = (Lc + Lf)·Ac / (Lc·A0 + Lf·Ac)
        #
        # Ac → A0 이면 정확히 1, Ac = 0.10 cm² 이면 0.079 (−22 dB). 단순 면적비
        # Ac/A0 를 쓰면 협착이 풀리는 도중을 과소평가한다 (해제 중반 0.23 vs 0.42).
        #
        # **이게 없으면 성문 잡음이 성도가 열려 있는 것처럼 방사된다.** 협착 뒤에
        # 갇혀 있어야 할 뒤공동 공진이 /s/ 스펙트럼에 서고, 실제 /s/ 에는 거의 없는
        # 1~2 kHz 가 마찰음을 어둡게 만든다 (v1 실측: 봉우리 대비 −1.8 dB, 에너지의
        # 9.8 %). v2 는 이 항이 없어서 적합기가 그걸 **`aspiration` 을 전 구간
        # 0.031 로 내려** 막고 있었다 — 협착 구간만 끌 수단이 없으니 모음의 기식까지
        # 같이 죽인 것이다 (docs/MEASUREMENTS.md §18.1 의 "갚아야 할 빚").
        #
        # 협착이 풀리면 1 로 가므로 전이에서 기식이 제 세기를 되찾는다. 발성이 아니라
        # **혀**가 여는 것이라 전이가 비지 않는다(성문파열음이 안 된다).
        trans = ((CONSTRICTION_LEN + FRONT_CAVITY_LEN) * a_c
                 / (CONSTRICTION_LEN * NEUTRAL_TRACT_AREA
                    + FRONT_CAVITY_LEN * a_c).clamp_min(1e-9)).clamp(0.0, 1.0)
        # 세기는 ΔPg 에 선형 (√ 로 두면 /s/ 중 기식이 실측보다 10 dB 크다 — 같은 화자 A/B).
        asp = ((1.0 - add) ** 2 * frac * trans
               * torch.sqrt(ps.clamp_min(0.0) + 1e-12) * c["aspiration"])
        return dict(f0=f0, amp=amp, amp_raw=amp_raw, rd=rd, ag_dc=ag_dc, asp=asp, pth=pth)

    # ---------------------------------------------------------- 파형
    def _lf_index(self, rd: torch.Tensor):
        """(B,N) Rd -> (i0, w). 하모닉 계수는 **차수별로** 표에서 뽑는다.

        전에는 (B,N,K) 복소 텐서를 통째로 만들고 루프에서 `[..., j]` 로 잘랐다. 그러면
        역전파가 차수마다 (B,N,K) 짜리 0 텐서를 만들어 채운다 — K=234, N=19200 이면 한 번에
        37 MB, 5937 번이면 역전파의 68 % 였다(프로파일). 표를 차수별로 인덱싱하면 (B,N) 이다.
        """
        g = self.rd_grid
        pos = (rd.clamp(g[0], g[-1]) - g[0]) / (g[-1] - g[0]) * (len(g) - 1)
        i0 = pos.floor().long().clamp(0, len(g) - 2)
        return i0, pos - i0.to(pos.dtype)

    def lf_noise_envelope(self, phase: torch.Tensor, rd: torch.Tensor,
                          voiced: torch.Tensor) -> torch.Tensor:
        """Unit-cycle-mean AM from normalized LF flow, with no unvoiced AM.

        Periodic Catmull–Rom interpolation is C1 across phase/table boundaries.
        Each Rd row is centered over a full cycle, not the current chunk; hence
        normalization needs neither future samples nor extra streaming state.
        The fixed 0.7 depth reuses aspiration's existing modulation strength.
        """
        i, wrd = self._lf_index(rd)
        size = self.lf_flow.shape[1]
        pos = torch.remainder(phase / (2 * math.pi), 1.0) * size
        j = pos.floor().long()
        w = pos - j.to(pos.dtype)
        points = []
        for offset in (-1, 0, 1, 2):
            col = (j + offset) % size
            points.append(self.lf_flow[i, col] * (1 - wrd)
                          + self.lf_flow[i + 1, col] * wrd)
        p0, p1, p2, p3 = points
        flow = p1 + 0.5 * w * (p2 - p0 + w * (
            2 * p0 - 5 * p1 + 4 * p2 - p3 + w * (3 * (p1 - p2) + p3 - p0)))
        return 1.0 + 0.7 * voiced * flow

    def forward(self, c: dict, phase0: torch.Tensor | None = None,
                rps: torch.Tensor | None = None, dispersion: float = 0.0,
                noise=None, frame0: int = 0,
                amp0: torch.Tensor | None = None, state: dict | None = None,
                emit: int | None = None) -> dict:
        """c: 프레임률 (B,T) dict. 반환 샘플률 텐서들 (B,N).

        noise : NoiseBank (위치 기반 난수). frame0 : 이 청크의 첫 프레임 번호.
        amp0  : 이전 청크 끝의 기동 진폭(스트리밍). state : 지터/시머 저역통과 상태.
        emit  : 내보낼 프레임 수(기본 전부). 스트리밍에서는 T+1 프레임을 주고 T 만 내보내
                마지막 프레임의 샘플 보간이 다음 프레임을 보게 한다(선행 1 프레임).
        """
        st = self.physiology(c, amp0=amp0)
        hop, fs = self.hop, self.fs
        b, t_all = st["f0"].shape
        t = t_all if emit is None else int(emit)
        n = t * hop
        up = lambda v: frames_to_samples(v.unsqueeze(-1), hop)[..., 0][:, :n]
        f0, amp, rd = up(st["f0"]), up(st["amp"]), up(st["rd"])
        tilt, jit, shm = up(c["tilt"]), up(c["jitter"]), up(c["shimmer"])
        # 지터/시머: 프레임률 백색 난수(위치 기반) -> 1 극 저역통과(상태 유지) -> 샘플률
        state = {} if state is None else state
        noise = noise or NoiseBank()
        zj = noise.white("jitter", frame0, t_all, b, f0.dtype, f0.device)
        zs_ = noise.white("shimmer", frame0, t_all, b, f0.dtype, f0.device)
        a = 0.6                                                     # ~ 50 Hz 모서리 @1 kHz 프레임률
        g = math.sqrt((1 - a) / (1 + a)) * 2.0
        zj, zj_state = tv_biquad(zj, g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("j"))
        zs_, zs_state = tv_biquad(zs_, g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("s"))
        if t != t_all:                       # 상태는 내보내는 마지막 프레임(t) 기준이어야 한다
            _, zj_state = tv_biquad(noise.white("jitter", frame0, t, b, f0.dtype, f0.device),
                                    g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("j"))
            _, zs_state = tv_biquad(noise.white("shimmer", frame0, t, b, f0.dtype, f0.device),
                                    g * (1 - a), 0.0, 0.0, -a, 0.0, zi=state.get("s"))
        state["j"], state["s"] = zj_state, zs_state
        zs = frames_to_samples(torch.stack([zj, zs_], -1), hop)[:, :n]
        f0 = f0 * (1.0 + jit * zs[..., 0])
        amp = amp * (1.0 + shm * zs[..., 1]).clamp_min(0.0)
        # 위상 누적은 float64 로 (float32 cumsum 오차 × 하모닉 차수가 청크 경계에서 보인다)
        phase64 = torch.cumsum(2 * math.pi * f0.double() / fs, dim=-1)
        if phase0 is not None:
            phase64 = phase64 + phase0.double()
        phase64 = torch.remainder(phase64, 2 * math.pi)
        phase = phase64.to(f0.dtype)                  # 상태(phase_last)는 float64 로 넘긴다:
        #                                               float32 위상 2e-7 rad × 하모닉 100 = 2e-5 rad 가 청크 경계에서 보였다
        # 하모닉 가산합성. **하모닉마다 순차 누적**한다 — (B,N,K).sum(-1) 은 텐서 크기에 따라
        # 축약 순서가 달라 float32 반올림이 청크 의존이 되고(3e-6), 그것이 고 Q 성도를 지나며
        # 스트리밍/오프라인 차이 1e-3 로 커졌다(측정). 순차 누적은 청크와 무관하고 메모리도 작다.
        i0, wrd = self._lf_index(rd)                               # (B,N), (B,N)
        k = self.k_idx
        f_nyq, width = 0.95 * fs / 2, 0.02 * fs / 2
        f_cut = f_nyq + 6.0 * width                    # 이 위는 마스크를 정확히 0 으로 (청크 무관 상한)
        log2f0 = torch.log2(f0.clamp_min(1.0) / 1000.0)
        du = torch.zeros_like(phase)
        # NaN 안전. f0 에 NaN 이 들어오면 ceil(NaN) 이 그대로 통과해 int(NaN) 에서
        # **예외로 터진다** — 적합기의 "손실이 비유한이면 중단" 가드가 손실을 보기도 전이라
        # 원인을 못 찾는다. 하모닉 상한만 정하는 값이므로 NaN 은 상한으로 접고, 잘못된
        # 값은 아래 du 계산에서 NaN 으로 **전파**시켜 가드가 잡게 둔다.
        f0_min = torch.nan_to_num(f0.detach(), nan=1.0, posinf=1.0).min().clamp_min(1.0)
        k_max = int(torch.clamp(torch.ceil(f_cut / f0_min), 1, len(k)).item())
        for j in range(k_max):
            kk = k[j]
            fk = f0 * kk
            live = fk <= f_cut
            if not bool(live.any()):
                continue
            cj = self.lf_coef[i0, j] * (1 - wrd) + self.lf_coef[i0 + 1, j] * wrd   # (B,N)
            mask = torch.sigmoid((f_nyq - fk) / width) * live
            gain = 10.0 ** (tilt * (log2f0 + math.log2(float(kk))) / 20.0)
            ph = phase * kk + torch.angle(cj)
            if rps is not None:
                ph = ph + rps[..., j]
            if dispersion != 0.0:
                # **하모닉 위상 분산.** 고역일수록 지연되는 이차 위상 −D·(f/f_nyq)²
                # (= 주파수에 선형인 그룹 지연 = 시간축으로 퍼지는 chirp). 실제 성대는
                # 딱딱하지 않아 점막파로 상하연이 시간차를 두고 닫히므로 이런 분산이
                # 생긴다. `rps` 와 달리 **샘플률 fk 를 그대로 써서** (B,N,K) 텐서를
                # 만들지 않는다 — K=240 이면 그것만 65 MB 다.
                xn = (fk / (0.5 * fs)).clamp(0.0, 1.0)
                ph = ph - dispersion * xn * xn
            du = du + 2.0 * cj.abs() * mask * gain * torch.cos(ph)
        du = du * amp
        # 성문 개방기 (LF: 0 ~ te 가 열림) -> 기식 AM 마스크
        frac = phase / (2 * math.pi)
        open_phase = torch.sin(math.pi * frac.clamp(0, 1) / 0.65).clamp_min(0.0) ** 2
        open_phase = torch.where(frac < 0.65, open_phase, torch.zeros_like(open_phase))
        asp = up(st["asp"])
        voiced = (amp > 1e-3).float()
        noise_am = None
        if self.noise_modulation == "lf":
            noise_am = self.lf_noise_envelope(phase, rd, voiced)
            # Legacy open mask mean = 0.65/2, so keep its mean source level
            # while replacing only the periodic shape (not an RMS guarantee).
            asp_env = asp * (1.0 + ASP_AM_DEPTH * voiced * (0.325 - 0.5)) * noise_am
        else:
            asp_env = asp * (1.0 + ASP_AM_DEPTH * voiced * (open_phase - 0.5))
        return dict(du=du, phase=phase, asp_env=asp_env.clamp_min(0.0),
                    noise_am=noise_am,
                    amp=amp, f0=f0, ag_dc=up(st["ag_dc"]), ag_dc_frames=st["ag_dc"], voiced=voiced,
                    physiology=st, amp_last=st["amp_raw"][:, t - 1], state=state,
                    phase_last=phase64[:, -1:])
