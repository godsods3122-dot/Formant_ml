"""성대 자가진동 물리모델 (2-mass / body-cover 계열).

Steinecke & Herzel (1995) 의 비대칭 2질량 모델을 기반으로 한다. 이 모델은
소스를 '주기파형 사전'이 아니라 진짜 미분방정식의 극한주기해로 만들기 때문에
다음이 자동으로 따라온다.

* 진동 모드(mode) 제어 : 강성 파라미터 q 와 좌우 비대칭 a 를 움직이면
  1:1 모드락 -> 서브하모닉(2:1, 3:1) -> 비주기(biphonic) 로 분기(bifurcation)가 생긴다.
  성구(chest/falsetto), 성대 결절성 음성, 이중음(diplophonia)이 같은 방정식에서 나온다.
* 지터/시머 : 인위적 난수 없이 방정식 자체의 과도상태에서 자연스럽게 생긴다.

주의: 샘플 단위 시간루프이므로 느리다. 기본 학습 경로는 LF 사전(glottal.py)이고,
이 모듈은 (a) 데이터 증강/사전학습용 시뮬레이터, (b) LF 파라미터의 물리적 해석,
(c) 짧은 구간의 truncated BPTT 미세조정에 쓰도록 설계했다.
"""
from __future__ import annotations

import math

from dataclasses import dataclass

import torch


@dataclass
class FoldParams:
    """모두 CGS 단위 (g, cm, s). 기본값은 성인 남성 성대 근사."""
    m1: float = 0.125       # 하부(cover) 질량 [g]
    m2: float = 0.025       # 상부(cover) 질량 [g]
    k1: float = 80_000.0    # 강성 [dyn/cm]
    k2: float = 8_000.0
    kc: float = 25_000.0    # 두 질량 간 결합 강성
    r1: float = 0.02        # 감쇠비
    r2: float = 0.02
    a01: float = 0.05       # 정지 시 성문 면적 [cm^2] (내전 정도)
    a02: float = 0.05
    length: float = 1.4     # 성대 길이 [cm]
    d1: float = 0.25        # 질량의 수직 두께 [cm]
    d2: float = 0.05
    rho: float = 1.14e-3    # 공기밀도 [g/cm^3]
    ps: float = 8_000.0     # 성문하압 [dyn/cm^2] (~8 cmH2O)
    q: float = 1.0          # 전체 긴장도 (성구/음높이 제어). q>1 -> 고음/가성쪽
    asym: float = 1.0       # 좌우 비대칭 (1.0 = 대칭). 낮추면 서브하모닉/이중음
    collision: float = 3.0  # 접촉 시 강성 배수


def simulate(params: FoldParams, n_samples: int, sample_rate: int = 24000,
             oversample: int = 4, device=None, dtype=torch.float64):
    """성대 진동을 적분해 성문 유량 U(t) [cm^3/s] 를 반환. (n_samples,)

    반환: (flow, x1, x2) — 유량과 두 질량의 변위 궤적(모드 분석용).
    """
    p = params
    dt = 1.0 / (sample_rate * oversample)
    q, asym = p.q, p.asym

    k1, k2, kc = p.k1 * q, p.k2 * q, p.kc * q
    m1, m2 = p.m1 / q, p.m2 / q          # 긴장 -> 유효질량 감소 -> 고음
    # 좌우 비대칭: 한쪽 성대의 강성만 스케일 (Steinecke-Herzel 의 파라미터 a)
    k1l, k2l = k1 * asym, k2 * asym

    x1 = torch.zeros((), device=device, dtype=dtype) + 0.01
    x2 = torch.zeros((), device=device, dtype=dtype)
    v1 = torch.zeros((), device=device, dtype=dtype)
    v2 = torch.zeros((), device=device, dtype=dtype)

    flow = torch.zeros(n_samples, device=device, dtype=dtype)
    tr1 = torch.zeros(n_samples, device=device, dtype=dtype)
    tr2 = torch.zeros(n_samples, device=device, dtype=dtype)

    c1 = 2.0 * p.r1 * (m1 * k1) ** 0.5
    c2 = 2.0 * p.r2 * (m2 * k2) ** 0.5
    two_l = 2.0 * p.length
    sqrt_2ps_rho = (2.0 * p.ps / p.rho) ** 0.5

    idx = 0
    for n in range(n_samples * oversample):
        a1 = p.a01 + two_l * x1
        a2 = p.a02 + two_l * x2
        amin = torch.minimum(a1, a2)
        open_ = (amin > 0).to(dtype)

        # 베르누이 압력 (Steinecke & Herzel 1995 식 (3))
        p1 = p.ps * (1.0 - open_ * (amin / a1.abs().clamp_min(1e-6)) ** 2) \
            * (a1 > 0).to(dtype)
        p2 = torch.zeros_like(p1)

        # 접촉(성문 폐쇄) 시 추가 강성
        col1 = torch.where(a1 < 0, p.collision * k1 * (a1 / two_l), torch.zeros_like(a1))
        col2 = torch.where(a2 < 0, p.collision * k2 * (a2 / two_l), torch.zeros_like(a2))

        f1 = (-c1 * v1 - 0.5 * (k1 + k1l) * x1 - kc * (x1 - x2) - col1
              + p.length * p.d1 * p1)
        f2 = (-c2 * v2 - 0.5 * (k2 + k2l) * x2 - kc * (x2 - x1) - col2
              + p.length * p.d2 * p2)

        v1 = v1 + dt * f1 / m1          # semi-implicit Euler (심플렉틱, 안정적)
        v2 = v2 + dt * f2 / m2
        x1 = x1 + dt * v1
        x2 = x2 + dt * v2

        if n % oversample == 0:
            u = torch.clamp(amin, min=0.0) * sqrt_2ps_rho
            flow[idx] = u
            tr1[idx] = x1
            tr2[idx] = x2
            idx += 1
    return flow, tr1, tr2


def cycle_rate(flow: torch.Tensor, sample_rate: int = 24000) -> float:
    """성문 폐쇄 주기의 발생률(Hz). 진동이 완전 주기적이 아닐 때도 안정적이다.

    (자가진동 모델은 서브하모닉 영역에 들어가면 자기상관/YIN 이 1/2, 1/3 배음을
    잡는다. 그건 버그가 아니라 분기 현상이며, 이 함수는 그와 무관하게 실제
    폐쇄 횟수를 센다.)
    """
    op = (flow > 0).to(torch.int8)
    onsets = int(((op[1:] - op[:-1]) == 1).sum())
    return onsets * sample_rate / max(len(flow), 1)


# ------------------------------------------------- z축 다질량 모델 (수직 적층)
@dataclass
class MultiMassParams:
    """수직(기류) 방향으로 N 개를 쌓은 성대 모델.

    2질량 모델의 한계는 접촉이다. 질량이 둘뿐이면 성대가 '아래 한 번, 위 한 번'
    으로만 닿아서, 실제로 일어나는 **아래에서 위로 지퍼처럼 닫히는 과정**과
    그 접촉면이 시간에 따라 자라는 과정을 표현할 수 없다. 그래서
    (a) 폐쇄율(closed quotient)이 비현실적으로 낮고,
    (b) 성문 폐쇄가 너무 갑작스러워 여기신호의 고역이 과장되며,
    (c) 점막파(mucosal wave)의 수직 위상차가 딱 한 값으로 고정된다.

    z 방향으로 N 개를 쌓으면 셋 다 방정식에서 저절로 나온다 (Titze 의 다질량
    모델 계열). 추가로 성문 안의 **압력 분포**를 층마다 계산할 수 있어서,
    유동 박리(flow separation) 지점 위쪽은 압력이 0 이라는 사실 —
    2질량 모델이 근사로 때우던 부분 — 을 그대로 넣을 수 있다.
    """
    n_masses: int = 8
    total_mass: float = 0.15      # 전체 [g] (층마다 나눠 갖는다)
    thickness: float = 0.30       # 성대 수직 두께 [cm]
    length: float = 1.4           # 성대 길이 [cm]
    # 층 수 N 에 무관하게 물리가 같아야 한다:
    #   질량 M/N, 횡강성 K/N  -> 고유진동수(=F0)가 N 에 안 변한다
    #   층간 결합은 **점막파 속도**로 준다: k_v = M*N*(c_wave/두께)^2
    # 수직 위상차 = 360 * F0 * 두께 / c_wave [도]. 실측 40~90도가 나오려면
    # c_wave 는 2~4 m/s 다(문헌의 점막파 속도 범위와 일치).
    # 기본값은 자가진동 + 완전폐쇄가 나오는 동작점을 실측으로 잡은 것이다
    # (F0 185 Hz, 폐쇄율 0.36, 최대 접촉 1.00 = 8개 층이 모두 닿는다).
    # 점막파가 **느려야** 층들이 따로 움직인다. 빠르면(>150 cm/s) 전체가 한 덩어리로
    # 움직여 수렴/발산 교대가 사라지고 진동이 죽는다(실측: c>=220 이면 정지).
    mucosal_wave_speed: float = 40.0    # [cm/s]
    k_lateral: float = 120_000.0  # 전체 횡강성 (층마다 K/N)
    damping: float = 0.05
    a0: float = 0.004             # 정지 시 성문 면적 [cm^2] (내전)
    taper: float = 0.8            # 하단이 더 열려 있는 정도(수렴형 성문)
    rho: float = 1.14e-3
    ps: float = 8_000.0
    q: float = 1.0                # 긴장도
    collision: float = 4.0        # 접촉 강성 배수
    collision_damp: float = 0.4   # 접촉 감쇠(에너지 흡수)


def simulate_multi(p: MultiMassParams, n_samples: int, sample_rate: int = 24000,
                   oversample: int = 4, device=None, dtype=torch.float64):
    """수직 적층 다질량 성대. 반환 (flow, x (n_samples, N), contact (n_samples,)).

    contact 는 닿아 있는 층의 비율 — 접촉면적의 대용값이고, 여기서 폐쇄율과
    지퍼 닫힘의 진행을 직접 볼 수 있다.
    """
    n = p.n_masses
    dt = 1.0 / (sample_rate * oversample)
    m = torch.full((n,), p.total_mass / n / p.q, device=device, dtype=dtype)
    kl = torch.full((n,), p.k_lateral / n * p.q, device=device, dtype=dtype)
    kv = p.total_mass * n * (p.mucosal_wave_speed / p.thickness) ** 2 * p.q
    d = p.thickness / n                                  # 층 두께
    two_l = 2.0 * p.length
    c_damp = 2.0 * p.damping * (m * kl).sqrt()

    # 수렴형 성문: 아래가 더 열려 있다 (z=0 하단 -> z=1 상단)
    z = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    a0 = p.a0 * (1.0 + p.taper * (1.0 - z))

    x = torch.zeros(n, device=device, dtype=dtype)
    x[0] = 0.005                                         # 하단을 살짝 밀어 시동
    v = torch.zeros(n, device=device, dtype=dtype)

    flow = torch.zeros(n_samples, device=device, dtype=dtype)
    traj = torch.zeros(n_samples, n, device=device, dtype=dtype)
    contact = torch.zeros(n_samples, device=device, dtype=dtype)
    sqrt_2ps_rho = (2.0 * p.ps / p.rho) ** 0.5
    idx = 0

    for step in range(n_samples * oversample):
        a = a0 + two_l * x
        a_pos = a.clamp_min(0.0)
        closed = (a <= 0)
        a_min = a_pos.min()
        open_ = (~closed).all()

        # 유동 박리 지점 = 가장 좁은 곳. 그 위쪽은 제트라 압력 회복이 없다.
        sep = int(a_pos.argmin())
        if open_:
            ratio = (a_min / a.clamp_min(1e-6)) ** 2
            press = p.ps * (1.0 - ratio)
            press[sep + 1:] = 0.0                        # 박리 하류 = 대기압
            u = a_min * sqrt_2ps_rho
        else:
            # 어디든 닿아 있으면 흐름이 막힌다. 가장 아래 닿은 층보다 아래쪽은
            # 성문하압을 그대로 받고(이게 다음 개방을 밀어낸다), 위쪽은 0 이다.
            first = int(closed.float().argmax())
            press = torch.zeros_like(a)
            press[:first] = p.ps
            u = torch.zeros((), device=device, dtype=dtype)

        # 층간 결합(라플라시안) — 이게 점막파를 만든다
        lap = torch.zeros_like(x)
        lap[1:-1] = x[:-2] - 2 * x[1:-1] + x[2:]
        lap[0] = x[1] - x[0]
        lap[-1] = x[-2] - x[-1]

        col = torch.where(closed, p.collision * kl * (a / two_l), torch.zeros_like(a))
        col_d = torch.where(closed, p.collision_damp * c_damp * v, torch.zeros_like(v))

        f = (-c_damp * v - kl * x + kv * lap - col - col_d
             + p.length * d * press)
        v = v + dt * f / m
        x = x + dt * v

        if step % oversample == 0:
            flow[idx] = u
            traj[idx] = x
            contact[idx] = closed.to(dtype).mean()
            idx += 1
    return flow, traj, contact


def vertical_phase_difference(traj: torch.Tensor, sample_rate: int = 24000) -> float:
    """최상층이 최하층보다 얼마나 **뒤처지는가** [도]. 점막파의 존재 증거.

    실제 성대에서 하단이 상단보다 앞선다(수렴 -> 발산 형상 교대). 이 위상차가
    0 이면 자가진동에 필요한 에너지 전달이 일어나지 않는다. 실측 40~90도.
    """
    lo = traj[:, 0] - traj[:, 0].mean()
    hi = traj[:, -1] - traj[:, -1].mean()
    n = len(lo)
    L = torch.fft.rfft(lo * torch.hann_window(n, dtype=lo.dtype))
    H = torch.fft.rfft(hi * torch.hann_window(n, dtype=hi.dtype))
    k = int(L.abs().argmax())
    d = float(torch.angle(L[k]) - torch.angle(H[k]))
    return math.degrees((d + math.pi) % (2 * math.pi) - math.pi)


def contact_progression(traj: torch.Tensor, p: MultiMassParams,
                        sample_rate: int = 24000) -> dict:
    """접촉이 아래에서 위로 진행하는가(지퍼 닫힘)를 잰다.

    2질량 모델로는 볼 수 없는 양이다. 층별 접촉 신호의 상호상관 지연을 써서
    최하층 대비 최상층의 접촉 시점 지연 [ms] 과, 층 순서와 접촉 시점의
    순위상관을 낸다(+1 이면 완전히 아래->위 순서).
    """
    n = p.n_masses
    z = torch.linspace(0.0, 1.0, n, device=traj.device, dtype=traj.dtype)
    a = p.a0 * (1.0 + p.taper * (1.0 - z)) + 2.0 * p.length * traj
    c = (a <= 0).to(traj.dtype)                       # (T, N) 접촉 여부
    ref = c[:, 0] - c[:, 0].mean()
    lags = []
    max_lag = int(sample_rate * 0.004)                # +-4 ms
    for i in range(n):
        y = c[:, i] - c[:, i].mean()
        if float(y.abs().sum()) < 1e-6:
            lags.append(float("nan"))
            continue
        r = [float((ref[max_lag:-max_lag] * y[max_lag - k: -max_lag - k or None]).sum())
             for k in range(-max_lag, max_lag)]
        lags.append((int(torch.tensor(r).argmax()) - max_lag) / sample_rate * 1000.0)
    valid = [(i, v) for i, v in enumerate(lags) if v == v]
    rho = float("nan")
    if len(valid) > 2:
        xi = torch.tensor([float(i) for i, _ in valid])
        yi = torch.tensor([v for _, v in valid])
        xi = xi - xi.mean(); yi = yi - yi.mean()
        rho = float((xi * yi).sum() / (xi.norm() * yi.norm()).clamp_min(1e-9))
    return {"lags_ms": lags, "bottom_to_top_ms": lags[-1] - lags[0],
            "order_correlation": rho,
            "closed_fraction": float((c.sum(-1) > 0).to(traj.dtype).mean()),
            "max_contact": float(c.mean(-1).max())}


# --------------------------------------------------- 호흡(성문하압) 비선형 응답
CM_H2O = 980.0          # 1 cmH2O = 980 dyn/cm^2


def pressure_sweep(params: FoldParams, pressures_cm: tuple = (1, 2, 3, 4, 6, 8, 12, 16),
                   n_samples: int = 9600, sample_rate: int = 24000,
                   skip: int = 2400) -> list[dict]:
    """성문하압을 훑으며 진동 여부/주기율/진폭을 잰다.

    성대의 비선형성은 따로 넣을 필요가 없다 — 이미 방정식 안에 세 군데 있다.
      1. 베르누이 압력이 면적의 **제곱**에 의존한다
      2. 성대가 닿으면 강성이 스위치되는 **접촉(collision)** 항
      3. 폐쇄 시 유량이 0 으로 잘리는 정류(rectification)
    그래서 압력 하나만 움직여도 사람 목소리의 비선형 특성이 그대로 따라온다.
    측정(기본 파라미터): **발성 역치압 1~2 cmH2O**, 압력 2배당 **약 7 dB** 증가.
    """
    out = []
    for cm in pressures_cm:
        flow, _, _ = simulate(FoldParams(**{**params.__dict__, "ps": cm * CM_H2O}),
                              n_samples, sample_rate, oversample=4)
        seg = flow[skip:]
        out.append({"ps_cm": float(cm), "rms": float(seg.std()),
                    "f0": cycle_rate(seg, sample_rate),
                    "oscillating": bool(seg.std() > 1.0 and cycle_rate(seg) > 20)})
    return out


def phonation_threshold(params: FoldParams, lo_cm: float = 0.2, hi_cm: float = 12.0,
                        steps: int = 7, **kw) -> float:
    """발성 역치압(PTP) [cmH2O]. 이 아래에서는 성대가 아예 진동하지 않는다.

    말끝이 스러지는 소리, /ㅎ/ 뒤의 유성 시작, 속삭임과 발성의 경계가 전부
    이 문턱 하나다. 크로스페이드로 흉내내면 그 순간이 '기계처럼' 들린다.
    """
    for _ in range(steps):
        mid = 0.5 * (lo_cm + hi_cm)
        r = pressure_sweep(params, (mid,), **kw)[0]
        lo_cm, hi_cm = (lo_cm, mid) if r["oscillating"] else (mid, hi_cm)
    return 0.5 * (lo_cm + hi_cm)


def pressure_to_source(ps_cm: torch.Tensor, ptp_cm: float = 2.0,
                       db_per_doubling: float = 7.0,
                       f0_hz_per_cm: float = 3.0,
                       rd_per_cm: float = -0.06) -> dict:
    """호흡 압력 하나 -> LF 소스 파라미터 (빠른 학습 경로용).

    2질량 ODE 는 느려서 학습 본선에 못 쓴다. 대신 위 측정에서 얻은 관계를
    **파라미터 사이의 구속조건**으로 넣는다. amp/F0/Rd 를 세 개의 자유로운
    손잡이로 두지 않고 압력 하나에 묶으면, 그 자체가 물리적 정규화가 된다
    (포먼트를 저차원 부분공간에 가두는 것과 같은 발상).

    반환: {"amp": 배수, "f0_shift": Hz, "rd_shift": Rd 증분}
    - amp: 역치 아래에서 0 (발성이 아예 없다), 위에서 압력 2배당 약 7 dB
    - f0_shift: +2~5 Hz/cmH2O (문헌값. 2질량 모델 자체는 이 결합이 약하다)
    - rd_shift: 압력이 높을수록 pressed 쪽(Rd 감소)
    """
    p = ps_cm.clamp_min(0.0)
    on = (p > ptp_cm).to(p.dtype)
    ratio = (p / ptp_cm).clamp_min(1e-3)
    amp = on * 10 ** (db_per_doubling * torch.log2(ratio) / 20.0)
    return {"amp": amp,
            "f0_shift": f0_hz_per_cm * (p - ptp_cm) * on,
            "rd_shift": rd_per_cm * (p - ptp_cm) * on}


def flow_to_excitation(flow: torch.Tensor) -> torch.Tensor:
    """유량 U(t) -> 유량미분 dU/dt (입술 방사 효과 포함, LF 소스와 동일 규격)."""
    d = torch.zeros_like(flow)
    d[1:] = flow[1:] - flow[:-1]
    m = d.abs().max().clamp_min(1e-9)
    return d / m
