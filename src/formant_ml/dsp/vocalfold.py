"""성대 자가진동 물리모델과 그 유량이 난류를 구동하는 방식.

성대 모형이 '왜 이 모양이냐'
----------------------------
자가진동은 성대의 **수직 위상차**(mucosal wave)가 있어야 생긴다. 성문이 열릴 때는
수렴형(convergent), 닫힐 때는 발산형(divergent)이 되어야 기류에서 순(net) 에너지를
받는다(Titze 1988, surface-wave model; Titze 2006, myoelastic-aerodynamic theory).
한 점 질량으로는 이 위상차가 없어 절대 떨지 않는다. 그래서 최소 두 개가 필요하다:

* **2질량**(Ishizaka & Flanagan 1972; 여기서는 Steinecke & Herzel 1995 축약):
  성대의 하연(lower margin)·상연(upper margin)을 각각 한 점으로. 수직 위상차와
  자가진동, 그리고 분기(성구/이중음)까지 나온다. `simulate`.
* **수직 다질량**(`simulate_stack`): 2질량을 위-아래로 n 개 쌓은 것. "2질량 여러 개
  위아래로" 가 바로 이것이고, **쓸모없는 게 아니라** mucosal wave 를 공간적으로
  이산화한 것이다 — 통로 모양이 n 개 마디를 가질 수 있어 점막파가 아래에서 위로
  전파하는 것이 궤적에 그대로 나타난다(2질량은 위상차만, 다질량은 파형까지).
* **body-cover 3질량**(Story & Titze 1995; `simulate_body_cover`): 갑상피열근(body)과
  점막(cover)을 분리한 표준 모형. 성구(chest/falsetto)를 body/cover 강성비로 낸다.

치찰음과 '연결'되는 지점
------------------------
핵심은 **하나의 기류(U)가 발성과 난류를 동시에 만든다**는 것이다(Titze 의
myoelastic-aerodynamic + Stevens 의 aeroacoustic). 성대가 만드는 성문 유량 U(t) 는:

1. 유량미분 dU/dt 이 성도를 여기해 **유성음**이 되고(`flow_to_excitation`),
2. 성문이 덜 닫혀 있으면 성문에서 **난류(기식)** 를 내고(`aeroacoustic.aspiration_*`),
3. 그 맥동이 구강 협착의 **마찰음을 성문동기로 변조**한다(Jackson & Shadle 2000).

즉 성대 모형은 난류의 *구동원* 이다. `glottal_flow_to_modulation` 이 이 유량을
합성기의 노이즈 변조 신호로 바꿔, 성대와 치찰음을 한 기류로 잇는다. 그래서
유성 마찰음 /z/ 이 자동으로 성문주기로 변조되고, 학습 시 소스와 노이즈가 같은
물리량으로 묶인다(따로 노는 게 아니다).

주의: 샘플 단위 시간루프이므로 느리다. 기본 학습 경로는 LF 사전(glottal.py)이고,
이 모듈은 (a) 데이터 증강/사전학습용 시뮬레이터, (b) LF 파라미터의 물리적 해석,
(c) 소스-난류 결합 검증, (d) 짧은 구간 truncated BPTT 미세조정에 쓴다.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------- 다중 질량
def simulate_stack(params: FoldParams, n_masses: int = 5, n_samples: int = 9600,
                   sample_rate: int = 24000, oversample: int = 4,
                   coupling: float = 0.25, device=None, dtype=torch.float64):
    # n_masses 기본 5: 성구(흉성/두성)를 '어느 층이 접지를 잃는가' 로 구분하려면
    # 상·하연에 수직 해상도가 필요하다. contact_quotient 주석의 측정 참고.
    """성대의 **수직 방향**을 n 개 질량으로 쌓은 모델. 반환 (flow, x (n_samples, n)).

    왜 2 개로는 부족한가
    --------------------
    2질량 모델은 성대의 하연(lower margin)과 상연(upper margin)을 각각 하나의
    점으로 본다. 그것만으로도 수직 위상차(하연이 먼저 열리고 먼저 닫힌다)와
    자가진동은 나오지만, 성문 통로의 **모양**은 두 점을 잇는 직선뿐이다.
    실제로는 통로가 열릴 때 수렴형(convergent), 닫힐 때 발산형(divergent)으로
    바뀌고 그 형상 변화가 에너지 전달을 지배한다 — 이것이 점막파(mucosal wave)다.

    질량을 n 개로 쌓으면 통로 모양이 n 개의 마디를 가질 수 있어, 점막파가
    아래에서 위로 전파하는 것이 궤적에 그대로 나타난다. 강성을 수직으로 분포시키면
    (아래가 단단하고 위가 무름) 전파 속도도 생긴다.

    압력 분포는 2질량 모델의 규칙을 일반화한다: 최소 단면(협착)의 상류는 Ps,
    하류는 대기압. 유량은 최소 단면이 정한다.

    `coupling` 은 수직 결합 강성의 배율이다. 크면 전체가 한 덩어리처럼 움직여
    점막파가 사라지고, 작으면 마디가 따로 놀아 비현실적으로 커진다.
    기본 0.25 에서 하연이 상연을 약 10~15% 주기만큼 앞선다(사람 0.5~1.5 ms).
    """
    p = params
    dt = 1.0 / (sample_rate * oversample)
    q = p.q
    n = n_masses
    idx = torch.arange(n, dtype=dtype, device=device)
    frac = idx / max(n - 1, 1)                     # 0 = 하연, 1 = 상연

    # 수직 분포: 아래가 무겁고 단단하다 (body 에 가깝다), 위로 갈수록 무르다.
    #
    # 분할 규칙이 중요하다. 성대를 n 조각으로 나누면 각 조각의 질량도 강성도
    # 1/n 로 준다 -> ω = sqrt(k/m) 는 n 과 무관해야 한다. (처음에 m 을 n 배,
    # k 를 1/n 배로 두었더니 F0 가 질량 수에 따라 130 -> 45 Hz 로 흘렀다.)
    # 반면 이웃 결합 kc 는 이산 라플라시안의 Δz² 를 상쇄해야 파동 속도가
    # 유지되므로 n 에 비례해서 커진다.
    scale = 2.0 / n_masses
    m = (p.m1 + (p.m2 - p.m1) * frac) * scale / q
    k = (p.k1 + (p.k2 - p.k1) * frac) * q * scale
    kc = p.kc * q * (n_masses / 2.0) * coupling
    kl = k * p.asym                                # 좌우 비대칭
    c = 2.0 * p.r1 * (m * k).sqrt()
    a0 = p.a01 + (p.a02 - p.a01) * frac
    d = (p.d1 + (p.d2 - p.d1) * frac) * scale      # 각 질량의 수직 두께
    two_l = 2.0 * p.length
    sqrt_2ps_rho = (2.0 * p.ps / p.rho) ** 0.5

    x = torch.zeros(n, device=device, dtype=dtype)
    x[0] = 0.01
    v = torch.zeros(n, device=device, dtype=dtype)
    flow = torch.zeros(n_samples, device=device, dtype=dtype)
    traj = torch.zeros(n_samples, n, device=device, dtype=dtype)

    out = 0
    for step in range(n_samples * oversample):
        a = a0 + two_l * x
        amin, imin = a.min(0)
        open_ = (amin > 0).to(dtype)
        # 협착 상류는 Ps, 하류는 0 (2질량 모델의 베르누이 규칙을 일반화)
        upstream = (idx <= imin).to(dtype)
        press = p.ps * upstream * (1.0 - open_ * (amin / a.abs().clamp_min(1e-6)) ** 2)
        press = press * (a > 0).to(dtype)
        coll = torch.where(a < 0, p.collision * k * (a / two_l),
                           torch.zeros_like(a))
        # 수직 결합 (라플라시안) — 이것이 점막파를 전파시킨다
        xl = torch.cat([x[:1], x[:-1]])
        xr = torch.cat([x[1:], x[-1:]])
        f = (-c * v - 0.5 * (k + kl) * x - kc * (2.0 * x - xl - xr) - coll
             + p.length * d * press)
        v = v + dt * f / m
        x = x + dt * v
        if step % oversample == 0:
            flow[out] = torch.clamp(amin, min=0.0) * sqrt_2ps_rho
            traj[out] = x
            out += 1
    return flow, traj


def simulate_body_cover(params: FoldParams, n_samples: int = 9600,
                        sample_rate: int = 24000, oversample: int = 4,
                        body_mass: float = 0.5, body_stiff: float = 40_000.0,
                        body_damp: float = 0.1, device=None,
                        dtype=torch.float64):
    """Story & Titze(1995) body-cover 3질량. 반환 (flow, x (n_samples, 3)).

    2질량(cover: 하연 m1·상연 m2)에 **body 질량**을 하나 더 붙인다. body 는
    갑상피열근(TA)이고 cover 는 그 위를 덮은 점막이다. cover 두 질량이 body 에
    스프링으로 매달리고, body 는 고정벽에 매달린다.

    이 분리가 필요한 이유: 성구(register)는 TA(body)와 CT 의 긴장 배분으로
    바뀐다. body 가 단단하고 cover 가 무르면 점막파가 잘 실려 chest, body 까지
    긴장하면 falsetto 다. 2질량만으로는 이 배분을 표현할 축이 없다.

    x = [x1(하연 cover), x2(상연 cover), xb(body)]. 유량은 cover 최소 단면이 정한다.
    """
    p = params
    dt = 1.0 / (sample_rate * oversample)
    q = p.q
    m1, m2, mb = p.m1 / q, p.m2 / q, body_mass / q
    k1, k2, kc = p.k1 * q, p.k2 * q, p.kc * q
    kb = body_stiff * q
    k1l = k1 * p.asym
    c1 = 2.0 * p.r1 * (m1 * k1) ** 0.5
    c2 = 2.0 * p.r2 * (m2 * k2) ** 0.5
    cb = 2.0 * body_damp * (mb * kb) ** 0.5
    two_l = 2.0 * p.length
    sqrt_2ps_rho = (2.0 * p.ps / p.rho) ** 0.5

    x1 = torch.tensor(0.01, device=device, dtype=dtype)
    x2 = torch.zeros((), device=device, dtype=dtype)
    xb = torch.zeros((), device=device, dtype=dtype)
    v1 = torch.zeros((), device=device, dtype=dtype)
    v2 = torch.zeros((), device=device, dtype=dtype)
    vb = torch.zeros((), device=device, dtype=dtype)
    flow = torch.zeros(n_samples, device=device, dtype=dtype)
    traj = torch.zeros(n_samples, 3, device=device, dtype=dtype)

    out = 0
    for n in range(n_samples * oversample):
        # cover 변위는 body 변위에 상대적이다(cover 가 body 위에 얹혀 있다)
        a1 = p.a01 + two_l * (x1 + xb)
        a2 = p.a02 + two_l * (x2 + xb)
        amin = torch.minimum(a1, a2)
        open_ = (amin > 0).to(dtype)
        p1 = p.ps * (1.0 - open_ * (amin / a1.abs().clamp_min(1e-6)) ** 2) \
            * (a1 > 0).to(dtype)
        col1 = torch.where(a1 < 0, p.collision * k1 * (a1 / two_l), torch.zeros_like(a1))
        col2 = torch.where(a2 < 0, p.collision * k2 * (a2 / two_l), torch.zeros_like(a2))
        # cover 두 질량: body 에 대해 매달림(k1,k2), 서로 결합(kc)
        f1 = -c1 * v1 - 0.5 * (k1 + k1l) * x1 - kc * (x1 - x2) - col1 \
            + p.length * p.d1 * p1
        f2 = -c2 * v2 - 0.5 * k2 * x2 - kc * (x2 - x1) - col2
        # body: 고정벽에 매달리고, cover 의 반작용을 받는다
        fb = (-cb * vb - kb * xb + 0.5 * (k1 * x1 + k2 * x2)
              + kc * 0.0)
        v1 = v1 + dt * f1 / m1
        v2 = v2 + dt * f2 / m2
        vb = vb + dt * fb / mb
        x1 = x1 + dt * v1
        x2 = x2 + dt * v2
        xb = xb + dt * vb
        if n % oversample == 0:
            flow[out] = torch.clamp(amin, min=0.0) * sqrt_2ps_rho
            traj[out] = torch.stack([x1, x2, xb])
            out += 1
    return flow, traj


def contact_quotient(traj: torch.Tensor, params: FoldParams,
                     skip: int = 2000) -> torch.Tensor:
    """수직 질량별 **접지율**(성문이 닫혀 있는 시간 비율). 반환 (n,) 하연->상연.

    성구(register)를 가르는 것이 이것이다. 측정(수직 5질량, ps=8000):

        q=1.0 내전 0.02 -> F0 151 Hz, 접지 [0.25 0.28 0.32 0.31 0.38], 점막파 0.58 ms
        q=2.4 내전 0.08 -> F0 326 Hz, 접지 [0.00 0.09 0.21 0.25 0.28], 점막파 0.17 ms

    즉 **흉성은 하연까지 접지하고, 두성은 하연이 접지를 잃는다**(0.00). 동시에
    점막파가 붕괴한다 — 하연이 닿지 않으면 아래에서 위로 전달할 변형이 없다.
    이 축을 표현하려면 상·하연을 여러 겹으로 쌓아야 한다: 2질량은 '위상차가
    있다/없다' 만, 3질량은 body/cover 배분만 말할 수 있고, '어느 층이 접지를
    잃는가' 는 수직 해상도가 있어야 나온다. 그래서 `simulate_stack` 의 기본이 5 다.
    """
    n = traj.shape[1]
    frac = torch.arange(n, dtype=traj.dtype, device=traj.device) / max(n - 1, 1)
    a0 = params.a01 + (params.a02 - params.a01) * frac
    area = a0 + 2.0 * params.length * traj
    return (area[skip:] < 0).to(torch.float32).mean(0)


def glottal_flow_to_modulation(flow: torch.Tensor, sample_rate: int = 24000,
                               hop: int = 240):
    """성문 유량 U(t) -> 프레임률 (성문개방 포락선, 맥동 변조 깊이).

    성대와 난류를 잇는 다리다. 반환:
      * `open_env` (1,T,1): 각 프레임의 평균 성문 개방(≈유량). 성문이 열려 있을 때
        기식/마찰 난류가 세다.
      * `mod_depth` (1,T,1): 유량의 주기적 맥동 깊이(AC/DC). 유성 마찰음에서
        마찰음이 성문주기로 변조되는 정도(Jackson & Shadle 2000).

    이 두 신호를 합성기의 `noise_bands`·`noise_am` 에 그대로 넣으면, 성대 모형이
    낸 유량이 난류(치찰음)를 실제로 구동/변조한다.
    """
    u = flow.detach().to(torch.float32)
    if u.dim() == 1:
        u = u[None]
    n = u.shape[-1]
    t = n // hop
    u = u[..., :t * hop].reshape(u.shape[0], t, hop)
    dc = u.mean(-1)                                    # 프레임 평균(개방도)
    ac = u.std(-1)                                     # 프레임 내 맥동
    open_env = (dc / dc.amax().clamp_min(1e-9)).unsqueeze(-1)
    mod = (ac / dc.clamp_min(1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
    return open_env, mod


def mucosal_wave_delay(traj: torch.Tensor, sample_rate: int = 24000) -> float:
    """점막파 지연 [ms]. **양수면 하연이 상연을 앞선다**(생리적으로 맞는 방향).

    사람은 대략 0.5~1.5 ms (주기의 5~15%). 성대의 아래쪽이 먼저 열리고 먼저
    닫히면서 그 변형이 위로 전파하는 것이 점막파이고, 이 위상차가 있어야
    성문이 열릴 때 수렴형·닫힐 때 발산형이 되어 기류에서 에너지를 받는다.
    위상차가 0 이면 자가진동의 동력 자체가 없다.
    """
    a = traj[:, 0] - traj[:, 0].mean()
    b = traj[:, -1] - traj[:, -1].mean()
    n = len(a)
    A = torch.fft.rfft(a, 2 * n)
    B = torch.fft.rfft(b, 2 * n)
    cc = torch.fft.irfft(A * B.conj(), 2 * n)
    # 탐색 범위를 **반주기 이내**로 묶는다. 안 그러면 상호상관이 한 주기 건너뛴
    # 봉우리를 잡아 108% 같은 값이 나온다(주기 신호에서는 τ 와 τ±T 가 구별되지 않는다).
    aa = torch.fft.irfft(A * A.conj(), 2 * n)[: n // 2]
    lo = max(int(sample_rate / 500), 2)
    period = int(aa[lo:min(int(sample_rate / 60), n // 2)].argmax()) + lo
    half = max(period // 2, 2)
    pos = cc[:half]
    neg = cc[-half:]
    lag = (int(pos.argmax()) if float(pos.max()) >= float(neg.max())
           else int(neg.argmax()) - half)
    # r(τ)=Σ a(t+τ)b(t) 의 최대가 양수 τ 면 a(하연)가 뒤진다는 뜻이므로 부호를 뒤집어,
    # **양수 = 하연이 앞선다** 로 맞춘다.
    return -lag * 1000.0 / sample_rate


def cycle_rate(flow: torch.Tensor, sample_rate: int = 24000) -> float:
    """성문 폐쇄 주기의 발생률(Hz). 진동이 완전 주기적이 아닐 때도 안정적이다.

    (자가진동 모델은 서브하모닉 영역에 들어가면 자기상관/YIN 이 1/2, 1/3 배음을
    잡는다. 그건 버그가 아니라 분기 현상이며, 이 함수는 그와 무관하게 실제
    폐쇄 횟수를 센다.)
    """
    op = (flow > 0).to(torch.int8)
    onsets = int(((op[1:] - op[:-1]) == 1).sum())
    return onsets * sample_rate / max(len(flow), 1)


def flow_to_excitation(flow: torch.Tensor) -> torch.Tensor:
    """유량 U(t) -> 유량미분 dU/dt (입술 방사 효과 포함, LF 소스와 동일 규격)."""
    d = torch.zeros_like(flow)
    d[1:] = flow[1:] - flow[:-1]
    m = d.abs().max().clamp_min(1e-9)
    return d / m
