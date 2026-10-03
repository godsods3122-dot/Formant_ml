"""**성도 속 공기의 물성** — 날숨의 조성(포화 수증기 · CO2)과 온도에서 (MEASUREMENTS §52.523).

사용자: *"H2O와 CO2 고려 … 비강(특히 코 옆에 붙은 뼈가 공기를 데워주는 역할) … 습기 이런 것들이 은근히 영향을 줄 것 같은데, 다 고려해봤어?
너무 rough하게 잡으면 안 돼"*. 관은 음속 350 m/s · 밀도 1.14 kg/m³ · 점성 1.86e-5 Pa·s · 비열비 1.4 를 상수로 써 왔다 (말소리 문헌의 관례값
"37 °C 습한 공기"). 여기서는 그 값을 조성과 온도에서 계산한다.

* 조성: 날숨의 건조 기준 CO2 `CO2_DRY` (말하는 동안의 혼합 날숨 ~4 %, 폐포 5.3 %), O2 는 소비된 몫만큼 줄고(호흡 상 0.8), Ar 0.93 %, 나머지 N2.
  수증기는 그 자리 온도의 포화 증기압 × 상대 습도 (Buck 1981).
* 순물질 물성 (300–320 K, NIST Chemistry WebBook · Poling, Prausnitz & O'Connell 2001): 몰 비열 cp, 점성 μ, 열전도 κ — 온도 의존은
  Sutherland 꼴로.
* 혼합: 몰 질량·비열은 몰 분율 가중, 점성·열전도는 Wilke (1950) 혼합 규칙 (열전도는 Mason–Saxena 꼴, 같은 Φ).
* 음속 c = √(γ R T / M), 밀도 ρ = p M / (R T).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

R_GAS = 8.314462618          # J/(mol·K)
P_ATM = 101325.0             # Pa
CO2_DRY = 0.040              # 날숨의 건조 기준 CO2 몰 분율 (말하는 동안, 혼합 날숨)
RQ = 0.8                     # 호흡 상 — 소비 O2 = 생성 CO2 / RQ

#: 순물질: 몰 질량 [kg/mol], cp(310 K) [J/(mol·K)], μ(300 K) [Pa·s], Sutherland 상수 S_μ [K], κ(300 K) [W/(m·K)], κ 의 온도 지수
_SPECIES = {
    "N2": (0.0280134, 29.13, 1.787e-5, 111.0, 0.02583, 0.80),
    "O2": (0.0319988, 29.43, 2.063e-5, 127.0, 0.02658, 0.84),
    "Ar": (0.039948, 20.79, 2.270e-5, 144.0, 0.01772, 0.73),
    "CO2": (0.0440095, 37.35, 1.502e-5, 240.0, 0.01662, 1.20),
    "H2O": (0.0180153, 33.60, 0.990e-5, 1064.0, 0.01864, 1.10),
}


def p_sat(T_c: float) -> float:
    """물의 포화 증기압 [Pa] (Buck 1981, 0–50 °C 오차 < 0.05 %)."""
    return 611.21 * math.exp((18.678 - T_c / 234.5) * (T_c / (257.14 + T_c)))


def composition(T_c: float, rh: float = 1.0, co2_dry: float = CO2_DRY) -> dict[str, float]:
    """몰 분율 — 건조 기준 날숨(CO2 · 줄어든 O2 · Ar · N2)에 그 온도의 수증기."""
    x_w = min(rh * p_sat(T_c) / P_ATM, 0.2)
    o2 = 0.2095 - co2_dry / RQ
    ar = 0.0093
    n2 = 1.0 - co2_dry - o2 - ar
    dry = {"N2": n2, "O2": o2, "Ar": ar, "CO2": co2_dry}
    x = {k: v * (1.0 - x_w) for k, v in dry.items()}
    x["H2O"] = x_w
    return x


def _mu_k(sp: str, T: float) -> tuple[float, float]:
    M, cp, mu0, S, k0, nk = _SPECIES[sp]
    mu = mu0 * (T / 300.0) ** 1.5 * (300.0 + S) / (T + S)
    k = k0 * (T / 300.0) ** nk
    return mu, k


def _wilke(x: dict, val: dict, M: dict, mu: dict) -> float:
    out = 0.0
    for i in x:
        den = 0.0
        for j in x:
            phi = (1.0 + math.sqrt(mu[i] / mu[j]) * (M[j] / M[i]) ** 0.25) ** 2 / math.sqrt(8.0 * (1.0 + M[i] / M[j]))
            den += x[j] * phi
        out += x[i] * val[i] / den
    return out


@dataclass(frozen=True)
class AirProps:
    T_c: float
    rho: float               # kg/m³
    c: float                 # m/s
    mu: float                # Pa·s
    kappa: float             # W/(m·K)
    cp: float                # J/(kg·K)
    gamma: float
    M: float                 # kg/mol
    x: dict

    def cgs(self) -> dict:
        """관 커널의 단위(CGS)로: C_SOUND cm/s, RHO g/cm³, MU dyn·s/cm² (= poise), LAMBDA_TH erg/(s·cm·K), CP erg/(g·K), GAMMA."""
        return dict(C_SOUND=self.c * 100.0, RHO=self.rho * 1e-3, MU=self.mu * 10.0, LAMBDA_TH=self.kappa * 1e5,
                    CP=self.cp * 1e4, GAMMA=self.gamma)


def props(T_c: float = 37.0, rh: float = 1.0, co2_dry: float = CO2_DRY, p: float = P_ATM) -> AirProps:
    """온도 T_c [°C], 상대 습도 rh, 건조 기준 CO2 에서의 물성."""
    T = T_c + 273.15
    x = composition(T_c, rh, co2_dry)
    Mi = {k: _SPECIES[k][0] for k in x}
    cpm = sum(x[k] * _SPECIES[k][1] for k in x)               # J/(mol·K)
    M = sum(x[k] * Mi[k] for k in x)
    gamma = cpm / (cpm - R_GAS)
    mus, ks = {}, {}
    for k in x:
        mus[k], ks[k] = _mu_k(k, T)
    mu = _wilke(x, mus, Mi, mus)
    kappa = _wilke(x, ks, Mi, mus)
    rho = p * M / (R_GAS * T)
    c = math.sqrt(gamma * R_GAS * T / M)
    return AirProps(T_c, rho, c, mu, kappa, cpm / M, gamma, M, x)
