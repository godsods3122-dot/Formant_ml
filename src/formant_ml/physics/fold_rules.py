"""근육 활성 → 성대 층의 세로 응력 → **몸체–덮개 집중 매개변수** (MEASUREMENTS §52.504).

사용자: *"성대 근육 규칙 … 후속 연구 등등으로부터 얻어낼 순 없어?"* — 후속 연구에서 얻었다 (모두 `third_party/` 에 로컬로):

* **Titze & Story 2002 규칙의 실제 구현** — Zañartu 연구실 코드 `PhonationModelsCode2/+vfsolver/getRules.m` 과
  `@MuscleActivation/Rule2BodyCoverParameters.m` (Serry·Zañartu·Peterson 2026 보–막 모형 저장소에 같이 실려 있다). Brad Story 와의 교신에 따른
  정오표가 들어 있다: 수렴 규칙의 부호, 점막 ε₂ = −0.35 (논문은 +0.35), 길이 L = L0(1+ε) (정오표의 L0/(1+ε) 는 "THIS IS WRONG" 으로 되돌렸다).
* **조직 응력–변형 판 넷** — 수정 Kelvin 모형의 정적 해 (Hunter & Titze 2004; Titze 2006):
  σ_p(ε) = −σ0/ε1·(ε − ε1)  (+ σ2·[e^{B(ε−ε2)} − 1 − B(ε−ε2)] , ε > ε2),  σ_a = a·σm·max(0, 1 − b(ε − εm)²) (갑상피열근만).
  "ts2002" getRules.m, "tit2006"·"alz2020" 근육 모형 파일(`+IntrinsicMuscles/*Model.m`), "alz2021" Alzamendi et al. 2021 표 IV,
  "serry2026" Serry et al. 2026 표 2.

**이 규칙들은 사용자의 막 밴드 구조를 한 점에서 잰 것이다.** 덮개 용수철 k = 2μ(LT/D) + π²σ(D/L)T 의 둘째 항은 세로 응력 σ 아래 막의 가로
가지를 k_z = π/L (양 끝 고정의 첫 정재파) 한 점에서 읽은 값이고, 첫 항은 단면의 전단이다. 경험 규칙으로 남은 것은 **마디점**
Zn = (1 + a_TA)·T/3 (아래·위 덮개의 경계), **진동 깊이** D_b = (a_TA·D_mus + ½D_lig)/(1 + 0.2ε) 과 두께 T0/(1 + 0.8ε) 이다 —
`fold_modes.layered_lumped` 가 이것들을 층 구조 막의 고유모드에서 물리로 뽑는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: 조직 밀도 [kg/m³] (Titze & Story 2002)
RHO = 1040.0

#: 조직 판: 이름 → {조직: (σ0, σ2, ε1, ε2, B, σm, εm, b)} [Pa, −]. σm 이 0 이면 능동 응력이 없다.
TISSUE_SETS = {
    "ts2002": {"muc": (500.0, 20000.0, -0.5, -0.35, 4.4, 0.0, 0.0, 0.0),
               "lig": (400.0, 1393.0, -0.5, 0.0, 17.0, 0.0, 0.0, 0.0),
               "ta": (1000.0, 1500.0, -0.5, -0.05, 6.5, 105e3, 0.4, 1.07)},
    "tit2006": {"muc": (1000.0, 9000.0, -0.5, -0.35, 4.4, 0.0, 0.0, 0.0),
                "lig": (1000.0, 1400.0, -0.5, 0.0, 17.0, 0.0, 0.0, 0.0),
                "ta": (1000.0, 1500.0, -0.5, -0.05, 6.5, 105e3, 0.4, 1.07)},
    "alz2020": {"muc": (1000.0, 1000.0, -0.5, -0.3, 6.0, 0.0, 0.0, 0.0),
                "lig": (500.0, 1000.0, -0.5, -0.3, 7.0, 0.0, 0.0, 0.0),
                "ta": (2000.0, 1000.0, -0.5, -0.05, 9.0, 100e3, 0.4, 1.07)},
    "alz2021": {"muc": (1000.0, 20000.0, -0.5, -0.3, 4.4, 0.0, 0.0, 0.0),
                "lig": (2000.0, 1400.0, -0.5, -0.3, 13.0, 0.0, 0.0, 0.0),
                "ta": (2000.0, 1500.0, -0.5, -0.05, 6.5, 150e3, 0.2, 1.07)},
    "serry2026": {"muc": (1000.0, 1000.0, -0.5, -0.3, 6.0, 0.0, 0.0, 0.0),
                  "lig": (500.0, 1000.0, -0.5, -0.3, 7.0, 0.0, 0.0, 0.0),
                  "ta": (2000.0, 1000.0, -0.5, -0.05, 9.0, 100e3, 0.3, 1.0)},
}

#: 휴지 치수 [m] (Titze & Story 2002, getRules.m) — 여성은 Titze 1989 척도.
DIMS = {"female": dict(L0=1.0e-2, T0=0.2e-2, Dmuc=0.15e-2, Dlig=0.15e-2, Dmus=0.3e-2),
        "male": dict(L0=1.6e-2, T0=0.3e-2, Dmuc=0.2e-2, Dlig=0.2e-2, Dmus=0.4e-2)}

#: 덮개·몸체 전단 계수 [Pa] — Zañartu 코드 `MuscleControlModel` (500·1000), Alzamendi 2021 본문 (300·600).
MU_COVER = 500.0
MU_BODY = 1000.0


def passive_stress(eps, p) -> np.ndarray:
    """수동 응력 [Pa] (getRules.m `sigma_p` — ε < ε1 이면 0)."""
    s0, s2, e1, e2, B = p[:5]
    e = np.asarray(eps, float)
    lin = -(s0 / e1) * (e - e1)
    x = np.maximum(e - e2, 0.0)
    ex = s2 * (np.exp(B * x) - 1.0 - B * x)
    return np.where(e < e1, 0.0, lin + ex)


def passive_modulus(eps, p) -> np.ndarray:
    """접선 영률 dσ_p/dε [Pa]."""
    s0, s2, e1, e2, B = p[:5]
    e = np.asarray(eps, float)
    x = np.maximum(e - e2, 0.0)
    return np.where(e < e1, 0.0, -(s0 / e1) + s2 * B * (np.exp(B * x) - 1.0))


def active_stress(eps, a, p) -> np.ndarray:
    """능동 응력 [Pa] a·σm·max(0, 1 − b(ε − εm)²)."""
    sm, em, b = p[5:8]
    return float(a) * sm * np.maximum(0.0, 1.0 - b * (np.asarray(eps, float) - em) ** 2)


def active_modulus(eps, a, p) -> np.ndarray:
    sm, em, b = p[5:8]
    e = np.asarray(eps, float)
    on = (1.0 - b * (e - em) ** 2) > 0.0
    return np.where(on, float(a) * sm * (-2.0 * b * (e - em)), 0.0)


def layer_stress(eps, a_ta: float = 0.0, tissue: str = "ts2002") -> dict:
    """층별 세로 응력 [Pa] {muc, lig, ta}."""
    ts = TISSUE_SETS[tissue]
    return {"muc": passive_stress(eps, ts["muc"]), "lig": passive_stress(eps, ts["lig"]),
            "ta": passive_stress(eps, ts["ta"]) + active_stress(eps, a_ta, ts["ta"])}


def layer_modulus(eps, a_ta: float = 0.0, tissue: str = "ts2002") -> dict:
    ts = TISSUE_SETS[tissue]
    return {"muc": passive_modulus(eps, ts["muc"]), "lig": passive_modulus(eps, ts["lig"]),
            "ta": passive_modulus(eps, ts["ta"]) + active_modulus(eps, a_ta, ts["ta"])}


@dataclass
class Lumped:
    """몸체–덮개 집중 매개변수 (SI). l = 아래 덮개, u = 위 덮개, b = 몸체. kc 는 아래–위 결합."""
    eps: float
    L: float
    T: float
    Tl: float
    Tu: float
    ml: float
    mu: float
    mb: float
    kl: float
    ku: float
    kc: float
    kb: float

    def matrices(self):
        M = np.diag([self.ml, self.mu, self.mb])
        K = np.array([[self.kl + self.kc, -self.kc, -self.kl],
                      [-self.kc, self.ku + self.kc, -self.ku],
                      [-self.kl, -self.ku, self.kl + self.ku + self.kb]])
        return M, K

    def modes(self):
        """고유진동수 [Hz] 와 모드 (열) — 선형, 흐름 없음."""
        from scipy.linalg import eigh
        M, K = self.matrices()
        w2, V = eigh(K, M)
        return np.sqrt(np.maximum(w2, 0.0)) / (2 * np.pi), V

    def cgs(self) -> dict:
        """`tube_td` 몸체–덮개(모드 2) 의 단위 — 길이 cm, 질량 g, 용수철 dyn/cm."""
        return dict(L=self.L * 100, d1=self.Tl * 100, d2=self.Tu * 100, m1=self.ml * 1e3, m2=self.mu * 1e3,
                    mb=self.mb * 1e3, k1=self.kl * 1e3, k2=self.ku * 1e3, kp=self.kc * 1e3, kb=self.kb * 1e3)


def ts_rules(eps: float, a_ta: float, sex: str = "female", tissue: str = "ts2002",
             mu_c: float = MU_COVER, mu_b: float = MU_BODY, scale: float = 1.0, lig_body: float = 0.5) -> Lumped:
    """Titze & Story 2002 규칙 (늘어남 ε 과 갑상피열근 활성 a_TA 를 받는다 — ε 은 자세 모형 또는 f0 에서).

    `CalcBodyCoverParameters.m` 그대로: 두께 T0/(1+0.8ε), 마디점 (1+a_TA)T/3, 깊이 (a_TA·D_mus + ½D_lig)/(1+0.2ε) ·
    (D_muc + ½D_lig)/(1+0.2ε), 층 응력의 깊이 가중 평균, k = 2μLT/D + π²σDT/L, m = ρLTD.
    """
    d = {k: v * scale for k, v in DIMS[sex].items()}
    st = layer_stress(eps, a_ta, tissue)
    Lg = d["L0"] * (1.0 + eps)
    Tg = d["T0"] / (1.0 + 0.8 * eps)
    Zn = (1.0 + a_ta) * Tg / 3.0
    Db = (a_ta * d["Dmus"] + lig_body * d["Dlig"]) / (1.0 + 0.2 * eps)
    Dc = (d["Dmuc"] + (1.0 - lig_body) * d["Dlig"]) / (1.0 + 0.2 * eps)
    sb = (lig_body * float(st["lig"]) * d["Dlig"] + float(st["ta"]) * d["Dmus"]) / max(Db, 1e-9)
    sc = ((1.0 - lig_body) * float(st["lig"]) * d["Dlig"] + float(st["muc"]) * d["Dmuc"]) / Dc
    z = Zn / Tg
    kl = 2 * mu_c * (Lg * Tg / Dc) * z + math.pi ** 2 * sc * (Dc / Lg) * Zn
    ku = 2 * mu_c * (Lg * Tg / Dc) * (1 - z) + math.pi ** 2 * sc * (Dc / Lg) * Tg * (1 - z)
    kc = (0.5 * mu_c * (Lg * Dc / Tg) / (1.0 / 3.0 - z * (1 - z)) - 2 * mu_c * (Lg * Tg / Dc)) * z * (1 - z)
    ml = RHO * Lg * Tg * Dc * z
    mu = RHO * Lg * Tg * Dc * (1 - z)
    kb = 2 * mu_b * (Lg * Tg / max(Db, 1e-9)) + math.pi ** 2 * sb * (Db / Lg) * Tg
    mb = RHO * Lg * Tg * Db
    return Lumped(eps, Lg, Tg, Zn, Tg - Zn, ml, mu, mb, kl, ku, kc, kb)


def string_f0(eps, sex: str = "female", tissue: str = "ts2002", layer: str = "cover") -> np.ndarray:
    """덮개 현의 진동수 (1/2L)√(σ/ρ) [Hz] — 비교용 (덮개 응력은 점막 + ½ 인대의 깊이 가중)."""
    d = DIMS[sex]
    e = np.asarray(eps, float)
    st = layer_stress(e, 0.0, tissue)
    sc = (0.5 * st["lig"] * d["Dlig"] + st["muc"] * d["Dmuc"]) / (d["Dmuc"] + 0.5 * d["Dlig"])
    return np.sqrt(np.maximum(sc, 0.0) / RHO) / (2 * d["L0"] * (1 + e))


def main() -> int:
    print("Titze–Story 규칙 (여성) — 선형 3 자유도 고유진동수 [Hz], 조직 판별")
    for tis in TISSUE_SETS:
        print(f"  [{tis}]")
        for a in (0.0, 0.3, 0.6):
            row = []
            for eps in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                f, _ = ts_rules(eps, a, tissue=tis).modes()
                row.append(f"{f[0]:6.0f}")
            print(f"    a_TA {a:.1f}: " + " ".join(row) + "   (ε −0.2 … 0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
