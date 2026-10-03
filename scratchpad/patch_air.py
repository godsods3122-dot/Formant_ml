p="src/formant_ml/engine/tube_td.py"; s=open(p,encoding="utf-8").read()
a='''C_SOUND = 35000.0        # cm/s (37 °C 습한 공기)
RHO = 1.14e-3            # g/cm³
MU = 1.86e-4             # dyn·s/cm²
NU = MU / RHO            # cm²/s
GAMMA = 1.4
LAMBDA_TH = 2.30e3
CP = 1.00e7
'''
b='''#: **공기 물성 — 날숨의 조성과 온도에서** (§52.523, `physics/air.py`). 예전 상수는 말소리 문헌의 관례값(음속 350 m/s, 비열비 1.4, 열전도
#: 0.023 W/m·K, 정압 비열 1000 J/kg·K)이었다. 날숨(건조 기준 CO2 4 %, 포화 수증기)을 성도 평균 온도 `AIR_T_C` 에서 계산하면 음속 +0.5 %,
#: 열전도 +11 %, 비열비 −0.8 %, 점성 −2.6 %. 성문(≈36.5 °C) → 입술(≈34.5 °C) 기울기가 평균 대비 포먼트를 옮기는 몫은 F1 에서 ~0.1 % 라 성도는
#: 평균 하나로 둔다 (비강·기관의 영역별 값은 따로). 바꾸려면 첫 커널 호출 전에 `set_air` 를 부른다 (numba 에 굳는 상수).
AIR_T_C = 35.5
AIR_RH = 1.0
from ..physics import air as _air_mod
_AIR = _air_mod.props(AIR_T_C, AIR_RH).cgs()
C_SOUND = _AIR["C_SOUND"]        # cm/s
RHO = _AIR["RHO"]                # g/cm³
MU = _AIR["MU"]                  # dyn·s/cm²
NU = MU / RHO                    # cm²/s
GAMMA = _AIR["GAMMA"]
LAMBDA_TH = _AIR["LAMBDA_TH"]
CP = _AIR["CP"]


def set_air(T_c: float = 35.5, rh: float = 1.0, co2_dry: float | None = None) -> dict:
    """공기 물성을 바꾼다 (첫 커널 호출 전에만 — numba 상수). 반환: 새 CGS 값."""
    global C_SOUND, RHO, MU, NU, GAMMA, LAMBDA_TH, CP, AIR_T_C, AIR_RH
    kw = {} if co2_dry is None else {"co2_dry": co2_dry}
    v = _air_mod.props(T_c, rh, **kw).cgs()
    C_SOUND, RHO, MU, GAMMA, LAMBDA_TH, CP = (v[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
    NU = MU / RHO
    AIR_T_C, AIR_RH = T_c, rh
    return v
'''
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
