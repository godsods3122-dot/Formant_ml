import re
p="src/formant_ml/engine/tube_td.py"; s=open(p,encoding="utf-8").read()
def rep(a,b,cnt):
    global s
    n=s.count(a); assert n==cnt,(a[:60],n,cnt); s=s.replace(a,b)
# 1. 상수 · set_air
rep('''CP = _AIR["CP"]


def set_air(''','''CP = _AIR["CP"]
#: 영역별 공기 (§52.524). **비강**: 날숨에서 비갑개(아래·중간)가 공기를 식혀 열을 되찾는다 — 상피 표면 비인두 34.3 · 전정 33.5 °C (Pless et al. 2004
#: 날숨 수치 모의, Keck · Lindemann 계열), 비강 공기 평균 `AIR_T_NASAL_C`. **기관**: 심부 체온 37 °C. 부비동 곁관은 비강, 이상와·후두실은 성도 물성.
AIR_T_NASAL_C = 34.5
AIR_T_TRACHEA_C = 37.0
_AIRN = _air_mod.props(AIR_T_NASAL_C, AIR_RH).cgs()
_AIRT = _air_mod.props(AIR_T_TRACHEA_C, AIR_RH).cgs()
C_N, RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N = (_AIRN[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
C_T, RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T = (_AIRT[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))


def set_air(''',1)
rep('''    NU = MU / RHO
    AIR_T_C, AIR_RH = T_c, rh
    return v''','''    NU = MU / RHO
    AIR_T_C, AIR_RH = T_c, rh
    return v


def set_air_regions(T_nasal: float = 34.5, T_trachea: float = 37.0, rh: float = 1.0) -> None:
    """비강·기관의 공기 물성 (첫 커널 호출 전에만)."""
    global C_N, RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N, C_T, RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T, AIR_T_NASAL_C, AIR_T_TRACHEA_C
    vn, vt = _air_mod.props(T_nasal, rh).cgs(), _air_mod.props(T_trachea, rh).cgs()
    C_N, RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N = (vn[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
    C_T, RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T = (vt[k] for k in ("C_SOUND", "RHO", "MU", "GAMMA", "LAMBDA_TH", "CP"))
    AIR_T_NASAL_C, AIR_T_TRACHEA_C = T_nasal, T_trachea''',1)
# 2. 경계층 계수 도우미
rep('''@njit(cache=True)
def _consts(fs):''','''@njit(cache=True)
def _kvkt(rho, mu, gamma, lam, cp):
    """경계층 손실 계수 (점성 Kv, 열 Kt) — 영역 물성으로 (§52.524). `_consts` 와 같은 꼴."""
    return (LOSS_WET * math.sqrt(rho * mu / 2.0) * math.sqrt(REF_W),
            LOSS_WET * (gamma - 1.0) * math.sqrt(lam / (2.0 * rho * cp)) * math.sqrt(REF_W))


@njit(cache=True)
def _consts(fs):''',1)
# 3. 두 커널: 상수 받은 뒤 영역 계수
rep('''    kc, kg, Kv, Kt = _consts(fs)
''','''    kc, kg, Kv, Kt = _consts(fs)
    KvN, KtN = _kvkt(RHO_N, MU_N, GAMMA_N, LAMBDA_N, CP_N)
    KvT, KtT = _kvkt(RHO_T, MU_T, GAMMA_T, LAMBDA_T, CP_T)
    rc2N = RHO_N * C_N * C_N
    rc2T = RHO_T * C_T * C_T
''',2)
# 4. 기관
rep("LgT = RHO * 0.5 * dxt / TRA[NT - 1] if NT > 0 else 0.0","LgT = RHO_T * 0.5 * dxt / TRA[NT - 1] if NT > 0 else 0.0",2)
rep("Jt = RHO * 0.5 * dxt / b2 / dt","Jt = RHO_T * 0.5 * dxt / b2 / dt",2)
rep("rl = RHO * C_SOUND / b2","rl = RHO_T * C_T / b2",2)
rep("Jt = 0.5 * RHO * dxt * (1.0 / b1 + 1.0 / b2) / dt","Jt = 0.5 * RHO_T * dxt * (1.0 / b1 + 1.0 / b2) / dt",2)
rep("rft = (2.0 * math.sqrt(math.pi * ah) * dxt / (ah * ah)) * Kv if loss_on else 0.0","rft = (2.0 * math.sqrt(math.pi * ah) * dxt / (ah * ah)) * KvT if loss_on else 0.0",2)
rep("r0t = 8.0 * math.pi * MU * dxt / (ah * ah) if loss_on else 0.0","r0t = 8.0 * math.pi * MU_T * dxt / (ah * ah) if loss_on else 0.0",2)
rep("C = bj * dxt / rc2 / dt","C = bj * dxt / rc2T / dt",2)
rep("g = (S * dxt / rc2) * Kt if loss_on else 0.0","g = (S * dxt / rc2T) * KtT if loss_on else 0.0",2)
# 5. 비강
rep("Jn = 0.5 * RHO * dxn * (1.0 / b1 + 1.0 / b2) / dt","Jn = 0.5 * RHO_N * dxn * (1.0 / b1 + 1.0 / b2) / dt",2)
rep("rfn = NAS_LOSS * (2.0 * math.sqrt(math.pi * ah) * dxn / (ah * ah)) * Kv if loss_on else 0.0","rfn = NAS_LOSS * (2.0 * math.sqrt(math.pi * ah) * dxn / (ah * ah)) * KvN if loss_on else 0.0",2)
rep("r0n = 8.0 * math.pi * MU * dxn / (ah * ah) if loss_on else 0.0","r0n = 8.0 * math.pi * MU_N * dxn / (ah * ah) if loss_on else 0.0",2)
rep("RrN = 128.0 * RHO * C_SOUND / (9.0 * math.pi * math.pi * an)","RrN = 128.0 * RHO_N * C_N / (9.0 * math.pi * math.pi * an)",2)
rep("LrN = 8.0 * RHO / (3.0 * math.pi * math.sqrt(math.pi * an))","LrN = 8.0 * RHO_N / (3.0 * math.pi * math.sqrt(math.pi * an))",2)
rep("IN_ = RHO * 0.5 * dxn / an / dt","IN_ = RHO_N * 0.5 * dxn / an / dt",2)
rep("C = bj * dxn / rc2 / dt","C = bj * dxn / rc2N / dt",2)
rep("g = NAS_LOSS * (S * dxn / rc2) * Kt if loss_on else 0.0","g = NAS_LOSS * (S * dxn / rc2N) * KtN if loss_on else 0.0",2)
# 6. 곁관 이음매 (부비동은 비강 물성)
rep('''                Jp = RHO * h_ * ln_ / an_ / dt + (RHO * 0.5 * dxa / aip / dt if jj == 0 else 0.0)
                rfp = (2.0 * math.sqrt(math.pi * an_) * h_ * ln_ / (an_ * an_)) * Kv if loss_on else 0.0
                r0p = 8.0 * math.pi * MU * h_ * ln_ / (an_ * an_) if loss_on else 0.0''','''                rho_b = RHO_N if nas else RHO                  # 부비동은 비강 공기, 후두 곁관은 성도 공기 (§52.524)
                Jp = rho_b * h_ * ln_ / an_ / dt + (rho_b * 0.5 * dxa / aip / dt if jj == 0 else 0.0)
                rfp = (2.0 * math.sqrt(math.pi * an_) * h_ * ln_ / (an_ * an_)) * (KvN if nas else Kv) if loss_on else 0.0
                r0p = 8.0 * math.pi * (MU_N if nas else MU) * h_ * ln_ / (an_ * an_) if loss_on else 0.0''',2)
# 7. 곁관 칸
rep('''            dxp = SBL[bb] / NPB
            for jj in range(NPB):
                j = bb * NPB + jj
                C = PA * dxp / rc2 / dt
                g = (2.0 * math.sqrt(math.pi * PA) * dxp / rc2) * Kt if loss_on else 0.0''','''            dxp = SBL[bb] / NPB
            nasb = SBI[bb] >= N
            rc2b = rc2N if nasb else rc2
            ktb = KtN if nasb else Kt
            for jj in range(NPB):
                j = bb * NPB + jj
                C = PA * dxp / rc2b / dt
                g = (2.0 * math.sqrt(math.pi * PA) * dxp / rc2b) * ktb if loss_on else 0.0''',2)
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
