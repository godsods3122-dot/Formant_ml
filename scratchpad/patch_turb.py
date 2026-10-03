p = "src/formant_ml/engine/tube_td.py"
s = open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (a[:80], n, cnt)
    s = s.replace(a, b)


rep("JET_SOFT_BETA = 3.0\n", '''JET_SOFT_BETA = 3.0
#: **난류 음원을 매끈하게** (§52.524). 레이놀즈 문턱 (1 − (Re_c/Re)²)₊, 동압 상한 min(ρv², 2P_s), 제트 판정의 참/거짓 스위치(a₂ > a₁), 좁은 쪽 min(a₁, a₂) 이
#: 면적에 대해 꺾여, 실제 판의 관 입력에서 면적을 흔든 차분이 보폭에 따라 흔들렸다 (잡음을 끄면 수렴). 난류 전이는 날카로운 문턱이 아니라 전이 구간이다:
#: 문턱은 폭 `TURB_GATE_W` 의 C¹ 양수부, 상한은 폭 `TURB_CAP_W` (상한 대비) 의 매끈한 최소, 제트 세기는 흐름 방향 위쪽 칸(목) 면적 a_위 와 아래쪽 넓어진
#: 면적 a_넓 의 (1 − a_위/a_넓)₊² — 스위치 없이 넓어짐이 사라지면 0 으로 간다. 성문 난류에도 문턱·상한을 같은 꼴로.
TURB_SMOOTH = True
TURB_GATE_W = 0.05
TURB_CAP_W = 0.05
''')
rep('''@njit(cache=True)
def _kvkt(rho, mu, gamma, lam, cp):''', '''@njit(cache=True)
def _spos(x, e):
    """C¹ 양수부 — (값, 도함수). |x| < e 에서 (x + e)²/(4e) (§52.524)."""
    if x >= e:
        return x, 1.0
    if x <= -e:
        return 0.0, 0.0
    return (x + e) * (x + e) / (4.0 * e), (x + e) / (2.0 * e)


@njit(cache=True)
def _smin(a, b, w):
    """매끈한 최소 (a, b) — (값, ∂/∂a, ∂/∂b). 폭 e = w·|b| 안에서 |a − b| 를 이차로 (§52.524)."""
    e = w * abs(b) + 1e-30
    d = a - b
    if d >= e:
        return b, 0.0, 1.0
    if d <= -e:
        return a, 1.0, 0.0
    sa = d * d / (2.0 * e) + 0.5 * e
    dsd = d / e
    dse = -d * d / (2.0 * e * e) + 0.5
    return 0.5 * (a + b - sa), 0.5 * (1.0 - dsd), 0.5 * (1.0 + dsd - dse * w * (1.0 if b >= 0.0 else -1.0))


@njit(cache=True)
def _kvkt(rho, mu, gamma, lam, cp):''')
# ---- 협착 앞 방향
rep('''            Re = abs(us) * 2.0 / (math.sqrt(math.pi * amin) * NU)
            if Re > remax:
                remax = Re
            pn = 0.0''', '''            aup = a1 if us >= 0.0 else a2                # 흐름 방향 위쪽 칸 — 제트가 나오는 목 (§52.524)
            Re = abs(us) * 2.0 / (math.sqrt(math.pi * (aup if TURB_SMOOTH else amin)) * NU)
            if Re > remax:
                remax = Re
            pn = 0.0''')
rep('''            if Re > RE_CRIT and jet:
                v = us / amin
                gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                av = abs(v)
                dyn = min(RHO * av * av, 2.0 * abs(Ps[n]))
                q = dyn * av * math.sqrt(amin) * gob / (V_REF * math.sqrt(A_REF))
                if JET_BC:
                    amx_, _dxj, _j0, _dr, _ns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)''', '''            if TURB_SMOOTH and JET_BC:
                hg = 0.0
                if Re > 0.0:
                    hg, _dh = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
                if hg > 0.0:
                    v = us / aup
                    gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                    av = abs(v)
                    dyn, _d1, _d2 = _smin(RHO * av * av, 2.0 * abs(Ps[n]), TURB_CAP_W)
                    q = dyn * av * math.sqrt(aup) * gob / (V_REF * math.sqrt(A_REF))
                    amx_, _dxj, _j0, _dr, _ns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                    rj = aup / amx_
                    if rj < 1.0:
                        pn = noise_c * q * (1.0 - rj) * (1.0 - rj) * hg * xi_c[n, i]
            elif Re > RE_CRIT and jet:
                v = us / amin
                gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                av = abs(v)
                dyn = min(RHO * av * av, 2.0 * abs(Ps[n]))
                q = dyn * av * math.sqrt(amin) * gob / (V_REF * math.sqrt(A_REF))
                if JET_BC:
                    amx_, _dxj, _j0, _dr, _ns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)''')
# ---- 협착 수반
rep('''            if JET_BC:
                jet = (us > 0.0 and a2 > a1) or (us < 0.0 and a1 > a2)
            else:
                jet = (us > 0.0 and a2 > 1.2 * a1) or (us < 0.0 and a1 > 1.2 * a2)
            if Re > RE_CRIT and jet:
                v = us / amin
                gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                av = abs(v)
                kq = gob / (V_REF * math.sqrt(A_REF))''', '''            if JET_BC:
                jet = (us > 0.0 and a2 > a1) or (us < 0.0 and a1 > a2)
            else:
                jet = (us > 0.0 and a2 > 1.2 * a1) or (us < 0.0 and a1 > 1.2 * a2)
            if TURB_SMOOTH and JET_BC:
                aup = a1 if us >= 0.0 else a2
                Re = abs(us) * 2.0 / (math.sqrt(math.pi * aup) * NU)
                hg = 0.0
                dhg = 0.0
                if Re > 0.0:
                    hg, dhg = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
                if hg > 0.0:
                    v = us / aup
                    gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                    av = abs(v)
                    kq = gob / (V_REF * math.sqrt(A_REF))
                    rv = RHO * av * av
                    ps2 = 2.0 * abs(Ps[n])
                    dyn, ddr, ddp = _smin(rv, ps2, TURB_CAP_W)
                    S = av * math.sqrt(aup) * kq
                    q = dyn * S
                    amx, jdx, j0, jdr, jns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                    rj = aup / amx
                    if rj < 1.0:
                        ej = (1.0 - rj) * (1.0 - rj)
                        x = xi_c[n, i]
                        gng[1] += lpn * q * ej * hg * x
                        L0 = lpn * noise_c * x
                        lq = L0 * ej * hg
                        lej = L0 * q * hg
                        lhg = L0 * q * ej
                        lS = lq * dyn
                        ldyn = lq * S
                        lv = lS * (1.0 if v >= 0.0 else -1.0) * math.sqrt(aup) * kq
                        laup = lS * av * kq * 0.5 / math.sqrt(aup)
                        lv += ldyn * ddr * 2.0 * RHO * v
                        gPs[n] += ldyn * ddp * 2.0 * (1.0 if Ps[n] >= 0.0 else -1.0)
                        lRe = lhg * dhg * 2.0 * RE_CRIT * RE_CRIT / (Re * Re * Re)
                        lus += lRe * Re / us + lv / aup
                        laup += lRe * (-0.5 * Re / aup) + lv * (-v / aup)
                        lrj = lej * (-2.0 * (1.0 - rj))
                        laup += lrj / amx
                        lamx = lrj * (-rj / amx)
                        if us >= 0.0:
                            la1 += laup
                        else:
                            la2 += laup
                        for s_ in range(jns):
                            j = j0 + s_ * jdr
                            gj = lamx * JG[s_]
                            if j == i:
                                la1 += gj
                            elif j == i + 1:
                                la2 += gj
                            elif A[n, j] > A_FLOOR:
                                gA[n, j] += gj
                        ldx += lamx * jdx
            elif Re > RE_CRIT and jet:
                v = us / amin
                gob = 1.0 + OBS_GAIN if (i + 1) >= OBS_FRAC * N else 1.0
                av = abs(v)
                kq = gob / (V_REF * math.sqrt(A_REF))''')
# ---- 성문 앞 방향
rep('''        pn_g = 0.0
        if Re_g > RE_CRIT:
            vg = ugs / ag
            q = min(RHO * vg * vg, 2.0 * abs(Ps[n]))''', '''        pn_g = 0.0
        hgg = 0.0
        if TURB_SMOOTH:
            if Re_g > 0.0:
                hgg, _dhg = _spos(1.0 - (RE_CRIT / Re_g) ** 2, TURB_GATE_W)
        elif Re_g > RE_CRIT:
            hgg = 1.0 - (RE_CRIT / Re_g) ** 2
        if hgg > 0.0:
            vg = ugs / ag
            if TURB_SMOOTH:
                q, _d1, _d2 = _smin(RHO * vg * vg, 2.0 * abs(Ps[n]), TURB_CAP_W)
            else:
                q = min(RHO * vg * vg, 2.0 * abs(Ps[n]))''')
rep('''            pn_g = noise_g * q * (1.0 - (RE_CRIT / Re_g) ** 2) * xi_g[n]''', '''            pn_g = noise_g * q * hgg * xi_g[n]''')
# ---- 성문 수반
rep('''        Re_g = 2.0 * ugs / (Lre * NU)
        if Re_g > RE_CRIT:
            vg = ugs / ag
            rv = RHO * vg * vg
            ps2 = 2.0 * abs(Ps[n])
            q0 = min(rv, ps2)''', '''        Re_g = 2.0 * ugs / (Lre * NU)
        hgg = 0.0
        dhgg = 0.0
        if TURB_SMOOTH:
            if Re_g > 0.0:
                hgg, dhgg = _spos(1.0 - (RE_CRIT / Re_g) ** 2, TURB_GATE_W)
        elif Re_g > RE_CRIT:
            hgg = 1.0 - (RE_CRIT / Re_g) ** 2
            dhgg = 1.0
        if hgg > 0.0:
            vg = ugs / ag
            rv = RHO * vg * vg
            ps2 = 2.0 * abs(Ps[n])
            if TURB_SMOOTH:
                q0, ddr, ddp = _smin(rv, ps2, TURB_CAP_W)
            else:
                q0 = min(rv, ps2)
                ddr = 1.0 if rv < ps2 else 0.0
                ddp = 0.0 if rv < ps2 else 1.0''')
rep('''            q = q0 * sc * fj
            h = 1.0 - (RE_CRIT / Re_g) ** 2
            x = xi_g[n]''', '''            q = q0 * sc * fj
            h = hgg
            x = xi_g[n]''')
rep('''            if rv < ps2:
                lvg += lq * sc * 2.0 * RHO * vg
            else:
                gPs[n] += lq * sc * 2.0 * (1.0 if Ps[n] >= 0.0 else -1.0)
            lRe = lh * 2.0 * RE_CRIT * RE_CRIT / (Re_g * Re_g * Re_g)''', '''            lvg += lq * sc * ddr * 2.0 * RHO * vg
            gPs[n] += lq * sc * ddp * 2.0 * (1.0 if Ps[n] >= 0.0 else -1.0)
            lRe = lh * dhgg * 2.0 * RE_CRIT * RE_CRIT / (Re_g * Re_g * Re_g)''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
