p = "src/formant_ml/engine/tube_td.py"
s = open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    assert n == cnt, (a[:80], n, cnt)
    s = s.replace(a, b)


def indent(block, k=4):
    return "\n".join((" " * k + ln) if ln.strip() else ln for ln in block.split("\n"))


# ---- 상수 + 세기 함수
rep("GLOT_NOISE_JET = True\n", '''GLOT_NOISE_JET = True
#: **성문 제트를 성대 길이 방향으로** (§52.524). 사용자: *"성대 중앙부에서 공기가 나가지만 가장자리는 기류가 상대적으로 느린 것"*. 예전 세기는 1 ms 로
#: 고른 유량을 그 순간의 면적으로 나눈 속도 하나였다 — 닫히는 순간 면적은 줄고 고른 유량은 뒤처져 속도가 치솟아 잡음이 닫힘마다 터졌다. 새 꼴:
#: 그 순간의 유량·면적에서 준정상 압력 강하 Δp = K ρu²/(2A²) + 12μd ℓ² u/A³ (흐름 식과 같은 꼴), 틈을 길이 방향으로 h(s) ∝ sin(πs) (막 첫 모드, ∫ = A/ℓ)
#: 로 `GN_PTS` 점에 두고 점마다 ½Kρv² + 12μd v/h² = Δp 를 풀어 그 자리 속도 — 좁은 가장자리는 점성으로 느리고 가운데가 빠르다. 점마다 Stevens 세기
#: ρv²(v/V_REF)√(a/A_REF) × 제트 도달 몫 × 레이놀즈 전이 (2vh/ν), 점들은 서로 무관한 음원 (세기 제곱합). 1 ms 난류 발달 평활은 유량이 아니라 이 세기에.
GLOT_NOISE_3D = True
GN_PTS = 8
''')
rep('''@njit(cache=True)
def _spos(x, e):''', '''@njit(cache=True)
def _gq(u, ag, lre):
    """성문 제트 난류 세기 (§52.524, `GLOT_NOISE_3D`) — 성대 길이 방향 점들의 Stevens 세기 제곱합의 제곱근 [동압 단위]."""
    au = abs(u)
    if au <= 1e-12 or ag <= 1e-9:
        return 0.0
    k2 = 0.5 * GLOTTIS_KE * RHO
    dp = k2 * au * au / (ag * ag) + 12.0 * MU * GLOTTIS_D * lre * lre * au / (ag * ag * ag)
    hpk = 0.5 * math.pi * ag / lre
    tot = 0.0
    for j in range(GN_PTS):
        h = hpk * math.sin(math.pi * (j + 0.5) / GN_PTS)
        c1 = 12.0 * MU * GLOTTIS_D / (h * h)
        v = 2.0 * dp / (c1 + math.sqrt(c1 * c1 + 4.0 * k2 * dp))          # ½Kρv² + c1 v = Δp 의 양근
        Re = 2.0 * v * h / NU
        if Re <= 0.0:
            continue
        g, _dg = _spos(1.0 - (RE_CRIT / Re) ** 2, TURB_GATE_W)
        if g <= 0.0:
            continue
        fj = (1.0 - math.exp(-GLOT_JET_CORE * h / GLOT_JET_D)) if GLOT_NOISE_JET else 1.0
        q = RHO * v * v * (v / V_REF) * math.sqrt(h * lre / GN_PTS / A_REF) * fj * g
        tot += q * q
    return math.sqrt(tot)


@njit(cache=True)
def _gq_d(u, ag, lre):
    """`_gq` 와 그 편미분 (∂/∂u, ∂/∂A, ∂/∂ℓ) — 매끈한 함수라 상대 보폭 1e-6 의 중앙 차분."""
    q0 = _gq(u, ag, lre)
    hu = 1e-6 * max(abs(u), 1e-3)
    ha = 1e-6 * ag
    hl = 1e-6 * lre
    du = (_gq(u + hu, ag, lre) - _gq(u - hu, ag, lre)) / (2.0 * hu)
    da = (_gq(u, ag + ha, lre) - _gq(u, ag - ha, lre)) / (2.0 * ha)
    dl = (_gq(u, ag, lre + hl) - _gq(u, ag, lre - hl)) / (2.0 * hl)
    return q0, du, da, dl


@njit(cache=True)
def _spos(x, e):''')
# ---- 앞 방향
a0 = "        ugs = TUgs[n] + kg * (abs(ug0) - TUgs[n])\n"
a1 = "            pn_g = noise_g * q * hgg * xi_g[n]\n"
i0 = s.index(a0)
i1 = s.index(a1) + len(a1)
old = s[i0:i1]
new = '''        if GLOT_NOISE_3D:
            Lre = Lc if (VF and RE_USE_LC) else GLOTTIS_LEN
            ugs = TUgs[n] + kg * (_gq(ug0, ag, Lre) - TUgs[n])         # 1 ms 로 고른 길이 방향 제트 세기 (§52.524)
            TUgs[n + 1] = ugs
            Re_g = 2.0 * abs(ug0) / (Lre * NU)
            pn_g = noise_g * ugs * xi_g[n]
        else:
''' + indent(old.rstrip("\n")) + "\n"
s = s[:i0] + new + s[i1:]
# ---- 수반
b0 = "        ugs = TUgs[n + 1]\n        lugs = lUgs\n"
b1 = "        mUg += lugs * kg * (1.0 if ug0 >= 0.0 else -1.0)\n"
j0 = s.index(b0)
j1 = s.index(b1, j0) + len(b1)
oldb = s[j0:j1]
newb = '''        if GLOT_NOISE_3D:
            ugs = TUgs[n + 1]
            Lre = Lc if (VF and RE_USE_LC) else GLOTTIS_LEN
            x = xi_g[n]
            gng[0] += lpng * ugs * x
            lugs = lUgs + lpng * noise_g * x
            q0_, dqu, dqa, dql = _gq_d(ug0, ag, Lre)
            lq = lugs * kg
            mUg += lq * dqu
            lag += lq * dqa
            if VF and RE_USE_LC:
                lLc += lq * dql
            mUgs += lugs * (1.0 - kg)
        else:
''' + indent(oldb.rstrip("\n")) + "\n"
s = s[:j0] + newb + s[j1:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
