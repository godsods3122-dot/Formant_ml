p="src/formant_ml/engine/tube_td.py"; s=open(p,encoding="utf-8").read()
# 상수
a="JET_LEN_CM = 1.0\n"
b='''JET_LEN_CM = 1.0
#: 제트가 보는 넓어진 면적을 **매끈하게** (§52.523). 발달 거리 안 칸들의 최댓값(argmax)은 넓이가 비슷한 두 칸 사이에서 손실을 꺾고(수반 대 수치 미분
#: 0.2 % 어긋남 — 26·27 칸이 같은 3.0 cm² 인 시험 배치), 거리를 칸 수로 반올림해 길이에 대해 계단이었다. 물리로도 제트는 정해진 거리에서 끊기지
#: 않고 점점 퍼진다 — 거리 d 의 코사인 무게 w = ½(1 + cos(π d / JET_LEN_CM)) (첫 칸 1) 와 무게 준 로그–합–지수 (날카로움 `JET_SOFT_BETA` cm⁻²):
#: 넓이가 같으면 그 값, 발달 거리 0 이면 옆 칸 그대로.
JET_SOFT_BETA = 3.0
'''
assert s.count(a)==1; s=s.replace(a,b,1)
# 함수 교체
i0=s.index("@njit(cache=True)\ndef _jet_far(")
i1=s.index("@njit(cache=True)\ndef _consts(")
new='''@njit(cache=True)
def _jet_far(A, n, i, us, a1, a2, N, dx, G):
    """제트가 퍼져 나가는 넓어진 면적 (매끈, §52.523) — 흐름 방향 `JET_LEN_CM` 안의 코사인 무게 로그–합–지수.
    반환 (amx, ∂amx/∂dx, 첫 칸, 방향, 칸 수); G[s] = ∂amx/∂a_(첫 칸 + s·방향)."""
    if us > 0.0:
        j0 = i + 1
        dr = 1
        a0 = a2
    else:
        j0 = i
        dr = -1
        a0 = a1
    num = 0.0
    den = 0.0
    dnum = 0.0
    dden = 0.0
    ns = 0
    for s_ in range(N):
        j = j0 + s_ * dr
        if j < 0 or j >= N:
            break
        d = s_ * dx
        if s_ > 0 and d >= JET_LEN_CM:
            break
        if s_ == 0:
            w = 1.0
            dw = 0.0
            aj = a0
        else:
            w = 0.5 * (1.0 + math.cos(math.pi * d / JET_LEN_CM))
            dw = -0.5 * math.sin(math.pi * d / JET_LEN_CM) * math.pi * s_ / JET_LEN_CM      # ∂w/∂dx
            aj = max(A[n, j], A_FLOOR)
        e = w * math.exp(JET_SOFT_BETA * (aj - a0))
        G[s_] = e
        num += e
        den += w
        dnum += dw * (e / w if w > 0.0 else 0.0)
        dden += dw
        ns = s_ + 1
    amx = a0 + math.log(num / den) / JET_SOFT_BETA
    for s_ in range(ns):
        G[s_] = G[s_] / num
    damx_dx = (dnum / num - dden / den) / JET_SOFT_BETA
    return amx, damx_dx, j0, dr, ns


'''
s=s[:i0]+new+s[i1:]
# 앞 방향 호출
a='''                if JET_BC:
                    amx_, jm_ = _jet_far(A, n, i, us, a1, a2, N, dx)
                    rj = amin / amx_'''
b='''                if JET_BC:
                    amx_, _dxj, _j0, _dr, _ns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                    rj = amin / amx_'''
assert s.count(a)==1; s=s.replace(a,b,1)
# 수반 호출
a='''                ej = 1.0
                amx = max(a1, a2)
                jm = i + 1 if a2 >= a1 else i
                if JET_BC:
                    amx, jm = _jet_far(A, n, i, us, a1, a2, N, dx)
                rj = amin / amx'''
b='''                ej = 1.0
                amx = max(a1, a2)
                jdx = 0.0
                j0 = i + 1 if a2 >= a1 else i
                jdr = 1
                jns = 0
                if JET_BC:
                    amx, jdx, j0, jdr, jns = _jet_far(A, n, i, us, a1, a2, N, dx, JG)
                rj = amin / amx'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''                if jm == i:
                    la1 += lamx
                elif jm == i + 1:
                    la2 += lamx
                elif A[n, jm] > A_FLOOR:
                    gA[n, jm] += lamx'''
b='''                if JET_BC:                                   # 매끈한 넓어진 면적의 몫을 창 안 칸마다 (§52.523)
                    for s_ in range(jns):
                        j = j0 + s_ * jdr
                        gj = lamx * JG[s_]
                        if j == i:
                            la1 += gj
                        elif j == i + 1:
                            la2 += gj
                        elif A[n, j] > A_FLOOR:
                            gA[n, j] += gj
                    ldx += lamx * jdx'''
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
