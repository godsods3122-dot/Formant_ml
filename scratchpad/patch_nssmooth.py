p="src/formant_ml/engine/tube_td.py"; s=open(p,encoding="utf-8").read()
# 상수
a="VF_SEP_P = 6.0\n"
b='''VF_SEP_P = 6.0
#: **점막 닫힘을 매끈하게** (§52.523). 성대 길이 16 점의 양수부 max(h, 0) 은 점이 하나씩 닫힐 때마다 면적 기울기를 꺾었고, 닿음 경사
#: clamp(h/EDGE + ½) 는 양 끝에서 꺾였고, 흐름 분리는 가장 좁은 줄의 hard min 과 달리는 최댓값이라 그 줄이 바뀔 때 꺾였다 — 손실이 제어에 대해
#: 매끈하지 않았다 (몸체 구동판의 기울기 점검이 차분 보폭에 따라 뒤집혔다). 점막 표면의 점액막·무른 상피는 닿음을 `VF_NS_EDGE` (50 μm) 폭에 걸쳐
#: 점진적으로 만든다: 양수부는 그 폭의 C¹ 이차 이음 (h+e)²/(4e), 닿음 무게는 smoothstep, 분리 기준 면적은 줄들의 매끈한 최소, 분리 몫은
#: χ_k = Π_{j≤k} (1 − w_j) (w 가 0/1 이면 1 − max 와 같다), w 는 smoothstep. 거짓이면 예전 꼴.
VF_NS_SMOOTH = True
'''
assert s.count(a)==1; s=s.replace(a,b,1)
# _ns_geom
a='''    for i in range(VF_NS_PTS):
        sv = (i + 0.5) * ds
        sn = math.sin(math.pi * sv)
        h = r * sv + x * sn
        if h > 0.0:
            a += h
            dar += sv
            dax += sn
        w = min(max(h / VF_NS_EDGE + 0.5, 0.0), 1.0)
        o += w * sn'''
b='''    e2 = 0.5 * VF_NS_EDGE
    for i in range(VF_NS_PTS):
        sv = (i + 0.5) * ds
        sn = math.sin(math.pi * sv)
        h = r * sv + x * sn
        if VF_NS_SMOOTH:
            if h >= e2:
                a += h
                dar += sv
                dax += sn
            elif h > -e2:                               # 점액막 폭의 C¹ 이음 (§52.523)
                a += (h + e2) * (h + e2) / (4.0 * e2)
                sp1 = (h + e2) / (2.0 * e2)
                dar += sp1 * sv
                dax += sp1 * sn
            u = min(max(h / VF_NS_EDGE + 0.5, 0.0), 1.0)
            w = u * u * (3.0 - 2.0 * u)
        else:
            if h > 0.0:
                a += h
                dar += sv
                dax += sn
            w = min(max(h / VF_NS_EDGE + 0.5, 0.0), 1.0)
        o += w * sn'''
assert s.count(a)==1; s=s.replace(a,b,1)
# _ns_geom_d
a='''        u = h / VF_NS_EDGE + 0.5
        w = min(max(u, 0.0), 1.0)
        wp = 1.0 / VF_NS_EDGE if (u > 0.0 and u < 1.0) else 0.0'''
b='''        u = h / VF_NS_EDGE + 0.5
        if VF_NS_SMOOTH:
            uc = min(max(u, 0.0), 1.0)
            w = uc * uc * (3.0 - 2.0 * uc)
            wp = 6.0 * uc * (1.0 - uc) / VF_NS_EDGE
        else:
            w = min(max(u, 0.0), 1.0)
            wp = 1.0 / VF_NS_EDGE if (u > 0.0 and u < 1.0) else 0.0'''
assert s.count(a)==1; s=s.replace(a,b,1)
# 앞 방향: 분리
a='''            mw = 0.0
            for k in range(NS3 + 1):
                if k < NS3:
                    wj = min(max(1.0 - (G3[k] - agm) / (VF_NS_SEP * agm), 0.0), 1.0)
                    if wj > mw:
                        mw = wj
                    rr = agm / G3[k]
                    pk_ = p0 + (ps - p0) * (1.0 - rr * rr) * (1.0 - mw)'''
b='''            mw = 0.0
            chi_ = 1.0
            agq = ag if (VF_NS_SMOOTH and VF_SEP_P > 0.0) else agm      # 분리 기준 = 매끈한 최소 (§52.523)
            for k in range(NS3 + 1):
                if k < NS3:
                    if VF_NS_SMOOTH:
                        zc = min(max(1.0 - (G3[k] - agq) / (VF_NS_SEP * agq), 0.0), 1.0)
                        chi_ *= 1.0 - zc * zc * (3.0 - 2.0 * zc)
                        rr = agq / G3[k]
                        pk_ = p0 + (ps - p0) * (1.0 - rr * rr) * chi_
                    else:
                        wj = min(max(1.0 - (G3[k] - agm) / (VF_NS_SEP * agm), 0.0), 1.0)
                        if wj > mw:
                            mw = wj
                        rr = agm / G3[k]
                        pk_ = p0 + (ps - p0) * (1.0 - rr * rr) * (1.0 - mw)'''
assert s.count(a)==1; s=s.replace(a,b,1)
# 수반: 분리
a='''            # 분리 몫 χ_k = 1 − max_{j≤k} w_j 를 다시 세운다 (앞으로): 달리는 최대와 그 자리
            mw = 0.0
            jm = -1
            for k in range(NS3):
                wj = min(max(1.0 - (G3[k] - agm) / (VF_NS_SEP * agm), 0.0), 1.0)
                W3[k] = wj
                if wj > mw:
                    mw = wj
                    jm = k
                MW3[k] = mw
                JM3[k] = jm
            for k in range(NS3):
                lF = T2 * MU3[k]
                rr = agm / G3[k]
                chi = 1.0 - MW3[k]'''
b='''            # 분리 몫 χ_k 를 다시 세운다 (앞으로). 예전 꼴: 1 − max_{j≤k} w_j (달리는 최대와 그 자리), 매끈한 꼴: Π_{j≤k} (1 − w_j) (§52.523)
            smooth3 = VF_NS_SMOOTH and VF_SEP_P > 0.0
            if smooth3:
                agm = ag
            mw = 0.0
            jm = -1
            chi_ = 1.0
            for k in range(NS3):
                zc = min(max(1.0 - (G3[k] - agm) / (VF_NS_SEP * agm), 0.0), 1.0)
                if VF_NS_SMOOTH:
                    wj = zc * zc * (3.0 - 2.0 * zc)
                    ZC3[k] = zc
                else:
                    wj = zc
                W3[k] = wj
                if wj > mw:
                    mw = wj
                    jm = k
                chi_ *= 1.0 - wj
                MW3[k] = (1.0 - chi_) if VF_NS_SMOOTH else mw
                JM3[k] = jm
            for k in range(NS3):
                lF = T2 * MU3[k]
                rr = agm / G3[k]
                chi = 1.0 - MW3[k]'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''                # χ_k = 1 − w_{jm(k)}: w_j = 1 − (G_j − ag)/(δ ag) 가 경사 안일 때만
                j = JM3[k]
                if j >= 0 and W3[j] > 0.0 and W3[j] < 1.0:
                    lw = -lpk * (ps - p0) * (1.0 - rr * rr)
                    lG3[j] += lw * (-1.0 / (VF_NS_SEP * agm))
                    lag3 += lw * G3[j] / (VF_NS_SEP * agm * agm)
            lG3[kmin3] += lag3'''
b='''                if VF_NS_SMOOTH:
                    # χ_k = Π_{j≤k} (1 − w_j), w_j = smoothstep(z_j), z_j = 1 − (G_j − ag)/(δ ag)
                    lchi = lpk * (ps - p0) * (1.0 - rr * rr)
                    for j in range(k + 1):
                        zc = ZC3[j]
                        if zc <= 0.0 or zc >= 1.0:
                            continue
                        pr = 1.0
                        for i2 in range(k + 1):
                            if i2 != j:
                                pr *= 1.0 - W3[i2]
                        lw = -lchi * pr * 6.0 * zc * (1.0 - zc)
                        lG3[j] += lw * (-1.0 / (VF_NS_SEP * agm))
                        lag3 += lw * G3[j] / (VF_NS_SEP * agm * agm)
                else:
                    # χ_k = 1 − w_{jm(k)}: w_j = 1 − (G_j − ag)/(δ ag) 가 경사 안일 때만
                    j = JM3[k]
                    if j >= 0 and W3[j] > 0.0 and W3[j] < 1.0:
                        lw = -lpk * (ps - p0) * (1.0 - rr * rr)
                        lG3[j] += lw * (-1.0 / (VF_NS_SEP * agm))
                        lag3 += lw * G3[j] / (VF_NS_SEP * agm * agm)
            if smooth3:                                     # 분리 기준 = 매끈한 최소: ∂ag/∂G_k = (ag/G_k)^(p+1)
                for k in range(NS3):
                    lG3[k] += lag3 * (agm / G3[k]) ** (p3 + 1.0)
            else:
                lG3[kmin3] += lag3'''
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
