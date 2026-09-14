"""과도응답 음원 — 매 주기의 **폐쇄 충격**이 모드를 때리고, 그 링다운이 열림 구간이다.

왜 이렇게 하는가
----------------
자려 진동(한계 순환)을 직접 적분하려 했으나 에너지 수지가 두세 자릿수 모자랐다
(`self_oscillation`). 단일층 연속체의 모드가 내측면을 거의 안 움직이기 때문이다.

그런데 발성의 실제 모습은 **주기마다의 충격 + 링다운**이다. 성대가 닫히는 순간 조직이
부딪히며 모든 모드를 한꺼번에 때리고, 다음 닫힘까지 그 모드들이 울리며 성문 틈을 만든다.
그러면 한계 순환을 풀지 않고도 **모드 구조·접촉 상태·늘어남**이 전부 음원에 실린다.

무엇이 랜덤인가
---------------
사용자: *"과도응답을 길게 뽑아서 통계로 feature 를 뽑고, 그 기저들의 랜덤 시드로."*
주기마다 (1) 충격의 세기, (2) 모드별 여기 계수, (3) 주기 길이가 조금씩 다르다. 그 변동이
지터·시머·주기 배증으로 나타난다. **위상 분산으로 배음을 뭉개는 것과는 다르다** — 배음은
또렷하게 서고(사용자: *"실제 음원은 배음이 죄다 퍼져있는 게 아니라 solid 한 가로 토막"*)
변동은 주기 구조에만 들어간다.

검증
----
* 링다운의 주파수가 `fold_modes` 의 고유진동수와 일치해야 한다.
* 감쇠율이 ζ 와 맞아야 한다.
* 여기를 끄면 조용해져야 한다 (에너지 보존).
"""
from __future__ import annotations

import numpy as np

from .fold_modes import geometry, bands
from .glottal_flow import A_SUB, MU_AIR, RHO_AIR


#: **접촉 벌칙 강성** [Pa/m] 과 감쇠 [Pa·s/m]. 닫히는 순간 조직이 눌리는 과정이다.
#: 이것이 없으면 면적이 0 에서 계단으로 잘려 음원의 고역이 안 줄어든다 — 실측으로
#: 2 차 이상 배음이 목표보다 **+16~18 dB** 떴다 (LF 음원은 +0.4~3.0 dB). LF 의 복귀
#: 상수 Ta 가 하는 일을 물리로 하는 자리가 여기다.
K_CONTACT = 6.0e6
C_CONTACT = 8.0e3


def modal_source(strain=0.15, contact=0.0, vib_depth=1.0, f0=240.0,
                 seconds=0.05, fs=48000.0, zeta=0.10, n_modes=6,
                 nx=14, ny=8, h0=0.15e-3, seed=0,
                 jitter=0.02, shimmer=0.10, mode_var=0.25,
                 open_ratio=1.6, k_contact=None, c_contact=None, p_sub_cm=6.0,
                 substeps=1, closure_tau=1.0e-4, open_amp=None,
                 l_supra=0.17, a_supra=3.0e-4, l_sub=0.0, a_sub=2.5e-4):
    """성문 **면적** 파형과 그 미분을 만든다.

    반환 dict: t, area [m²], flow_deriv (음원), pulses [s], modes [Hz].
    """
    # **울리는 것은 열린 구간의 모드다.** `contact` 은 `fold_modes` 에서 *닫힌 구간의*
    # 경계조건인데, 그것을 진동 내내 걸어 두면 내측면이 맨 위 한 점만 남는다 (실측: 8 점 중
    # 7 점이 정확히 0 → 면적이 계단으로 변하고 시머도 수직 위상차도 없어짐). 닫힘은 경계가
    # 아니라 **주기의 한 구간**이므로 아래 접촉 처리(면적이 0 으로 닫히는 것)가 담당한다.
    # 기본값을 0 으로 둔다. 그래도 닫힌 모드를 보고 싶으면 명시적으로 넘긴다.
    L = geometry(strain)[0]
    f, vec, meta = bands([np.pi / L], strain, nx=nx, ny=ny, n_modes=n_modes,
                         shapes=True, contact=contact, vib_depth=vib_depth)
    f, vec = f[0], vec[0]
    n = nx * ny
    u = vec[:n]
    med = np.array([j for j in range(ny)])          # x = 0 열 (내측면)
    phi = u[med]                                    # (ny, n_modes)
    good = np.isfinite(f) & (f > 1.0)
    f, phi = f[good], phi[:, good]
    # 내측면을 실제로 움직이는 모드만 (참여도 상위)
    part = np.abs(phi).mean(0)
    keep = part > 0.05 * part.max()
    f, phi = f[keep], phi[:, keep]
    nq = len(f)
    w = 2 * np.pi * f

    # **충격은 표면 충격량을 모드에 투영한 것이다** — J_n = Σ_y φ_n(y)·j(y)·dz.
    #
    # 예전 판은 모드마다 양의 속도를 그냥 더했다(부호가 φ 와 무관). 그런데 ARPACK 은 호출마다
    # 고유벡터의 **부호를 임의로** 준다. 그러면 같은 인자로 부를 때마다 모드의 기여가 뒤집혀
    # 성문 면적이 달라진다 — 접촉을 끈 선형 계조차 적분 간격과 무관하게 개방률이
    # 0.23~0.60 으로 흩어졌다 (행렬 지수 기준 0.251). 투영하면 φ → −φ 일 때 J 와 q 가 같이
    # 뒤집혀 h = h0 + Φq 가 그대로다. 퇴화 쌍이 회전해도 ΦΦᵀj 는 그 부분공간 안에서 불변이다.
    #
    # j(y) 는 내측면 전체에 고르게 바깥쪽(여는 쪽)으로 준다. 세기는 첫 반주기의 최대 벌어짐이
    # `open_ratio × h0` 가 되도록 정한다: 충격 뒤 q_n ≈ (J_n/ω_n)·sin(ω_n t) 이므로
    # 벌어짐의 상한 추정은 max_y |Σ_n φ_n(y) J_n/ω_n| 이다.
    T_th0 = geometry(strain)[1]
    dz0 = T_th0 / (ny - 1)
    j_unit = np.ones(ny)
    J_unit = phi.T @ (j_unit * dz0)                 # 단위 표면 충격량의 모달 투영
    open_est = np.abs(phi @ (J_unit / np.maximum(w, 1e-9))).max()
    # **벌어짐의 크기는 정지 틈과 독립이다.** 예전에는 `open_ratio × h0` 라 기하가 닮은꼴이 되어
    # h0 를 바꿔도 개방률이 0.770 에 붙어 있었고, h0 = 0(완전 내전)이면 아예 진동하지 않았다.
    # 물리적으로 벌어짐은 압력과 조직이 정한다. `open_amp` [m] 를 주면 그것을 쓰고,
    # h0 는 정지 틈(음수면 눌러 닫은 압착)으로만 쓴다. 안 주면 예전 거동을 유지한다.
    amp_target = float(open_amp) if open_amp is not None else open_ratio * h0
    J_kick = J_unit * (amp_target / max(open_est, 1e-30))
    kc = K_CONTACT if k_contact is None else float(k_contact)
    cc = C_CONTACT if c_contact is None else float(c_contact)
    T_th = geometry(strain)[1]
    dz_m = T_th / (ny - 1)                          # 내측면 높이 방향 간격 [m]
    rng = np.random.default_rng(seed)
    nt = int(seconds * fs)
    t = np.arange(nt) / fs
    dt_out = 1.0 / fs
    dt = dt_out / max(int(substeps), 1)        # 내부 적분 간격 — 수렴 검증용
    q = np.zeros(nq); qd = np.zeros(nq)
    area = np.empty(nt)
    flow_u = np.empty(nt)
    U = 0.0
    d_g = T_th                                  # 성문의 두께 방향 길이
    p_drive = float(p_sub_cm) * 98.0665
    # **성도·기관 공기 기둥의 관성** (음향 질량 ρℓ/A). 성문만의 관성 ρd/a 로는 유량 펄스가 거의
    # 대칭이라 낮은 배음 모양이 LF 보다 3 dB 나빴다 (기울기 제거 후 0-1.5 kHz 6.09 dB 대 2.94).
    # 성문 위 공기 기둥이 흐름의 변화를 늦추면 펄스가 닫힘 쪽으로 기운다 (Rothenberg 1981,
    # Titze 2008 의 비선형 음원-필터 결합). 사용자가 강조한 "임피던스와 리액턴스" 가 이 항이다.
    # 기본값은 균일관 근사: 성도 17 cm × 3 cm², 기관은 0 (저주파에서 컴플라이언스 쪽이라 뺌).
    L_tract = RHO_AIR * (float(l_supra) / float(a_supra) + float(l_sub) / float(a_sub))
    pulses = []
    next_t = 0.0
    for it in range(nt):
        if t[it] >= next_t:
            # **폐쇄 충격** — 모드를 한꺼번에 때린다. 세기와 배분이 주기마다 흔들린다.
            g = max(0.0, 1.0 + shimmer * rng.normal())
            c = np.abs(rng.normal(1.0, mode_var, nq))
            # 모드별 배분 흔들림 c 는 **크기에만** 곱한다 — 부호는 투영이 정한다.
            qd += g * c * J_kick
            pulses.append(t[it])
            per = (1.0 / f0) * (1.0 + jitter * rng.normal())
            next_t = t[it] + max(per, 1.0 / 800.0)
        # 모달 + 유동 적분을 한 출력 표본 안에서 substeps 번 돈다 (시간 간격 수렴 검증용).
        for _sub in range(max(int(substeps), 1)):
            # **접촉**: 틈이 음수로 내려가면 조직이 눌린다. 벌칙 강성 + 감쇠를 모드에 투영한다.
            # 이것이 닫힘에 유한한 기울기를 주고, 그 기울기가 음원의 고역 감쇠를 만든다.
            h_now = h0 + phi @ q
            hd_now = phi @ qd
            pen = np.where(h_now < 0.0,
                           -kc * h_now - cc * np.minimum(hd_now, 0.0), 0.0)
            fq = phi.T @ (pen * dz_m)
            qdd = fq - 2 * zeta * w * qd - (w ** 2) * q
            qd += qdd * dt
            q += qd * dt
            h = h0 + phi @ q
            # **성문 면적은 높이 방향의 최소 단면**이다. 어느 한 높이가 닿으면 흐름이 막힌다.
            a_now = 2.0 * max(h.min(), 0.0) * L
            area[it] = a_now
            # **유량은 면적에 비례하지 않는다.** 베르누이 + 공기 기둥 관성으로 푼다:
            #     ρ(d/a)·dU/dt + (ρ/2)U²(1/a² − 1/A_sub²) + 12μdℓ²/a³·U = ΔP
            #
            # 면적을 그냥 미분하면 음원의 고역이 통째로 빈다 (실측: 기울기 +6 dB, LF 는 +24).
            # 고역은 성대의 진동 모드에서 오지 않는다 — 막의 고유모드는 20 개를 뽑아도
            # **425 Hz** 를 못 넘는데 음원은 4~8 kHz 가 필요하다. 고역은 **닫힐 때 유량이
            # 비선형으로 끊기는 것**에서 온다.
            # **암시적(후진) 오일러로 푼다.** 명시적 오일러는 틈이 좁아지면 점성 항의 강성
            # rate = (ρU/a² + 12μdℓ²/a³)/(ρd/a) 가 10⁶~10⁹ /s 로 올라가 rate·dt 가 수십~수만이
            # 된다 — 발산을 max(U, 0) 가 가려서 결과가 적분 간격에 따라 +0.9~+19 dB 로
            # 널뛰었다. 후진 오일러는
            #     L(U' − U)/dt = P − R_k U'² − R_v U'
            #  →  R_k U'² + (R_v + L/dt) U' − (P + L U/dt) = 0
            # 의 양의 근이고 무조건 안정하다. a → 0 이면 R_v → ∞ 라 U' → 0 으로 **저절로** 닫힌다 —
            # 임의로 두었던 닫힘 감쇠 상수(closure_tau)가 필요 없어진다.
            if a_now <= 0.0:
                U = 0.0
            else:
                R_k = 0.5 * RHO_AIR * (1.0 / a_now ** 2 - 1.0 / A_SUB ** 2)
                R_v = 12.0 * MU_AIR * d_g * L ** 2 / a_now ** 3
                Li = RHO_AIR * d_g / a_now + L_tract
                bq = R_v + Li / dt
                cq = p_drive + Li * U / dt
                disc = bq * bq + 4.0 * R_k * cq
                U = (2.0 * cq) / (bq + np.sqrt(disc))   # 근의 공식의 안정한 형태 (소거 오차 없음)
        flow_u[it] = U
    deriv = np.gradient(flow_u, dt_out)

    return {"t": t, "area": area, "flow": flow_u, "flow_deriv": deriv,
            "pulses": np.array(pulses), "modes": f}


def _check() -> int:
    import numpy.fft as nfft
    print("과도응답 음원 검증")
    r = modal_source(seconds=0.2, f0=240.0, jitter=0.0, shimmer=0.0, mode_var=0.0)
    print(f"  모드 {np.round(r['modes'], 1)} Hz,  펄스 {len(r['pulses'])} 개")

    # 1. 여기를 끈 링다운의 주파수가 고유진동수와 맞는가
    r2 = modal_source(seconds=0.2, f0=2.0, jitter=0.0, shimmer=0.0, mode_var=0.0)
    x = r2["area"][int(0.01 * 48000):int(0.06 * 48000)]
    x = x - x.mean()
    X = np.abs(nfft.rfft(x * np.hanning(len(x))))
    fr = nfft.rfftfreq(len(x), 1 / 48000.0)
    pk = fr[np.argmax(X)]
    near = r["modes"][np.argmin(np.abs(r["modes"] - pk))]
    print(f"  1 링다운 정점 {pk:7.1f} Hz   가장 가까운 고유모드 {near:7.1f}   "
          f"오차 {100*(pk-near)/near:+5.1f} %")

    # 2. 주기성 — 지터 0 이면 펄스 간격이 정확히 1/f0
    d = np.diff(r["pulses"])
    print(f"  2 펄스 간격 {1000*d.mean():6.3f} ms (목표 {1000/240:.3f}), 표준편차 {1000*d.std():.4f}")

    # 3. 지터·시머를 켜면 변동이 생긴다
    r3 = modal_source(seconds=0.2, f0=240.0, jitter=0.03, shimmer=0.15, seed=1)
    d3 = np.diff(r3["pulses"])
    amp = np.array([r3["area"][int(p*48000):int(p*48000)+100].max() for p in r3["pulses"][:-1]])
    print(f"  3 지터 켬: 간격 변동 {100*d3.std()/d3.mean():5.2f} %   "
          f"진폭 변동 {100*amp.std()/max(amp.mean(),1e-30):5.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(_check())
