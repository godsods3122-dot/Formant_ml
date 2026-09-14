"""물리 층 검증 — 전부 **독립적으로 세운 해석해·기준**과 대조한다 (MEASUREMENTS §52).

같은 식을 같은 식으로 비교하는 순환 검증은 이미 한 번 버그(점성 항 ℓ 배)를 숨겼다.
여기 있는 기준은 모두 모듈 코드와 다른 경로로 계산한다.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from formant_ml.physics import fold_modes as FM
from formant_ml.physics import glottal_flow as GF
from formant_ml.physics import glottal_noise as GN
from formant_ml.physics import lungs as LU
from formant_ml.physics.transient import modal_source


def test_fold_transverse_branch_follows_string_formula(monkeypatch):
    """장력이 탄성보다 압도적이면 가로 가지는 현 공식 (1/2L)√(σ/ρ) 로 간다.

    '최저 모드' 가 아니라 **가로 에너지가 지배적인 모드**를 본다 — σ 가 크면 최저가
    세로 모드로 갈아탄다 (§52.1 에서 이 가정 때문에 검증을 잘못 읽었다).
    """
    monkeypatch.setattr(FM, "SIGMA0", 50.0e3)
    monkeypatch.setattr(FM, "STRAIN_A", 0.0)
    L = FM.L0
    f, _v, meta = FM.bands([np.pi / L], 0.0, nx=16, ny=12, n_modes=10, shapes=True)
    tr = [i for i in range(len(f[0])) if meta[0][i, 0] > 0.6]
    assert tr, "가로 모드가 없다"
    want = (1.0 / (2.0 * L)) * np.sqrt(50.0e3 / FM.RHO)
    assert abs(f[0][tr[0]] / want - 1.0) < 0.03


def test_fold_grid_convergence():
    L = FM.geometry(0.2)[0]
    a = FM.bands([np.pi / L], 0.2, nx=16, ny=12, n_modes=2)[0][0]
    b = FM.bands([np.pi / L], 0.2, nx=22, ny=16, n_modes=2)[0][0]
    assert abs(a / b - 1.0) < 0.01


def test_fold_f0_range_matches_speaker():
    """화자 물성(L0 11 mm, σ0 3 kPa)이 wavs/ 실측 하위(231 Hz)와 프로파일 상한(580)을 낸다."""
    # 격자는 보고한 표와 같은 기본값(16×12)이어야 한다. ny=10 에 접촉 0.95 를 주면
    # round(0.95×10) = 10 이라 내측면 전체가 고정되어 면적 모드가 없다 (nan).
    lo, _ = FM.phonation_mode(0.02, 0.95, 1.00)
    hi, _ = FM.phonation_mode(0.45, 0.00, 0.24)
    assert 215.0 < lo < 250.0
    assert 520.0 < hi < 620.0


def test_flow_inviscid_limit_is_bernoulli(monkeypatch):
    monkeypatch.setattr(GF, "MU_AIR", 0.0)
    a = np.full(6, 1.0e-5)
    U, _p, _i = GF.flow(a, 0.5e-3, 11e-3, 800.0)
    want = a[0] * np.sqrt(2 * 800.0 / (GF.RHO_AIR * (1 - (a[0] / GF.A_SUB) ** 2)))
    assert abs(U / want - 1.0) < 0.02


def test_flow_viscous_limit_is_parallel_plate_poiseuille():
    """h 로 쓴 평행판 공식 Q = ℓ(2h)³ΔP/(12μd) 와 대조한다 (모듈의 a 표기와 독립)."""
    L, dz, h = 11e-3, 0.5e-3, 2.0e-6
    a = GF.duct_area(np.full(6, h), L)
    U, _p, _i = GF.flow(a, dz, L, 100.0)
    want = L * (2 * h) ** 3 * 100.0 / (12 * GF.MU_AIR * dz * 6)
    assert abs(U / want - 1.0) < 0.05


def test_flow_separation_does_not_recover_pressure():
    a = np.array([2.0, 1.0, 1.5, 2.0, 3.0, 4.0]) * 1e-5
    _U, p, i = GF.flow(a, 0.5e-3, 11e-3, 800.0)
    assert i == 2 and np.allclose(p[i + 1:], 0.0)


def test_whisper_needs_posterior_chink():
    """유성 중 누설(2 mm²)은 난류 문턱 아래, 속삭임(12 mm²)은 위."""
    L, dz, P = 11e-3, 0.4e-3, 6 * 98.0665
    leak = GN.turbulence(GN.solve_flow(np.full(6, 2e-6), dz, L, P)[0], 2e-6, L)
    whis = GN.turbulence(GN.solve_flow(np.full(6, 12e-6), dz, L, P)[0], 12e-6, L)
    assert leak == 0.0 and whis > 0.0


def test_lung_recoil_and_whisper_budget():
    assert abs(LU.relaxation_pressure(0.0)) < 0.5
    assert abs(LU.relaxation_pressure(1.0) - 35.0) < 3.0
    assert abs(LU.relaxation_pressure(-0.25) + 20.0) < 3.0
    r = LU.breath_budget(420.0) / LU.breath_budget(180.0)
    assert 0.25 < r < 0.55


def _exact_reference(seconds, f0, h0, orr, n_modes=12, eps=0.15, nx=14, ny=8, zeta=0.10,
                     fs=48000.0):
    """같은 선형 계를 **행렬 지수**로 정확히 이산화한다 (접촉·유동 없음)."""
    L, T, _D = FM.geometry(eps)
    f, vec, _m = FM.bands([np.pi / L], eps, nx=nx, ny=ny, n_modes=n_modes, shapes=True)
    f, vec = f[0], vec[0]
    phi = vec[:nx * ny][np.arange(ny)]
    good = np.isfinite(f) & (f > 1.0)
    f, phi = f[good], phi[:, good]
    part = np.abs(phi).mean(0)
    keep = part > 0.05 * part.max()
    f, phi = f[keep], phi[:, keep]
    w = 2 * np.pi * f
    J = phi.T @ (np.ones(ny) * T / (ny - 1))
    J = J * (orr * h0 / np.abs(phi @ (J / w)).max())
    P = [expm(np.array([[0, 1], [-wk ** 2, -2 * zeta * wk]]) / fs) for wk in w]
    nt = int(seconds * fs)
    q = np.zeros(len(w)); qd = np.zeros(len(w)); area = np.empty(nt); nxt = 0.0
    for it in range(nt):
        if it / fs >= nxt:
            qd += J; nxt = it / fs + 1.0 / f0
        for k in range(len(w)):
            q[k], qd[k] = P[k] @ np.array([q[k], qd[k]])
        area[it] = 2 * max((h0 + phi @ q).min(), 0.0) * L
    return area


def test_transient_linear_core_converges_to_exact_reference():
    """접촉·유동을 끈 선형 핵이 행렬 지수 기준으로 **1 차 수렴**한다 (§52.1 버그 9·10)."""
    ref = _exact_reference(0.05, 240.0, 0.10e-3, 4.0)
    nrm = lambda x: np.sqrt(np.mean(x ** 2))                     # noqa: E731
    errs = []
    for ss in (1, 2, 4):
        r = modal_source(seconds=0.05, f0=240.0, jitter=0.0, shimmer=0.0, mode_var=0.0,
                         seed=3, n_modes=12, h0=0.10e-3, open_ratio=4.0,
                         k_contact=0.0, c_contact=0.0, substeps=ss)
        errs.append(nrm(r["area"] - ref) / nrm(ref))
    assert errs[-1] < 1e-3
    assert 1.8 < errs[0] / errs[1] < 2.2 and 1.8 < errs[1] / errs[2] < 2.2


def test_transient_is_reproducible_despite_random_eigenvector_signs():
    """ARPACK 은 호출마다 고유벡터 부호를 임의로 준다. 음원은 그것에 불변이어야 한다."""
    kw = dict(seconds=0.04, f0=240.0, jitter=0.0, shimmer=0.0, mode_var=0.0, seed=3,
              n_modes=12, h0=0.10e-3, open_ratio=4.0, substeps=2)
    a, b = modal_source(**kw), modal_source(**kw)
    assert np.abs(a["area"] - b["area"]).max() < 1e-15
    assert np.abs(a["flow"] - b["flow"]).max() < 1e-12


def test_tract_inertance_skews_pulse_and_keeps_convergence():
    """성도 관성(ρℓ/A)이 유량 펄스를 닫힘 쪽으로 기울이고, 적분은 여전히 수렴한다 (§52.3)."""
    fs, f0 = 48000.0, 240.0

    def skew(r):
        U, per, sq = r["flow"], int(fs / f0), []
        for p in r["pulses"][3:-1]:
            seg = U[int(p * fs):int(p * fs) + per]
            op = np.flatnonzero(seg > 0.05 * seg.max()) if seg.max() > 0 else []
            if len(op) < 3:
                continue
            a, b = op[0], op[-1]
            k = a + int(np.argmax(seg[a:b + 1]))
            if b > k:
                sq.append((k - a) / (b - k))
        return np.median(sq)

    kw = dict(seconds=0.08, f0=f0, jitter=0.0, shimmer=0.0, mode_var=0.0, seed=3, n_modes=12,
              h0=0.0, open_amp=0.30e-3)
    s0 = skew(modal_source(substeps=2, l_supra=0.0, **kw))
    s1 = skew(modal_source(substeps=2, l_supra=0.30, **kw))
    assert s1 > s0 + 0.2
    a = modal_source(substeps=2, **kw)
    b = modal_source(substeps=8, **kw)
    nrm = lambda x: np.sqrt(np.mean(x ** 2))                      # noqa: E731
    assert nrm(a["flow"] - b["flow"]) / nrm(b["flow"]) < 0.02


def test_cycle_jitter_decorrelates_high_harmonics_more_than_low():
    """주기 지터는 k·f0·δT 위상 오차라 **고역 배음만** 흩는다 — 고정 위상 분산의 물리적 대체."""
    from scipy.signal import stft
    fs, f0 = 48000.0, 240.0

    def harm(dv, lo, hi):
        f, _t, Z = stft(dv[int(0.05 * fs):], fs, nperseg=4096, noverlap=4096 - 512)
        S = 20 * np.log10(np.abs(Z) + 1e-10)
        lag = int(round(f0 / (f[1] - f[0])))
        s = (f >= lo) & (f < hi)
        vals = []
        for j in range(S.shape[1]):
            y = S[s, j]
            y = y - np.convolve(y, np.ones(21) / 21, "same")
            y = y - y.mean()
            ac = np.correlate(y, y, "full")[len(y) - 1:]
            vals.append(ac[lag - 1:lag + 2].max() / max(ac[0], 1e-12))
        return np.median(vals)

    kw = dict(seconds=0.4, f0=f0, shimmer=0.0, mode_var=0.0, seed=7, n_modes=12, substeps=2,
              h0=0.0, open_amp=0.30e-3)
    clean = modal_source(jitter=0.0, **kw)["flow_deriv"]
    jit = modal_source(jitter=0.025, **kw)["flow_deriv"]
    low_keep = harm(jit, 300, 2000) / harm(clean, 300, 2000)
    high_keep = harm(jit, 8000, 12000) / harm(clean, 8000, 12000)
    assert high_keep < 0.3 and low_keep > 0.5 and high_keep < low_keep


def test_self_oscillation_reaches_a_limit_cycle_and_f0_rises_with_pressure():
    """접촉 경계를 상시로 걸지 않으면(§52.4) 자려 진동이 한계 순환에 이르고 F0 가 압력을 따라 오른다."""
    from formant_ml.physics import self_oscillation as S

    def amp(p):
        r = S.simulate(p_sub_cm=p, contact=0.0, h0=0.03e-3, seconds=0.16, n_modes=6)
        U = r["U"]; n = len(U)
        return r["f0"], np.ptp(U[3 * n // 4:]), np.ptp(U[n // 2:3 * n // 4])

    f_lo, _a, _b = amp(8.0)
    f_hi, late, mid = amp(12.0)
    assert late > 1e-5                        # 진동이 선다
    assert 0.9 < late / mid < 1.1             # 포화 = 한계 순환
    assert f_hi > f_lo                        # 압력이 F0 를 올린다


def test_limit_cycle_f0_rises_monotonically_with_strain():
    """성도 관성을 넣은 자려 진동에서 늘어남이 F0 를 단조로 올리고, 닫힘이 유지된다 (§52.5).

    9 cmH₂O, h0 −0.02 mm 에서 실측: ε 0.02/0.10/0.20/0.30 -> 246/267/299/340 Hz, 개방률 0.67~0.69.
    """
    from formant_ml.physics import self_oscillation as S
    f0s, oqs = [], []
    for eps in (0.02, 0.20, 0.30):
        r = S.simulate(p_sub_cm=9.0, strain=eps, h0=-0.02e-3, seconds=0.08, n_modes=6)
        f0s.append(r["f0"]); oqs.append(r["oq"])
    assert f0s[0] < f0s[1] < f0s[2]
    assert all(0.4 < q < 0.8 for q in oqs)


def test_glottis_refactor_and_noise_bypass_are_exact():
    """`Glottis` 로 통합한 뒤에도 요동을 끄면 이전 결과와 정확히 같다 (§52.8)."""
    from formant_ml.physics import self_oscillation as S
    assert S.P_NOISE == 0.0 and S.F_NOISE == 0.0
    r = S.simulate(p_sub_cm=9.0, h0=-0.02e-3, seconds=0.12, n_modes=6)
    U = r["U"]; n = len(U)
    assert abs(r["f0"] - 282.1) < 0.1
    assert abs(r["oq"] - 0.681) < 0.002
    assert abs(U[n // 2:].max() * 1e6 - 460.6) < 0.5


def test_pressure_noise_scales_jitter(monkeypatch):
    """압력 요동은 주기 간 변동을 만든다 (선형에 가깝게) — §52.9."""
    from formant_ml.physics import self_oscillation as S
    fs = 200000.0

    def jit(pn):
        monkeypatch.setattr(S, "P_NOISE", pn)
        U = S.simulate(p_sub_cm=9.0, h0=-0.02e-3, seconds=0.25, n_modes=6, seed=5)["U"]
        x = U[len(U) // 4:]; thr = 0.05 * x.max()
        on = np.flatnonzero((x[1:] > thr) & (x[:-1] <= thr))
        T = np.diff(on) / fs
        return 100 * np.mean(np.abs(np.diff(T))) / np.mean(T)

    j0, j1 = jit(0.0), jit(0.10)
    assert j0 < 0.05 and j1 > 0.5


def test_render_calibration_table_is_strictly_monotone():
    """교정표는 F0 가 누적 최대보다 큰 점만 남겨야 한다 — 아니면 np.interp 가 조용히 틀린다 (§52.10)."""
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "physics_render_lc.py"
    spec = importlib.util.spec_from_file_location("physics_render_lc", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    es, fs_ = mod.monotone_table([-0.15, -0.10, -0.05, 0.0, 0.02, 0.10],
                                 [229.6, 238.1, 259.4, 242.1, 246.6, 267.4])
    assert list(es) == [-0.15, -0.10, -0.05, 0.10]
    assert np.all(np.diff(fs_) > 0)


def test_f0_feedback_correction_does_not_wind_up():
    """도달할 수 없는 목표 앞에서 보정이 한계를 넘어 쌓이지 않아야 한다 (§52.11 적분 와인드업)."""
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "physics_render_lc.py"
    spec = importlib.util.spec_from_file_location("physics_render_lc", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    lo, hi, eps_ff = -0.15, 0.50, -0.15
    corr = 0.0
    for _ in range(200):                         # 목표가 하한보다 낮다 -> 오차가 계속 음수
        corr = mod.update_correction(corr, eps_ff, err=-0.3, slope=0.8, lo=lo, hi=hi, gain=0.6)
    assert eps_ff + corr >= lo - 1e-12           # 한계를 넘지 않는다
    # 목표가 범위 안으로 돌아오면 곧바로 풀려야 한다 (쌓인 음의 보정이 없으므로)
    eps_ff2 = 0.10
    corr2 = mod.update_correction(corr, eps_ff2, err=+0.05, slope=0.8, lo=lo, hi=hi, gain=0.6)
    assert abs((eps_ff2 + corr2) - 0.10) < 0.1


def test_glottis_reshape_with_same_strain_does_not_disturb_oscillation():
    """같은 늘어남으로 모드를 다시 세워도(ARPACK 부호가 바뀌어도) 진동이 그대로다 (§52.13)."""
    from formant_ml.physics import self_oscillation as S
    fs, P = 200000.0, 9.0 * 98.0665
    dt, nt, seg = 1.0 / fs, int(0.08 * fs), int(0.005 * fs)

    def run(reshape):
        g = S.Glottis(strain=0.15, h0=-0.02e-3, n_modes=6, seed=3)
        U = np.empty(nt)
        for it in range(nt):
            if reshape and it > 0 and it % seg == 0:
                g.reshape(0.15)
            U[it] = g.step(P, dt)[0]
        return U

    a, b = run(False), run(True)
    nrm = lambda x: np.sqrt(np.mean(x ** 2))                     # noqa: E731
    assert nrm(a - b) / nrm(a) < 0.01


def test_glottis_reshape_projects_whole_field_without_gaining_energy():
    # §52.18 — 내측면만 pinv 로 이으면 rank 5 라 내부 변형을 버렸다. 변위장 전체를 M 가중으로 사영한다.
    from formant_ml.physics import self_oscillation as S
    g = S.Glottis(strain=0.15, h0=-0.02e-3, n_modes=6, seed=3)
    assert np.abs(g.vec.T @ (g.m_cell * g.vec) - np.eye(g.nq)).max() < 1e-9
    P = 8.0 * 98.0665
    for _ in range(4000):
        g.step(P, 1.0 / 200000.0)

    def energy(gl):
        return 0.5 * np.sum(gl.qd ** 2) + 0.5 * np.sum((gl.w * gl.q) ** 2)

    for de in (0.005, 0.02):
        e0, h0 = energy(g), g.phi @ g.q
        g.reshape(0.15 + de)
        e1, h1 = energy(g), g.phi @ g.q
        # 모드 에너지(각 모드의 ω 로 잰 것) 는 진동수가 조금 바뀌므로 정확히 같지는 않지만, 사영이 키우면 안 된다.
        assert 0.85 < e1 / e0 < 1.0 + 4 * de
        assert np.abs(h1 - h0).max() < 0.1 * np.abs(h0).max()
        g.reshape(0.15)


def _steady_f0(r, fs=200000.0):
    U = r["U"][len(r["U"]) // 2:]
    thr = 0.05 * U.max()
    i = np.flatnonzero((U[1:] > thr) & (U[:-1] <= thr))
    return fs / np.median(np.diff(i))


def test_ta_zero_leaves_everything_unchanged():
    from formant_ml.physics import fold_modes as FM
    from formant_ml.physics import self_oscillation as S
    assert FM.geometry(0.1) == FM.geometry(0.1, 0.0)
    a = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.02, n_modes=6)
    b = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.02, n_modes=6, ta=0.0)
    assert np.array_equal(a["U"], b["U"])        # ARPACK 시작 벡터를 고정했으므로 비트 단위로 같아야 한다 (§52.23)


def test_ta_contraction_lowers_f0_monotonically_below_strain_floor():
    # §52.22 — 늘어남만으로는 약 222 Hz 아래로 못 갔다. TA 두께·깊이 축이 목표 저음(186~199 Hz) 에 닿아야 한다.
    from formant_ml.physics import self_oscillation as S
    f = [_steady_f0(S.simulate(p_sub_cm=8.0, strain=-0.15, h0=-0.02e-3, seconds=0.3, n_modes=6, ta=ta))
         for ta in (0.0, 0.5, 1.0)]
    assert f[0] > f[1] > f[2]
    assert f[2] < 200.0


def _render_lc_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "physics_render_lc.py"
    spec = importlib.util.spec_from_file_location("physics_render_lc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_posture_rule_inverts_log_linear_table_and_skips_nan():
    R = _render_lc_module()
    strains = np.array(R.POSTURE_STRAINS)
    tas = np.array(R.POSTURE_TAS)
    F = 250.0 * np.exp(1.5 * strains[None, :] - 0.3 * tas[:, None])
    F[0, 0] = np.nan
    assert R.posture_ta(R.TA_KNEE_HZ + 10) == 0.0 and R.posture_ta(R.TA_LOW_HZ - 10) == 1.0
    for f0 in (200.0, 240.0, 300.0, 400.0):
        eps, ta = R.posture_for(f0, strains, tas, F)
        assert abs(250.0 * np.exp(1.5 * eps - 0.3 * ta) / f0 - 1.0) < 1e-6


def test_steady_f0_rejects_unsettled_oscillation():
    R = _render_lc_module()
    fs = 200000.0
    t = np.arange(int(0.2 * fs)) / fs
    assert abs(R.steady_f0(np.maximum(0, np.sin(2 * np.pi * 250 * t)), fs) - 250.0) < 0.5
    chirp = np.maximum(0, np.sin(2 * np.pi * (150 * t + 900 * t ** 2)))
    assert np.isnan(R.steady_f0(chirp, fs))


def test_set_h0_matches_constructor_rest_shape():
    from formant_ml.physics import self_oscillation as S
    a = S.Glottis(strain=0.1, h0=0.3e-3, n_modes=6)
    b = S.Glottis(strain=0.1, h0=-0.02e-3, n_modes=6)
    b.set_h0(0.3e-3)
    assert np.allclose(a.h_rest, b.h_rest, rtol=0.0, atol=1e-15)


def test_abducted_glottis_holds_steady_flow_without_oscillating():
    # §52.25 — 외전 반틈새 0.4 mm 는 자세 격자의 어느 끝에서도 떨지 않고 직류만 흘려야 한다 (무성·속삭임).
    from formant_ml.physics import self_oscillation as S
    for eps, ta in ((-0.1, 0.0), (0.3, 1.0 / 3.0), (0.45, 0.0), (-0.2, 1.0)):
        r = S.simulate(p_sub_cm=8.0, strain=eps, h0=0.4e-3, seconds=0.1, n_modes=6, ta=ta)
        U = r["U"][len(r["U"]) // 2:]
        assert U.mean() > 0 and U.std() < 0.02 * U.mean()


def test_abduction_removes_voicing_offset_click():
    # 옛 렌더러는 무성 프레임에서 성대를 없애 유량이 순간에 끊겼다. 외전으로 끝내면 경계의 유량 변화가 발성 중 펄스보다 크지 않다.
    R = _render_lc_module()
    strains = np.array(R.POSTURE_STRAINS)
    tas = np.array(R.POSTURE_TAS)
    F = 250.0 * np.exp(1.5 * strains[None, :] - 0.3 * tas[:, None])
    f0 = np.full(90, 250.0)
    voiced = np.zeros(90, bool)
    voiced[10:60] = True
    fs = 200000
    U, _A, _L = R.render_source(f0, 1.0, voiced=voiced, posture=(strains, tas, F),
                                abduct=dict(h0_open=0.4e-3, tau_s=0.02, lead_ms=10.0))
    dU = np.abs(np.diff(U))
    steady = dU[int(0.030 * fs):int(0.050 * fs)].max()
    edge = dU[int(0.055 * fs):int(0.085 * fs)].max()
    assert steady > 0 and edge <= 1.2 * steady
    U0, _A0, _L0 = R.render_source(f0, 1.0, voiced=voiced, posture=(strains, tas, F), abduct=None)
    assert np.all(U0[int(0.061 * fs):] == 0.0)          # 옛 방식은 무성에서 유량이 정확히 0 (대조)


def test_outlet_none_and_infinite_outlet_are_bitwise_identical():
    from formant_ml.physics import self_oscillation as S
    runs = []
    for a_out in (None, float("inf")):
        g = S.Glottis(strain=0.1, h0=-0.02e-3, n_modes=6, seed=1, ta=1.0 / 3.0)
        g.a_out = a_out
        runs.append(np.array([g.step(8.0 * 98.0665, 1.0 / 200000.0)[0] for _ in range(4000)]))
    assert np.array_equal(runs[0], runs[1])


def test_supraglottal_constriction_matches_two_orifice_bernoulli(monkeypatch):
    # §52.30 — 독립 해석해: 비점성 두 오리피스 직렬의 정상 유량 U = √(2P / ρ(1/a_sep² − 1/A_sub² + 1/A_c²)).
    from formant_ml.physics import self_oscillation as S
    monkeypatch.setattr(S, "MU_AIR", 1e-12)
    P = 8.0 * 98.0665
    for A_c in (0.4e-4, 0.1e-4):
        g = S.Glottis(strain=0.3, h0=1.0e-3, n_modes=6, seed=0)
        g.a_out = A_c
        for _ in range(int(0.15 * 200000)):
            U, _h = g.step(P, 1.0 / 200000.0)
        h = g.h_rest + g.phi @ g.q
        a = 2.0 * np.maximum(h, 0.0) * g.ell
        assert np.all(np.diff(a) < 0)                 # 수렴형 — 분리 없이 맨 위가 최소
        u_ref = np.sqrt(2.0 * P / (S.RHO_AIR * (1.0 / a[-1] ** 2 - 1.0 / S.A_SUB ** 2 + 1.0 / A_c ** 2)))
        assert abs(U / u_ref - 1.0) < 2e-3
        assert abs(g.p_sup / (0.5 * S.RHO_AIR * U * U / A_c ** 2) - 1.0) < 1e-9


def _physics_source_module():
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("physics_source", root / "scripts" / "physics_source.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frame_f0_from_flow_reads_known_period_and_skips_dc():
    M = _physics_source_module()
    fs = 200000.0
    t = np.arange(int(0.2 * fs)) / fs
    U = np.maximum(0.0, np.sin(2 * np.pi * 250.0 * t)) * 3e-4
    U[int(0.1 * fs):] = 2e-4                      # 뒤 절반은 벌린 성문의 직류
    f = M.frame_f0_from_flow(U, 200, 1.0, fs)
    assert np.nanmax(np.abs(f[5:95] - 250.0)) < 0.5
    assert np.all(np.isnan(f[110:]))


def test_refine_command_removes_constant_ratio_error_and_respects_limit():
    M = _physics_source_module()
    tgt = np.full(100, 250.0)
    voiced = np.ones(100, bool)
    cmd = tgt.copy()
    meas = tgt * 2 ** (-40 / 1200)                # 명령대로 내면 40 센트 낮게 나오는 계
    for _ in range(6):
        cmd = M.refine_command(cmd, tgt, meas, voiced)
        meas = cmd * 2 ** (-40 / 1200)
    assert np.max(np.abs(1200 * np.log2(meas / tgt))) < 0.5
    wild = M.refine_command(tgt, tgt, tgt * 0.25, voiced)
    assert np.max(1200 * np.log2(wild / tgt)) <= 300.0 + 1e-6


def test_reshape_across_vibration_depth_projects_without_gaining_energy():
    # §52.32 — 성구 축: 진동 깊이를 바꿔도 변위장 사영이 에너지를 키우지 않고, 되돌리면 같은 모드 공간으로 돌아온다.
    from formant_ml.physics import self_oscillation as S
    g = S.Glottis(strain=0.3, h0=-0.02e-3, n_modes=6, seed=5)
    for _ in range(4000):
        g.step(8.0 * 98.0665, 1.0 / 200000.0)

    def energy(gl):
        return 0.5 * np.sum(gl.qd ** 2) + 0.5 * np.sum((gl.w * gl.q) ** 2)

    field0 = g.vec @ g.q
    e0 = energy(g)
    g.reshape(0.3, vib_depth=0.4)
    assert np.abs(g.vec.T @ (g.m_cell * g.vec) - np.eye(g.nq)).max() < 1e-9
    field1 = g.vec @ g.q
    # 얕아진 모드 공간으로의 M-직교 사영은 운동에너지·변위 크기를 늘릴 수 없다
    assert np.sqrt(np.sum(g.m_cell * field1 ** 2)) <= np.sqrt(np.sum(g.m_cell * field0 ** 2)) * (1 + 1e-9)
    assert energy(g) < 4.0 * e0                     # 강성이 오르므로 위치 에너지는 늘 수 있지만 폭주하지 않는다
    g.reshape(0.3, vib_depth=1.0)
    assert g.vib_depth == 1.0 and np.all(np.isfinite(g.q))


def test_register_rule_endpoints_and_minimal_register_choice():
    # §52.33 — 흉성으로 닿으면 r 0, 고음은 표 안쪽에서 낼 수 있는 가장 작은 r, 뒤집기는 정확해야 한다.
    R = _render_lc_module()
    assert R.register_params(0.0) == (1.0, 0.0, 1.0)
    vd, dh0, ps = R.register_params(1.0)
    assert vd == R.VD_FALSETTO and dh0 == R.H0_FALSETTO_SHIFT and ps == 1.0 + R.P_FALSETTO_GAIN
    strains = np.array(R.REGISTER_STRAINS)
    rows = np.array(R.REGISTER_ROWS)
    F = 250.0 * np.exp(1.5 * strains[None, :] + 0.9 * rows[:, None])
    F[1, 0] = np.nan
    chest_max = 360.0
    f0 = np.array([300.0, 700.0, 0.0])
    voiced = np.array([True, True, False])
    r = R.register_track(f0, voiced, strains, rows, F, chest_max, frame_ms=1.0)   # 창 30 프레임 > 길이 — 평활 전 값을 따로 본다
    raw = R.register_track(f0, voiced, strains, rows, F, chest_max, frame_ms=1000.0)
    assert raw[0] == 0.0 and raw[2] == 0.0
    need = np.log(700.0 / R.REGISTER_MARGIN / (250.0 * np.exp(1.5 * strains[-1]))) / 0.9
    assert need <= raw[1] < need + 0.05 + 1e-9
    assert np.all((r >= 0) & (r <= 1))
    eps, vd, dh0, ps = R.register_posture(700.0, raw[1], strains, rows, F)
    assert abs(250.0 * np.exp(1.5 * eps + 0.9 * raw[1]) / 700.0 - 1.0) < 1e-6


def test_register_smoothing_never_pushes_a_low_frame_below_its_row_floor():
    # §52.34 — 고음 프레임 옆의 낮은 F0 프레임에 성구가 번져도, 그 행이 그 F0 까지 내려갈 수 있어야 한다.
    R = _render_lc_module()
    strains = np.array(R.REGISTER_STRAINS)
    rows = np.array(R.REGISTER_ROWS)
    F = 250.0 * np.exp(1.5 * strains[None, :] + 0.9 * rows[:, None])   # 행 바닥 = 250·e^{0.15 + 0.9 r}
    f0 = np.full(80, 300.0)
    f0[40:50] = 700.0
    f0[52:56] = 295.0                                  # 고음 바로 뒤의 낮은 음
    voiced = np.ones(80, bool)
    r = R.register_track(f0, voiced, strains, rows, F, 360.0, frame_ms=1.0)
    floor = 250.0 * np.exp(1.5 * strains[0] + 0.9 * r)
    assert np.all(floor[52:56] <= f0[52:56] + 1e-9)
    assert np.all(r[40:50] > 0.0)


def test_lung_pressure_track_inverts_level_and_silences():
    # §52.36 — 목표 유성 수준 6 dB 차이는 폐압 10^(6/(20γ)) 배, 무음은 0, 말 안의 무성은 이웃 유성 사이.
    M = _physics_source_module()
    # 비교 프레임은 모든 전이에서 230 프레임 넘게 떨어뜨린다 — 앞뒤 1 차 지연(τ 30 ms) 의 잔차가 e^(−230/30) < 1e-3.
    # 무음도 말에서 250 프레임 넘게 떨어뜨린다 (앞뒤 지연이 말 앞뒤로 압력을 미리 올리고 늦게 내리는 것은 설계다).
    n = 1900
    level = np.full(n, -90.0)
    level[350:850] = -20.0
    level[850:880] = -35.0                            # 말 안의 무성 (유성 중앙 −25 dB 안)
    level[880:1400] = -14.0
    voiced = np.zeros(n, bool)
    voiced[350:850] = True
    voiced[880:1400] = True
    p = M.lung_pressure_track(level, voiced, 8.0, 1.0, gamma=2.4)
    ratio = p[1130] / p[600]
    assert abs(ratio - 10 ** (6.0 / 48.0)) < 0.002
    assert p[:60].max() < 0.05 and p[1690:].max() < 0.05
    assert min(p[600], p[1130]) - 1e-6 <= p[865] <= max(p[600], p[1130]) + 1e-6
    assert np.all(p >= 0.0)


def test_open_curve_normalizes_voiced_area_and_clips_breathy():
    # §52.38 — 물리 성문 면적에서 만든 개방 곡선: 유성 99 % 분위가 1, 닫힘 0, 넓게 벌린 무성은 1 에서 잘린다.
    M = _physics_source_module()
    fs_sim, fs_out, spf = 200000.0, 48000, 48
    t = np.arange(int(0.2 * fs_sim)) / fs_sim
    area = np.maximum(0.0, np.sin(2 * np.pi * 250.0 * t)) * 1e-5
    area[int(0.1 * fs_sim):] = 5e-5                   # 뒤 절반은 벌린 성문
    voiced = np.zeros(200, bool)
    voiced[:100] = True
    g, a48 = M.open_curve(area, voiced, 200 * spf, spf, fs_sim, fs_out)
    assert len(g) == len(a48) == 200 * spf
    assert g.min() == 0.0 and abs(np.percentile(g[:100 * spf][a48[:100 * spf] > 0], 99) - 1.0) < 1e-9
    assert np.all(g[110 * spf:] == 1.0)


def test_sanitize_f0_removes_impossible_dip_but_keeps_real_glide():
    # §52.39 — 6 ms 안에 7 반음 내려갔다 오는 V 자는 분석기 오류, 100 ms 에 걸친 6 반음 글라이드는 진짜.
    M = _physics_source_module()
    f0 = np.full(400, 260.0)
    f0[100:106] = 260.0 * 2 ** (-np.array([2, 5, 7, 5, 2, 0]) / 12)
    f0[200:300] = 260.0 * 2 ** (np.linspace(0, 6, 100) / 12)
    f0[300:] = 260.0 * 2 ** (6 / 12)
    voiced = np.ones(400, bool)
    out, n_fix = M.sanitize_f0(f0, voiced, 1.0)
    assert np.max(np.abs(1200 * np.log2(out[95:112] / 260.0))) < 30.0
    assert np.allclose(out[200:300], f0[200:300])
    assert n_fix >= 3


def test_calibration_key_tracks_physics_constants_and_workers_receive_them():
    # §52.43 — 접촉 강성을 바꾸면 캐시 키가 달라져야 하고, 작업자는 넘겨받은 상수로 교정하며 잡음은 끈다.
    R = _render_lc_module()
    from formant_ml.physics import self_oscillation as S
    base = R.physics_consts()
    old_k, old_jet = S.K_CONTACT, S.JET_P_NOISE
    try:
        S.K_CONTACT = old_k * 3.0
        changed = R.physics_consts()
        assert changed != base
        S.K_CONTACT = old_k
        S.JET_P_NOISE = 0.2
        R.apply_physics_consts(changed)
        assert S.K_CONTACT == old_k * 3.0 and S.JET_P_NOISE == 0.0
    finally:
        S.K_CONTACT, S.JET_P_NOISE = old_k, old_jet


def test_jet_development_distance_zero_is_bitwise_bypass():
    # §52.44 — JET_X 0 은 예전 제트 요동과 비트 단위로 같아야 한다 (잡음 씨앗 같음).
    from formant_ml.physics import self_oscillation as S
    old = (S.JET_P_NOISE, S.JET_X)
    try:
        S.JET_P_NOISE = 0.2
        S.JET_X = 0.0
        a = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.03, n_modes=6, ta=1.0 / 3.0, seed=2)["U"]
        S.JET_X = 3e-3
        b = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.03, n_modes=6, ta=1.0 / 3.0, seed=2)["U"]
        S.JET_X = 0.0
        c = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.03, n_modes=6, ta=1.0 / 3.0, seed=2)["U"]
    finally:
        S.JET_P_NOISE, S.JET_X = old
    assert np.array_equal(a, c) and not np.array_equal(a, b)


def test_jet_order_unit_variance_and_bypass():
    # §52.45 — 2·3 차 직렬 저역통과도 정상 상태 분산 1 (고정 계수로 독립 점검), 1 차는 예전과 비트 단위로 같다.
    from scipy.signal import lfilter
    rng = np.random.default_rng(0)
    for a in (0.05, 0.3):
        b = 1.0 - a
        for k, g2 in ((2, a ** 4 * (1 + b * b) / (1 - b * b) ** 3), (3, a ** 6 * (1 + 4 * b * b + b ** 4) / (1 - b * b) ** 5)):
            x = rng.normal(size=400000) / np.sqrt(g2)
            for _ in range(k):
                x = lfilter([a], [1, -b], x)
            assert abs(np.var(x[20000:]) - 1.0) < 0.03
    from formant_ml.physics import self_oscillation as S
    old = (S.JET_P_NOISE, S.JET_ORDER)
    try:
        S.JET_P_NOISE = 0.2
        S.JET_ORDER = 1
        u1 = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.02, n_modes=6, seed=3)["U"]
        S.JET_ORDER = 2
        u2 = S.simulate(p_sub_cm=8.0, strain=0.1, h0=-0.02e-3, seconds=0.02, n_modes=6, seed=3)["U"]
    finally:
        S.JET_P_NOISE, S.JET_ORDER = old
    assert not np.array_equal(u1, u2)


def test_tract_compliance_zero_is_bitwise_bypass():
    from formant_ml.physics import self_oscillation as S
    runs = []
    for comp in (0.0, 0.0):
        old = S.TRACT_COMPLIANCE
        S.TRACT_COMPLIANCE = comp
        try:
            g = S.Glottis(strain=0.1, h0=-0.02e-3, n_modes=6, seed=1)
            g.a_out = 0.4e-4
            runs.append(np.array([g.step(8.0 * 98.0665, 1.0 / 200000.0)[0] for _ in range(3000)]))
        finally:
            S.TRACT_COMPLIANCE = old
    assert np.array_equal(runs[0], runs[1])


def test_tract_compliance_steady_state_and_closed_outlet_integral(monkeypatch):
    # §52.49 — (1) 정상 상태는 독립 해석해(비점성 두 오리피스 직렬) 와 같다. (2) 출구를 막으면 압력 = ∫U dt / C.
    from formant_ml.physics import self_oscillation as S
    monkeypatch.setattr(S, "MU_AIR", 1e-12)
    monkeypatch.setattr(S, "TRACT_COMPLIANCE", 5e-9)
    P, dt = 8.0 * 98.0665, 1.0 / 200000.0
    g = S.Glottis(strain=0.3, h0=1.0e-3, n_modes=6, seed=0)
    g.a_out = 0.4e-4
    for _ in range(int(0.3 / dt)):
        U, _h = g.step(P, dt)
    h = g.h_rest + g.phi @ g.q
    a = 2.0 * np.maximum(h, 0.0) * g.ell
    u_ref = np.sqrt(2.0 * P / (S.RHO_AIR * (1.0 / a[-1] ** 2 - 1.0 / S.A_SUB ** 2 + 1.0 / g.a_out ** 2)))
    assert abs(U / u_ref - 1.0) < 3e-3
    g2 = S.Glottis(strain=0.3, h0=1.0e-3, n_modes=6, seed=0)
    g2.a_out = 1e-12
    us = []
    for _ in range(400):                              # 2 ms
        us.append(g2.step(P, dt)[0])
    integral = np.sum(us) * dt / 5e-9
    assert integral > 50.0 and abs(g2.p_sup / integral - 1.0) < 0.02


def test_voicing_persists_longer_during_closure_with_more_compliance(monkeypatch):
    # §52.49 — 입을 막으면 성문 위 압력이 쌓여 발성이 멎는다. 순응도가 크면 더 오래 이어진다 (유성 폐쇄음의 voice bar).
    from formant_ml.physics import self_oscillation as S
    P, dt = 8.0 * 98.0665, 1.0 / 200000.0
    dur = {}
    for comp in (3e-9, 2e-8):
        monkeypatch.setattr(S, "TRACT_COMPLIANCE", comp)
        g = S.Glottis(strain=0.1, h0=-0.02e-3, n_modes=6, seed=1, ta=1.0 / 3.0)
        g.a_out = 3e-4
        pre = np.array([g.step(P, dt)[0] for _ in range(int(0.12 / dt))])[-int(0.03 / dt):]
        ac0 = pre.std()
        g.a_out = 1e-9
        post = np.array([g.step(P, dt)[0] for _ in range(int(0.25 / dt))])
        w = int(0.01 / dt)
        live = [post[i:i + w].std() > 0.1 * ac0 for i in range(0, len(post) - w, w)]
        dur[comp] = (live.index(False) if False in live else len(live)) * 0.01
    assert dur[2e-8] > dur[3e-9]
    assert dur[3e-9] < 0.25


def test_pulse_phase_errors_recover_offset_and_wrap():
    # §52.56 — 조각마다 다른 상수 어긋남(한 조각은 주기 경계를 넘어 감싼다) 을 빼면 남는 오차가 드러난다.
    M = _physics_source_module()
    T = 1.0 / 250.0
    tgt = np.concatenate([np.arange(0.10, 0.40, T), np.arange(0.50, 0.80, T)])
    err = 0.05 * np.sin(np.arange(len(tgt)) / 5.0)                 # 주기 몫
    off = np.where(tgt < 0.45, 0.30, -0.45)                        # 두 번째 조각은 −0.45 (감쌈 경계 근처)
    phys = tgt + (off + err) * T
    tj, e, Tj, seg = M.pulse_phase_errors(phys, tgt)
    d, R = M.remove_segment_offsets(e, seg)
    truth = np.interp(tj, tgt, err)
    for sid in np.unique(seg):
        m = seg == sid
        assert np.max(np.abs((d[m] - d[m].mean()) - (truth[m] - truth[m].mean()))) < 0.01
    assert R > 0.95


def test_phase_lock_command_raises_frequency_before_late_closure():
    M = _physics_source_module()
    f = np.full(400, 250.0)
    tj = np.array([0.2]); d = np.array([0.2]); Tj = np.array([0.004])
    g = M.phase_lock_command(f, 1.0, tj, d, Tj, gain=0.5, cycles=4.0)
    before = g[int((0.2 - 0.008) * 1000)]
    assert before > 250.0 and abs(before / 250.0 - 1.0) <= 0.03 + 1e-12
    assert g[0] == 250.0 and g[399] == 250.0
