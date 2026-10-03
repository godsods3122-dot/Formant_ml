"""시간 영역 관 (`engine/tube_td.py`) — 해석해와 물리 효과, 수반 기울기 (MEASUREMENTS §52.476)."""
import numpy as np
import pytest
import torch
from scipy.signal import find_peaks

from formant_ml.engine import tube_td as td

FS = td.FS_SIM
N = td.N_SECT


def _peaks(out, lo=200.0, hi=5000.0, k=4):
    T = len(out)
    X = np.abs(np.fft.rfft(out * np.hanning(T), 1 << 20))
    f = np.fft.rfftfreq(1 << 20, 1 / FS)
    db = 20 * np.log10(X + 1e-30)
    m = (f > lo) & (f < hi)
    pk, _ = find_peaks(db[m], prominence=6)
    return f[m][pk][:k]


def _impulse(walls=True, L=17.5):
    T = int(0.25 * FS)
    A = np.full((T, N), 3.0)
    Ps = np.zeros(T)
    Ps[10:20] = 1000.0
    return td.simulate(A, np.full(T, L), np.full(T, 0.02), Ps, walls=walls)[0]


def _vowel(cons=None, L=17.0, noise=(0.0, 0.0), T_s=0.3):
    T = int(T_s * FS)
    t = np.arange(T) / FS
    ph = (t * 200.0) % 1.0
    Ag = 0.01 + 0.15 * np.where(ph < 0.6, 0.5 * (1 - np.cos(2 * np.pi * ph / 0.6)), 0.0)
    A = np.full((T, N), 3.0)
    A[:, :8] = 1.5
    A[:, 8:] = 4.5
    if cons is not None:
        A[:, 22] = cons
    return td.simulate(A, np.full(T, L), Ag, np.full(T, 8 * td.CMH2O), noise_g=noise[0], noise_c=noise[1]), Ag


def test_uniform_tube_hits_the_end_corrected_quarter_wave_series():
    """입술 방사 리액턴스(질량)는 관을 배플 피스톤 끝 보정 0.85 r 만큼 길게 한다 — F2~F4 가 (2n−1)c/4(L+0.85r) 에 1~3 % 안."""
    f = _peaks(_impulse(walls=False))
    Leff = 17.5 + 0.85 * np.sqrt(3.0 / np.pi)
    for k in range(1, 4):
        want = (2 * (k + 1) - 1) * td.C_SOUND / (4 * Leff)
        assert abs(f[k] - want) / want < 0.03, (f, want)


def test_yielding_walls_raise_f1():
    """벽의 질량·강성 리액턴스가 F1 을 올린다 (닫힌 관의 벽 공진, Fant) — 끄면 내려간다."""
    f_on, f_off = _peaks(_impulse(True))[0], _peaks(_impulse(False))[0]
    assert f_on > f_off * 1.05, (f_on, f_off)


def test_glottal_flow_is_physical_and_skewed_by_the_tract():
    """비선형 성문: 8 cmH2O 에서 최대 유량 200~800 cm³/s, 유량 정점이 면적 정점보다 늦다 (성도 관성의 기울임, Rothenberg)."""
    (out, rec), Ag = _vowel()
    s = int(0.2 * FS)
    per = int(FS / 200)
    Ug = rec[s:, 0]
    assert 200 < Ug.max() < 800
    assert np.argmax(rec[s:s + per, 0]) > np.argmax(Ag[s:s + per])


@pytest.mark.parametrize("L", [11.0, 19.0])
def test_stable_with_a_tight_constriction(L):
    (out, rec), _ = _vowel(cons=0.02, L=L, noise=(0.05, 0.05))
    assert np.isfinite(out).all() and np.abs(out).max() < 1e9


def test_turbulence_follows_the_reynolds_number():
    """난류는 레이놀즈 수가 임계를 넘는 **좁은** 곳에서 세다 — 같은 난류 세기에서 협착 0.02 cm² 가 모음보다 4–10 kHz 에서 훨씬 크다."""
    s = int(0.2 * FS)

    def hf(o):
        X = np.abs(np.fft.rfft(o[s:])) ** 2
        f = np.fft.rfftfreq(len(o) - s, 1 / FS)
        return X[(f > 4000) & (f < 10000)].mean()
    (v_on, _), _ = _vowel(noise=(0.0, 0.05))
    (v_off, _), _ = _vowel(noise=(0.0, 0.0))
    (c_on, rc), _ = _vowel(cons=0.02, noise=(0.0, 0.05))
    (c_off, _), _ = _vowel(cons=0.02, noise=(0.0, 0.0))
    assert rc[s:, 4].max() > td.RE_CRIT
    assert 10 * np.log10(hf(c_on) / hf(c_off)) > 10 * np.log10(hf(v_on) / hf(v_off)) + 10.0


def test_adjoint_matches_finite_differences():
    rng = np.random.default_rng(1)
    T = 1500
    t = np.arange(T) / FS
    ph = (t * 250.0) % 1.0
    A0 = np.tile(np.linspace(1.5, 4.5, N), (T, 1))
    A0[:, 20] = 0.12 + 0.05 * np.sin(2 * np.pi * 40 * t)
    base = {"A": A0, "L": np.full(T, 16.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
            "Ag": 0.01 + 0.15 * np.where(ph < 0.6, 0.5 * (1 - np.cos(2 * np.pi * ph / 0.6)), 0),
            "Ps": np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.005),
            "Av": 0.3 + 0.25 * np.sin(2 * np.pi * 20 * t) ** 2, "ng": np.array(0.03), "nc": np.array(0.05),
            "Q": 20.0 * np.sin(2 * np.pi * 15 * t)[:, None] * np.ones((1, N))}
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc", "Q")
    w = torch.tensor(rng.standard_normal(T))

    def loss(v):
        return (td.tube_torch(*[v[k] for k in keys[:-1]], seed=3, Q=v["Q"]) * w).sum() * 1e-6
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    for k in keys:
        d = rng.standard_normal(base[k].shape)
        h = {"A": 1e-5, "L": 1e-5, "Ag": 1e-7, "Ps": 1e-3, "Av": 1e-6, "ng": 1e-6, "nc": 1e-6, "Q": 1e-3}[k]
        vp = {q: torch.tensor(base[q]) for q in keys}
        vm = {q: torch.tensor(base[q]) for q in keys}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)


def test_velum_opening_adds_a_nasal_murmur_and_antiresonance():
    """연구개가 열리면 콧구멍으로 소리가 나고(비강 유량 > 0) 구강 스펙트럼 모양이 바뀐다 — 닫히면 비강 유량은 0."""
    T = int(0.25 * FS)
    t = np.arange(T) / FS
    ph = (t * 200.0) % 1.0
    Ag = 0.01 + 0.15 * np.where(ph < 0.6, 0.5 * (1 - np.cos(2 * np.pi * ph / 0.6)), 0.0)
    A = np.full((T, N), 3.0)
    A[:, -3:] = 0.0001                                   # 입술 닫힘 (/m/ 머머)
    Ps = np.full(T, 8 * td.CMH2O)
    o_closed, r_closed = td.simulate(A, np.full(T, 17.0), Ag, Ps, Av=np.zeros(T))
    o_open, r_open = td.simulate(A, np.full(T, 17.0), Ag, Ps, Av=np.full(T, 1.0))
    s = int(0.15 * FS)
    assert np.abs(r_closed[s:, 6]).max() < 1e-3 * np.abs(r_open[s:, 6]).max()
    assert np.std(o_open[s:]) > 10 * np.std(o_closed[s:])


@pytest.mark.parametrize("a_ip", [0.05, 0.16])
def test_piriform_branch_is_stable_when_its_junction_cell_is_narrow(a_ip):
    """이상와 입구 관성에 구강 칸 반을 직렬로 넣었다 — 곁관 몫만 두면 갈라지는 칸이 0.16 cm² 로 좁아질 때 발산했다 (§52.478)."""
    T = int(0.15 * FS)
    t = np.arange(T) / FS
    ph = (t * 230.0) % 1.0
    Ag = 0.02 + 0.15 * np.where(ph < 0.6, 0.5 * (1 - np.cos(2 * np.pi * ph / 0.6)), 0.0)
    A = np.full((T, N), 3.0)
    ip = int(td.PIR_FRAC * N)
    A[:, ip - 1:ip + 2] = a_ip
    out, _ = td.simulate(A, np.full(T, 14.0), Ag, np.full(T, 8 * td.CMH2O), noise_g=0.01, noise_c=0.01, pir=(1.6, 1.2))
    assert np.isfinite(out).all() and np.abs(out).max() < 1e8


def test_piriform_fossa_carves_an_antiresonance():
    """닫힌 곁관의 1/4 파장 반공명 — 곁관이 있으면 그 근처(5–8 kHz)에 골이 선다 (Dang & Honda 1997; 여성 5–6 kHz).
    골 깊이는 곁관 벽의 손실이 정한다 — 젖은 점막 손실(`LOSS_WET`, √f)에서 균일관 기준 ~9 dB (마른 1 kHz 고정 손실에서는 15 dB 넘었다, §52.482)."""
    T = int(0.25 * FS)
    A = np.full((T, N), 3.0)
    Ps = np.zeros(T)
    Ps[10:20] = 1000.0
    def db(pir):
        o, _ = td.simulate(A, np.full(T, 15.0), np.full(T, 1e-4), Ps, pir=pir)
        X = np.abs(np.fft.rfft(o, 1 << 20)); f = np.fft.rfftfreq(1 << 20, 1 / FS)
        return f, 20 * np.log10(X + 1e-30)
    f, d0 = db((0.0, 1.2))
    _, d1 = db((0.8, 1.2))
    m = (f > 4500) & (f < 9000)
    assert (d1 - d0)[m].min() < -7.0


def test_velar_port_is_stable_when_the_oral_cell_under_it_is_narrow():
    """연구개 포트 관성에 구강 칸 반을 직렬로 넣었다 — 포트 몫만 두면 연구개 자리 0.1 cm² · 포트 0.47 cm² 에서 발산했다 (§52.478)."""
    T = int(0.15 * FS)
    t = np.arange(T) / FS
    ph = (t * 230.0) % 1.0
    Ag = 0.02 + 0.15 * np.where(ph < 0.6, 0.5 * (1 - np.cos(2 * np.pi * ph / 0.6)), 0.0)
    A = np.full((T, N), 3.0)
    iv = int(td.VEL_FRAC * N)
    A[:, iv - 2:iv + 2] = 0.05
    out, _ = td.simulate(A, np.full(T, 14.0), Ag, np.full(T, 8 * td.CMH2O), Av=np.full(T, td.A_VMAX), noise_g=0.01, noise_c=0.01)
    assert np.isfinite(out).all() and np.abs(out).max() < 1e8


def _bw(out, lo, hi):
    """봉우리 (주파수, -3 dB 대역폭) — 방사 미분을 걷은 유량 전달에서."""
    X = np.fft.rfft(out, 1 << 21)
    f = np.fft.rfftfreq(1 << 21, 1 / FS)
    db = 20 * np.log10(np.abs(X) / np.maximum(2 * np.pi * f, 1.0) + 1e-30)
    m = np.where((f > lo) & (f < hi))[0]
    pk, _ = find_peaks(db[m], prominence=6)
    res = []
    for p in m[pk]:
        i, j = p, p
        while db[i] > db[p] - 3: i -= 1
        while db[j] > db[p] - 3: j += 1
        res.append((f[p], f[j] - f[i]))
    return res


def test_boundary_layer_bandwidth_grows_as_sqrt_f_with_wet_mucosa_loss():
    """점성·열 경계층 손실은 √(jω) — 공명마다 더하는 대역폭이 LOSS_WET·(S/A)/2π·[√(νω/2) + (γ−1)√(κω/2)] (§52.482).
    예전 1 kHz 고정 손실은 모든 공명에 같은 대역폭을 더해 고역 공명이 좁았다."""
    T = int(0.25 * FS)
    A = np.full((T, N), 3.0)
    Ps = np.zeros(T)
    Ps[10:20] = 1000.0
    on = _bw(td.simulate(A, np.full(T, 17.5), np.full(T, 1e-4), Ps, walls=False, pir=(0.0, 1.0))[0], 300, 9000)
    off = _bw(td.simulate(A, np.full(T, 17.5), np.full(T, 1e-4), Ps, walls=False, losses=False, pir=(0.0, 1.0))[0], 300, 9000)
    nu, kap = td.MU / td.RHO, td.LAMBDA_TH / (td.RHO * td.CP)
    sa = 2 * np.sqrt(np.pi * 3.0) / 3.0
    for (f1, b1), (f0, b0) in zip(on[:5], off[:5]):
        w = 2 * np.pi * f1
        want = td.LOSS_WET * sa / (2 * np.pi) * (np.sqrt(nu * w / 2) + (td.GAMMA - 1) * np.sqrt(kap * w / 2))
        assert abs((b1 - b0) - want) < 0.2 * want, (f1, b1 - b0, want)


def test_closure_blocks_the_steady_flow():
    """√(jω) 손실은 직류에서 0 이다 — 폐쇄를 막는 것은 정상류 점성(푸아죄유 8πμ dx/A²) 저항이다. 없으면 폐쇄 칸 관성만 남아 유량이 샌다."""
    T = int(0.12 * FS)
    A = np.full((T, N), 3.0)
    A[:, 20] = td.A_FLOOR
    Ps = np.full(T, 8 * td.CMH2O) * np.minimum(1, np.arange(T) / (0.01 * FS))
    _, rec = td.simulate(A, np.full(T, 16.0), np.full(T, 0.2), Ps)
    assert np.abs(rec[int(0.08 * FS):, 2]).mean() < 1.0      # 입술 유량 [cm³/s]


def test_fricative_flow_is_set_by_the_jet_losses_of_both_constrictions():
    """ㅅ 배치 (성문 0.4 cm² 벌림, 치경 협착 0.1 cm², 8 cmH2O): 흐름은 두 좁은 곳의 동압 손실이 정한다 —
    U = √(P_s / (ρ/2 · (k_g/A_g² + (1/A_c − 1/A_뒤)²))) (Borda–Carnot; Stevens 1998 무성 마찰 300–500 cm³/s). 협착의 분출 손실이 없으면
    성문만 막아 ~1500 cm³/s 가 흘렀다 (§52.483)."""
    T = int(0.1 * FS)
    A = np.full((T, N), 3.0)
    A[:, 25] = 0.1
    Ps = np.full(T, 8 * td.CMH2O) * np.minimum(1, np.arange(T) / (0.01 * FS))
    _, rec = td.simulate(A, np.full(T, 15.0), np.full(T, 0.4), Ps)
    U = rec[int(0.06 * FS):, 2].mean()
    want = np.sqrt(8 * td.CMH2O / (0.5 * td.RHO * (td.GLOTTIS_KE / 0.4 ** 2 + (1 / 0.1 - 1 / 3.0) ** 2)))
    assert abs(U - want) < 0.05 * want, (U, want)


def test_adjoint_matches_finite_differences_when_turbulence_dominates():
    """치경 마찰 배치(협착 0.08 cm², 성문 0.3 cm²)에서 난류가 출력보다 13 dB 크다 — 난류 음원(동압 × Stevens 배율, 상한 가지 포함)과
    분출 손실의 수반을 본다. 모음 배치의 수반 시험은 난류 몫이 작아 난류 기울기의 결함을 못 잡았다."""
    rng = np.random.default_rng(2)
    T = 2000
    t = np.arange(T) / FS
    A0 = np.full((T, N), 3.0)
    A0[:, 25] = 0.08 + 0.02 * np.sin(2 * np.pi * 60 * t)
    A0[:, 24] = 0.5
    base = {"A": A0, "L": np.full(T, 15.0), "Ag": np.full(T, 0.3) + 0.02 * np.sin(2 * np.pi * 40 * t),
            "Ps": np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.003), "Av": np.zeros(T), "ng": np.array(0.0),
            "nc": np.array(0.02), "Q": np.zeros((T, N))}
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc", "Q")
    w = torch.tensor(rng.standard_normal(T))

    def loss(v):
        return (td.tube_torch(*[v[k] for k in keys[:-1]], seed=3, Q=v["Q"]) * w).sum() * 1e-6
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    for k in ("A", "Ps", "nc", "Ag"):
        d = rng.standard_normal(base[k].shape)
        h = {"A": 1e-6, "Ps": 1e-3, "nc": 1e-7, "Ag": 1e-7}[k]
        vp = {q: torch.tensor(base[q]) for q in keys}
        vm = {q: torch.tensor(base[q]) for q in keys}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)


def _vf_run(f0, r, ch, T=int(0.3 * FS), ps=8.0):
    x = (np.arange(N) + 0.5) / N
    A = np.tile(np.where(x < 0.45, 0.8, 3.5), (T, 1))                   # /a/ 비슷한 두 관
    Ps = np.full(T, ps * td.CMH2O) * np.minimum(1.0, np.arange(T) / (0.01 * FS))
    Q = np.full(T, 1.0 + (f0 - td.VF_NATURAL_F0) / td.VF_DF0_DQ)
    return td.simulate(A, np.full(T, 15.0), np.full(T, ch), Ps, vf=dict(Q=Q, R=np.tile(r, (T, 1))), tapes=True)


def test_vocal_folds_self_oscillate_near_the_commanded_f0_and_close_when_adducted():
    """자기 진동 성대 (§52.488): 면적을 주지 않아도 폐압만으로 떨고, f0 는 긴장 Q 가 정한다 (명령의 ±10 %). 내전(쉼 변위 ≈ 0)이면
    주기마다 닫히고(닫힌 몫 > 0.3), 벌리면(외전 0.2 mm) 막부가 끝까지 닫히지 않는다 (삼각형의 앞쪽만 거의 닿는다). 아래 질량이 위보다 먼저 열리고(점막파), 유량 봉우리는
    면적 봉우리보다 늦다(관성의 기울임)."""
    s = int(0.15 * FS)
    for f0 in (180.0, 260.0):
        _, rec, X = _vf_run(f0, (0.00495, 0.0004), 0.0002)
        a = np.minimum(rec[s:, 7], rec[s:, 8])
        z = a - a.mean()
        up = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
        f = FS / np.median(np.diff(up))
        assert abs(f / f0 - 1.0) < 0.1, (f0, f)
        assert (a < 0.001).mean() > 0.3
        per = int(FS / f)

        def lag(x, y):                                                   # y 가 x 보다 늦은 표본 수 (여러 주기의 상호상관, ± 1/4 주기)
            x, y = x - x.mean(), y - y.mean()
            n = len(x) - per
            ls = np.arange(-per // 4, per // 4 + 1)
            return ls[np.argmax([np.dot(x[per // 2:n], y[per // 2 + l:n + l]) for l in ls])]
        assert lag(rec[s:, 7], rec[s:, 8]) > 0                           # 아래가 먼저 (점막파)
        assert lag(a, rec[s:, 0]) > 0                                    # 유량이 늦다
    _, rec, _ = _vf_run(220.0, (0.02, 0.02), 0.0002)
    assert np.minimum(rec[s:, 7], rec[s:, 8]).min() > 0.0002 + 1e-4


def test_vocal_fold_adjoint_matches_finite_differences():
    """성대 운동(접촉·닫힘 포함) 을 거친 수반이 유한 차분과 맞는다 — 긴장·쉼 변위·뒤쪽 틈·폐압·성도 면적."""
    rng = np.random.default_rng(1)
    T = 3000
    t = np.arange(T) / FS
    A0 = np.tile(np.linspace(1.5, 4.5, N), (T, 1))
    A0[:, 20] = 0.5 + 0.05 * np.sin(2 * np.pi * 40 * t)
    base = {"A": A0, "L": np.full(T, 16.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
            "Ag": np.full(T, 0.004) + 0.002 * np.sin(2 * np.pi * 25 * t),
            "Ps": np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.003), "Av": np.zeros(T),
            "ng": np.array(0.0), "nc": np.array(0.0),
            "VQ": np.full(T, 1.6) + 0.1 * np.sin(2 * np.pi * 20 * t),
            "VR": np.stack([np.full(T, 0.006), np.full(T, 0.002) + 0.001 * np.sin(2 * np.pi * 30 * t)], 1),
            "VS": td.VF_STATIC.copy()}
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc", "VQ", "VR", "VS")
    w = torch.tensor(rng.standard_normal(T))

    def loss(v):
        return (td.tube_torch(*[v[k] for k in keys[:7]], seed=3, vf=(v["VQ"], v["VR"], v["VS"])) * w).sum() * 1e-6
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    assert (np.minimum(td.TubeFn.last_rec[:, 7], td.TubeFn.last_rec[:, 8]) < 0.0045).sum() > 100     # 닫힘을 지난다
    for k in ("A", "L", "Ag", "Ps", "VQ", "VR", "VS"):
        d = rng.standard_normal(base[k].shape) * (base[k] if k == "VS" else 1.0)     # 정적 변수는 상대 흔들림
        h = {"A": 1e-5, "L": 1e-5, "Ag": 1e-7, "Ps": 1e-3, "VQ": 1e-6, "VR": 1e-8, "VS": 1e-7}[k]
        vp = {q: torch.tensor(base[q]) for q in keys}
        vm = {q: torch.tensor(base[q]) for q in keys}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)


def test_abducted_vocal_folds_stay_stable():
    """성대를 크게 벌리고(쉼 변위 1.75 mm) 긴장이 높고(Q 3.25) 입술이 거의 닫힌 자리에서도 풀이가 유한하다 (§52.488). 성문 관성에
    성도 첫 칸의 반을 넣기 전에는 성문이 벌어져 ρd/A 가 작아지자 유량–첫 칸 압력이 발산했다 (p_0 → −10⁸, 변위 → 수 m)."""
    T = int(0.15 * FS)
    A = np.tile(np.linspace(1.0, 2.8, N), (T, 1))
    A[:, -3] = 0.001
    Ps = np.full(T, 5.0 * td.CMH2O) * np.minimum(1.0, np.arange(T) / (0.01 * FS))
    out, rec, X = td.simulate(A, np.full(T, 14.4), np.full(T, 0.0038), Ps,
                              vf=dict(Q=np.full(T, 3.25), R=np.tile([0.175, 0.170], (T, 1))), tapes=True)
    assert np.isfinite(out).all()
    assert np.abs(rec[:, 1]).max() < 3.0 * Ps.max()
    assert np.abs(X).max() < 0.1


def test_side_branch_adjoint_with_helmholtz_neck_matches_finite_differences():
    """곁관 여럿 (좌우 이상와 + 목 있는 후두실, §52.489) 을 거친 수반이 유한 차분과 맞는다."""
    old = td.SIDE_BRANCHES
    td.SIDE_BRANCHES = [(0.12, 0.4, 1.024), (0.12, 0.4, 1.024), (0.0, 0.07, 0.8, 0.15, 0.2)]
    try:
        rng = np.random.default_rng(2)
        T = 3000
        t = np.arange(T) / FS
        ph = (t * 220) % 1
        base = {"A": np.tile(np.linspace(1.0, 3.5, N), (T, 1)), "L": np.full(T, 15.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
                "Ag": 0.01 + 0.12 * np.where(ph < 0.6, 0.5 * (1 - np.cos(2 * np.pi * ph / 0.6)), 0),
                "Ps": np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.003)}
        w = torch.tensor(rng.standard_normal(T))
        z = torch.zeros(T, dtype=torch.float64)

        def loss(v):
            return (td.tube_torch(v["A"], v["L"], v["Ag"], v["Ps"], z, torch.tensor(0.0), torch.tensor(0.0), seed=3) * w).sum() * 1e-6
        vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
        loss(vals).backward()
        for k, h in (("A", 1e-5), ("L", 1e-5), ("Ag", 1e-7), ("Ps", 1e-3)):
            d = rng.standard_normal(base[k].shape)
            vp = {q: torch.tensor(base[q]) for q in base}
            vm = {q: torch.tensor(base[q]) for q in base}
            vp[k] = torch.tensor(base[k] + h * d)
            vm[k] = torch.tensor(base[k] - h * d)
            fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
            ad = float((vals[k].grad.numpy() * d).sum())
            assert np.isfinite(ad) and abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)
    finally:
        td.SIDE_BRANCHES = old


def test_trachea_adjoint_matches_finite_differences_and_default_is_unchanged():
    """기관 (§52.491): 켜면 성문은 기관 윗칸 압력으로 구동되고 수반이 유한 차분과 맞는다 (성대 모드 포함)."""
    rng = np.random.default_rng(4)
    T = 2500
    t = np.arange(T) / FS
    A = np.tile(np.linspace(1.0, 3.5, N), (T, 1))
    Ps = np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.003)
    vf = (torch.tensor(np.full(T, 1.6)), torch.tensor(np.tile([0.006, 0.002], (T, 1))), None)
    td.set_trachea(True)
    try:
        base = {"A": A, "L": np.full(T, 15.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
                "Ag": np.full(T, 0.004) + 0.002 * np.sin(2 * np.pi * 25 * t), "Ps": Ps}
        w = torch.tensor(rng.standard_normal(T))
        z = torch.zeros(T, dtype=torch.float64)

        def loss(v):
            return (td.tube_torch(v["A"], v["L"], v["Ag"], v["Ps"], z, torch.tensor(0.0), torch.tensor(0.0), seed=3, vf=vf) * w).sum() * 1e-6
        vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
        loss(vals).backward()
        for k, h in (("A", 1e-5), ("L", 1e-5), ("Ag", 1e-7), ("Ps", 1e-3)):
            d = rng.standard_normal(base[k].shape)
            vp = {q: torch.tensor(base[q]) for q in base}
            vm = {q: torch.tensor(base[q]) for q in base}
            vp[k] = torch.tensor(base[k] + h * d)
            vm[k] = torch.tensor(base[k] - h * d)
            fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
            ad = float((vals[k].grad.numpy() * d).sum())
            assert np.isfinite(ad) and abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)
    finally:
        td.set_trachea(False)


def test_glottal_area_output_gradient_matches_finite_differences():
    """성문 면적 출력 (`tube_torch(return_area=True)`, §52.491) 의 기울기 — 성대 모드에서 긴장·쉼 변위·폐압·성도 면적으로."""
    rng = np.random.default_rng(5)
    T = 2500
    t = np.arange(T) / FS
    base = {"A": np.tile(np.linspace(1.0, 3.5, N), (T, 1)), "Ps": np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.003),
            "VQ": np.full(T, 1.6) + 0.1 * np.sin(2 * np.pi * 20 * t), "VR": np.tile([0.006, 0.002], (T, 1))}
    w = torch.tensor(rng.standard_normal(T))
    z = torch.zeros(T, dtype=torch.float64)

    def loss(v):
        _, ga = td.tube_torch(v["A"], torch.full((T,), 15.0, dtype=torch.float64), torch.full((T,), 0.004, dtype=torch.float64), v["Ps"], z,
                              torch.tensor(0.0), torch.tensor(0.0), seed=3, vf=(v["VQ"], v["VR"], None), return_area=True)
        return (ga * w).sum()
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    for k, h in (("A", 1e-5), ("Ps", 1e-3), ("VQ", 1e-6), ("VR", 1e-8)):
        d = rng.standard_normal(base[k].shape)
        vp = {q: torch.tensor(base[q]) for q in base}
        vm = {q: torch.tensor(base[q]) for q in base}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)


def test_phase_locked_loop_pins_fold_closures_to_a_jittered_schedule():
    """위상 고정 고리 (§52.493): 명령 f0 가 7 % 틀리고 목표 주기가 ±3 % 씩 떨려도, 풀이기 안의 고리가 닫힘을 목표 시각표에 붙인다
    (고리 없이는 어긋남이 주기의 몫으로 표류한다). 고친 긴장은 VQ 로 돌려주고, 흔드는 폭은 작다."""
    T = int(0.4 * FS)
    x = (np.arange(N) + 0.5) / N
    A = np.tile(np.where(x < 0.45, 0.8, 3.5), (T, 1))
    Ps = np.full(T, 8.0 * td.CMH2O) * np.minimum(1.0, np.arange(T) / (0.01 * FS))
    Q = np.full(T, 1.0 + (200.0 - td.VF_NATURAL_F0) / td.VF_DF0_DQ)
    R = np.tile((0.00495, 0.0004), (T, 1))
    rng = np.random.default_rng(1)
    t, pt = 0.03 * FS, []
    while t < T:
        pt.append(t)
        t += FS / 214.0 * (1 + 0.03 * rng.standard_normal())
    pt = np.array(pt)

    def err(ev):
        late = ev[ev > 0.12 * FS]
        return np.median(np.abs([e - pt[np.argmin(np.abs(pt - e))] for e in late])) / FS * 1000.0
    g0 = dict(td.PLL_GAINS)
    td.PLL_GAINS.update(kp=0.7, ki=0.2, ff=1.0, tau_ms=0.0)            # 빠른 고리: 주기별 떨림까지 쫓는다
    try:
        _pll_fast(td, A, Ps, Q, R, T, pt, err)
    finally:
        td.PLL_GAINS.clear()
        td.PLL_GAINS.update(g0)
    # 기본(부드러운) 고리: 원본 같은 떨림(주기별 1 %), 기본 긴장은 f0 표에서 3 % 어긋나게 (적합에서는 음높이 맞춤이 표로 맞춘다) —
    # 표류는 잡고(0.2 s 뒤), 주기별 떨림은 쫓지 않는다
    import torch
    from formant_ml.engine import voice_td as vt
    Q = np.full(T, float(vt.vf_tension(torch.tensor([210.0 * 1.03], dtype=torch.float64))[0]))
    rng = np.random.default_rng(2)
    t, p2 = 0.03 * FS, []
    while t < T:
        p2.append(t)
        t += FS / 210.0 * (1 + 0.01 * rng.standard_normal())
    p2 = np.array(p2)
    td.simulate(A, np.full(T, 15.0), np.full(T, 0.0002), Ps, vf=dict(Q=Q, R=R, pll=p2))
    ev = td.simulate.last_ev
    late = ev[ev > 0.2 * FS]
    soft = np.median(np.abs([e - p2[np.argmin(np.abs(p2 - e))] for e in late])) / FS * 1000.0
    lq = np.log(td.simulate.last_q / Q)[int(0.2 * FS):]
    assert soft < 0.35 and np.abs(np.diff(lq)).max() < 1e-3, soft          # 계단 없이 (τ)


def _pll_fast(td, A, Ps, Q, R, T, pt, err):
    td.simulate(A, np.full(T, 15.0), np.full(T, 0.0002), Ps, vf=dict(Q=Q, R=R))
    free = err(td.simulate.last_ev)
    td.simulate(A, np.full(T, 15.0), np.full(T, 0.0002), Ps, vf=dict(Q=Q, R=R, pll=pt))
    locked = err(td.simulate.last_ev)
    lq = np.log(td.simulate.last_q / Q)[int(0.12 * FS):]
    assert free > 0.5 and locked < 0.1, (free, locked)
    assert lq.std() < 0.05 and np.abs(lq).max() <= td.PLL_GAINS["umax"] + 1e-9
    det = td.PLL_GAINS["det"]
    try:                                                                  # 최대 유량 감소율로 잡아도 같이 붙는다
        td.PLL_GAINS["det"] = 1
        td.simulate(A, np.full(T, 15.0), np.full(T, 0.0002), Ps, vf=dict(Q=Q, R=R, pll=pt))
        assert err(td.simulate.last_ev) < 0.1
    finally:
        td.PLL_GAINS["det"] = det


def test_body_cover_vocal_fold_adjoint_matches_finite_differences():
    """몸체–덮개 성대(모드 2, §52.500)의 수반이 유한 차분과 맞는다 — 몸체 질량·강성·감쇠·비선형 계수(정적 변수)와 틀마다의 몸체 강성
    배율(VR 셋째 열, 갑상피열근의 대리)까지. 몸체가 실제로 움직이고(폭 > 0.01 mm) 성대가 닫힘을 지나는 자리에서 잰다."""
    rng = np.random.default_rng(2)
    T = 3000
    t = np.arange(T) / FS
    A0 = np.tile(np.linspace(1.5, 4.5, N), (T, 1))
    A0[:, 20] = 0.5 + 0.05 * np.sin(2 * np.pi * 40 * t)
    base = {"A": A0, "L": np.full(T, 16.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
            "Ag": np.full(T, 0.004) + 0.002 * np.sin(2 * np.pi * 25 * t),
            "Ps": np.full(T, 8 * td.CMH2O) * np.minimum(1, t / 0.003), "Av": np.zeros(T),
            "ng": np.array(0.0), "nc": np.array(0.0),
            "VQ": np.full(T, 1.6) + 0.1 * np.sin(2 * np.pi * 20 * t),
            "VR": np.stack([np.full(T, 0.006), np.full(T, 0.002) + 0.001 * np.sin(2 * np.pi * 30 * t),
                            np.full(T, 0.4) + 0.1 * np.sin(2 * np.pi * 15 * t)], 1),
            "VS": td.vf_static_bc()}
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc", "VQ", "VR", "VS")
    w = torch.tensor(rng.standard_normal(T))

    def loss(v):
        return (td.tube_torch(*[v[k] for k in keys[:7]], seed=3, vf=(v["VQ"], v["VR"], v["VS"]), vf_mode=2) * w).sum() * 1e-6
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    assert (np.minimum(td.TubeFn.last_rec[:, 7], td.TubeFn.last_rec[:, 8]) < 0.0045).sum() > 100
    for k in ("A", "Ps", "VQ", "VR", "VS"):
        d = rng.standard_normal(base[k].shape) * (base[k] if k == "VS" else 1.0)
        h = {"A": 1e-5, "Ps": 1e-3, "VQ": 1e-6, "VR": 1e-8, "VS": 1e-7}[k]
        vp = {q: torch.tensor(base[q]) for q in keys}
        vm = {q: torch.tensor(base[q]) for q in keys}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, ad, fd)
    # 정적 변수 가운데 몸체 몫(14–18)만 따로
    d = np.zeros_like(base["VS"])
    d[14:19] = rng.standard_normal(5) * base["VS"][14:19]
    vp = {q: torch.tensor(base[q]) for q in keys}
    vm = {q: torch.tensor(base[q]) for q in keys}
    vp["VS"] = torch.tensor(base["VS"] + 1e-6 * d)
    vm["VS"] = torch.tensor(base["VS"] - 1e-6 * d)
    fd = (float(loss(vp)) - float(loss(vm))) / 2e-6
    ad = float((vals["VS"].grad.numpy() * d).sum())
    assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), ("몸체", ad, fd)


def test_ns_strip_vocal_fold_adjoint_matches_finite_differences():
    """N 줄 보–막 성대(모드 3, §52.504)의 수반이 유한 차분과 맞는다 — 쉼 변위·틀별 물리 계수 13 개(VR 열 3–15)·긴장·정적 변수(입구·출구·
    접촉·결합 비)까지. 성대가 닫힘을 지나는 자리에서 잰다."""
    from formant_ml.physics import beam_membrane as BM
    rng = np.random.default_rng(2)
    T = 3000
    t = np.arange(T) / FS
    A0 = np.tile(np.linspace(1.5, 4.5, N), (T, 1))
    A0[:, 20] = 0.5 + 0.05 * np.sin(2 * np.pi * 40 * t)
    VR = np.zeros((T, 16))
    VR[:, 0] = 0.0045 + 0.001 * np.sin(2 * np.pi * 30 * t)
    VR[:, 1] = 0.001 * np.sin(2 * np.pi * 20 * t)
    VR[:, 3:16] = BM.ns_coefs(0.15, 0.3, td.VF_NSTRIP, k_scale=3.0)[None, :] * (1.0 + 0.05 * np.sin(2 * np.pi * 15 * t))[:, None]
    base = {"A": A0, "L": np.full(T, 16.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
            "Ag": np.full(T, 0.004) + 0.002 * np.sin(2 * np.pi * 25 * t),
            "Ps": np.full(T, 10 * td.CMH2O) * np.minimum(1, t / 0.003), "Av": np.zeros(T),
            "ng": np.array(0.0), "nc": np.array(0.0),
            "VQ": np.full(T, 1.0) + 0.05 * np.sin(2 * np.pi * 20 * t), "VR": VR, "VS": BM.vf_static_ns()}
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc", "VQ", "VR", "VS")
    w = torch.tensor(rng.standard_normal(T))

    def loss(v):
        y, ga = td.tube_torch(*[v[k] for k in keys[:7]], seed=3, vf=(v["VQ"], v["VR"], v["VS"]), vf_mode=3, return_area=True)
        return (y * w).sum() * 1e-6 + (ga * w).sum() * 10.0
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    assert (td.TubeFn.last_rec[:, 7] < 0.0045).sum() > 100
    for k, h, cols in (("A", 1e-5, None), ("Ps", 1e-3, None), ("Ag", 1e-8, None), ("VQ", 1e-7, None),
                       ("VR", 1e-7, [0, 1]), ("VR", 1e-7, list(range(3, 16))), ("VS", 1e-7, [12, 13, 14, 15, 16])):
        d = rng.standard_normal(base[k].shape)
        if k == "VR":
            m = np.zeros_like(d)
            m[:, cols] = d[:, cols] * (np.abs(base[k][:, cols]) + 1e-3)
            d = m
        if k == "VS":
            m = np.zeros_like(d)
            m[cols] = d[cols] * np.abs(base[k][cols])
            d = m
        vp = {q: torch.tensor(base[q]) for q in keys}
        vm = {q: torch.tensor(base[q]) for q in keys}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, cols, ad, fd)


def _ns_lr_case(T=3000):
    from formant_ml.physics import beam_membrane as BM
    t = np.arange(T) / FS
    A0 = np.tile(np.linspace(1.5, 4.5, N), (T, 1))
    A0[:, 20] = 0.5 + 0.05 * np.sin(2 * np.pi * 40 * t)
    VR = np.zeros((T, 16))
    VR[:, 0] = 0.0045 + 0.001 * np.sin(2 * np.pi * 30 * t)
    VR[:, 1] = 0.001 * np.sin(2 * np.pi * 20 * t)
    VR[:, 3:16] = BM.ns_coefs(0.15, 0.3, td.VF_NSTRIP, k_scale=3.0)[None, :] * (1.0 + 0.05 * np.sin(2 * np.pi * 15 * t))[:, None]
    return {"A": A0, "L": np.full(T, 16.0) + 0.3 * np.sin(2 * np.pi * 30 * t),
            "Ag": np.full(T, 0.004) + 0.002 * np.sin(2 * np.pi * 25 * t),
            "Ps": np.full(T, 10 * td.CMH2O) * np.minimum(1, t / 0.003), "Av": np.zeros(T),
            "ng": np.array(0.0), "nc": np.array(0.0),
            "VQ": np.full(T, 1.0) + 0.05 * np.sin(2 * np.pi * 20 * t), "VR": VR}


def test_ns_left_right_folds_symmetric_equal_single_fold(monkeypatch):
    """좌우 성대 따로 (§52.533) 가 대칭 (긴장 · 질량 비대칭 0) 이면 단일 성대와 같은 소리 · 성문 면적."""
    from formant_ml.physics import beam_membrane as BM
    base = _ns_lr_case()
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc")
    run = lambda VS: td.tube_torch(*[torch.tensor(base[k]) for k in keys], seed=3,
                                   vf=(torch.tensor(base["VQ"]), torch.tensor(base["VR"]), torch.tensor(VS)), vf_mode=3, return_area=True)
    y1, g1 = run(BM.vf_static_ns())
    monkeypatch.setattr(BM, "LR_FOLDS", True)
    y2, g2 = run(BM.vf_static_ns())
    assert float(torch.abs(g1).max()) > 0.01
    assert float(torch.abs(y1 - y2).max()) <= 1e-9 * float(torch.abs(y1).max())
    assert float(torch.abs(g1 - g2).max()) <= 1e-9 * float(torch.abs(g1).max())


def test_ns_left_right_folds_adjoint_matches_finite_differences(monkeypatch):
    """좌우 성대 따로 (§52.533, 긴장 +4 % / −4 %, 질량 ±3 %) 의 수반이 유한 차분과 맞는다 — 쉼 변위 · 계수 · 긴장 · 정적 변수 (비대칭 VS[21] · VS[22] 포함)."""
    from formant_ml.physics import beam_membrane as BM
    monkeypatch.setattr(BM, "LR_FOLDS", True)
    monkeypatch.setattr(BM, "LR_DQ", 0.04)
    monkeypatch.setattr(BM, "LR_DM", 0.03)
    rng = np.random.default_rng(5)
    base = _ns_lr_case()
    base["VS"] = BM.vf_static_ns()
    T = base["A"].shape[0]
    keys = ("A", "L", "Ag", "Ps", "Av", "ng", "nc", "VQ", "VR", "VS")
    w = torch.tensor(rng.standard_normal(T))

    def loss(v):
        y, ga = td.tube_torch(*[v[k] for k in keys[:7]], seed=3, vf=(v["VQ"], v["VR"], v["VS"]), vf_mode=3, return_area=True)
        return (y * w).sum() * 1e-6 + (ga * w).sum() * 10.0
    vals = {k: torch.tensor(v, requires_grad=True) for k, v in base.items()}
    loss(vals).backward()
    assert (td.TubeFn.last_rec[:, 7] < 0.0045).sum() > 100
    for k, h, cols in (("A", 1e-5, None), ("Ps", 1e-3, None), ("VQ", 1e-7, None), ("VR", 1e-7, [0, 1]), ("VR", 1e-7, list(range(3, 16))),
                       ("VS", 1e-7, [12, 13, 14, 15, 16]), ("VS", 1e-6, [21, 22])):
        d = rng.standard_normal(base[k].shape)
        if k == "VR":
            m = np.zeros_like(d)
            m[:, cols] = d[:, cols] * (np.abs(base[k][:, cols]) + 1e-3)
            d = m
        if k == "VS":
            m = np.zeros_like(d)
            m[cols] = d[cols] * (np.abs(base[k][cols]) + (1.0 if 21 in cols else 0.0))
            d = m
        vp = {q: torch.tensor(base[q]) for q in keys}
        vm = {q: torch.tensor(base[q]) for q in keys}
        vp[k] = torch.tensor(base[k] + h * d)
        vm[k] = torch.tensor(base[k] - h * d)
        fd = (float(loss(vp)) - float(loss(vm))) / (2 * h)
        ad = float((vals[k].grad.numpy() * d).sum())
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1e-12), (k, cols, ad, fd)
