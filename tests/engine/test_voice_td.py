"""시간 영역 경로의 성문 면적 (`engine/voice_td.py`) — 접촉이 닫힘을 만들고, 프레임 이음은 매끈하다 (MEASUREMENTS §52.477)."""
import numpy as np
import torch

from formant_ml.engine import voice_td as vt


CHINK = 0.02


def _area(a_rest, a_vib, n=4000):
    phi = torch.arange(n, dtype=torch.float64) / n
    return (CHINK + vt.contact(a_rest + 0.5 * a_vib * vt.fold_shape(phi))).numpy()


def test_folds_touch_only_when_the_vibration_exceeds_the_rest_gap():
    """떨림이 쉼 틈보다 작으면 닿지 않는다 (닫힌 몫 0, 모서리 없음 — 첫머리), 크면 닫힌 몫이 생기고 닫힘이 꺾인다 (고역)."""
    soft, hard = _area(0.10, 0.08), _area(0.05, 0.30)
    assert soft.min() > CHINK + 0.05
    closed = (hard < CHINK + 1e-3).mean()
    assert 0.2 < closed < 0.6

    def hf(a):                                             # 면적 미분의 20~60 배음 몫 (방사 고역의 대리)
        X = np.abs(np.fft.rfft(np.diff(np.tile(a, 4))))[4::4]
        return X[20:60].sum() / X[1:4].sum()
    assert hf(hard) > 10 * hf(soft)


def test_spline_upsampling_has_no_slope_corners_at_frame_boundaries():
    """직선 보간은 프레임마다 기울기가 꺾여 방사(미분)에서 1 kHz 계단이 된다 — 스플라인은 기울기가 표본마다 조금씩만 바뀐다."""
    x = torch.tensor(np.random.default_rng(0).standard_normal(40), dtype=torch.float64).view(1, -1, 1)
    hop = 96
    lin = vt.frames_to_samples(x, hop)[0, :, 0].numpy()
    spl = vt.spline_up(x, hop)[0, :, 0].numpy()

    def corner(y):                                         # 기울기의 최대 한 표본 변화 / 기울기의 전형적 크기
        y = y[hop:-3 * hop]
        return np.abs(np.diff(y, 2)).max() / np.median(np.abs(np.diff(y)))
    assert corner(lin) > 2.0                                 # 직선: 모서리에서 기울기가 통째로 바뀐다
    assert corner(spl) < 0.3                                 # 스플라인: 한 프레임에 걸쳐 고르게


def test_vocal_fold_mode_phonates_from_controls_and_passes_gradients():
    """성대 모드 (§52.488): 프레임률 제어(폐압·내전·f0)만으로 떨고 (주기가 f0 의 ±15 %), 기울기가 폐압·내전·f0 로 흐른다.
    내전을 0 으로 벌리면 떨지 않는다 (무성)."""
    old = vt.GLOTTIS
    vt.GLOTTIS = "vf"
    try:
        fs, hop, T = 48000.0, 48, 80
        path = vt.TDPath(fs, hop).double()

        def ctrl(add):
            c = {k: torch.full((1, T), v, dtype=torch.float64) for k, v in
                 dict(p_sub=8.0, adduction=add, rd_offset=0.0, velum=0.0, tract_len=15.0, a_c=3.0, c_place=0.9,
                      oral_open=1.0, voice_gain=0.0).items()}
            for k in range(1, 11):
                c[f"art{k}"] = torch.zeros(1, T, dtype=torch.float64)
            return c
        c = ctrl(vt.VF_ADD_MODAL)
        for k in ("p_sub", "adduction"):
            c[k].requires_grad_(True)
        f0 = torch.full((1, T), 230.0, dtype=torch.float64, requires_grad=True)
        st = dict(f0=f0, amp=torch.ones(1, T, dtype=torch.float64), ag_dc=torch.zeros(1, T, dtype=torch.float64))
        pp = torch.zeros(1, T * hop, dtype=torch.float64)
        y = path(c, st, pp, T * hop)["audio"]
        idx, per = vt.TDPath.measure_f0(path.last_rec[0])
        assert idx.size > 8 and abs(np.median(1.0 / per) / 230.0 - 1.0) < 0.15
        (y[0, T * hop // 2:] ** 2).sum().backward()
        for g in (c["p_sub"].grad, c["adduction"].grad, f0.grad):
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        path(ctrl(0.0), dict(st, f0=f0.detach()), pp, T * hop)
        idx, _ = vt.TDPath.measure_f0(path.last_rec[0])
        assert idx.size == 0
    finally:
        vt.GLOTTIS = old


def test_vocal_fold_pressure_rule_follows_level_and_keeps_closures_pressurised():
    """성대 모드의 폐압 초기값 (§52.489): 음압 9 dB 당 폐압 두 배, 구간 앞뒤 무음만 숨 멈춤, 안쪽 무음(파열 폐쇄)은 양옆에서 잇는다."""
    L = np.r_[np.full(50, -60.0), np.full(100, -20.0), np.full(30, -60.0), np.full(100, -29.0), np.full(40, -60.0)]
    v = np.r_[np.zeros(50), np.ones(100), np.zeros(30), np.ones(100), np.zeros(40)].astype(bool)
    P, n = vt.vf_psub_rule(L, v, np.zeros(L.size, bool), 1.0)
    assert n == 90
    assert abs(P[0] - vt.VF_PS_SIL) < 1e-9 and abs(P[-1] - vt.VF_PS_SIL) < 1e-9
    assert abs(P[100] / P[250] - 2.0) < 0.05                     # 9 dB → 두 배
    assert P[165] > 0.9 * min(P[100], P[250])                    # 폐쇄 중에도 폐압이 남는다
