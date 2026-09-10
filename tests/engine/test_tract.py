"""성도 — 포먼트 위치, 노치, 비강, 고차 극 보정, 스트리밍 상태."""
import numpy as np
import torch

from formant_ml.engine.tract import VocalTract
from formant_ml.engine.control import PARAM_NAMES, INDEX, default_vector

FS, HOP = 44100, 44


def _ctrl(t, **kw):
    v = np.tile(default_vector(), (t, 1))
    for k, x in kw.items():
        v[:, INDEX[k]] = x
    ten = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    return {n: ten[..., INDEX[n]] for n in PARAM_NAMES}


def _impulse_response(tr, c, n_frames):
    n = n_frames * HOP
    x = torch.zeros(1, n); x[0, 10] = 1.0
    z = torch.zeros(1, n)
    with torch.no_grad():
        y = tr(x, z, z, z, c)["audio"][0].numpy()
    f = np.fft.rfftfreq(n, 1 / FS)
    return f, 20 * np.log10(np.abs(np.fft.rfft(y)) + 1e-12)


def test_formant_peaks_at_targets():
    tr = VocalTract(FS, HOP)
    f, H = _impulse_response(tr, _ctrl(60, f1=945, f2=1590, f3=2850), 60)
    for target in (945, 1590, 2850):
        m = (f > target * 0.85) & (f < target * 1.15)
        assert abs(f[m][H[m].argmax()] - target) < 40


def test_higher_pole_correction_keeps_highs_alive():
    """K=8 로 자르면 9~13 kHz 가 죽는다(v1 §6.8). 고차 극이 그것을 메운다."""
    a = VocalTract(FS, HOP, n_extra=0)
    b = VocalTract(FS, HOP)
    fa, Ha = _impulse_response(a, _ctrl(60, f1=945, f2=1590, f3=2850), 60)
    fb, Hb = _impulse_response(b, _ctrl(60, f1=945, f2=1590, f3=2850), 60)
    m = (fa > 9000) & (fa < 13000)
    assert Hb[m].mean() > Ha[m].mean() + 6
    assert (fb > 13000).any() and Hb[fb > 15500].max() < Hb[(fb > 800) & (fb < 2000)].max() + 20


def test_lateral_notch_is_local_and_flat_elsewhere():
    tr = VocalTract(FS, HOP)
    f, H0 = _impulse_response(tr, _ctrl(60, f1=459, f2=1914, f3=2907), 60)
    f, H1 = _impulse_response(tr, _ctrl(60, f1=459, f2=1914, f3=2907, lat_z1=3300,
                                       lat_bw=350, lat_mix=1.0), 60)
    d = H1 - H0
    assert d[(f > 3200) & (f < 3400)].min() < -8
    assert abs(d[(f > 300) & (f < 1500)]).max() < 1.0      # 저역은 그대로
    assert abs(d[(f > 8000) & (f < 12000)]).mean() < 1.5   # 고역도 그대로 (DC 정규화 영점의 +25 dB 없음)


def test_nasal_branch_is_parallel_and_off_is_identity():
    """비강은 캐스케이드에 얹은 필터가 아니라 **병렬 분기**다 (ADR 0010).

    * velum=0 이면 출력이 비강 없는 것과 정확히 같다.
    * velum=1, oral_open=0 (머머) 이면 저역은 살아 있고 중역만 죽는다 —
      실측에서 80~300 Hz 가 모음과 ±2 dB 로 연속이었던 그 성질.
    """
    tr = VocalTract(FS, HOP)
    f, H0 = _impulse_response(tr, _ctrl(60, f1=700, f2=1200, f3=2500), 60)
    f, Hoff = _impulse_response(tr, _ctrl(60, f1=700, f2=1200, f3=2500, velum=0.0), 60)
    assert np.abs(Hoff - H0).max() < 1e-6
    f, Hm = _impulse_response(tr, _ctrl(60, f1=700, f2=1200, f3=2500, velum=1.0,
                                        oral_open=0.0, nasal_f=350, nasal_f2=1200,
                                        nasal_f3=2000, nasal_z=1400, nasal_damp=1.2), 60)
    lo = slice(*np.searchsorted(f, [100, 400]))
    mid = slice(*np.searchsorted(f, [900, 2500]))
    assert Hm[lo].mean() > H0[lo].mean() - 8          # 저역은 모음과 비슷하게 남는다
    # 머머는 모음보다 훨씬 저역 지배적이다 (저역−중역 차이가 크게 벌어진다)
    assert (Hm[lo].mean() - Hm[mid].mean()) > (H0[lo].mean() - H0[mid].mean()) + 8
    # 비음화 모음: 구강도 열려 있으면 두 분기가 더해진다 (간섭 = 극-영점 쌍)
    f, Hn = _impulse_response(tr, _ctrl(60, f1=700, f2=1200, f3=2500, velum=0.6,
                                        oral_open=1.0, nasal_f=350), 60)
    assert not np.allclose(Hn, H0, atol=0.5)


def test_streaming_state_matches_offline():
    tr = VocalTract(FS, HOP)
    t = 80; n = t * HOP
    g = torch.Generator().manual_seed(0)
    du, fr, asp, ev = (torch.randn(1, n, generator=g) for _ in range(4))
    c = _ctrl(t, f1=700, f2=1200, f3=2500, velum=0.3, lat_z1=3000, ap1_f=2000, ap1_r=0.6, a_c=0.2)
    c["f1"][:, 40:] = 400.0                                 # 급변
    torch.set_grad_enabled(False)
    full = tr(du, fr, asp, ev, c)["audio"]
    st = {}
    outs = []
    for i in range(0, t, 17):
        j = min(t, i + 17); s = slice(i * HOP, j * HOP)
        o = tr(du[:, s], fr[:, s], asp[:, s], ev[:, s], {k: v[:, i:j] for k, v in c.items()}, state=st)
        st = o["state"]; outs.append(o["audio"])
    torch.set_grad_enabled(True)
    assert torch.allclose(torch.cat(outs, -1), full, atol=1e-4, rtol=1e-4)


def test_side_branches_do_not_boost_the_high_band():
    """**곁가지는 '꺼짐' 에서 통과여야 하고, 켜도 고역을 들어올리면 안 된다.**

    영점쌍만 DC 에서 정규화하면 먼 대역이 통째로 뜬다. 실측 (MEASUREMENTS §34):

    * 설측: 예전에는 깊이를 `lat_bw / lat_mix` 로 줬다. 적합값(lat_mix 0.065,
      lat_bw 296)이면 실효 대역폭 4554 Hz 가 되어 **8 k +7.4 / 12 k +11.0 /
      16 k +12.6 dB**. 영점이 둘이라 합치면 +20 dB 를 넘는다. 적합기는 이것을
      공짜 고역 셸프로 썼고, 그것이 파찰음 /ㅊ/ 의 "튀는 파형" 이었다
      (첨도 14.15, 목표 3.49; 설측을 끄면 9.45).
    * 비강 안티포먼트: `antiresonator_coeffs` (영점만) 로 1400 Hz 영점을 두면
      **12 k +35 / 20 k +41 dB**.

    둘 다 극쌍 보정(`notch_coeffs`)과 젖음/마름 섞기로 고쳤다.
    """
    import numpy as np
    import torch

    from formant_ml.engine.control import INDEX, PARAM_NAMES, default_vector
    from formant_ml.engine.tract import VocalTract

    fs, hop, n = 48000, 48, 200
    N = n * hop

    def ctrl(**kw):
        v = np.tile(default_vector(), (n, 1))
        for k, x in kw.items():
            v[:, INDEX[k]] = x
        t = torch.tensor(v, dtype=torch.float64).unsqueeze(0)
        return {m: t[..., INDEX[m]] for m in PARAM_NAMES}

    torch.manual_seed(0)
    x = torch.randn(1, N, dtype=torch.float64)
    f = np.fft.rfftfreq(N, 1.0 / fs)

    def band_db(y):
        Y = torch.fft.rfft(y[0]).abs().numpy()
        return np.array([10 * np.log10((Y[(f >= lo) & (f < hi)] ** 2).mean() + 1e-30)
                         for lo, hi in ((200, 1000), (1000, 4000), (4000, 8000),
                                        (8000, 12000), (12000, 20000))])

    base = band_db(x)
    for mix in (0.02, 0.065, 0.3, 1.0):
        tr = VocalTract(fs, hop)
        tr._n_emit = N
        y = tr._lateral(x.clone(), ctrl(lat_mix=mix, lat_bw=296.0,
                                        lat_z1=3300.0, lat_z2=4400.0), {})
        d = band_db(y) - base
        assert d[3] < 1.0 and d[4] < 1.0, f"lat_mix={mix} 에서 고역이 {d[3]:.1f}/{d[4]:.1f} dB 떴다"
        if mix <= 0.065:                       # 거의 꺼진 상태는 전 대역 통과
            assert np.abs(d).max() < 0.5, f"lat_mix={mix} 인데 {np.abs(d).max():.2f} dB 바뀐다"


def _tract_db(tract, formants, n=1 << 14, fs=48000.0, hop=48):
    """성도 종속(구강 + 고차 극)의 크기응답 [dB], 100 Hz 기준 정규화."""
    import numpy as np
    T = n // hop + 2
    c = {}
    for k in range(1, 9):
        c[f"f{k}"] = torch.full((1, T), float(formants[k - 1]))
        c[f"bw{k}"] = torch.zeros(1, T)
    c["velum"] = torch.zeros(1, T)
    x = torch.zeros(1, n)
    x[0, 0] = 1.0
    tract._n_emit = n
    tracks = tract._formant_tracks(c)
    st = {}
    y = tract._extra_cascade(tract._cascade(x, tracks, st, "f"), tracks, st)
    Y = np.fft.rfft(y[0].detach().numpy())
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    d = 20 * np.log10(np.abs(Y) + 1e-30)
    return fr, d - d[int(np.argmin(np.abs(fr - 100.0)))]


def _worst_hole(fr, d, lo, hi, width=3000.0):
    """국소 포락(width 이동평균) 대비 가장 깊이 파인 곳 [dB]."""
    import numpy as np
    k = int(width / (fr[1] - fr[0])) | 1
    env = np.convolve(d, np.ones(k) / k, mode="same")
    m = (fr > lo) & (fr < hi)
    return float((d - env)[m].min())


def test_extra_poles_follow_the_last_formant():
    """고차 극이 F_K 에 묶여 있어야 F_K 가 움직여도 그 위에 구멍이 남지 않는다.

    고정 위치 (2n−1)c/4L 에 두면 F8 이 피팅으로 내려갈 때 F8 과 첫 고차 극 사이가
    벌어진다. 그것이 §7.6 에 미해결로 적혀 있던 8990 Hz 구멍이다.
    """
    from formant_ml.engine.noise import C_SOUND
    from formant_ml.engine.tract import VocalTract
    tr = VocalTract(48000.0, 48)
    f1 = C_SOUND / (4.0 * 14.6)
    base = [(2 * n - 1) * f1 for n in range(1, 9)]
    for f8 in (8990.0, 8400.0, 7900.0):
        form = list(base)
        form[7] = f8
        fr, d = _tract_db(tr, form)
        # 첫 고차 극이 F8 바로 위 c/2L 에 있는지. 전체 기울기가 가파르므로
        # 국소 포락을 뺀 잔차에서 본다 — 봉우리라면 잔차가 양수여야 한다.
        import numpy as np
        k = int(3000.0 / (fr[1] - fr[0])) | 1
        r = d - np.convolve(d, np.ones(k) / k, mode="same")
        want = f8 + C_SOUND / (2.0 * 14.6)
        seg = (fr > want - 350) & (fr < want + 350)
        assert r[seg].max() > 0.5, (f8, r[seg].max())
        # 그 봉우리가 실제로 국소 최대인지 — 양옆 골보다 높아야 한다.
        gap = (fr > want - 900) & (fr < want - 500)
        assert r[seg].max() > r[gap].min() + 1.0, f8


def test_high_band_has_no_structural_hole():
    """5.5~11 kHz 에 구조적 구멍이 없어야 한다 — F_K 를 어디에 두든."""
    from formant_ml.engine.noise import C_SOUND
    from formant_ml.engine.tract import VocalTract
    tr = VocalTract(48000.0, 48)
    f1 = C_SOUND / (4.0 * 14.6)
    base = [(2 * n - 1) * f1 for n in range(1, 9)]
    for f8 in (8990.0, 8400.0, 7900.0):
        form = list(base)
        form[7] = f8
        fr, d = _tract_db(tr, form)
        assert _worst_hole(fr, d, 5500.0, 11000.0) > -4.5, f8


def test_high_band_is_not_sixty_db_down():
    """8~10 kHz 가 DC 대비 −30 dB 아래로 무너지면 안 된다.

    극을 0.60·fs/2 까지만 Q≈1 로 채우던 구조에서는 F8=7900 일 때 −68 dB 였다.
    그 구멍을 곁가지의 고역 셸프가 메우고 있었고, 셸프를 고치자 점수가 떨어졌다.
    """
    from formant_ml.engine.noise import C_SOUND
    from formant_ml.engine.tract import VocalTract
    import numpy as np
    tr = VocalTract(48000.0, 48)
    f1 = C_SOUND / (4.0 * 14.6)
    form = [(2 * n - 1) * f1 for n in range(1, 9)]
    form[7] = 7900.0
    fr, d = _tract_db(tr, form)
    band = (fr >= 8000) & (fr < 10000)
    assert d[band].mean() > -30.0, d[band].mean()


def test_extra_poles_stay_below_nyquist():
    """상한이 있어야 한다. Klatt 공명기는 z=−1 에서 이득이 (1+r)²/(1−r)² 로 뛴다."""
    from formant_ml.engine.tract import VocalTract
    tr = VocalTract(48000.0, 48)
    cap = tr.extra_cap * tr.fs / 2.0
    top = 12000.0 + tr.n_extra * tr.extra_spacing
    assert min(top, cap) <= cap < 0.75 * tr.fs / 2.0
    fr, d = _tract_db(tr, [600, 1800, 3000, 4200, 5400, 6600, 7800, 11500.0])
    import numpy as np
    assert np.isfinite(d).all() and d[(fr > 20000)].max() < 20.0
