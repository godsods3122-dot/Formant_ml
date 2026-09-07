"""녹음 -> 제어열 초기값. 복사합성 적합의 출발점을 만든다.

목표 화자의 F0 가 450 Hz 까지 올라가면 Praat 의 LPC 포먼트가 하모닉을 문다. 그래서
**참 포락선(true envelope)** 을 먼저 구해 하모닉 빗살을 지운 뒤 그 포락선에 전극 모형을
맞춘다 (Röbel & Rodet 의 true-envelope). 그러면 F0 와 무관하게 포먼트가 잡힌다.
"""
from __future__ import annotations

import numpy as np

from .control import INDEX, PARAM_NAMES, ControlTrack, default_vector
from .profile import SpeakerProfile


def true_envelope(logmag: np.ndarray, n_cep: int, iters: int = 40) -> np.ndarray:
    """하모닉 빗살을 지운 로그 스펙트럼 포락선 (Röbel & Rodet 2005)."""
    v = logmag.copy()
    n = 2 * (len(logmag) - 1)
    for _ in range(iters):
        c = np.fft.irfft(v, n)
        c[n_cep:n - n_cep] = 0.0
        e = np.real(np.fft.rfft(c, n))
        v = np.maximum(logmag, e)
    c = np.fft.irfft(v, n); c[n_cep:n - n_cep] = 0.0
    return np.real(np.fft.rfft(c, n))


def envelope_to_lpc(env_db: np.ndarray, order: int) -> np.ndarray:
    """포락선(dB) -> 전극 계수 (자기상관 + Levinson-Durbin)."""
    p = 10.0 ** (env_db / 10.0)
    n = 2 * (len(p) - 1)
    r = np.real(np.fft.irfft(p, n))[:order + 1]
    a = np.zeros(order + 1); a[0] = 1.0
    e = r[0] + 1e-20
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1 - k * k)
        if e <= 0:
            break
    return a


def envelope_peaks(env_db: np.ndarray, f: np.ndarray, n: int = 4,
                   fmin: float = 150.0, fmax: float = 5500.0):
    """참 포락선의 국소 최대 -> [(F, BW)]. 근 풀이보다 튼튼하다.

    LPC 근을 쓰면 차수가 높을 때 저역에 가짜 극이 생기고, 그걸 F1 으로 집으면 궤적이
    통째로 어긋난다(실제로 겪었다: F2 평균이 831 Hz 로 나왔다). 포락선의 봉우리는
    그런 실패가 없다. 대역폭은 봉우리의 −3 dB 폭에서 읽는다.
    """
    out = []
    for i in range(1, len(env_db) - 1):
        if not (fmin <= f[i] <= fmax):
            continue
        if env_db[i] > env_db[i - 1] and env_db[i] >= env_db[i + 1]:
            half = env_db[i] - 3.0
            lo = i
            while lo > 0 and env_db[lo] > half:
                lo -= 1
            hi = i
            while hi < len(env_db) - 1 and env_db[hi] > half:
                hi += 1
            out.append((f[i], max(40.0, f[hi] - f[lo]), env_db[i]))
    out.sort(key=lambda t: -t[2])                     # 큰 봉우리부터
    out = sorted(out[:max(n, 6)], key=lambda t: t[0])
    return [(a, b) for a, b, _ in out[:n]]


def lpc_formants(a: np.ndarray, sr: int, fmin: float = 120.0, fmax: float = 12000.0,
                 bw_max: float = 900.0):
    """LPC 계수 -> [(F, BW)] (주파수 오름차순).

    대역폭 상한이 있어야 한다. 기울기를 흉내내는 가짜 극은 대역폭이 1 kHz 를
    넘게 퍼지는데, 그걸 포먼트로 세면 배정이 한 칸 밀린다.
    """
    rts = np.roots(a)
    rts = rts[np.imag(rts) > 0]
    f = np.angle(rts) * sr / (2 * np.pi)
    bw = -np.log(np.abs(rts) + 1e-12) * sr / np.pi
    m = (f > fmin) & (f < fmax) & (bw < bw_max)
    f, bw = f[m], bw[m]
    o = np.argsort(f)
    return list(zip(f[o], bw[o]))


def glottal_pulses(y: np.ndarray, sr: int, prof: SpeakerProfile) -> np.ndarray:
    """성문 폐쇄 시각(초). Praat 의 상호상관 PointProcess.

    F0 를 주기별로 정확히 주는 것은 피치 궤적이 아니라 **펄스 열**이다. 궤적은 프레임마다
    독립 추정이라 0.5 % 씩 흔들리고, 위상은 F0 의 누적합이라 그 흔들림이 쌓인다. 200 ms
    (48 주기) 면 0.5 % 오차가 0.24 주기의 어긋남이 된다 — 크기는 맞는데 펄스가 어긋난
    소리가 나온다. 펄스 열에서 F0 를 만들면 그 누적 오차가 원리적으로 없다.
    """
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(y.astype(np.float64), sr)
    pt = snd.to_pitch(time_step=0.001, pitch_floor=max(60.0, prof.f0_lo * 0.6),
                      pitch_ceiling=prof.f0_hi * 1.3)
    pp = call([snd, pt], "To PointProcess (cc)")
    n = int(call(pp, "Get number of points"))
    return np.array([call(pp, "Get time from index", i + 1) for i in range(n)])


def f0_from_pulses(pulses: np.ndarray, t_grid: np.ndarray, f0_lo: float,
                   f0_hi: float) -> np.ndarray | None:
    """펄스 열 -> 프레임별 F0. 주기의 중점에 1/T 를 놓고 선형 보간한다."""
    if len(pulses) < 3:
        return None
    d = np.diff(pulses)
    ok = (d > 1.0 / (f0_hi * 1.3)) & (d < 1.0 / (f0_lo * 0.6))
    if ok.sum() < 2:
        return None
    tm, f0 = 0.5 * (pulses[:-1] + pulses[1:])[ok], (1.0 / d)[ok]
    return np.interp(t_grid, tm, f0)


FRICATIVE_HL_DB = 0.0      # 고역(3~16k)/저역(0.1~1k) 비가 이보다 크면 마찰로 본다


def smooth_track(x: np.ndarray, med: int, avg: int) -> np.ndarray:
    """중앙값 -> 이동평균. 프레임별 독립 추정의 흔들림을 지운다.

    분석창이 25 ms 인데 1 ms 마다 독립으로 재면 이웃 프레임이 상관 없는 잡음만큼
    흔들린다. 그 흔들림을 그대로 시변 IIR 계수로 넣으면 **계수가 kHz 로 변조되어**
    광대역 잡음이 생긴다 (실측: 12~24 kHz 대역이 −38 dB 여야 하는데 −5 dB 로 떴다.
    소스를 다 꺼도 남았다 — 소스가 아니라 계수 변조가 만든 에너지였다).
    조음기관은 그렇게 빨리 못 움직인다. 포먼트 전이는 30~80 ms 다.
    """
    n = len(x)
    if n < 3:
        return x
    if med >= 3:
        k = min(med | 1, n if n % 2 else n - 1)
        pad = k // 2
        xp = np.pad(x, pad, mode="edge")
        x = np.median(np.stack([xp[i:i + n] for i in range(k)]), axis=0)
    if avg >= 2:
        k = min(avg, n)
        w = np.hanning(k + 2)[1:-1]
        w = w / w.sum()
        x = np.convolve(np.pad(x, k // 2, mode="edge"), w, mode="same")[k // 2:k // 2 + n]
    return x


def analyze(y: np.ndarray, sr: int, prof: SpeakerProfile, hop: int,
            n_formants: int = 4, order: int | None = None,
            t0: float = 0.0, full: np.ndarray | None = None,
            pulses: np.ndarray | None = None) -> ControlTrack:
    """녹음 -> 제어열 초기값 (프레임 = hop 샘플).

    F0·유성도는 Praat, 포먼트는 참 포락선 + LPC, 세기는 프레임 RMS,
    마찰은 고역/저역 비로 협착 면적을 역산한다.
    """
    import parselmouth
    n = int(len(y) // hop)
    step = hop / sr
    # F0·HNR 은 **파일 전체**에서 잰다. 300 ms 짜리 조각에 Praat 을 걸면 NaN 이 돌아오고
    # 그러면 초기값이 통째로 기본값으로 무너진다(실제로 겪었다).
    src = full if full is not None else y
    if pulses is None:
        pulses = glottal_pulses(src, sr, prof)
    snd = parselmouth.Sound(src.astype(np.float64), sr)
    pt = snd.to_pitch(time_step=step, pitch_floor=max(60.0, prof.f0_lo * 0.6),
                      pitch_ceiling=prof.f0_hi * 1.3)
    hn = snd.to_harmonicity_cc(time_step=step, minimum_pitch=70)
    # 포먼트는 **16 kHz 로 내려서** 잰다. 48 kHz 에서 켑스트럼 리프터를 걸면 포락선이
    # 너무 잘아서 봉우리가 수십 개 나오고, 그중 아무거나 F1 으로 집힌다(측정: F1 320 Hz).
    # 16 kHz + 차수 14 는 포먼트 추적의 고전적 설정이고 5.5 kHz 까지 다 담는다.
    from scipy.signal import resample_poly
    g = np.gcd(int(sr), 16000)
    ya = resample_poly(y, 16000 // g, int(sr) // g)
    # **프리엠퍼시스.** 이걸 빼면 성문 소스의 −12 dB/oct 기울기가 LPC 극 하나를
    # 저역에서 통째로 잡아먹고, 그 가짜 극이 F1 로 집혀 배정이 한 칸씩 밀린다
    # (실측: 여성 /아/ 가 F1 376 / F2 962 로 나왔다. 실제는 F1 900 / F2 1560).
    ya = np.append(ya[0], ya[1:] - 0.97 * ya[:-1])
    sra = 16000
    win = int(0.025 * sra) // 2 * 2
    w = np.hanning(win)
    order = order or 14
    vals = np.tile(default_vector(), (n, 1))
    rms_db = np.zeros(n)
    voi = np.zeros(n, dtype=bool)
    rband = np.zeros(n)
    f0_last = prof.f0_nominal
    # 무성 구간에서 유지할 자세. **첫 프레임이 무성일 수 있으므로** 프로파일의 중립
    # 모음으로 씨앗을 준다 — 그러지 않으면 무성 첫 프레임의 잡음 봉우리를 끝까지 끌고 간다.
    nv = list(prof.vowels.get("a", [700.0, 1200.0, 2600.0]))
    _sp = 35000.0 / (2.0 * prof.tract_length_cm)
    while len(nv) < n_formants:
        nv.append(nv[-1] + _sp)
    f_hold: list[tuple[float, float]] = [(f, 60.0 + 0.06 * f) for f in nv[:n_formants]]
    for i in range(n):
        t = (i + 0.5) * step
        ca = int(t * sra) - win // 2
        sa = ya[max(ca, 0):max(ca, 0) + win]
        if len(sa) < win:
            sa = np.pad(sa, (0, win - len(sa)))
        f0_raw = pt.get_value_at_time(t + t0)
        # **유성 판정은 피치가 잡혔는가로 한다.** HNR 로 가르면 마찰음의 우연한 값이
        # 0~11 dB 로 나와 모음(10~22 dB)과 겹친다(실측: 남성 /사/). 피치는 안 겹친다.
        voiced = f0_raw is not None and f0_raw == f0_raw
        f0 = f0_raw if voiced else f0_last
        f0_last = f0
        # 리프터 컷은 F0 주기(퀘프런시 sra/f0)의 절반보다 아래여야 빗살이 지워진다.
        n_cep = int(np.clip(0.45 * sra / max(f0, 80.0), 14, 45))
        S = np.abs(np.fft.rfft(sa * w)) + 1e-9
        env_db = 20.0 / np.log(10) * true_envelope(np.log(S), n_cep)
        fmts = lpc_formants(envelope_to_lpc(env_db, order), sra, 150.0, 5500.0)
        h = hn.get_value(t + t0)
        h = -20.0 if (h is None or h != h) else h
        wn = int(0.025 * sr) // 2 * 2
        c2 = max(int(t * sr) - wn // 2, 0)
        seg = y[c2:c2 + wn]
        if len(seg) < wn:
            seg = np.pad(seg, (0, wn - len(seg)))
        rms = float(np.sqrt((seg ** 2).mean()) + 1e-12)
        fr = np.fft.rfftfreq(wn, 1 / sr)
        p = np.abs(np.fft.rfft(seg * np.hanning(wn))) ** 2 + 1e-20
        lo = p[(fr >= 100) & (fr < 1000)].sum() + 1e-20
        hi = p[(fr >= 3000) & (fr < 16000)].sum() + 1e-20
        vals[i, INDEX["f0_target"]] = f0
        # 세기 -> 폐압. 발성 역치가 3~5 cmH2O 이므로 그 위에서 로그로 편다.
        rms_db[i] = 20 * np.log10(rms)
        voi[i] = voiced
        vals[i, INDEX["adduction"]] = np.clip(0.10 + 0.5 * (h + 5) / 25.0, 0.02, 0.85)
        vals[i, INDEX["tension"]] = np.clip(
            np.log(f0 / prof.f0_lo) / np.log(prof.f0_hi / prof.f0_lo), 0.0, 1.0)
        # **무성 구간의 포먼트는 믿으면 안 된다.** 포먼트는 성문이 성도를 울릴 때만
        # 뜻이 있다. 마찰음 구간에 LPC 를 걸면 잡음의 우연한 봉우리가 나오고(실측:
        # 남성 /사/ 마찰부에서 F1 이 3539 → 486 → 2134 Hz 로 널뛰었다), 아래의 순서
        # 규칙이 그것들을 120 Hz 간격으로 **겹쳐 쌓아** 4 중 고 Q 극을 만든다.
        # 그 극이 성도 종속을 +88 dB 로 만들어(du rms 0.04 → glottal_path 1028)
        # 복사합성이 통째로 무너졌다. 무성 구간에서는 **직전 유성 프레임의 자세를
        # 유지한다** — 조음기관은 무성 구간에도 그 자리에 있다.
        if voiced:
            # 못 찾은 상위 포먼트는 균등 간격 c/2L 로 채운다. 프로파일 기본값을
            # 그냥 넣으면 F4 < F3 처럼 순서가 뒤집혀 캐스케이드가 엉킨다.
            spacing = 35000.0 / (2.0 * prof.tract_length_cm)
            prev, cur = 0.0, []
            for k in range(1, n_formants + 1):
                if len(fmts) >= k:
                    f, bw = fmts[k - 1]
                    bwv = float(np.clip(bw, 40.0, 900.0))
                else:
                    f, bwv = prev + spacing, 200.0 + 60.0 * k
                f = max(f, prev + 120.0)
                prev = f
                cur.append((f, bwv))
            f_hold = cur
        for k, (f, bwv) in enumerate(f_hold, start=1):
            vals[i, INDEX[f"f{k}"]] = f
            vals[i, INDEX[f"bw{k}"]] = bwv
        # 마찰: 고역/저역 비가 크면 협착이 좁다 (초기값일 뿐, 적합이 다듬는다)
        r = 10 * np.log10(hi / lo)
        rband[i] = r
        vals[i, INDEX["a_c"]] = float(np.clip(3.0 * 10 ** (-(r + 10) / 25.0), 0.06, 3.0))
        vals[i, INDEX["front_len"]] = prof.sib_front_len_cm

    # **세기를 폐압에만 실으면 안 된다.** 폐는 20 ms 만에 압력을 못 바꾼다. 자음의
    # 세기 골(탄음 −9 dB / 40 ms, 비음 폐쇄, 파열음)은 **구강이 닫혀 방사가 줄어서**
    # 생긴다. 전부 p_sub 로 보내면 −9 dB 가 −1.1 dB 로 뭉개진다(실측: 여성 탄음에서
    # p_sub 진폭이 1.05 cmH2O 뿐이었다). 느린 성분(호흡, 150 ms)만 폐압에 싣고
    # 빠른 성분(조음, 10~40 ms)은 성도 출력 이득으로 보낸다.
    # 느린 성분은 **유성 프레임에서만** 잰다. /s/ 동안에도 폐압은 모음과 거의 같다 —
    # 난류가 소리로 바뀌는 효율이 낮아서 조용한 것이지 폐가 쉰 것이 아니다. 무성 구간의
    # 낮은 세기를 그대로 넣으면 폐압이 3.5 cmH2O 까지 떨어지고, 마찰 세기는 레이놀즈
    # 게이트를 통과한 **비선형**이라 그 순간 마찰음이 통째로 사라진다(실측: 남 /사/
    # 마찰부가 −42.8 dB). 유성 프레임 사이를 이어 붙여 호흡의 연속성을 지킨다.
    fm_ms = 1000.0 * hop / sr
    idx = np.arange(n)
    if voi.any():
        slow_v = smooth_track(rms_db[voi], max(3, int(round(75.0 / fm_ms)) | 1),
                              max(2, int(round(150.0 / fm_ms))))
        slow = np.interp(idx, idx[voi], slow_v)
    else:
        slow = smooth_track(rms_db, max(3, int(round(75.0 / fm_ms)) | 1),
                            max(2, int(round(150.0 / fm_ms))))
    vals[:, INDEX["p_sub"]] = np.clip(3.0 + 5.0 * (slow + 60) / 40.0, 0.0, 16.0)
    # 빠른 성분은 **구강 폐쇄**의 표현이다. 다만 마찰 구간에서는 세기를 소스가 정하므로
    # 거기서 이득을 깎으면 안 된다 — 그건 `fric_gain` 의 몫이다.
    #
    # **무성이라는 것만으로 마찰이라고 보면 안 된다.** 비음 머머는 유성인데 약해서
    # 피치 검출이 자주 놓치고(실측: 남 /나/ 300 프레임 중 176 이 무성으로 잡혔다),
    # 그걸 마찰로 오인하면 폐쇄의 세기 골이 통째로 사라진다. 마찰은 **고역/저역비**로
    # 가른다: 실측 무성부 r 중앙값이 /ㅆ/ +16.2, /ㅅ/ +15.9 인데 비음 머머는 −26.2 이라
    # 0 dB 로 깨끗이 갈린다.
    fric = (~voi) & (rband > FRICATIVE_HL_DB)
    fast = np.where(fric, 1.0, np.clip(10.0 ** ((rms_db - slow) / 20.0), 0.05, 4.0))
    vals[:, INDEX["tract_gain"]] = fast
    tr = ControlTrack(vals, frame_ms=1000.0 * hop / sr)
    # 프레임률에 맞춰 평활. 5 ms 프레임이면 창 수를 줄인다.
    fm = tr.frame_ms
    m = max(3, int(round(9.0 / fm)) | 1)
    a_f = max(2, int(round(12.0 / fm)))
    a_s = max(2, int(round(20.0 / fm)))
    for k in range(1, n_formants + 1):
        tr[f"f{k}"] = smooth_track(tr[f"f{k}"], m, a_f)
        tr[f"bw{k}"] = smooth_track(tr[f"bw{k}"], m, a_s)
    for k in ("f0_target", "adduction", "tension", "a_c"):
        tr[k] = smooth_track(tr[k], m, a_f)
    tr["tract_gain"] = smooth_track(tr["tract_gain"], m, max(2, int(round(6.0 / fm))))
    # **F0 는 펄스 열이 우선한다.** 피치 궤적은 프레임마다 독립이라 0.5 % 씩 흔들리고,
    # 위상은 그 누적합이라 200 ms 면 0.24 주기가 어긋난다. 펄스에서 만든 F0 는 그 누적
    # 오차가 원리적으로 없다. 무성 구간에서는 펄스가 없으므로 궤적 값을 그대로 둔다.
    t_grid = (np.arange(n) + 0.5) * step + t0
    pf = f0_from_pulses(pulses, t_grid, prof.f0_lo, prof.f0_hi) if pulses is not None else None
    if pf is not None:
        near = np.zeros(n, dtype=bool)
        if len(pulses):
            k = np.searchsorted(pulses, t_grid).clip(1, len(pulses) - 1)
            near = np.minimum(np.abs(t_grid - pulses[k]),
                              np.abs(t_grid - pulses[k - 1])) < 0.02
        tr["f0_target"] = np.where(near, pf, tr["f0_target"])
    tr.voiced = voi
    tr.fricative = fric
    if pulses is not None and len(pulses):
        rel = pulses - t0
        tr.pulses = rel[(rel >= 0.0) & (rel < n * step)]
    else:
        tr.pulses = np.zeros(0)
    return tr.clamp()
