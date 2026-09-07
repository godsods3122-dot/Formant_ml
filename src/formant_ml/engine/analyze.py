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
            t0: float = 0.0, full: np.ndarray | None = None) -> ControlTrack:
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
    f0_last = prof.f0_nominal
    for i in range(n):
        t = (i + 0.5) * step
        ca = int(t * sra) - win // 2
        sa = ya[max(ca, 0):max(ca, 0) + win]
        if len(sa) < win:
            sa = np.pad(sa, (0, win - len(sa)))
        f0 = pt.get_value_at_time(t + t0)
        f0 = f0_last if (f0 is None or f0 != f0) else f0
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
        vals[i, INDEX["p_sub"]] = np.clip(3.0 + 5.0 * (20 * np.log10(rms) + 60) / 40.0, 0.0, 16.0)
        vals[i, INDEX["adduction"]] = np.clip(0.10 + 0.5 * (h + 5) / 25.0, 0.02, 0.85)
        vals[i, INDEX["tension"]] = np.clip(
            np.log(f0 / prof.f0_lo) / np.log(prof.f0_hi / prof.f0_lo), 0.0, 1.0)
        # 못 찾은 상위 포먼트는 **균등 간격 c/2L** 로 채운다. 프로파일 기본값을
        # 그냥 넣으면 F4 < F3 처럼 순서가 뒤집혀 캐스케이드가 엉킨다.
        spacing = 35000.0 / (2.0 * prof.tract_length_cm)
        prev = 0.0
        for k in range(1, n_formants + 1):
            if len(fmts) >= k:
                f, bw = fmts[k - 1]
                bwv = float(np.clip(bw, 40.0, 900.0))
            else:
                f, bwv = prev + spacing, 200.0 + 60.0 * k
            f = max(f, prev + 120.0)
            prev = f
            vals[i, INDEX[f"f{k}"]] = f
            vals[i, INDEX[f"bw{k}"]] = bwv
        # 마찰: 고역/저역 비가 크면 협착이 좁다 (초기값일 뿐, 적합이 다듬는다)
        r = 10 * np.log10(hi / lo)
        vals[i, INDEX["a_c"]] = float(np.clip(3.0 * 10 ** (-(r + 10) / 25.0), 0.06, 3.0))
        vals[i, INDEX["front_len"]] = prof.sib_front_len_cm
        vals[i, INDEX["tract_gain"]] = 1.0
    tr = ControlTrack(vals, frame_ms=1000.0 * hop / sr)
    # 프레임률에 맞춰 평활. 5 ms 프레임이면 창 수를 줄인다.
    fm = tr.frame_ms
    m = max(3, int(round(9.0 / fm)) | 1)
    a_f = max(2, int(round(12.0 / fm)))
    a_s = max(2, int(round(20.0 / fm)))
    for k in range(1, n_formants + 1):
        tr[f"f{k}"] = smooth_track(tr[f"f{k}"], m, a_f)
        tr[f"bw{k}"] = smooth_track(tr[f"bw{k}"], m, a_s)
    for k in ("f0_target", "p_sub", "adduction", "tension", "a_c"):
        tr[k] = smooth_track(tr[k], m, a_f)
    return tr.clamp()
