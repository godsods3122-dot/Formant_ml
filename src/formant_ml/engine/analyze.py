"""녹음 -> 제어열 초기값. 복사합성 적합의 출발점을 만든다.

목표 화자의 F0 가 450 Hz 까지 올라가면 Praat 의 LPC 포먼트가 하모닉을 문다. 그래서
**참 포락선(true envelope)** 을 먼저 구해 하모닉 빗살을 지운 뒤 그 포락선에 전극 모형을
맞춘다 (Röbel & Rodet 의 true-envelope). 그러면 F0 와 무관하게 포먼트가 잡힌다.
"""
from __future__ import annotations

import math

import numpy as np

from .control import INDEX, N_FORMANTS, PARAM_NAMES, ControlTrack, default_vector
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


#: **포먼트 배정에 연속성을 건다** (MEASUREMENTS §51.21). 사용자: *"파라미터의 최종값이 다음 창의
#: 초기값이 되어야 하잖아."*
#:
#: 예전에는 창마다 LPC 봉우리를 **주파수 순서로만** F1~F_K 에 넣었다. 그러면 가짜 봉우리가 하나 생기거나
#: 진짜 봉우리가 하나 사라질 때마다 그 위의 슬롯이 통째로 한 칸씩 밀려, 궤적에 창 하나짜리 큰 계단이 생긴다.
#: 직전 창의 값을 전혀 보지 않으므로 그 계단을 막을 방법도 없었다 — 분석 궤적의 움직임 에너지 가운데
#: 20~60 Hz 몫이 F2 28 %, F3 30 %, F4 22 % 였던 것이 이것이다(§51.5). 그 흔들림이 종속 가지의 고역 수준을
#: 흔들어 모음 고역의 거칠기가 됐고, 적합기는 그 출발점 위에서 적합했다.
#:
#: 이제 **직전 창의 값을 기준으로** 후보를 슬롯에 붙인다 (작은 동적 계획법: 후보를 건너뛰거나 슬롯을 비워
#: 두는 것도 값으로 친다). 붙인 뒤에는 **속도 상한**을 건다 — 포먼트 `TRACK_VMAX_F`, 대역폭 `TRACK_VMAX_BW`
#: [로그/ms]. 조음기관은 움직이는 물건이라 한 창에 몇 %를 넘게 뛰지 않는다.
TRACK_CONTINUITY = True

#: **성대 밴드 추적** (MEASUREMENTS §51.23, `engine/bands.py`). 켜면 목표에서 음원 기울기(H1–H2, H2–H4,
#: H4–H8)를 재고, 촘촘한 Rd 격자(밴드) 위를 발화 전체 비터비로 걸어 `rd_offset` 의 초기 궤적을 만든다.
#: 그동안 `rd_offset` 의 초기값은 **전 구간 0** 이었다 — 음원 모양에 대한 관측이 하나도 없었다.
BAND_TRACK = True

#: **비음·폐쇄 분석** (MEASUREMENTS §51.28). 켜면 목표에서 비음 머머와 구강 폐쇄를 찾아 `velum`·`oral_open`·
#: `nasal_f`·`nasal_z` 의 초기 궤적을 만든다.
#:
#: 왜: 그동안 `velum` 은 **전 구간 0**, `oral_open` 은 **전 구간 1** 이었다 — 즉 연구개가 한 번도 열리지 않고
#: 구강이 한 번도 닫히지 않았다. 그런데 목표 문장("아, 일인일실이래. 코로나 땜에…")의 프레임 가운데
#: **19~21 % 가 비음 머머**다. 적합기는 그 구간을 F1 이 낮고 조용한 **모음**으로 흉내 냈고(대역 오차 ±1 dB),
#: 스펙트럼 수준은 맞지만 기제가 달라 ㄴ·ㅁ 이 뭉개진 모음으로 들린다. 머머→모음 전이(장소 단서)도 잘못 난다.
#:
#: 판정 근거(§2 실측): 비음 머머는 0~250 Hz 가 지배하고 2~3 kHz 가 −40 dB 로 죽는다. 모음보다 5~7 dB 조용하다.
NASAL_TRACK = True
#: 머머로 보는 문턱 — (0~500 Hz) − (1~3 kHz) [dB].
NASAL_HL_DB = 22.0
#: 연구개가 열리고 닫히는 데 걸리는 시간 [ms] (실측 머머→모음 전이 80 ms 의 절반쯤).
NASAL_RAMP_MS = 35.0
#: 머머 판정의 둘째 조건 — 1.2~2 kHz 가 0~500 Hz 보다 이만큼 아래여야 한다 [dB]. 구강이 닫혔다는 증거다.
NASAL_F2_DB = 28.0
#: **연구개를 실제로 연다** (MEASUREMENTS §51.34). 오래 꺼 두었던 이유는 켜면 머머 스펙트럼 오차가
#: 4.5 → 8.9~15.3 dB 로 나빠졌기 때문인데, 원인이 분기 자체가 아니라 **적합하는 식과 렌더하는 식이 달랐던 것**
#: 이었다. 셋을 고쳤다 — (1) 적합 식에 엔진의 여분 극 16 개와 영점 폭 `fz×0.80×damp` 를 넣고,
#: (2) 아날로그 근사 대신 엔진의 바이쿼드 계수를 그대로 평가하고, (3) 머머 관측에서 **음원을 먼저 나눈다**
#: (안 나누면 음원 기울기가 두 번 세인다). 그 결과 머머 오차가 13.24 / 20.70 → **3.90 / 4.97 dB** 가 되어
#: `s040` 은 연구개를 켠 쪽이 끈 쪽(4.61)을 이긴다.
NASAL_VELUM = True
TRACK_VMAX_F = 0.020        # 2 %/ms = 10 ms 에 20 %
TRACK_VMAX_BW = 0.060
TRACK_MAX_JUMP = 0.35       # 이보다 먼 후보는 그 슬롯의 것이 아니다 (로그 비)
TRACK_MISS = 0.30           # 슬롯을 비워 두는 값


def track_formants(cands, voiced, ref, k_max, w_ref=0.10, w_bw=0.20, w_trans=1.0,
                   n_keep=8):
    """창마다의 봉우리 후보를 **발화 전체를 보는 동적 계획법**으로 F1~F_k 궤적에 배정한다.

    `TRACK_CONTINUITY` 의 구현이다. 한 창씩 앞에서 뒤로 정하면(탐욕) 한 번 잘못된 가지에
    물렸을 때 빠져나오지 못한다 — 실측에서 F1 이 2183 Hz 까지 끌려갔다. 전체를 놓고 비터비로
    풀면 뒤의 증거가 앞의 선택을 되돌린다 (Praat 의 `Formant: Track...` 과 같은 구성).

    값 = Σ_k [ `w_ref`·|log(f_k/기준_k)| + `w_bw`·bw_k/f_k ] + `w_trans`·Σ_k |log(f_k/직전 f_k)|.
    기준은 화자 프로파일의 중립 모음이고, 전이 값이 연속성이다.
    """
    n = len(cands)
    F = np.zeros((n, k_max))
    BW = np.zeros((n, k_max))
    lref = np.log(np.asarray(ref[:k_max], float))
    combos, locals_ = [], []
    idx = [i for i in range(n) if voiced[i] and len(cands[i]) >= 1]
    for i in idx:
        c = list(cands[i])
        if len(c) > n_keep:                       # 기준에서 먼 후보부터 버린다
            c.sort(key=lambda fb: min(abs(math.log(max(fb[0], 1.0)) - r) for r in lref))
            c = sorted(c[:n_keep])
        f = np.array([q[0] for q in c], float)
        bw = np.array([q[1] for q in c], float)
        # 후보가 모자라면 기준 자리에 넓은 가짜 극을 세운다 (조합이 늘 존재하도록)
        if len(f) < k_max:
            add = k_max - len(f)
            f = np.concatenate([f, np.exp(lref[-add:])])
            bw = np.concatenate([bw, np.full(add, 900.0)])
            o = np.argsort(f); f, bw = f[o], bw[o]
        from itertools import combinations
        cb = np.array(list(combinations(range(len(f)), k_max)), dtype=np.int64)
        lf = np.log(np.maximum(f[cb], 1.0))
        loc = (w_ref * np.abs(lf - lref[None, :]).sum(1)
               + w_bw * (bw[cb] / np.maximum(f[cb], 1.0)).sum(1))
        combos.append(lf)
        locals_.append(loc)
    if not idx:
        return F, BW, []
    # 비터비
    back, cost = [], locals_[0]
    for t in range(1, len(idx)):
        d = np.abs(combos[t][:, None, :] - combos[t - 1][None, :, :]).sum(2)   # (M_t, M_{t-1})
        tot = cost[None, :] + w_trans * d
        b = tot.argmin(1)
        back.append(b)
        cost = tot[np.arange(tot.shape[0]), b] + locals_[t]
    path = [int(cost.argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    from itertools import combinations
    out = []
    for t, i in enumerate(idx):
        c = list(cands[i])
        if len(c) > n_keep:
            c.sort(key=lambda fb: min(abs(math.log(max(fb[0], 1.0)) - r) for r in lref))
            c = sorted(c[:n_keep])
        f = np.array([q[0] for q in c], float); bw = np.array([q[1] for q in c], float)
        if len(f) < k_max:
            add = k_max - len(f)
            f = np.concatenate([f, np.exp(lref[-add:])]); bw = np.concatenate([bw, np.full(add, 900.0)])
            o = np.argsort(f); f, bw = f[o], bw[o]
        sel = np.array(list(combinations(range(len(f)), k_max)), dtype=np.int64)[path[t]]
        F[i] = f[sel]; BW[i] = np.clip(bw[sel], 40.0, 900.0)
        out.append(i)
    return F, BW, out


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


#: 앞니 다이폴의 **바닥값**. 0 이 아니어야 하는 이유는 위 주석 참조 (로짓이
#: 죽는다). 0.02 는 마찰 소스에 2 % 만 섞이는 값이라 음향적으로 무의미하고,
#: 오직 적합기의 기울기가 흐르게 하는 몫이다.
OBSTACLE_FLOOR = 0.02


#: LPC 로 **실측**하는 포먼트 수. 16 kHz·차수 14 는 5.5 kHz 까지만 담는다.
#: 그 위는 여기서 채우지 않고 **0 으로 남긴다** — `VocalTract._formant_tracks` 가
#: "빠진 포먼트 = 직전 + c/(2L)" 로 만들어 준다. 적합된 F4 를 따라 사다리 전체가
#: 함께 움직여야 하므로, 분석 시점의 F4 로 굳혀 두면 안 된다 (MEASUREMENTS §40).
N_MEASURED = 4


#: **파열(스톱 버스트) 찾기.** 켜면 분석이 목표에서 파열 자리를 찾아 `ControlTrack.bursts` 에 넣는다.
#:
#: 왜: 엔진의 과도음 이벤트는 복사합성에서 **한 번도 켜진 적이 없다**(events = 0). 그래서 파열이
#: 통째로 빠진다 — 목표의 2~8 kHz 는 3 ms 안에 13~31 dB 서는데 합성은 −7.5 ~ +9.2 dB 다
#: (`out/_tmp/burst2.py`, `L53`·`L85` 두 파일 일곱 자리). 파열 뒤 10 ms 의 **수준**은 ±3 dB 로
#: 맞는데 **모서리**만 없다 — 제어가 10~20 ms 격자에 묶여 3 ms 계단을 못 만들고, 손실도 멜 5 ms
#: 홉에 `_expect` 25 ms 평활이라 그 계단을 못 본다. 파열은 폐쇄 위치의 주 단서이므로(Stevens,
#: Blumstein & Stevens 1979) 명료도에 직접 걸린다.
#: **고역 극(F5~F8)을 적합 대상으로 푼다** (MEASUREMENTS §51.42). 분석이 사다리 값으로 출발점을
#: 채우고, 적합은 `--formant-band` 의 구역(섭동 이론 ×0.80~1.25) 안에서만 움직인다.
#:
#: 왜: 사용자: *"고역에서 음성 특성이 전혀 안 나오잖아. f5-f8을 고정시켜놓은 걸로 기억하는데,
#: 얘네 다 풀고 차라리 제약을 걸어. 고역에서 다 고정된 포먼트가 나오니까 백색소음이 되잖아."*
#: 실측이 그대로다 — 켑스트럼 포락의 봉우리−골이 5~10 kHz 에서 목표 11.8 dB 대 합성 **6.6 dB**,
#: 그리고 `HF_Q`·`NOISE_BW_REL`·`NOISE_EXTRA_BW` 를 어떻게 돌려도 6.6~6.7 에서 안 움직인다.
#: 고정된 극은 Q 를 올려도 **같은 자리**에 설 뿐이라 시간에 따라 변하는 구조가 안 생긴다.
HF_FREE = False
C_SOUND_CM = 34000.0

BURST_TRACK = True
BURST_BAND = (2000.0, 8000.0)
BURST_RISE_DB = 12.0
BURST_RISE_MS = 3.0
BURST_BACK_MS = 15.0
BURST_MIN_GAP_MS = 25.0


def find_bursts(y: np.ndarray, sr: float, hop: int) -> np.ndarray:
    """2~8 kHz 가 `BURST_RISE_MS` 안에 `BURST_RISE_DB` 넘게 서는 프레임. `BURST_TRACK` 참조."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = BURST_BAND
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], "bandpass", output="sos")
    b = sosfiltfilt(sos, np.asarray(y, float))
    n = len(b) // hop
    if n < 8:
        return np.zeros(0, dtype=int)
    lv = 20.0 * np.log10(np.sqrt((b[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12)
    ms = 1000.0 * hop / sr
    k = max(1, int(round(BURST_RISE_MS / ms)))
    back = max(1, int(round(BURST_BACK_MS / ms)))
    gap = max(1, int(round(BURST_MIN_GAP_MS / ms)))
    out: list[int] = []
    for i in range(back + 1, n - k - 1):
        if lv[i + k] - lv[i - back:i].max() > BURST_RISE_DB and lv[i + k] > lv.max() - 40.0:
            if not out or i - out[-1] > gap:
                out.append(i)
    return np.asarray(out, dtype=int)


def _nasal_source_db(freqs: np.ndarray, rd: float, f0: float, sr: float) -> np.ndarray:
    """성문 음원의 크기 [dB] 를 `freqs` 위에서 준다 (100 Hz 기준 0 dB). 엔진의 LF 표를 그대로 쓴다."""
    from .glottis import lf_table
    rd = float(np.clip(rd, 0.3, 2.7))
    f0 = float(f0) if 60.0 < float(f0) < 600.0 else 200.0
    rds, coef = lf_table(64, n_harm=256)
    j = int(np.argmin(np.abs(rds.numpy() - rd)))
    mag = np.abs(coef.numpy()[j])
    hk = (np.arange(len(mag)) + 1) * f0
    db = 20.0 * np.log10(mag + 1e-20)
    m = hk < sr / 2.0
    out = np.interp(np.asarray(freqs, float), hk[m], db[m], left=db[m][0], right=db[m][-1])
    return out - np.interp(100.0, hk[m], db[m])


def analyze(y: np.ndarray, sr: int, prof: SpeakerProfile, hop: int,
            n_formants: int = N_MEASURED, order: int | None = None,
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
    cand_all: list = [[] for _ in range(n)]
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
        cand_all[i] = fmts
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
                    # 채움 대역폭은 손실 법칙 (`VocalTract.default_bw` 와 같은 식).
                    # 예전의 `200 + 60k` 는 6~10 kHz 에서 물리값의 1.3~1.5 배였다.
                    f = prev + spacing
                    bwv = 40.0 + 0.05 * f
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
        # **앞니 다이폴을 켜 둔다.** 이게 0 이면 적합기가 **원리적으로 못 켠다** —
        # `fit._to_raw` 가 로짓이라 0 은 x 를 1e-4 로 자르고, 거기서 시그모이드
        # 기울기가 1e-4 로 죽는다. 게다가 `PRIOR_W["obstacle"] = 10` 이 초기값에
        # 세게 묶는다. 이중 잠금이라 실제로 **적합된 트랙 전 구간이 0.0001**(= 그
        # 클램프 값) 이었다 — 즉 실제 녹음 복사합성에서는 치찰음의 다이폴 소스가
        # 통째로 꺼져 있었고, 적합기는 그 빈자리를 `fric_gain` 을 중앙 10.2 / 최대
        # 67 까지 올려 레벨로 때웠다(측정, out/long/s101_track.npz).
        # `lat_mix` 를 0.05 로 켜 둔 것과 **같은 부류의 버그**다(바로 아래 주석).
        #
        # 초기값은 같은 `r` 에서 낸다. a_c 가 다이폴 기준 면적(noise.OBSTACLE_A_REF
        # = 0.1 cm²)을 지나는 구간이 r ≈ 27, 협착이 0.5 cm² 로 넓어지는 구간이
        # r ≈ 9.5 다. 그 사이를 부드럽게 잇고, 밖에서는 **바닥이 아니라 작은 값**을
        # 준다 — 모음에서도 기울기가 흘러야 적합기가 되돌릴 수 있다. 그 작은 값이
        # 모음을 더럽히지는 않는다: 다이폴은 마찰 소스에만 걸리고, 그 위에 기하
        # 효율 (0.1/a_c)^2.5 가 또 곱해져 a_c = 3 에서 6e-5 배가 된다.
        g = float(np.clip((r - 9.5) / (27.0 - 9.5), 0.0, 1.0))
        g = g * g * (3.0 - 2.0 * g)                      # smoothstep — 꺾임을 만들지 않는다
        vals[i, INDEX["obstacle"]] = OBSTACLE_FLOOR + (
            float(prof.sibilant.get("obstacle", 0.15)) - OBSTACLE_FLOOR) * g
        # **측지 분기를 켜 둔다(깊이 0 에 가깝게).** 설측음의 정의적 특징은 혀 옆으로
        # 공기가 흐르며 생기는 반공진인데, 지금까지 `lat_*` 가 전부 0 이라 적합기에
        # 그걸 만들 수단이 아예 없었다(실측: 설측 위상 오차 89.5°, 사전을 40 -> 0.5 로
        # 풀어도 변화 없음 — 자유도가 묶인 게 아니라 **없었다**).
        # 주파수는 로그 파라미터라 0 이면 적합 대상에서 빠지므로 미리 켜고, 깊이만
        # `lat_mix` 로 0 부근에서 시작한다. 정확히 0 이면 `_lateral` 이 조기 반환해
        # 기울기가 안 흐르므로 0.05 로 둔다 (대역폭이 20 배로 벌어져 음향 효과는 없다).
        vals[i, INDEX["lat_z1"]] = float(prof.lateral["zeros"][0])
        vals[i, INDEX["lat_z2"]] = float(prof.lateral["zeros"][1])
        vals[i, INDEX["lat_bw"]] = float(prof.lateral["zero_bw"])
        vals[i, INDEX["lat_mix"]] = 0.05

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
    _nasal_pend = None
    if NASAL_TRACK:
        # **비음 머머와 구강 폐쇄를 찾는다** (§51.28). 0~500 Hz 가 1~3 kHz 를 크게 누르는 유성 구간이 머머다.
        w_ = 1024
        fr_ = np.fft.rfftfreq(w_, 1.0 / sr)
        m_lo = (fr_ >= 0) & (fr_ < 500)
        m_mid = (fr_ >= 1000) & (fr_ < 3000)
        m_f2 = (fr_ >= 1200) & (fr_ < 2000)      # 설측·유음은 여기가 살아 있다
        m_z = (fr_ >= 700) & (fr_ < 2500)
        win_ = np.hanning(w_)
        hl = np.zeros(n)
        f2d = np.zeros(n)
        f1n = np.full(n, 280.0)
        fz = np.full(n, 1400.0)
        for i in range(n):
            c = max(i * hop + hop // 2 - w_ // 2, 0)
            s_ = y[c:c + w_]
            if len(s_) < w_:
                s_ = np.pad(s_, (0, w_ - len(s_)))
            P = np.abs(np.fft.rfft(s_ * win_)) ** 2 + 1e-20
            hl[i] = 10 * np.log10(P[m_lo].sum() / P[m_mid].sum())
            f2d[i] = 10 * np.log10(P[m_f2].mean() / P[m_lo].mean())
            k = int(np.argmax(P[m_lo])) if m_lo.any() else 0
            f1n[i] = float(np.clip(fr_[m_lo][k], 150.0, 600.0))
            fz[i] = float(np.clip(fr_[m_z][int(np.argmin(P[m_z]))], 400.0, 4000.0))
        # 유성이면서 (i) 저역이 중역을 크게 누르고 (ii) **F2 대역이 죽어 있어야** 한다.
        # (ii) 가 없으면 설측·유음(§1 의 "설측 닐" F2 2000~2300)까지 비음으로 잡힌다 — 실측에서
        # 그렇게 잡은 판은 머머 스펙트럼 오차가 5.8 → 14.9 dB 로 오히려 나빠졌다.
        nasal = voi[:n] & (hl > NASAL_HL_DB) & (f2d[:n] < -NASAL_F2_DB)
        # 12 ms 미만은 버린다 (머머는 그보다 길다), 그리고 경사로로 연다.
        run, s0 = np.zeros(n), None
        for i, v in enumerate(list(nasal) + [False]):
            if v and s0 is None:
                s0 = i
            if not v and s0 is not None:
                if i - s0 >= 12:
                    run[s0:i] = 1.0
                s0 = None
        if run.any():
            k_ = max(1, int(round(NASAL_RAMP_MS / (1000.0 * hop / sr))))
            ker = np.hanning(2 * k_ + 1); ker /= ker.sum()
            vel = np.convolve(run, ker, mode="same")
            if NASAL_VELUM:
                vals[:, INDEX["velum"]] = np.clip(vel, 0.0, 1.0)
                # 비음은 **구강이 닫힌다** — 그게 비강 파열음을 만든다.
                vals[:, INDEX["oral_open"]] = np.clip(1.0 - vel, 0.02, 1.0)
            sel = run > 0.5
            if sel.sum() >= 8:
                # **비강 분기 적합은 밴드 추적 뒤로 미룬다.** 여기서는 머머 스펙트럼만 모은다 —
                # 음원을 나누려면 Rd 가 필요한데 그건 밴드 추적이 낸다(§51.34).
                S_ = []
                for _i in np.flatnonzero(sel)[::2]:
                    c_ = max(_i * hop + hop // 2 - w_ // 2, 0)
                    q_ = y[c_:c_ + w_]
                    if len(q_) == w_:
                        S_.append(np.abs(np.fft.rfft(q_ * win_)) ** 2)
                if S_:
                    _nasal_pend = (sel.copy(), np.mean(S_, 0), fr_.copy())
            print(f"  비음 머머 {100 * sel.mean():.1f} % (연구개 최대 {vel.max():.2f})", flush=True)
            # **머머 안의 구강 포먼트는 관측되지 않는다.** 구강이 닫혀 있으므로 LPC 봉우리는 비강의 것이거나
            # 잡음이다. 그것을 그대로 구강 포먼트로 쓰면 렌더에서 F2 가 머머 안에서 **3.4~5.5 배 범위를
            # 휩쓴다**(실측). 후보를 비워 추적기가 건너뛰게 하고, 뒤에서 로그 선형으로 이어 준다 —
            # 혀는 폐쇄 동안에도 다음 모음을 향해 움직인다.
            for _i in np.flatnonzero(sel):
                cand_all[_i] = []
            nasal_hold = sel.copy()
    if TRACK_CONTINUITY:
        # **창마다 따로 고른 봉우리를 발화 전체의 연속성으로 다시 배정한다** (§51.21).
        # 순서만으로 배정하면 가짜 봉우리 하나에 슬롯이 통째로 밀린다. 무성 구간은
        # 직전 유성 자세를 유지한다 (조음기관은 그 자리에 있다).
        ref0 = list(prof.vowels.get("a", [700.0, 1200.0, 2600.0]))
        _sp0 = 35000.0 / (2.0 * prof.tract_length_cm)
        while len(ref0) < n_formants:
            ref0.append(ref0[-1] + _sp0)
        Ft, Bt, done = track_formants(cand_all, voi, ref0, n_formants)
        if done:
            hold = None
            for i in range(n):
                if voi[i] and Ft[i, 0] > 0:
                    hold = (Ft[i], Bt[i])
                if hold is not None:
                    for k in range(n_formants):
                        vals[i, INDEX[f"f{k + 1}"]] = hold[0][k]
                        vals[i, INDEX[f"bw{k + 1}"]] = hold[1][k]
            # 관측이 없던 구간(비음 머머)은 **양쪽을 로그 선형으로 잇는다**.
            _skip = locals().get("nasal_hold")
            if _skip is not None and _skip.any():
                good = np.flatnonzero(~_skip[:n] & (Ft[:n, 0] > 0))
                if good.size >= 2:
                    for k in range(1, n_formants + 1):
                        for nm_ in (f"f{k}", f"bw{k}"):
                            col = vals[:n, INDEX[nm_]]
                            vals[:n, INDEX[nm_]] = np.exp(np.interp(
                                np.arange(n), good, np.log(np.maximum(col[good], 1e-6))))
    if BAND_TRACK:
        # **성대 밴드 위를 걷는다** (§51.23). 음원 모양(Rd)의 초기 궤적을 목표에서 관측한다.
        try:
            from . import bands as _bands
            _rds, _feat, _tot = _bands.dense_profile()
            _F = np.stack([vals[:, INDEX[f"f{k}"]] for k in range(1, n_formants + 1)], 1)
            _B = np.stack([vals[:, INDEX[f"bw{k}"]] for k in range(1, n_formants + 1)], 1)
            _obs, _lev, _ok = _bands.measure_slopes(
                y, sr, vals[:, INDEX["f0_target"]], voi, _F, _B, hop,
                spacing=35000.0 / (2.0 * prof.tract_length_cm))
            if _ok.sum() >= 32:
                # 화자 고유의 상수 몫을 떼고, 밴드 추적은 그 둘레의 변화분만 따른다.
                _off, _i0 = _bands.speaker_offset(_obs, _ok, _feat)
                _flip = _bands.flow_flip(vals[:, INDEX["a_c"]], vals[:, INDEX["p_sub"]], voi)
                _path = _bands.track_bands(_obs - _off[None, :], _lev, _ok, _feat, _tot, _flip,
                                           d_rd=float(_rds[1] - _rds[0]))
                print("  음원 밴드: 화자 상수 " + " ".join(
                    f"{n}{v:+.1f}" for n, v in zip(("H1-H2 ", "H2-H4 ", "H4-H8 "), _off))
                    + f" dB, 기준 Rd {_rds[_i0]:.2f}", flush=True)
                _rd = _rds[_path]
                # 유성 구간만 관측이다. 무성 구간은 직전 유성 값을 유지한다.
                _last = float(_rd[np.flatnonzero(_ok)[0]])
                for _i in range(n):
                    if _ok[_i]:
                        _last = float(_rd[_i])
                    else:
                        _rd[_i] = _last
                _add = np.clip(vals[:, INDEX["adduction"]], 0.0, 1.0)
                _base = 0.3 + 2.4 * (1.0 - _add) ** 1.5        # glottis.physiology 의 식
                vals[:, INDEX["rd_offset"]] = np.clip(_rd - _base, -1.5, 1.5)
        except Exception as _e:                                # 관측이 모자라면 예전대로 0
            print(f"  (밴드 추적 건너뜀: {_e})")
    if _nasal_pend is not None:
        # **비강 분기를 머머 스펙트럼에 맞춰서 잰다** (§51.34). 관측 스펙트럼은 음원 × 분기이므로
        # 음원을 먼저 나눈다 — 안 나누면 적합된 "전달함수" 가 음원 기울기를 품고, 엔진이 그것을
        # 다시 음원에 곱해 **음원이 두 번 세인다**.
        _sel, _P, _fr = _nasal_pend
        try:
            from .bands import fit_nasal
            _add = float(np.median(np.clip(vals[_sel, INDEX["adduction"]], 0.0, 1.0)))
            _rdm = (0.3 + 2.4 * (1.0 - _add) ** 1.5
                    + float(np.median(vals[_sel, INDEX["rd_offset"]])))
            _src = _nasal_source_db(_fr, _rdm, float(np.median(vals[_sel, INDEX["f0_target"]])), sr)
            nf1, nf2, nf3, nz, ndamp, nres = fit_nasal(_fr, 10.0 * np.log10(_P + 1e-30) - _src)
            vals[:, INDEX["nasal_f"]] = nf1
            vals[:, INDEX["nasal_f2"]] = nf2
            vals[:, INDEX["nasal_f3"]] = nf3
            vals[:, INDEX["nasal_z"]] = nz
            vals[:, INDEX["nasal_damp"]] = float(np.clip(ndamp, 0.3, 3.0))
            print(f"  비강 적합: 극 {nf1:.0f}/{nf2:.0f}/{nf3:.0f} Hz, 영점 {nz:.0f}, "
                  f"감쇠 {ndamp:.2f}, 잔차 {nres:.1f} dB (Rd {_rdm:.2f})", flush=True)
        except Exception as _e:
            print(f"  (비강 적합 건너뜀: {_e})", flush=True)
    if HF_FREE:
        # **고역 극의 출발점을 사다리에서 채운다** (MEASUREMENTS §51.42). 0 으로 두면 엔진이
        # 내부 사다리를 쓰는데, 그러면 적합 대상이 될 수 없다 — 로그 파라미터의 0 은 시그모이드의
        # 끝이라 기울기가 죽는다(`aux{k}_mix` 에서 같은 실수를 했다).
        sp = C_SOUND_CM / (2.0 * prof.tract_length_cm)
        for k in range(n_formants + 1, 9):
            prev = vals[:, INDEX[f"f{k-1}"]]
            fk = np.where(prev > 0, prev + sp, (2 * k - 1) * sp / 2.0)
            vals[:, INDEX[f"f{k}"]] = fk
            vals[:, INDEX[f"bw{k}"]] = 40.0 + 0.05 * fk
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
    # obstacle 도 함께 뭉갠다 — a_c 와 **같은 r 에서 낸 값**이라 프레임마다 같이
    # 튄다. 안 뭉개면 소스 스펙트럼이 프레임률(1 kHz)로 흔들려 그 자체가
    # 거친 변조가 된다.
    for k in ("f0_target", "adduction", "tension", "a_c", "obstacle"):
        tr[k] = smooth_track(tr[k], m, a_f)
    if BAND_TRACK:
        # 음원 모양은 후두 근육이 움직이는 양이라 포먼트보다 느리다 (20 ms).
        tr["rd_offset"] = smooth_track(tr["rd_offset"], m, a_s)
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
    if BURST_TRACK:
        tr.bursts = find_bursts(y, sr, hop)
        if len(tr.bursts):
            print("  파열 %d 곳: %s" % (len(tr.bursts),
                  ", ".join("%.3f s" % (i * hop / sr) for i in tr.bursts)))
    if pulses is not None and len(pulses):
        rel = pulses - t0
        tr.pulses = rel[(rel >= 0.0) & (rel < n * step)]
    else:
        tr.pulses = np.zeros(0)
    return tr.clamp()
