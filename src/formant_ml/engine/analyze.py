"""녹음 -> 제어열 초기값. 복사합성 적합의 출발점을 만든다.

목표 화자의 F0 가 450 Hz 까지 올라가면 Praat 의 LPC 포먼트가 하모닉을 문다. 그래서
**참 포락선(true envelope)** 을 먼저 구해 하모닉 빗살을 지운 뒤 그 포락선에 전극 모형을
맞춘다 (Röbel & Rodet 의 true-envelope). 그러면 F0 와 무관하게 포먼트가 잡힌다.
"""
from __future__ import annotations

import math

import numpy as np

from .control import INDEX, N_FORMANTS, PARAM_NAMES, PARAMS, ControlTrack, default_vector
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
#: **밴드 잔차에서 `tilt` 을 관측한다** (§52.101). 끄면 예전처럼 분석기가 `tilt` 을 건드리지 않고,
#: 제어열 기본값 2.0 이 그대로 남아 적합기가 **관측 없이** 자유롭게 움직인다.
BAND_TILT = False
#: `tilt` 관측의 평활 폭 [ms] 과 상한 [dB/oct]. 음원 기울기는 발성 노력이라 음절 안에서 튀지 않는다.
BAND_TILT_MS = 80.0
BAND_TILT_MAX = 6.0

#: **유성 꼬리의 이력(hysteresis)** (MEASUREMENTS §52.416). 사용자: *"1:333 to 1:458 에 왜 끊김이 있는지 모르겠네."*
#: 목표 C2 1.37~1.50 s 는 720~790 Hz 의 목소리가 −26 → −65 dB 로 **잦아드는** 꼬리다(자기상관 0.55~0.7). Praat 의 기본 문턱
#: (무음 0.03 = 파일 최대의 −30 dB, 유성 0.45) 은 그 꼬리를 무성으로 자르고, 그러면 내전 초기값이 0.02 로 떨어져 성문이
#: 발성 문턱 밑에 갇힌다 — 거기서는 `glottis` 의 게이트 기울기가 0 이라 적합기가 되살리지 못한다. 그래서 **이미 유성인
#: 구간의 가장자리에서만** 느슨한 문턱(`VOICE_TAIL_SIL`, `VOICE_TAIL_VOI`)의 피치를 이어 붙인다. 이어 붙이는 동안 F0 는
#: 직전 값에서 `VOICE_TAIL_JUMP` (로그) 안에 있어야 한다 — 연속된 성대 진동이라는 조건이고, 잡음의 우연한 피치를 막는다.
VOICE_TAIL = True
VOICE_TAIL_SIL = 0.003
VOICE_TAIL_VOI = 0.30
VOICE_TAIL_JUMP = 0.12
#: 이어 붙이는 방향 — 1 = 유성 구간의 **끝**(잦아드는 꼬리), −1 = 시작. §52.420: 시작 쪽은 기식 발성 시작을 유성으로 바꿔 전체 적합을 흔들었다.
VOICE_TAIL_DIRS = (1,)
#: **유성이면 성문은 발성 문턱 위에 있어야 한다** (§52.416). 내전 초기값은 HNR 에서 오는데, 잦아드는 꼬리의 HNR 은 **녹음
#: 바닥** 때문에 낮다 — 성문이 벌어져서가 아니다(사슬의 다른 단계의 값을 성문에 넣은 것). 유성 틀에서는 내전을
#: `glottis.threshold(f0, add) ≤ PHON_MARGIN · p_sub` 가 되는 최소값 위로 올린다.
PHON_FLOOR = True
PHON_MARGIN = 0.9
PHON_ADD_MIN = 0.35
PHON_P_MAX = 18.0

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
#: **/이/ 가림** (MEASUREMENTS §52.445). 이 화자의 /이/ 는 F1 ≈ 300 Hz, F2 ≈ 2.7~3.2 kHz 라 0.5~2.5 kHz 가 깊게 파여 (i)(ii) 를 다 통과하고
#: 비음 머머로 잡혔다(C2 "이게" 0.54~0.76 s, K1 "피" 0.19~0.22 s — 초기값이 입을 닫고 연구개를 1.0 으로 열었다). 비음 머머는 입이 닫혀
#: 코로만 방사하므로 고역 포먼트가 약하고, /이/ 는 앞공동 공진이 2.5~4 kHz 에 봉우리를 세운다. 그래서 (iii) 2.5–4 kHz 최대가 1.2–2 kHz
#: 평균보다 `NASAL_I_PROM_DB` 넘게 솟으면 비음이 아니다. 실측 두드러짐: 비음 −11~+6 dB, /이/ +9.8~+17 dB. None 이면 끔(옛 거동).
NASAL_I_PROM_DB: float | None = None
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


#: **갈라지는 저음 발성(creak, vocal fry)의 펄스** (MEASUREMENTS §52.441). Praat 의 펄스는 유성 판정(자기상관 0.45) 을 넘는
#: 틀에서만 나온다. creak 은 주기가 불규칙하고(주기 간 ±20 % 이상) 저주파(60~150 Hz)라 상관이 0.2~0.5 에 머물러 **펄스가 통째로
#: 빠진다** — C2 0.400~0.472 s 는 72.6 ms 동안 펄스가 없었고, 그러면 펄스 차단 게이트(`voicing_gate_from_pulses`)가 그 구간의
#: 배음 음원을 막아 적합기가 성문을 활짝 열고 기식 잡음(속삭임)으로 메웠다 — 사용자가 들은 "0:443~0:447 끊김".
#: 켜면 Praat 펄스 사이의 틈 가운데 **말 수준이고 저역이 주도하는** 곳에서 LPC 잔차 포락의 봉우리를 성문 폐쇄로 잡아 펄스에 더한다.
CREAK_PULSES = False
CREAK_GAP_MUL = 1.8        # 국소 주기의 이 배를 넘는 펄스 간격을 틈으로 본다 (`VGATE_GAP` 와 같은 값)
CREAK_MAX_GAP = 0.150      # 이보다 긴 틈은 쉼으로 보고 건드리지 않는다 [s]
CREAK_LEVEL_DB = 30.0      # 틈의 20 ms 수준이 파일 99 분위의 이 dB 안이어야 한다 (말)
CREAK_LOW_SHARE = 0.5      # 80–1500 Hz 가 0.08–16 kHz 에너지의 이만큼 넘어야 한다 (마찰·파열이 아님)
CREAK_MIN_INT = 0.0025     # 펄스 사이 최소 간격 [s] (이중 펄스까지 허용)
CREAK_MAX_INT = 0.016      # 펄스 사이 최대 간격 [s] (62 Hz)
#: creak 틀의 내전 초기값 바닥 — creak 은 성대를 **세게 붙이고**(큰 내전) 세로 긴장을 늦춘 발성이다 (Gordon & Ladefoged 2001,
#: *Phonation types: a cross-linguistic overview*; Keating et al. 2015). HNR 에서 온 초기값은 불규칙 주기 때문에 낮게 나온다.
CREAK_ADD = 0.6


#: **유성 폐쇄·유성 개방의 펄스** (MEASUREMENTS §52.443). 모음 사이 평음은 폐쇄 중에도 성대가 떤다(보이스 바, Lisker & Abramson 1964)
#: — 그러나 약해서 Praat 이 펄스를 안 낸다. C2 0.137~0.232 s 는 폐쇄·개방·개방 뒤 유성까지 95 ms 동안 펄스가 없어 펄스 게이트가 성대 음원을
#: 막았고, 모든 판이 이 파열의 저역을 15~40 dB 못 냈다. 파열이 든 틈(creak 조건에서 빠지는 곳)에서도 80–1000 Hz 가 **이웃 주기 ±25 % 에서
#: 주기적**(20 ms 창 자기상관 > `VGAP_R`)이고 틈 앞 저역의 −35 dB 안인 창을 유성으로 보고 그 안에 펄스를 둔다. 무성 파열은 폐쇄가 조용하고
#: 기식이 비주기라 걸리지 않는다.
VGAP_R = 0.5            # 25 ms 창, 이웃 주기 ±25 % 지연의 자기상관 (잡음 50/90/99 분위 0.14/0.29/0.43 — §52.443 눈금)
VGAP_LEVEL_DB = 25.0    # 창의 80–1000 Hz 수준이 틈 앞 15 ms 의 이 dB 안 (보이스 바는 모음의 −15~−20 dB)
VGAP_LOW_SHARE = 0.6    # 창의 80–1000 Hz 가 80–16000 Hz 에너지의 이만큼 넘어야 — 치찰·기식은 고역이 주도한다


def _lowband_peaks(x_al, c0, c1, sr, dist):
    """정렬된 극성의 60–2000 Hz 파형에서 봉우리 시각 [s] (두드러짐 ≥ 국소 ±12 ms 최대의 25 %)."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import find_peaks
    seg = x_al[c0:c1]
    if seg.size < 3:
        return np.zeros(0)
    lmax = maximum_filter1d(np.abs(seg), int(0.012 * sr) * 2 + 1)
    pk, pr = find_peaks(seg, distance=dist, prominence=0.0)
    keep = (pr["prominences"] >= 0.25 * lmax[pk]) & (seg[pk] > 0)
    return (c0 + pk[keep]) / sr


def creak_pulses(y: np.ndarray, sr: int, pulses: np.ndarray) -> tuple[np.ndarray, list]:
    """Praat 펄스 열의 **안쪽 틈**에서 성대 펄스를 되찾는다. 반환 (더할 펄스 [s], 채운 틈 [(a, b, 개수, 방식)]).

    방식 "creak" — 불규칙 발성 (§52.441): 20 ms 수준이 말 수준이고 저역 몫이 크며 저역이 이어지고 파열이 없는 틈 전체에서, 60–2000 Hz 파형의
    봉우리(간격 ≥ 2.5 ms, 두드러짐 ≥ 국소 ±12 ms 최대의 25 %)를 폐쇄로 잡는다.
    방식 "voiced" — 유성 폐쇄·개방 (§52.443): creak 조건에서 빠진 틈에서, 저역이 이웃 주기 ±25 % 에서 주기적인 창들 안에서만 같은 봉우리를
    잡는다(간격 ≥ 0.7 주기, 파열 ±3 ms 는 뺀다, 한 무리에 3 개 이상).
    **극성과 시각 기준은 틈 앞뒤의 Praat 펄스에 맞춘다** — Praat 의 점은 주기 안의 특정 자리라, 기준이 다르면 틈 경계에서 펄스 위상이 튄다.
    앞뒤 Praat 펄스 각 6 개에서 두 극성의 최근접 봉우리 시각 차를 재어 흩어짐(MAD)이 작은 극성을 고르고, 그 중앙값만큼 옮긴다.
    (LPC 잔차 포락은 이 화자의 불규칙 발성에서 봉우리가 고르게 깔려 가를 수 없었다 — §52.441.)
    """
    from scipy.signal import butter, find_peaks, sosfiltfilt
    p = np.sort(np.asarray(pulses, dtype=np.float64))
    if p.size < 4:
        return np.zeros(0), []
    y = np.asarray(y, dtype=np.float64)
    d = np.diff(p)
    w20 = int(0.02 * sr)
    top = np.percentile(20.0 * np.log10(np.sqrt(np.convolve(y ** 2, np.ones(w20) / w20, mode="same")) + 1e-12), 99)
    lo_band = sosfiltfilt(butter(4, [60.0 / (sr / 2), 2000.0 / (sr / 2)], "bandpass", output="sos"), y)
    sos_lo = butter(4, [80.0 / (sr / 2), 1500.0 / (sr / 2)], "bandpass", output="sos")
    sos_all = butter(4, [80.0 / (sr / 2), 16000.0 / (sr / 2)], "bandpass", output="sos")
    per_band = sosfiltfilt(butter(4, [80.0 / (sr / 2), 1000.0 / (sr / 2)], "bandpass", output="sos"), y)
    dist = max(1, int(CREAK_MIN_INT * sr))
    # **파열이 든 틈은 creak 이 아니다** — 무성 파열(폐쇄 + 개방 + 기식)도 Praat 펄스의 틈이고 저역 몫이 클 수 있다(K1 "커피" ㅍ 에서
    # 오경보). 분석의 파열 검출(`find_bursts`, 1 ms 홉)이 틈 ±5 ms 안에서 하나라도 찾으면 creak 에서 빼고 유성 폐쇄 방식으로만 본다.
    burst_t = find_bursts(y, sr, int(sr // 1000)) / 1000.0
    lo_env = sosfiltfilt(sos_lo, y) ** 2
    w10 = int(0.010 * sr)
    extra, filled = [], []
    for i in range(len(d)):
        loc = np.median(d[max(0, i - 6):i]) if i > 0 else np.median(d[:6])
        a, b = p[i], p[i + 1]
        if d[i] <= CREAK_GAP_MUL * loc or d[i] > CREAK_MAX_GAP:
            continue
        ia, ib = int(a * sr), int(b * sr)
        seg = y[ia:ib]
        if seg.size < int(0.01 * sr):
            continue
        # 극성·기준점: 틈 앞뒤 Praat 펄스 각 6 개의 최근접 봉우리
        ref = np.concatenate((p[max(0, i - 5):i + 1], p[i + 1:i + 7]))
        best = None
        for sign in (1.0, -1.0):
            x = sign * lo_band
            r0, r1 = max(0, int((ref.min() - 0.01) * sr)), min(len(x), int((ref.max() + 0.01) * sr))
            pk_r, _ = find_peaks(x[r0:r1], distance=dist)
            if pk_r.size < 2:
                continue
            tpk = (r0 + pk_r) / sr
            off = np.array([tpk[np.argmin(np.abs(tpk - q))] - q for q in ref])
            mad = float(np.median(np.abs(off - np.median(off))))
            if best is None or mad < best[0]:
                best = (mad, sign, float(np.median(off)))
        if best is None:
            continue
        _, sign, off = best
        x_al = sign * lo_band
        c0, c1 = max(0, ia - w20), min(len(y), ib + w20)
        has_burst = bool(np.any((burst_t > a - 0.005) & (burst_t < b + 0.005)))
        pre = 10 * np.log10(np.mean(lo_env[max(0, ia - int(0.015 * sr)):ia]) + 1e-20)
        inside = [10 * np.log10(np.mean(lo_env[k:k + w10]) + 1e-20) for k in range(ia, max(ia + 1, ib - w10 + 1), w10 // 2)]
        lv = [20 * np.log10(np.sqrt(np.mean(seg[k:k + w20] ** 2)) + 1e-12)
              for k in range(0, max(1, seg.size - w20 + 1), max(1, w20 // 4))]
        ctx = y[c0:c1]
        e_lo = np.sum(sosfiltfilt(sos_lo, ctx)[ia - c0:ib - c0] ** 2)
        e_all = np.sum(sosfiltfilt(sos_all, ctx)[ia - c0:ib - c0] ** 2) + 1e-20
        creak_ok = (not has_burst and min(inside) >= pre - 20.0
                    and np.mean(np.asarray(lv) > top - CREAK_LEVEL_DB) >= 0.8 and e_lo / e_all >= CREAK_LOW_SHARE)
        if creak_ok:
            tp = _lowband_peaks(x_al, c0, c1, sr, dist) - off
            tp = tp[(tp > a + CREAK_MIN_INT) & (tp < b - CREAK_MIN_INT)]
            seq = np.concatenate(([a], tp, [b]))
            if tp.size and np.diff(seq).max() <= CREAK_MAX_INT and np.diff(seq).min() >= 0.5 * CREAK_MIN_INT:
                extra.extend(tp.tolist())
                filled.append((a, b, int(tp.size), "creak"))
                continue
        # 유성 폐쇄·개방: 이웃 주기 근처에서 주기적이고, 저역이 주도하며, 틈 앞 모음의 −25 dB 안인 창만
        T0 = float(np.median(np.concatenate((d[max(0, i - 6):i], d[i + 1:i + 7]))))
        lags = np.arange(max(1, int(0.8 * T0 * sr)), int(1.25 * T0 * sr) + 1)
        pre_lo = 10 * np.log10(np.mean(per_band[max(0, ia - int(0.015 * sr)):ia] ** 2) + 1e-20)
        hop2 = int(0.002 * sr)
        w25 = int(0.025 * sr)
        all_band = sosfiltfilt(sos_all, ctx)
        centers = np.arange(ia + w25 // 2, ib - w25 // 2 + 1, hop2)
        voiced_c = []
        for cc in centers:
            s_ = per_band[cc - w25 // 2:cc + w25 // 2]
            s_ = s_ - s_.mean()
            e_w = np.mean(s_ ** 2)
            if 10 * np.log10(e_w + 1e-20) < pre_lo - VGAP_LEVEL_DB:
                continue
            e_a = np.mean(all_band[cc - w25 // 2 - c0:cc + w25 // 2 - c0] ** 2) + 1e-20
            if e_w / e_a < VGAP_LOW_SHARE:
                continue
            best_r = 0.0
            for L in lags:
                u, v = s_[:-L], s_[L:]
                eu, ev = np.dot(u, u), np.dot(v, v)
                if eu < 1e-18 or ev < 1e-18 or eu > 4 * ev or ev > 4 * eu:
                    continue
                best_r = max(best_r, np.dot(u, v) / np.sqrt(eu * ev))
            if best_r > VGAP_R:
                voiced_c.append(cc)
        if not voiced_c:
            continue
        vc = np.asarray(voiced_c)
        runs, st = [], 0
        for k in range(1, len(vc) + 1):
            if k == len(vc) or vc[k] - vc[k - 1] > 2 * hop2:
                if k - st >= 3:                        # 10 ms 넘게 이어진 무리만
                    runs.append((vc[st] - w25 // 2, vc[k - 1] + w25 // 2))
                st = k
        dist_v = max(1, int(0.7 * T0 * sr))

        def _per15(t_s):
            """t_s 부터 15 ms 창의 이웃 주기 자기상관 — 파열 **뒤만** 보는 창."""
            i0 = int(t_s * sr)
            q = per_band[i0:i0 + int(0.015 * sr)]
            if q.size < int(0.015 * sr):
                return 0.0
            q = q - q.mean()
            rr = 0.0
            for L in lags:
                u, v = q[:-L], q[L:]
                eu, ev = np.dot(u, u), np.dot(v, v)
                if eu < 1e-18 or ev < 1e-18 or eu > 4 * ev or ev > 4 * eu:
                    continue
                rr = max(rr, np.dot(u, v) / np.sqrt(eu * ev))
            return rr
        in_gap_b = burst_t[(burst_t > a) & (burst_t < b)]
        post_ok = {float(tb): _per15(tb + 0.002) > VGAP_R for tb in in_gap_b}
        tp_all = []
        for r0_, r1_ in runs:
            # 펄스는 유성 창의 **중심** 범위 안에만 — 25 ms 창이 파열 앞 유성 꼬리를 걸쳐 잡아 파열 뒤 기식에 펄스를 두었다(K1 "커피" ㅍ)
            c0_, c1_ = r0_ + w25 // 2, r1_ - w25 // 2
            tp = _lowband_peaks(x_al, max(0, c0_ - w10), min(len(y), c1_ + w10), sr, dist_v) - off
            tp = tp[(tp >= c0_ / sr - 0.5 * T0) & (tp <= c1_ / sr + 0.5 * T0) & (tp > a + 0.5 * T0) & (tp < b - 0.5 * T0)]
            if burst_t.size and tp.size:
                tp = tp[np.all(np.abs(tp[:, None] - burst_t[None, :]) > 0.003, axis=1)]
            # 파열 뒤 펄스는 파열 2 ms 뒤부터의 15 ms 창이 그 자체로 주기적일 때만 (유성 개방 — 무성 파열의 기식은 비주기)
            for tb, ok in post_ok.items():
                if not ok:
                    tp = tp[(tp < tb) | (tp > b)]
            if tp.size >= 3:
                tp_all.extend(tp.tolist())
        if tp_all:
            tq = np.sort(np.asarray(tp_all))
            tq = tq[np.concatenate(([True], np.diff(tq) > 0.001))]      # 무리가 겹쳐 같은 봉우리를 두 번 잡은 것을 지운다
            extra.extend(tq.tolist())
            filled.append((a, b, int(tq.size), "voiced"))
    return np.asarray(extra, dtype=np.float64), filled


#: **치찰 협착 관측** (MEASUREMENTS §52.446). 지금까지의 **모든 판**에서 치찰 구간의 협착 면적이 2.4~3.1 cm²(모음처럼 열림), 마찰 이득 ≈ 0,
#: 성문 활짝(내전 0.02~0.08)·기식 0.6~0.8 이었다 — 적합기가 /ㅅ/ 을 협착의 마찰이 아니라 **성문 기식을 성도에 통과시켜** 흉내 냈다
#: (기식 세기에 곱하는 성문 압력 몫 a_c²/(a_c²+Ag²) 을 살리려고 협착을 열었다). 기식은 성도 전체의 포먼트를 지나므로 2–4 kHz 가 5~13 dB 넘친다
#: (C1b 의 무게중심 9.6~10 kHz 치찰). 켜면 목표의 치찰 틀(무성, 10 ms 창의 4–16 kHz 가 1 kHz 위 에너지의 절반 이상, 저역 비주기, 정점 −45 dB
#: 안, 15 ms 이상 — `scripts/diag/obstruent.py` 의 눈금 맞춘 검출과 같은 정의)을 `track.sibilant` 에 싣고 초기 협착을 `SIB_AC`, 앞니
#: 쌍극 몫을 프로파일 값으로 둔다. 적합 쪽 분포 사전은 `fit.SIB_PRIOR_W`.
SIB_OBS = False
SIB_AC = 0.10              # /ㅅ/ 협착 단면적 [cm²] — 0.05~0.2 (Stevens 1998, 8 장), `noise.OBSTACLE_A_REF` 와 같은 근거


def sibilant_frames(y: np.ndarray, sr: int, n: int, hop: int, voi: np.ndarray) -> np.ndarray:
    """목표의 치찰 틀 (n,) bool — 무성이고 고역(4–16 kHz)이 1 kHz 위 에너지의 절반 이상, 저역 비주기, 말 수준, 15 ms 이상 이어짐."""
    from scipy.signal import butter, sosfiltfilt
    y = np.asarray(y, dtype=np.float64)
    win, st = int(0.010 * sr), int(0.005 * sr)
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, 1.0 / sr)
    hi, ab = (f >= 4000) & (f <= 16000), f >= 1000
    e5 = 10 * np.log10(np.convolve(y ** 2, np.ones(st) / st, mode="same") + 1e-20)
    top = np.percentile(e5, 99)
    lo = sosfiltfilt(butter(4, [80.0 / (sr / 2), 1000.0 / (sr / 2)], "bandpass", output="sos"), y)
    m = max(1, (len(y) - win) // st + 1)
    flag = np.zeros(m, bool)
    for i in range(m):
        seg = y[i * st:i * st + win]
        if seg.size < win:
            break
        P = np.abs(np.fft.rfft(seg * w)) ** 2
        lv = 10 * np.log10(P.sum() / win + 1e-20)
        if lv < top - 45 or P[ab].sum() <= 0 or P[hi].sum() / P[ab].sum() < 0.5:
            continue
        lseg = lo[i * st:i * st + win] - lo[i * st:i * st + win].mean()
        if 10 * np.log10(np.mean(lseg ** 2) + 1e-20) > lv - 20:
            best = 0.0
            for lag in range(int(sr / 950), int(sr / 120)):
                u, v = lseg[:-lag], lseg[lag:]
                eu, ev = np.dot(u, u), np.dot(v, v)
                if eu < 1e-18 or ev < 1e-18 or eu > 4 * ev or ev > 4 * eu:
                    continue
                best = max(best, np.dot(u, v) / np.sqrt(eu * ev))
            if best > 0.4:
                continue
        flag[i] = True
    out = np.zeros(n, bool)
    i = 0
    while i < m:
        if flag[i]:
            j = i
            while j + 1 < m and flag[j + 1]:
                j += 1
            if (j - i + 1) * 5 >= 15:
                a, b = int(i * st / hop), min(n, int((j * st + win) / hop))
                out[a:b] = True
            i = j + 1
        else:
            i += 1
    return out & ~np.asarray(voi[:n], bool)


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
BURST_BAND2 = None          # 두 번째 검출 대역 (예: (5000, 14000)) — 합친다 (§52.481)
BURST_BAND = (2000.0, 8000.0)
BURST_RISE_DB = 12.0
BURST_RISE_MS = 3.0
BURST_BACK_MS = 15.0
BURST_MIN_GAP_MS = 25.0
#: 파열 앞 구간에서 **기준 수준을 잡는 분위** [%]. 100 이면 최댓값(옛 동작)이다.
#:
#: 왜 고쳤나 (MEASUREMENTS §52.395): 사용자가 짚은 스윕 자리(파일 1.443 s)에 **+21 dB** 짜리
#: 파열이 있는데 검출이 놓쳤다. 폐쇄가 완전히 조용하지 않아 앞 15 ms 의 **최댓값**이 높게
#: 잡혔고, 그래서 상승이 +10.9 dB 로 나와 문턱 12.0 을 **1.1 dB 차로** 못 넘었다. 한 칸의
#: 잔향이 폐쇄를 가린 것이다 — 폐쇄는 "푹 꺼졌다가 선다" 로 정의되므로 기준은 최댓값이 아니라
#: 분위여야 한다. 90 분위로 잡으면 그 파열과 2.463 s 의 파열(손으로 확인: 5 ms 고역 +13.2 /
#: +10.9 dB)을 되찾는다. 코퍼스 15 개에서 검출 수는 169 -> 222 로 는다.
BURST_BACK_PCT = 100.0


#: `smooth_over_bursts` 가 이어 붙이는 제어.
#:
#: **협착 계열(`a_c`·`obstacle`·`front_len`)이 여기 꼭 있어야 한다** (MEASUREMENTS §52.338).
#: 파열 순간 레이놀즈 수는 임계(1800)의 1.5~7.6 배로 제대로 터지는데 소리가 안 났다. 까닭은
#: 적합기가 그 순간의 `a_c` 를 0.41 → 0.21~0.30 으로 **조여** 놓아 난류가 성도에서 눌렸기
#: 때문이고, 그 조임은 분석기가 파열을 조음으로 오독한 결과다. 포먼트만 이어 붙이고 협착을
#: 놔두면 반쪽이다 — 파열의 소리는 조음 궤적이 아니라 **과도음 이벤트**가 진다.
BURST_SMOOTH_PARAMS = ("f0_target", "f1", "f2", "f3", "f4",
                       "bw1", "bw2", "bw3", "bw4",
                       "a_c", "obstacle", "front_len")


def smooth_over_bursts(track, half_ms: float, params=None):
    """파열 자리 ±`half_ms` 의 **조음 궤적을 양끝으로 이어 붙인다** (MEASUREMENTS §52.329).

    까닭: 파열은 광대역 과도음인데 포먼트 추적기는 거기에 공진을 맞추려 든다. 실측(0 번 파일
    0.451~3.271 s): 파열에서 4 ms 안쪽의 관측 속도가 f2 **110.22 Hz/ms**(바깥 3.10, 36 배),
    f0 **15.00**(바깥 0.44, 34 배)이다. f2 110 Hz/ms 는 생리 상한(93)마저 넘는다 — 조음이 아니다.
    8~15 ms 까지 남고 15~25 ms 면 가라앉으므로 창은 그만큼만 잡는다.

    적합기는 이 궤적을 그대로 따라가므로(그리고 움직임 예산의 기준도 여기서 나온다) 그 순간의
    급발진이 **활공으로 번져** 사용자가 들은 sweep 이 된다. 파열의 소리는 조음 궤적이 아니라
    **과도음 이벤트**가 져야 한다 (`scripts/eventfit.py`).

    양끝 값을 선형으로 이어 덮어쓴다. 제자리에서 바꾸고 바꾼 프레임 수를 돌려준다.
    """
    import numpy as _np
    if half_ms <= 0:
        return 0
    b = _np.asarray(getattr(track, "bursts", _np.zeros(0, dtype=int)), dtype=int)
    if b.size == 0:
        return 0
    vals = track.values
    n = vals.shape[0]
    w = max(1, int(round(half_ms / float(track.frame_ms))))
    names = tuple(params) if params else BURST_SMOOTH_PARAMS
    cols = [INDEX[p] for p in names if p in INDEX]
    touched = _np.zeros(n, dtype=bool)
    for i in b:
        lo, hi = max(0, int(i) - w), min(n - 1, int(i) + w)
        if hi - lo < 2:
            continue
        touched[lo:hi + 1] = True
        for c in cols:
            a0, a1 = vals[lo, c], vals[hi, c]
            vals[lo:hi + 1, c] = _np.linspace(a0, a1, hi - lo + 1)
    return int(touched.sum())


#: **파열 앞의 폐쇄를 관측한다** (MEASUREMENTS §52.353, 사용자의 "파열음 구현이 미흡" 지적).
#:
#: `a_c` 의 초기값은 고역/저역 비 `r` 하나에서 나온다 — 곧 **소리가 있어야** 협착을 잴 수 있다.
#: 그런데 파열음의 폐쇄는 **소리가 없는 것**이므로 그 자로는 원리적으로 안 보이고, `r` 이 낮게
#: 나와 `a_c` 가 3 cm² (활짝 열림) 로 간다. 실측(`L11`): 파열 다섯 중 **셋에서 `a_c` 가
#: 2.18~2.92 cm²** 였다 — 파열음인데 폐쇄가 없다. 그리고 바로 그 자리에서 포먼트가 수백 Hz 를
#: 휩쓸어(f2 683 Hz) 사용자가 듣는 sweep 이 된다.
#:
#: 그래서 무음을 **직접** 본다: 파열 앞 `STOP_CLOSE_MS` 안에서 국소 발화 레벨보다
#: `STOP_CLOSE_DROP_DB` 아래로 떨어진 이어진 구간을 폐쇄로 보고 `a_c` 를 `STOP_CLOSE_AC` 로 놓는다.
#: `smooth_over_bursts` **뒤에** 걸어야 한다 — 그쪽이 `a_c` 를 파열 너머로 이어 붙이기 때문이다.
STOP_CLOSE = False
STOP_CLOSE_MS = 140.0          # 파열 앞에서 폐쇄를 찾는 창
STOP_CLOSE_DROP_DB = 16.0      # 국소 발화 레벨 대비 이만큼 아래면 폐쇄
#: 폐쇄 구간의 `a_c` [cm²] — **공기역학적으로 닫힌 값** (MEASUREMENTS §52.435). 예전 0.04 는 /ㅅ/(≈0.1) 급 틈이라 벌린 성문·폐압 20 에서
#: 레이놀즈 수 8728 (문턱 1800 의 5 배), 마찰 구동이 /ㅅ/ 의 2.6 배 — 폐쇄마다 쉿 소리를 냈다(사용자: "파형이 분리돼서 들려").
#: 문턱 밑 조건: Re = v·d/ν < 1800, v ≤ √(2·20 cmH2O/ρ) ≈ 5.9 m/s·10 → d < 0.46 mm → A < 0.0017 cm². 여유를 두어 0.001 (Re ≈ 1385).
STOP_CLOSE_AC = 0.001
#: **폐쇄 구간의 `oral_open`** — 이것이 없으면 `a_c` 만 조이는 것은 폐쇄가 아니라 **마찰**이다.
#:
#: `N2` 가 가르쳐 준 것: `a_c` 를 0.04 로 닫아 놨더니 적합기가 파열 셋에서 2.4~2.9 로 도로 열었다.
#: 까닭은 `oral_open` 이 **전 발화 1.000** 이기 때문이다 — `analyze.py` 는 `oral_open` 을
#: `clip(1 − velum, 0.02, 1)` 로만 정하고 비음 밖에서 `velum`≈0 이므로 입이 한 번도 안 닫힌다.
#: 유량이 그대로인 채 협착만 좁아지면 레이놀즈 수가 올라 **마찰이 커진다.** 손실은 그 잡음을
#: 없애려고 `a_c` 를 여는 것이 맞다 — 적합기가 옳았고 관측이 반쪽이었다.
#:
#: 실측(`N2` 적합 트랙): 폐쇄 구간의 `tract_gain` 이 0.19~0.54 인데 모음은 0.87 이다. 곧 적합기는
#: 기하학적 폐쇄를 **이득으로 대체**하고 있었다 — 사용자가 막으라고 한 보상 그 자체다.
#:
#: 물리적 폐쇄는 **협착이 좁고 유량이 없는 것**이다. 둘을 같이 줘야 한다.
STOP_CLOSE_OPEN = 0.02         # `oral_open` 의 하한과 같은 값 (구강 분기를 닫는다)
STOP_CLOSE_MIN_MS = 20.0       # 이보다 짧으면 폐쇄로 안 본다
STOP_CLOSE_RAMP_MS = 5.0       # 양끝을 이 폭으로 잇는다 (계단을 만들지 않는다)
STOP_CLOSE_VBAR_DB = 15.0      # (쓰지 않음 — §52.430: 스펙트럼 비로는 모음과 보이스 바가 안 갈렸다)
STOP_CLOSE_GAP_MS = 15.0       # 폐쇄 끝과 파열 검출 사이 허용 간격
#: **개방을 버스트 자리에 날카롭게** (MEASUREMENTS §52.451). 예전 관측은 폐쇄 끝(음량 문턱 아래의 마지막 틀)에서 `STOP_CLOSE_RAMP_MS`(5 ms) 경사로
#: 열었고 그 끝 자체가 파열 검출 틀(상승이 **시작**되는 틀)보다 앞이라, 공기역학적 개방이 실제 버스트보다 5~8 ms 일렀다(K1b ㅍ Δt −6 ms, K3 ㄲ +5 ms).
#: 파열의 개방은 조음 가운데 가장 빠른 사건이다(접촉이 떨어지면 면적이 불연속으로 커진다). 켜면 폐쇄를 1–16 kHz 1 ms 포락의 **가장 가파른 상승**
#: 바로 앞까지 잇고 개방 경사를 `STOP_RELEASE_RAMP_MS` 로 둔다. 닫히는 쪽 경사는 그대로.
STOP_RELEASE_SHARP = False
STOP_RELEASE_RAMP_MS = 1.0


def close_stops(track, y: np.ndarray, sr: float, hop: int) -> int:
    """파열 앞의 **무음 구간을 폐쇄로 관측**해 `a_c` 를 닫는다. `STOP_CLOSE` 참조.

    바꾼 프레임 수를 돌려준다. 파열이 없으면 아무것도 안 한다.
    """
    b = np.asarray(getattr(track, "bursts", np.zeros(0, dtype=int)), dtype=int)
    if b.size == 0 or STOP_CLOSE_AC <= 0.0:
        return 0
    vals = track.values
    n = vals.shape[0]
    fm = float(track.frame_ms)
    w = max(1, int(round(fm * 1e-3 * sr)))
    m = min(n, len(y) // w)
    if m < 4:
        return 0
    lev = np.full(n, -120.0)
    seg = y[:m * w].reshape(m, w)
    lev[:m] = 20.0 * np.log10(np.sqrt((seg ** 2).mean(1)) + 1e-12)
    speech = float(np.quantile(lev[:m], 0.90))          # 국소가 아니라 발화 전체의 말 레벨
    thr = speech - STOP_CLOSE_DROP_DB
    # **모음은 폐쇄가 아니다** (MEASUREMENTS §52.429). 음량만으로 가르면 약하게 시작하는 모음이 "무음" 으로 잡혀
    # 폐쇄로 닫혔다 — C2 "방" 의 /아/ 가 76~140 ms 동안 입을 닫은 채(oral_open 0.02) 합성됐고, 사용자가 "첫 유성음 직전
    # 입술 무성음이 반영 안 됐다" 고 들었다. 유성(엄격)이면서 중역(0.5~3 kHz)이 저역(80~500 Hz)의 −`STOP_CLOSE_VBAR_DB`
    # 안에 살아 있으면 모음이다. 폐쇄 중 유성(보이스 바)은 저역만 있어 그대로 폐쇄로 남는다.
    vowel_like = np.zeros(n, dtype=bool)
    voi_s = np.asarray(getattr(track, "voiced_strict", getattr(track, "voiced", np.zeros(0))), bool)
    if voi_s.size:
        from scipy.signal import butter, sosfiltfilt
        yl = sosfiltfilt(butter(4, (80.0, 500.0), "bp", fs=sr, output="sos"), y)[:m * w].reshape(m, w)
        ym = sosfiltfilt(butter(4, (500.0, 3000.0), "bp", fs=sr, output="sos"), y)[:m * w].reshape(m, w)
        el = 10.0 * np.log10((yl ** 2).mean(1) + 1e-20)
        em = 10.0 * np.log10((ym ** 2).mean(1) + 1e-20)
        k = min(m, voi_s.size)
        vowel_like[:k] = voi_s[:k] & (em[:k] > el[:k] - STOP_CLOSE_VBAR_DB)
    back = max(1, int(round(STOP_CLOSE_MS / fm)))
    need = max(1, int(round(STOP_CLOSE_MIN_MS / fm)))
    ramp = max(1, int(round(STOP_CLOSE_RAMP_MS / fm)))
    col = INDEX["a_c"]
    col_o = INDEX["oral_open"]
    touched = np.zeros(n, dtype=bool)
    for i in b:
        i = int(i)
        # 파열 **직전**만 보면 안 된다 — 파열 검출 자리와 개방 사이에 몇 프레임이 뜨는 일이 흔하다.
        # 창 안에서 **가장 긴** 무음 구간을 찾는다.
        a0 = max(0, i - back)
        q = lev[a0:i] < thr
        lo = hi = -1
        k = 0
        while k < len(q):
            if not q[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(q) and q[j + 1]:
                j += 1
            # **파열에 가장 가까이 끝나는** 조용한 구간을 고른다 (§52.430). "가장 긴" 구간을 고르면 파열 앞 창에 발화 전 무음과
            # 약한 모음 시작이 함께 들어올 때 그쪽을 골라, 진짜 폐쇄(C2 136~176 ms) 대신 모음(76~140 ms)을 닫았다.
            if (j - k + 1) >= need and (a0 + j) > hi:
                lo, hi = a0 + k, a0 + j
            k = j + 1
        if lo < 0 or hi - lo + 1 < need or (i - hi) > int(round(STOP_CLOSE_GAP_MS / fm)):
            continue
        ramp_r = ramp
        if STOP_RELEASE_SHARP:
            # 가장 가파른 1–16 kHz 상승 (검출 틀 앞 2 ms ~ 뒤 BURST_RISE_MS + 3 ms) — 그 틀 **앞**까지 닫는다
            if "_hf_lev" not in locals():
                from scipy.signal import butter as _bt, sosfiltfilt as _sf
                _yh = _sf(_bt(4, (1000.0, min(16000.0, 0.45 * sr)), "bp", fs=sr, output="sos"), y)[:m * w].reshape(m, w)
                _hf_lev = 10.0 * np.log10((_yh ** 2).mean(1) + 1e-20)
            j0, j1 = max(1, i - 2), min(m - 1, i + int(round((BURST_RISE_MS + 3.0) / fm)))
            if j1 > j0:
                r_ref = j0 + int(np.argmax(np.diff(_hf_lev[j0 - 1:j1 + 1])))
                hi = max(hi, r_ref - 1)
            ramp_r = max(1, int(round(STOP_RELEASE_RAMP_MS / fm)))
        touched[lo:hi + 1] = True
        base = vals[lo, col]
        base_o = vals[lo, col_o]
        base_r, base_ro = vals[min(n - 1, hi + 1), col], vals[min(n - 1, hi + 1), col_o]
        for k in range(lo, hi + 1):
            u_on, u_off = (k - lo) / ramp, (hi - k) / ramp_r
            u = min(u_on, u_off, 1.0)
            b_a, b_o = (base, base_o) if (u_on <= u_off or not STOP_RELEASE_SHARP) else (base_r, base_ro)
            vals[k, col] = (1.0 - u) * b_a + u * STOP_CLOSE_AC
            vals[k, col_o] = (1.0 - u) * b_o + u * STOP_CLOSE_OPEN
    return int(touched.sum())


#: **`front_len` 을 프레임마다 관측한다** (MEASUREMENTS §52.353·354, 사용자의 "ㄱ" 지적).
#:
#: 지금은 분석기가 **전 프레임을 `prof.sib_front_len_cm` 하나로 박는다** (치찰음 값 0.92 cm).
#: 그래서 적합된 궤적이 두 판 모두 **2.4 cm 를 안 넘는다** — 성도 14.6 cm 에서 연구개 폐쇄는
#: 앞공동이 6~8 cm 여야 하므로 **연구개음을 낼 자리가 원리적으로 없다.** `PRIOR_W["front_len"]`
#: 이 10 이라 그 상수에 세게 묶이는 것이 이중 잠금이다 (`obstacle` 과 같은 부류의 결함).
#:
#: 앞공동 공진은 1/4 파장이므로 `front_len = c/(4·f_p)` 다. `f_p` 는 **잡음이 우세한 프레임**의
#: 스펙트럼 정점으로 관측한다 — 유성 구간에서는 배음이 덮어 안 보이므로 붙잡아 잇는다.
#:
#: **한계를 적어 둔다.** 연구개 협착은 앞공동이 길어 `f_p` 가 1~2.5 kHz 로 내려오고 거기서는
#: F2·F3 와 섞인다. 그래서 연구개 쪽 관측은 약하다 — 이 관측의 값은 "관측했다" 보다 **"상수에
#: 박힌 것을 풀었다"** 에 있다. §52.354 가 보였듯 모양 오차는 `front_len` 이 흡수하므로 자리만
#: 풀리면 적합이 나머지를 한다.
FRONT_LEN_OBS = False
FRONT_LEN_BAND = (1200.0, 18000.0)     # `f_p` 를 찾는 범위 [Hz]
FRONT_LEN_MS = 25.0                    # 관측 평활 폭
FRONT_LEN_LO, FRONT_LEN_HI = 0.4, 6.0  # [cm]
#: **주기성**이 이보다 낮아야 잡음 우세로 본다 (F0 지연 자기상관의 최고점).
FRONT_LEN_PERIODIC = 0.45
#: **그리고 마찰이 우세해야 한다** — 조용한 고역/저역 비 [dB] 의 하한.
#:
#: 여기서 두 번 틀렸고 두 번 다 `N1` 이 잡았다.
#:
#: (1) 처음에는 고역비 하나로 갈랐다. 그러면 앞공동이 긴 연구개(마루 1.2~2.5 kHz)를 못 본다 —
#:     ㄱ 을 고치려는 관측이 ㄱ 에서만 안 되는 자가당착이었다.
#: (2) 그래서 비주기성 하나로 바꿨더니 **더 나빴다.** 기식(ㅎ)·유성 개시의 비주기 프레임에서
#:     스펙트럼 정점이 앞공동이 아니라 **모음 포먼트(F1·F2)** 로 잡히고, 그 긴 앞공동을 보간이
#:     발화 전체로 퍼뜨렸다. `N1` 실측: `front_len` 중앙 **5.43 cm** (= 극 1565 Hz), 그래서
#:     6~10 kHz 를 만들던 앞공동이 F2 자리로 내려와 그 대역이 **83.04 → 73.64** 로 무너졌다.
#:
#: 물리가 말하는 것은 이렇다 — **앞공동 공진이 스펙트럼을 지배하는 것은 치찰음뿐이다.** 기식은
#: 성문 잡음이 성도 전체를 지나므로 정점이 모음 포먼트에 선다. 그러니 둘 다 요구한다:
#: 비주기적이고 **또한** 고역이 실려 있을 것.
#:
#: 연구개는 어떻게 하나 — **관측하지 않고 자유만 준다.** `--front-free` 가 `PRIOR_W["front_len"]`
#: 을 낮춰 손실이 요구할 때 적합기가 뒤로 보낼 수 있게 한다. 관측이 안 되는 것을 관측한 척하지
#: 않는 편이 낫다 (§52.354: 모양 오차는 `front_len` 이 흡수하므로 자리만 풀리면 된다).
FRONT_LEN_FRIC_DB = -12.0
#: 관측이 없는 구간을 보간으로 잇는 최대 길이 [ms]. 이보다 길면 **화자 기본값으로 되돌린다** —
#: `N1` 의 붕괴는 전 구간 보간이 만들었다.
FRONT_LEN_HOLD_MS = 60.0


def observe_front_len(track, y: np.ndarray, sr: float, hop: int) -> int:
    """잡음 우세 프레임의 스펙트럼 정점에서 **앞공동 길이를 관측**한다. `FRONT_LEN_OBS` 참조.

    관측된 프레임 수를 돌려준다. 관측이 없으면 트랙을 건드리지 않는다.
    """
    vals = track.values
    n = vals.shape[0]
    fm = float(track.frame_ms)
    w = max(1, int(round(fm * 1e-3 * sr)))
    nfft = 1 << int(np.ceil(np.log2(max(w * 4, 512))))
    half = nfft // 2
    fr = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (fr >= FRONT_LEN_BAND[0]) & (fr < FRONT_LEN_BAND[1])
    lo_b = (fr >= 200.0) & (fr < 1000.0)
    hi_b = (fr >= 4000.0) & (fr < 12000.0)
    win = np.hanning(nfft)
    obs = np.full(n, np.nan)
    for i in range(n):
        c0 = i * w
        a, b2 = c0 - half, c0 + half
        if a < 0 or b2 > len(y):
            continue
        seg = y[a:b2]
        sg = seg - seg.mean()
        e = float((sg * sg).sum())
        if e <= 0.0:
            continue
        r = np.correlate(sg, sg, "full")[len(sg) - 1:]
        k0, k1 = int(sr / 400.0), min(int(sr / 70.0), len(r) - 1)
        if k1 > k0 and float(r[k0:k1].max()) / e > FRONT_LEN_PERIODIC:
            continue                                  # 배음이 덮는 프레임 — 관측 없음
        P = np.abs(np.fft.rfft(seg * win)) ** 2 + 1e-20
        if 10.0 * np.log10(P[hi_b].mean() / P[lo_b].mean()) < FRONT_LEN_FRIC_DB:
            continue                                  # 마찰이 아니다 — 정점이 앞공동이 아니라 포먼트다
        fp = float(fr[band][np.argmax(P[band])])
        obs[i] = float(np.clip(C_SOUND_CM / (4.0 * fp), FRONT_LEN_LO, FRONT_LEN_HI))
    ok = ~np.isnan(obs)
    if not ok.any():
        return 0
    idx = np.arange(n)
    filled = np.interp(idx, idx[ok], obs[ok])
    # **긴 빈틈은 보간하지 않는다.** 전 구간 보간이 `N1` 을 무너뜨렸다 (`FRONT_LEN_FRIC_DB` 참조).
    hold = max(1, int(round(FRONT_LEN_HOLD_MS / fm)))
    base = vals[:, INDEX["front_len"]].copy()
    pos = np.where(ok)[0]
    far = np.full(n, True)
    for a_ in pos:
        far[max(0, a_ - hold):a_ + hold + 1] = False
    filled[far] = base[far]
    k = max(3, int(round(FRONT_LEN_MS / fm)) | 1)
    # **관측 구간과 기본값의 이음매도 평활한다** (MEASUREMENTS §52.463). 중앙값만 거치면 이음매의 계단이 그대로 남는다(중앙값은 계단을 지킨다) —
    # W1 분석 출발점에서 0.553·0.725 s 에 한 틀에 범위의 28 %·100 % 가 뛰었다. 다른 조음 궤적과 같은 12 ms 이동평균을 더한다.
    filled = smooth_track(filled, k, max(2, int(round(12.0 / fm))))
    vals[:, INDEX["front_len"]] = np.clip(filled, FRONT_LEN_LO, FRONT_LEN_HI)
    return int(ok.sum())


def find_bursts(y: np.ndarray, sr: float, hop: int) -> np.ndarray:
    """2~8 kHz 가 `BURST_RISE_MS` 안에 `BURST_RISE_DB` 넘게 서는 프레임. `BURST_TRACK` 참조.

    `BURST_BAND2` 가 있으면 그 대역의 검출을 합친다 (§52.481) — 치조·파찰의 파열은 5–14 kHz 에 선다. 038 "시청" 의 ㅊ(0.601 s)은
    앞 모음 ㅣ 의 F3–F4 가 2–8 kHz 를 채워 그 대역의 오름이 문턱에 못 미쳤고, 폐쇄·개방 관측 없이 적합기가 파열을 만들지 못했다."""
    out = _find_bursts_band(y, sr, hop, BURST_BAND)
    if BURST_BAND2 is not None:
        gap = max(1, int(round(BURST_MIN_GAP_MS / (1000.0 * hop / sr))))
        extra = _find_bursts_band(y, sr, hop, BURST_BAND2)
        keep = [int(e) for e in extra if not out.size or np.min(np.abs(out - e)) > gap]
        out = np.sort(np.concatenate([out, np.asarray(keep, dtype=int)])).astype(int)
    return out


def _find_bursts_band(y: np.ndarray, sr: float, hop: int, band) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    lo, hi = band
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
        ref = (lv[i - back:i].max() if BURST_BACK_PCT >= 100.0
               else float(np.percentile(lv[i - back:i], BURST_BACK_PCT)))
        if lv[i + k] - ref > BURST_RISE_DB and lv[i + k] > lv.max() - 40.0:
            if not out or i - out[-1] > gap:
                out.append(i)
    return np.asarray(out, dtype=int)


def _nasal_source_db(freqs: np.ndarray, rd: float, f0: float, sr: float) -> np.ndarray:
    """성문 음원의 크기 [dB] 를 `freqs` 위에서 준다 (100 Hz 기준 0 dB). 엔진의 LF 표를 그대로 쓴다."""
    from .glottis import lf_table
    rd = float(np.clip(rd, 0.3, 2.7))
    f0 = float(f0) if 60.0 < float(f0) < 600.0 else 200.0
    rds, coef = lf_table(64, n_harm=256, rd_max=2.7)
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
    creak_gaps: list = []
    if pulses is None:
        pulses = glottal_pulses(src, sr, prof)
        if CREAK_PULSES:
            _ex, creak_gaps = creak_pulses(src, sr, pulses)
            if _ex.size:
                pulses = np.sort(np.concatenate([np.asarray(pulses, dtype=np.float64), _ex]))
            print(f"  creak·유성 폐쇄 펄스: 틈 {len(creak_gaps)} 곳에 {_ex.size} 개 더함 "
                  + ", ".join(f"{a:.3f}–{b:.3f} s({k}, {m})" for a, b, k, m in creak_gaps), flush=True)
    snd = parselmouth.Sound(src.astype(np.float64), sr)
    pt = snd.to_pitch(time_step=step, pitch_floor=max(60.0, prof.f0_lo * 0.6),
                      pitch_ceiling=prof.f0_hi * 1.3)
    hn = snd.to_harmonicity_cc(time_step=step, minimum_pitch=70)
    _vmask, _vf0 = None, None
    if VOICE_TAIL:
        _pl = snd.to_pitch_ac(time_step=step, pitch_floor=max(60.0, prof.f0_lo * 0.6),
                              pitch_ceiling=prof.f0_hi * 1.3, silence_threshold=VOICE_TAIL_SIL,
                              voicing_threshold=VOICE_TAIL_VOI)
        _ts = (np.arange(n) + 0.5) * step + t0
        _fs = np.array([pt.get_value_at_time(t) for t in _ts], float)
        _fl = np.array([_pl.get_value_at_time(t) for t in _ts], float)
        _vmask, _vf0 = np.isfinite(_fs), _fs.copy()
        _added = 0
        for _dir in VOICE_TAIL_DIRS:
            _rng = range(n) if _dir == 1 else range(n - 1, -1, -1)
            _last = np.nan
            for _i in _rng:
                if np.isfinite(_fs[_i]):
                    _last = _fs[_i]
                elif (np.isfinite(_last) and not _vmask[_i] and np.isfinite(_fl[_i])
                      and abs(np.log(_fl[_i] / _last)) < VOICE_TAIL_JUMP):
                    _vmask[_i], _vf0[_i], _last = True, _fl[_i], _fl[_i]
                    _added += 1
                elif not _vmask[_i]:
                    _last = np.nan
        print(f"  유성 꼬리 이력: 느슨한 문턱으로 이어 붙인 틀 {_added} 개 "
              f"(무음 {VOICE_TAIL_SIL:g}, 유성 {VOICE_TAIL_VOI:g}, F0 도약 < {VOICE_TAIL_JUMP:g})", flush=True)
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
    creak_f = np.zeros(n, dtype=bool)   # 되찾은 펄스가 있는 틈의 틀 (파일 시각으로 판정) — 유성으로, F0 는 펄스에서
    creak_only = np.zeros(n, dtype=bool)   # 그중 creak 방식 — 내전 바닥 `CREAK_ADD`
    _pf_creak = None
    if creak_gaps:
        _tg = (np.arange(n) + 0.5) * step + t0
        _pp = np.asarray(pulses, dtype=np.float64)
        for _a, _b, _k, _m in creak_gaps:
            if _m == "creak":
                _in = (_tg > _a) & (_tg < _b)
                creak_only |= _in
            else:
                # 유성 폐쇄 방식은 **펄스 곁만** 유성 — 틈 안의 무성 부분(개방 버스트 등)은 그대로 둔다
                _q = _pp[(_pp > _a) & (_pp < _b)]
                _in = np.zeros(n, dtype=bool)
                if _q.size:
                    _T = float(np.median(np.diff(_q))) if _q.size > 1 else 0.004
                    for _t in _q:
                        _in |= np.abs(_tg - _t) < 0.75 * _T
            creak_f |= _in
        _pf_creak = f0_from_pulses(np.asarray(pulses, dtype=np.float64), _tg, prof.f0_lo, prof.f0_hi)
        if _pf_creak is None:
            creak_f[:] = False
    voi_f = np.zeros(n, dtype=bool)     # 엄격한 유성 — 포먼트·비음·밴드처럼 **측정**에 쓰는 것 (§52.416)
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
        f0_raw = pt.get_value_at_time(t + t0) if _vmask is None else (_vf0[i] if _vmask[i] else None)
        if creak_f[i]:
            f0_raw = float(_pf_creak[i])
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
        strict = voiced and (_vmask is None or bool(np.isfinite(_fs[i])))
        voi_f[i] = strict
        vals[i, INDEX["adduction"]] = np.clip(0.10 + 0.5 * (h + 5) / 25.0, 0.02, 0.85)
        if creak_only[i]:
            vals[i, INDEX["adduction"]] = max(vals[i, INDEX["adduction"]], CREAK_ADD)
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
        # 꼬리 이력으로 이어 붙인 틀은 성문 몫(F0·유성)만 받는다 — 약하고 잡음 섞인 틀의 LPC 는 포먼트를
        # 1.9 kHz 까지 흔들어 첫 렌더를 +13 dB 로 터뜨렸다 (yang_00000101 0.15 s, 초기 포락 35 → −2 %).
        if strict:
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
    if voi_f.any():
        slow_v = smooth_track(rms_db[voi_f], max(3, int(round(75.0 / fm_ms)) | 1),
                              max(2, int(round(150.0 / fm_ms))))
        slow = np.interp(idx, idx[voi_f], slow_v)
    else:
        slow = smooth_track(rms_db, max(3, int(round(75.0 / fm_ms)) | 1),
                            max(2, int(round(150.0 / fm_ms))))
    vals[:, INDEX["p_sub"]] = np.clip(3.0 + 5.0 * (slow + 60) / 40.0, 0.0, 16.0)
    if PHON_FLOOR and voi.any():
        # glottis.threshold: pth = (1.5 + 0.7 (F0/F0_nom)²) · (0.6 + 2.4 (1 − add)²). 고음일수록 문턱이 오른다
        # (Titze 1989). 유성인데 문턱 밑이면 (1) 내전을 게이트가 열리는 값 위로, (2) 폐압을 문턱 위로 올리고,
        # (3) 폐압이 상한에 닿으면 남는 몫을 내전으로 메운다. 폐압은 호흡이라 올린 몫을 30 ms 로 퍼뜨린다.
        from scipy.ndimage import maximum_filter1d, uniform_filter1d
        _f0 = vals[:, INDEX["f0_target"]]
        _k = 1.5 + 0.7 * (_f0 / prof.f0_nominal) ** 2
        _add = vals[:, INDEX["adduction"]].copy()
        _ps = vals[:, INDEX["p_sub"]].copy()
        _add = np.where(voi, np.maximum(_add, PHON_ADD_MIN), _add)
        _need = _k * (0.6 + 2.4 * (1.0 - _add) ** 2) / PHON_MARGIN
        _q = (PHON_MARGIN * PHON_P_MAX / _k - 0.6) / 2.4
        _amax_p = np.where(_q > 0, 1.0 - np.sqrt(np.clip(_q, 0.0, None)), 0.85)
        _over = voi & (_need > PHON_P_MAX)
        _add = np.where(_over, np.minimum(np.maximum(_add, _amax_p), 0.85), _add)
        _need = _k * (0.6 + 2.4 * (1.0 - _add) ** 2) / PHON_MARGIN
        _def = np.where(voi, np.clip(np.minimum(_need, PHON_P_MAX) - _ps, 0.0, None), 0.0)
        _w = max(3, int(round(30.0 / fm_ms)) | 1)
        _def = uniform_filter1d(maximum_filter1d(_def, _w), _w)
        n_add = int((voi & (_add > vals[:, INDEX["adduction"]] + 1e-9)).sum())
        vals[:, INDEX["adduction"]] = _add
        vals[:, INDEX["p_sub"]] = np.clip(_ps + _def, 0.0, 20.0)
        print(f"  발성 문턱 바닥: 유성 틀 {int(voi.sum())} 개 중 내전 올림 {n_add}, 폐압 올림 {int((_def > 0.05).sum())} "
              f"(최대 +{_def.max():.1f} cmH2O; pth ≤ {PHON_MARGIN:g}·p_sub, 내전 ≥ {PHON_ADD_MIN:g})", flush=True)
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
        m_hi_ = (fr_ >= 2500) & (fr_ < 4000)
        win_ = np.hanning(w_)
        hl = np.zeros(n)
        f2d = np.zeros(n)
        iprom = np.zeros(n)
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
            iprom[i] = 10 * np.log10(P[m_hi_].max() / P[m_f2].mean())
            k = int(np.argmax(P[m_lo])) if m_lo.any() else 0
            f1n[i] = float(np.clip(fr_[m_lo][k], 150.0, 600.0))
            fz[i] = float(np.clip(fr_[m_z][int(np.argmin(P[m_z]))], 400.0, 4000.0))
        # 유성이면서 (i) 저역이 중역을 크게 누르고 (ii) **F2 대역이 죽어 있어야** 한다.
        # (ii) 가 없으면 설측·유음(§1 의 "설측 닐" F2 2000~2300)까지 비음으로 잡힌다 — 실측에서
        # 그렇게 잡은 판은 머머 스펙트럼 오차가 5.8 → 14.9 dB 로 오히려 나빠졌다.
        nasal = voi_f[:n] & (hl > NASAL_HL_DB) & (f2d[:n] < -NASAL_F2_DB)
        if NASAL_I_PROM_DB is not None:
            _n0 = int(nasal.sum())
            nasal &= iprom[:n] < NASAL_I_PROM_DB
            print(f"  비음 /이/ 가림: 2.5–4 kHz 봉우리가 1.2–2 kHz 보다 {NASAL_I_PROM_DB:g} dB 넘게 솟는 틀 {_n0 - int(nasal.sum())} 개를 "
                  f"비음에서 뺐다 ({_n0} → {int(nasal.sum())})", flush=True)
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
        Ft, Bt, done = track_formants(cand_all, voi_f, ref0, n_formants)
        if done:
            hold = None
            for i in range(n):
                if voi_f[i] and Ft[i, 0] > 0:
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
                y, sr, vals[:, INDEX["f0_target"]], voi_f, _F, _B, hop,
                spacing=35000.0 / (2.0 * prof.tract_length_cm))
            if _ok.sum() >= 32:
                # 화자 고유의 상수 몫을 떼고, 밴드 추적은 그 둘레의 변화분만 따른다.
                _off, _i0 = _bands.speaker_offset(_obs, _ok, _feat)
                _flip = _bands.flow_flip(vals[:, INDEX["a_c"]], vals[:, INDEX["p_sub"]], voi_f,
                                          f0=vals[:, INDEX["f0_target"]])
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
                vals[:, INDEX["rd_offset"]] = np.clip(_rd - _base, PARAMS["rd_offset"].lo, PARAMS["rd_offset"].hi)
                if BAND_TILT:
                    # **tilt 을 관측한다** (MEASUREMENTS §52.101). 지금까지 분석기는 `tilt` 을 한 번도
                    # 건드리지 않아 코퍼스 80 파일에서 중앙 2.000·**사분범위 0** 이었다 — 관측 없이 떠 있는
                    # 자유도이고, 적합은 그것을 고역 결손을 가리는 데 썼다 (M30 +2.0 상한, M14 중앙 +4.19).
                    #
                    # Rd 가 설명하지 못한 몫이 곧 추가 기울기다: 잔차 = (관측 − 화자상수) − 그 Rd 의 예측.
                    # 세 번째 지문은 H3–H4 대 H5–H8 이고 `band_profile` 이 `저역 − 고역` 으로 만드므로,
                    # 잔차가 양수면 목표가 모델보다 더 떨어진다 -> tilt 은 음수다. 두 무게중심의 간격은
                    # log2(√(5·8)/√(3·4)) = **0.87447 옥타브**다.
                    _res = (_obs - _off[None, :]) - _feat[_path]
                    _oct = float(np.log2(np.sqrt(5.0 * 8.0) / np.sqrt(3.0 * 4.0)))
                    # **잔차는 프레임마다 거칠다** — 성도 오차까지 섞여 들어온다. 실측: 그대로 쓰면
                    # 사분범위 5.13 dB/oct 로 ±12 상한에 붙고 **초기 포락이 27.5 % -> −6.8 %** 로 무너졌다.
                    # 음원 기울기는 발성 노력이라 음절 안에서 튀지 않으므로 `BAND_TILT_MS` 로 느린 성분만 남긴다.
                    _tl = -_res[:, 2] / _oct
                    _w = max(3, int(round(BAND_TILT_MS / (hop * 1000.0 / sr))) | 1)
                    _ok_f = _ok.astype(float)
                    _num = np.convolve(np.where(_ok, _tl, 0.0), np.ones(_w), "same")
                    _den = np.convolve(_ok_f, np.ones(_w), "same")
                    _tl = np.clip(np.where(_den > 0, _num / np.maximum(_den, 1e-9), 0.0),
                                  -BAND_TILT_MAX, BAND_TILT_MAX)
                    _lastt = float(_tl[np.flatnonzero(_ok)[0]])
                    for _i in range(n):
                        if _ok[_i]:
                            _lastt = float(_tl[_i])
                        else:
                            _tl[_i] = _lastt
                    vals[:, INDEX["tilt"]] = _tl
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
    # **치찰 협착 관측은 평활 앞에서 싣는다** (MEASUREMENTS §52.462). 평활 뒤에 치찰 틀만 덮어쓰면 틀 경계에서 협착 면적이 1 ms 만에
    # 0.41 → 0.08, 0.10 → 0.90 cm² 로 뛰고, 구강압 상미분방정식이 그 계단을 짧은 파열처럼 울린다 — `121` "마신" 의 ㅅ 양 끝
    # (0.626·0.656 s) 에서 원본에 없는 2~3 ms 의 7~9 kHz 클릭이 원본의 두 배 진폭으로 났다. 혀끝은 1 ms 에 협착을 만들거나 풀지
    # 못하므로, 다른 조음 궤적과 **같은 평활**을 거쳐 경계가 경사가 되게 한다.
    tr.sibilant = np.zeros(n, dtype=bool)
    if SIB_OBS:
        _sib = sibilant_frames(y, sr, n, hop, voi)
        if _sib.any():
            _ac = np.asarray(tr["a_c"], float).copy()
            _ob = np.asarray(tr["obstacle"], float).copy()
            _ac[_sib] = SIB_AC
            _ob[_sib] = float(prof.sibilant.get("obstacle", 0.15))
            tr["a_c"], tr["obstacle"] = _ac, _ob
        tr.sibilant = _sib
    # obstacle 도 함께 뭉갠다 — a_c 와 **같은 r 에서 낸 값**이라 프레임마다 같이
    # 튄다. 안 뭉개면 소스 스펙트럼이 프레임률(1 kHz)로 흔들려 그 자체가
    # 거친 변조가 된다.
    for k in ("f0_target", "adduction", "tension", "a_c", "obstacle"):
        tr[k] = smooth_track(tr[k], m, a_f)
    if SIB_OBS:
        _sib = tr.sibilant
        _edge = ""
        if _sib.any():
            _e = np.flatnonzero(np.diff(_sib.astype(np.int8)))
            _dl = np.abs(np.diff(np.log(np.maximum(np.asarray(tr["a_c"], float), 1e-6))))
            _w = [_dl[max(0, i - a_f):i + a_f + 1].max() for i in _e]
            _edge = f" — 경계 {len(_e)} 곳, 한 틀 협착 변화 최대 ×{float(np.exp(max(_w))):.2f}" if _w else ""
        print(f"  치찰 협착 관측: {int(_sib.sum())} 틀을 a_c {SIB_AC:g} cm², 앞니 쌍극 {float(prof.sibilant.get('obstacle', 0.15)):g} 로 (평활 앞){_edge}", flush=True)
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
            # "펄스 곁" 은 **주기로** 잰다 — 20 ms 는 720 Hz 에서 14 주기라, 펄스 검출이 끊긴 잦아드는 꼬리에
            # 묵은 펄스 간격(577 Hz)이 궤적(720~760 Hz)을 덮었다 (§52.416). 2.5 주기 안만 펄스가 이긴다.
            _per = 1.0 / np.maximum(np.asarray(tr["f0_target"], float), 50.0)
            near = np.minimum(np.abs(t_grid - pulses[k]),
                              np.abs(t_grid - pulses[k - 1])) < np.minimum(0.02, 2.5 * _per)
        # 펄스를 하나 놓치면 간격이 길어져 F0 가 **분수배**로 떨어진다 (1.370 s: 궤적 774 → 펄스 494 Hz, §52.416).
        # 궤적과 25 % 넘게 어긋나는 펄스 F0 는 쓰지 않는다 — 펄스가 이기는 것은 누적 위상의 미세 오차 때문이지
        # 주기 수를 세는 데서가 아니다.
        _tf = np.asarray(tr["f0_target"], float)
        near &= np.abs(np.log(np.maximum(pf, 1.0) / np.maximum(_tf, 1.0))) < 0.25
        near |= creak_f             # creak 틀은 펄스 간격이 곧 주기다 (궤적은 틈 앞 값을 붙들고 있다)
        tr["f0_target"] = np.where(near, pf, tr["f0_target"])
    tr.voiced = voi
    tr.voiced_strict = voi_f
    tr.fricative = fric
    if BURST_TRACK:
        # **전체 파일 맥락에서 찾는다** (MEASUREMENTS §52.290). 잘린 구간만 보면 가장자리의 파열을 놓친다 —
        # 검출기는 `BURST_BACK_MS`(15 ms) 만큼 되돌아봐 기준선을 잡는데, 구간 앞이 잘려 있으면 기준선이
        # 올라가 상승량이 줄어든다. 실측: 0.463 s 의 구강 클릭이 파일 전체에서는 잡히는데(25 개 중 첫째)
        # 0.451 s 부터 자른 구간에서는 상승 11.3 dB 로 문턱(12) 아래라 **0 개**였다.
        if full is not None:
            b_all = find_bursts(full, sr, hop)
            off = int(round(t0 * sr / hop))
            b = b_all - off
            tr.bursts = b[(b >= 0) & (b < n)]
        else:
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


#: 펄스 사이 간격이 중앙값의 이 배를 넘으면 **끊긴 것**으로 본다.
VGATE_GAP = 1.8
#: 묶음의 앞뒤로 이만큼(국소 주기의 몫) 여유를 준다.
VGATE_PAD = 0.75
#: 여유 끝에서 0 까지 내려가는 경사 (국소 주기의 몫). 계단은 클릭을 만든다.
VGATE_RAMP = 0.75


def voicing_gate_from_pulses(pulses: np.ndarray, n: int, sr: float) -> np.ndarray:
    """목표의 성문 폐쇄 시각에서 **배음 음원 차단 게이트**(표본률 0~1)를 만든다.

    사용자 지시: "음원에서 펄스를 인지하지 못했다면, 그 구간에 펄스를 아예 못 쓰게 하도록 해."
    연속한 펄스 묶음마다 [첫 펄스 − 여유, 끝 펄스 + 여유] 를 1 로 두고, 그 바깥은
    반주기 코사인으로 0 까지 내린다. 펄스가 없으면 전부 0 이다(배음을 아예 안 만든다).
    """
    g = np.zeros(int(n), dtype=np.float32)
    p = np.asarray(pulses, dtype=np.float64)
    p = p[(p >= 0.0) & (p * sr < n)]
    if p.size < 2:
        return g
    d = np.diff(p)
    per = float(np.median(d))
    if not np.isfinite(per) or per <= 0:
        return g
    if CREAK_PULSES:
        # 국소 중앙값 (앞뒤 6 간격) — creak 의 불규칙 간격이 전역 중앙값 기준으로는 끊김으로 읽힌다 (§52.441)
        loc = np.array([np.median(d[max(0, k - 6):k + 7]) for k in range(len(d))])
        breaks = np.flatnonzero(d > VGATE_GAP * loc)
    else:
        breaks = np.flatnonzero(d > VGATE_GAP * per)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(p) - 1]))
    ramp = max(1, int(round(VGATE_RAMP * per * sr)))
    win = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp + 2)[1:-1]))
    for i0, i1 in zip(starts, ends):
        if i1 <= i0:
            continue
        loc = float(np.median(d[i0:i1])) if i1 > i0 else per
        pad = int(round(VGATE_PAD * loc * sr))
        a = int(round(p[i0] * sr)) - pad
        b = int(round(p[i1] * sr)) + pad
        lo, hi = max(0, a), min(int(n), b)
        if hi > lo:
            g[lo:hi] = 1.0
        if a - ramp >= 0 and a > 0:
            g[max(0, a - ramp):a] = np.maximum(g[max(0, a - ramp):a], win[:a - max(0, a - ramp)])
        if b < n:
            e = min(int(n), b + ramp)
            g[b:e] = np.maximum(g[b:e], win[::-1][:e - b])
    return g
