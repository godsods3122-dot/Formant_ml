"""난류를 재는 자 — 파형이 아니라 **통계**로.

왜 파형으로 재면 안 되는가
--------------------------
마찰 잡음의 표본 하나하나는 미시적 소용돌이가 정한다. 완벽한 물리 모형이라도 그
*실현*은 재현하지 못한다 — 재현할 수 있는 것은 그 통계뿐이다. 그런데 지금까지의
일치율(`fit.FitReport.env` / `.fine`)은 크기 스펙트럼을 프레임마다 직접 뺀 값이라,
같은 스펙트럼의 두 독립 실현조차 크게 틀리다고 말한다.

얼마나 틀리다고 말하는지 재 봤다 (`tests/engine/test_turbulence.py::test_realization_floor`):

    | 두 독립 실현            | 정밀 일치 | 포락 일치 |
    |-------------------------|-----------|-----------|
    | 백색 잡음               |    33.3 % |    46.6 % |
    | /s/ 모양 대역잡음       |    33.3 % |    53.7 % |

이론값도 같다. STFT 빈의 크기는 레일리 분포이므로 두 독립 실현의 스펙트럼 수렴도는

    E‖|S₁|−|S₂|‖² / E‖|S₁|‖² = (4−π)/2·σ²/(2σ²)  ->  SC = √((4−π)/2) = 0.655

즉 독립 레일리 실현의 기대 정밀 점수는 약 34.5 % 다. 이는 유한 표본의 엄밀한
상한도, 모든 치찰음에 적용되는 상한도 아니다.

무엇으로 바꾸는가
-----------------
1. **편향 보정 스펙트럼 수렴도** (`corrected_sc`). 같은 파라미터를 시드만 바꿔 한 번 더
   합성하면 그 둘의 거리가 곧 "완벽한 모형이라도 남는" 실현 잡음이다. 그 바닥을
   제곱 영역에서 빼면 모형 편향의 휴리스틱 추정이 남는다. 분산 추정과 정상성
   가정에 민감하며, 음질이나 지각적 동등성의 증거가 아니다.

   `trust` / `resolved` 는 잔여 거리 비율과 그 임계 판정이지 통계적 신뢰도가 아니다.
   분해가 안 된 점수는 **미분해 추정치**이며 하한이나 신뢰구간이 아니다.
   완벽한 모형이 거기 오고(편향이 실제로 0 이니까) 잡음이 과한 합성도 거기 온다 —
   **둘을 가르는 것은 `noise_ratio`** (합성/목표 실현 분산 비) 다. 실측 대조군:

    | 대조군            | 정밀  | 보정  | 잔여비 | 분해 | 잡음비 |
    |-------------------|-------|-------|--------|------|--------|
    | 완벽 (시드만 다름)| 33.9  | 97.2  |  0.007 | ✗    |  0.97  |
    | 1 kHz 어긋남      | 29.4  | 74.3  |  0.105 | ✓    |  0.97  |
    | 크게 어긋남       | −5.8  |  6.9  |  0.600 | ✓    |  0.98  |
    | 6 dB 조용         | 31.7  | 51.0  |  0.399 | ✓    |  0.24  |
    | 같은 하모닉       | 100.0 | 100.0 |  1.000 | ✓    |  1.00  |

   잡음비가 6 dB 조용한 합성에서 0.24 로 나온다 — 진폭 0.5 배의 분산비 0.25 다.

2. **치찰음 척도** (`sibilant_features`). 문헌이 조음 위치를 가른다고 확인한 양들:
   스펙트럼 봉우리 위치와 네 모멘트(무게중심·표준편차·왜도·첨도)
   [Jongman, Wayland & Wong (2000), JASA 108(3):1252 — "spectral peak location,
   spectral moments, and both normalized and relative amplitude serve to distinguish
   all four places of fricative articulation"].

3. **저분산 스펙트럼 추정** (`multitaper_psd`). 난류 스펙트럼을 창 하나로 재면 추정
   분산이 추정값만큼 크다. Reidy (2015) JASA 137(4):EL248 이 사이비언트 분석에
   저분산 추정기를 권한 이유다. DPSS 멀티테이퍼를 쓴다.

4. **시간 구조** (`modulation_bands`, `amplitude_kurtosis`). 크기 스펙트럼이 같아도
   포락이 출렁이면 "지글거린다". 대조군이 반드시 필요하다 — 순수 가우시안 대역잡음도
   800~2000 Hz 변조 대역에 40 % 남짓을 갖는다 (docs/HANDOFF.md §7 에서 한 번
   오독했다가 철회했다).

무엇을 재지 *않는가*
--------------------
난류의 위상. 재도 뜻이 없고, 손실에 넣으면 해롭다 — `fit.CopySynthFitter.phase_loss`
참조.
"""
from __future__ import annotations

import numpy as np

# 치찰음 모멘트를 재는 대역. 아래를 500 Hz 에서 자르는 이유: 유성 마찰이나 앞뒤 모음의
# 꼬리가 조금만 섞여도 저역 하모닉이 무게중심을 통째로 끌어내린다 (실측: 300 Hz 부터
# 재면 /ㅆ/ 무게중심이 8.1 -> 5.4 kHz). 위는 나이퀴스트에 맡긴다.
SIB_FMIN = 500.0
# 대역 LTAS 의 경계. 치찰음이 사는 곳(3~16 kHz)을 촘촘히 본다.
LTAS_EDGES = (0.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
              8000.0, 10000.0, 12000.0, 16000.0, 24000.0)
# 포락 변조 스펙트럼의 대역. 5~60 Hz 는 제트 사행(의도한 것), 그 위는 질감이다.
MOD_EDGES = (5.0, 60.0, 400.0, 800.0, 1200.0, 2000.0)
# Heuristic residual-distance threshold, not statistical confidence or a bound.
TRUST_MIN = 0.10


# ----------------------------------------------------------------- 스펙트럼 추정
def multitaper_psd(x: np.ndarray, fs: float, nw: float = 4.0, k: int = 7):
    """DPSS 멀티테이퍼 PSD -> (주파수 (F,), 파워 (F,)).

    창 하나로 잰 주기도표는 추정 분산이 추정값과 같다(자유도 2). 난류처럼 스펙트럼
    자체가 확률변수인 신호에서는 그 분산이 곧 우리가 재려는 차이만큼 크다. 서로
    직교하는 테이퍼 k 개의 평균은 자유도를 2k 로 올려 분산을 1/k 로 줄인다.

    Reidy (2015) 가 사이비언트 분석에 저분산 추정기를 권한 것이 이 이유다.
    """
    from scipy.signal.windows import dpss
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 32:
        f = np.fft.rfftfreq(max(n, 2), 1.0 / fs)
        return f, np.zeros(len(f))
    k = max(1, min(int(k), int(2 * nw) - 1))
    tapers = dpss(n, nw, k)
    X = np.fft.rfft(tapers * x[None, :], axis=-1)
    p = (np.abs(X) ** 2).mean(0) / (fs * n)
    return np.fft.rfftfreq(n, 1.0 / fs), p


def welch_psd(x: np.ndarray, fs: float, win: int = 1024):
    """겹침 평균 PSD. 구간이 길 때는 멀티테이퍼보다 싸고 분산도 충분히 낮다."""
    from scipy.signal import welch
    n = min(win, max(64, len(x) // 4 * 2))
    return welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=n,
                 noverlap=n // 2, detrend=False)


def average_psd(x: np.ndarray, fs: float):
    """구간 길이에 맞는 저분산 PSD 를 고른다. 짧으면 멀티테이퍼, 길면 Welch."""
    n = len(x)
    if n < int(0.06 * fs):          # 60 ms 미만 — 겹침 평균할 조각이 부족하다
        return multitaper_psd(x, fs)
    return welch_psd(x, fs)


# --------------------------------------------------------------- 치찰음 특징
def spectral_moments(f: np.ndarray, p: np.ndarray, fmin: float = SIB_FMIN,
                     fmax: float | None = None) -> dict:
    """파워 스펙트럼의 네 모멘트 (Forrest et al. 1988; Jongman et al. 2000).

    스펙트럼을 주파수 위의 확률분포로 보고 평균·표준편차·왜도·첨도를 낸다. 무게중심은
    협착의 앞공동 길이를, 왜도는 스펙트럼이 어느 쪽으로 기우는지를, 첨도는 봉우리가
    얼마나 뾰족한지를 잡는다. /s/ 와 /ʃ/ 를 가르는 표준 자다.

    첨도는 **초과 첨도**(정규분포 = 0)로 돌려준다.
    """
    fmax = fmax or float(f[-1])
    m = (f >= fmin) & (f <= fmax)
    w = np.maximum(np.asarray(p, dtype=np.float64)[m], 0.0)
    fr = np.asarray(f, dtype=np.float64)[m]
    s = w.sum()
    if s <= 0 or len(fr) < 4:
        nan = float("nan")
        return dict(centroid=nan, sd=nan, skew=nan, kurt=nan)
    w = w / s
    c = float((fr * w).sum())
    v = float(((fr - c) ** 2 * w).sum())
    sd = float(np.sqrt(max(v, 1e-12)))
    sk = float(((fr - c) ** 3 * w).sum() / sd ** 3)
    ku = float(((fr - c) ** 4 * w).sum() / sd ** 4) - 3.0
    return dict(centroid=c, sd=sd, skew=sk, kurt=ku)


def spectral_peak(f: np.ndarray, p: np.ndarray, fmin: float = 1000.0,
                  fmax: float | None = None, smooth_hz: float = 300.0) -> float:
    """스펙트럼 봉우리 주파수 (Hz). 평활한 뒤 최대를 찾는다.

    평활 없이 최대를 집으면 난류의 우연한 빈이 잡힌다 (실측: 같은 /ㅆ/ 를 두 번 재서
    8.9 kHz 와 6.2 kHz 가 나왔다). 300 Hz 폭이면 앞공동 극의 폭보다 좁아 봉우리 자체는
    안 옮긴다.
    """
    fmax = fmax or float(f[-1])
    m = (f >= fmin) & (f <= fmax)
    if m.sum() < 4:
        return float("nan")
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    k = max(1, int(round(smooth_hz / max(df, 1e-9))))
    w = np.hanning(k + 2)[1:-1]
    w = w / w.sum()
    q = np.convolve(np.asarray(p)[m], w, mode="same")
    return float(f[m][int(np.argmax(q))])


def band_levels_db(f: np.ndarray, p: np.ndarray, edges=LTAS_EDGES) -> np.ndarray:
    """대역별 레벨 (dB, 전체 합 대비). 가장 읽기 쉬운 스펙트럼 요약이다."""
    tot = float(np.trapezoid(p, f)) + 1e-30
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        e = float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 0.0
        out.append(10.0 * np.log10(e / tot + 1e-12))
    return np.array(out)


def sibilance_db(f: np.ndarray, p: np.ndarray) -> float:
    """치찰 정도 — 치찰 대역(3~16 kHz) 과 저역(0.3~1 kHz) 의 레벨 차 (dB).

    Reidy (2015) 의 "degree of sibilance" 와 같은 취지이되, 대역은 이 화자의 실측
    (`profiles/yang_female.json` §sibilant) 에 맞춰 우리가 정한 것이다. 무게중심이
    같아도 이 값이 다르면 "쉬" 와 "스" 처럼 다르게 들린다.
    """
    def lvl(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 0.0
    hi = lvl(3000.0, min(16000.0, float(f[-1])))
    lo = lvl(300.0, 1000.0)
    return 10.0 * np.log10((hi + 1e-30) / (lo + 1e-30))


def sibilant_features(x: np.ndarray, fs: float) -> dict:
    """한 구간의 치찰음 지문. 목표와 합성에 똑같이 걸어 비교한다."""
    f, p = average_psd(x, fs)
    m = spectral_moments(f, p)
    return dict(**m,
                peak=spectral_peak(f, p),
                sibilance=sibilance_db(f, p),
                bands=band_levels_db(f, p),
                mod=modulation_bands(x, fs),
                kurtosis_t=amplitude_kurtosis(x, fs),
                rms_db=float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)))


# ----------------------------------------------------------------- 시간 구조
def _band_envelope(x: np.ndarray, fs: float, lo: float, hi: float):
    from scipy.signal import butter, hilbert, sosfiltfilt
    hi = min(hi, 0.45 * fs)
    if hi <= lo or len(x) < 64:
        return None, None
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    b = sosfiltfilt(sos, np.asarray(x, dtype=np.float64))
    return b, np.abs(hilbert(b))


def modulation_bands(x: np.ndarray, fs: float, lo: float = 4000.0,
                     hi: float = 12000.0, edges=MOD_EDGES) -> np.ndarray:
    """고역 포락의 변조 스펙트럼을 대역별 비율(%)로. 지글거림의 자다.

    **대조군 없이 읽으면 안 된다.** 순수 가우시안 대역잡음도 800~2000 Hz 에 40 % 를
    갖는다. 비교 대상은 언제나 목표 녹음의 같은 값이다.
    """
    b, e = _band_envelope(x, fs, lo, hi)
    if b is None:
        return np.full(len(edges) - 1, np.nan)
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    fm = np.fft.rfftfreq(len(e), 1.0 / fs)
    tot = E[(fm >= edges[0]) & (fm < edges[-1])].sum()
    if tot <= 1e-20:
        return np.full(len(edges) - 1, np.nan)
    return np.array([100.0 * E[(fm >= a) & (fm < b_)].sum() / tot
                     for a, b_ in zip(edges[:-1], edges[1:])])


def amplitude_kurtosis(x: np.ndarray, fs: float, lo: float = 4000.0,
                       hi: float = 12000.0) -> float:
    """고역 대역통과 파형의 첨도. 가우시안 난류면 3, 버스트성이면 그 위."""
    b, _ = _band_envelope(x, fs, lo, hi)
    if b is None or b.std() < 1e-12:
        return float("nan")
    return float(((b / b.std()) ** 4).mean())


# -------------------------------------------------- 실현 잡음을 뺀 일치도
def _sq(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.shape[-1], b.shape[-1])
    d = a[..., :m] - b[..., :m]
    return float((d * d).sum())


def _realization_var(mag: np.ndarray, k: int = 9) -> float:
    """시간 평활 잔차로 실현 분산을 추정한다.

    신호의 실제 변화(마찰 시작, 포먼트 전이)도 섞이므로 **절대값은 못 믿는다.** 목표와
    합성에 똑같이 걸었을 때의 **비**만 쓴다 — 그 용도로는 편향이 대부분 상쇄된다.
    """
    from scipy.ndimage import uniform_filter1d
    a = np.asarray(mag, dtype=np.float64)
    t = a.shape[-1]
    if t < 3:
        return 0.0
    k = max(3, min(k, t // 2 * 2 + 1))
    sm = uniform_filter1d(a, k, axis=-1, mode="nearest")
    return float(((a - sm) ** 2).sum())


def corrected_sc(mag_t: np.ndarray, mag_p: np.ndarray,
                 mag_p2: np.ndarray | None) -> dict:
    """실현 잡음을 뺀 스펙트럼 수렴도.

    -> `{"sc", "floor", "trust", "noise_ratio"}`

    크기 스펙트럼을 `|S| = m + n` (기대값 + 실현 잡음) 으로 쓰면

        D_tp  = E‖|S_t|−|S_p|‖²  = ‖m_t−m_p‖² + V_t + V_p
        D_pp' = E‖|S_p|−|S_p'|‖² = 2·V_p            (같은 파라미터, 시드만 다름)

    이므로 **빼야 할 것은 V_t + V_p** 이고 V_p 는 합성 쌍으로 추정한다 (= D_pp'/2).

    **V_t ≈ V_p 를 가정하고 D_pp' 를 통째로 빼면 안 된다.** 합성의 실현 분산이 목표보다
    크면 (실제로 그렇다 — 녹음은 위너 차감을 거쳐 스펙트럼이 평활해져 있다) 과하게
    빼서 **잡음 투성이 합성이 100 % 를 받는다.** 실측: 치찰음 구간 넷 전부 100.00.

    그래서 V_t 를 따로 추정한다. 시간 평활 잔차는 절대 눈금이 안 맞지만, 합성 쪽에서는
    실현 쌍의 추정값(D_pp'/2)으로 눈금을 교정해 목표에 옮긴다:

        V_t ≈ V_t^est · (D_pp'/2) / V_p^est

    V_p = 2·V_t 인 경우로 확인하면 정확히 ‖m_t−m_p‖² 가 남는다.

    돌려주는 값
    -----------
    * `sc`    — 보정 SC 추정. 0 으로 클리핑될 수 있다; 완벽함을 뜻하지 않는다.
    * `floor` — 추정 실현 잡음 기여량. 원래 SC 의 엄밀한 하한이 아니다.
    * `trust` — `(D_tp − 뺀 양) / D_tp`. **0 에 가까우면 보정값을 믿으면 안 된다** —
      실현 잡음이 거리를 통째로 설명해 버려 편향을 못 잰다. 그럴 때는
      `spectrum_match`(시간 평균 스펙트럼)를 대신 본다.
    * `noise_ratio` — V_p/V_t 추정. 1 보다 크면 합성이 목표보다 잡음이 심하다.
      **이건 보정으로 지워지는 양이 아니라 별개의 결함이다** (잡음량도 물리
      파라미터다). 모양이 맞는지와 잡음량이 맞는지를 갈라서 봐야 한다.

    성질:
    * 독립 레일리 실현의 원래 SC 기대값은 약 0.655 (일치 34.5 %).
    * 결정적 신호(하모닉) -> D_pp' ≈ 0 이므로 보정이 아무 일도 안 한다.
    """
    n_t = float((np.asarray(mag_t, dtype=np.float64) ** 2).sum())
    if n_t <= 1e-24 or not np.isfinite(n_t):
        return dict(sc=float("nan"), floor=float("nan"), trust=float("nan"),
                    noise_ratio=float("nan"))
    d_tp = _sq(mag_t, mag_p)
    if mag_p2 is None:
        return dict(sc=float(np.sqrt(d_tp / max(n_t, 1e-30))), floor=0.0,
                    trust=1.0, noise_ratio=float("nan"))
    d_pp = _sq(mag_p, mag_p2)
    vt_est, vp_est = _realization_var(mag_t), _realization_var(mag_p)
    ratio = float(vp_est / max(vt_est, 1e-30))
    # 두 신호가 사실상 같으면 잴 것이 없다 — 실현 잡음도 0 이라 아래의 비가 0/0 이 된다.
    if d_tp <= 1e-12 * max(n_t, 1e-30):
        return dict(sc=0.0, floor=0.0, trust=1.0, noise_ratio=ratio)
    v_p = 0.5 * d_pp
    v_t = vt_est * (v_p / max(vp_est, 1e-30))
    sub = min(v_t + v_p, d_tp)                   # 실제 거리보다 더 뺄 수는 없다
    bias = max(d_tp - sub, 0.0)
    denom = max(n_t - v_t, 1e-30)
    return dict(
        sc=float(np.sqrt(bias / denom)),
        floor=float(np.sqrt(min(sub, d_tp) / max(n_t, 1e-30))),
        trust=float((d_tp - sub) / d_tp),
        noise_ratio=ratio,
    )


def spectrum_match(target: np.ndarray, synth: np.ndarray, fs: float) -> float:
    """시간 평균 파워 스펙트럼의 일치율 (%). 난류에 쓰는 자.

    프레임별 실현이 아니라 **기대 스펙트럼**을 비교한다. 시간 평균이 실현 잡음을
    프레임 수의 제곱근만큼 줄이므로, 완벽한 모형이면 100 % 로 수렴한다. 편향 보정과
    달리 두 번째 합성이 필요 없다.

    dB 로 비교하는 이유: 선형 파워로 재면 가장 큰 대역 하나가 값을 독차지한다.
    """
    if min(len(target), len(synth)) < 32:
        return float("nan")
    if not all(np.isfinite(x).all() and np.max(np.abs(x)) > 1e-12
               for x in (np.asarray(target), np.asarray(synth))):
        return float("nan")
    ft, pt = average_psd(target, fs)
    fp, pp = average_psd(synth, fs)
    n = min(len(pt), len(pp))
    a = 10.0 * np.log10(pt[:n] + 1e-20)
    b = 10.0 * np.log10(pp[:n] + 1e-20)
    keep = a > a.max() - 60.0            # 정점 −60 dB 아래는 잡음 바닥이다
    if keep.sum() < 8:
        return float("nan")
    a, b = a[keep], b[keep]
    a = a - a.mean(); b = b - b.mean()   # 전역 레벨은 이득의 몫, 여기선 모양만 본다
    if np.linalg.norm(a) < 1e-9:
        return float("nan")
    return float(100.0 * (1.0 - np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)))


def compare(target: np.ndarray, synth: np.ndarray, fs: float) -> dict:
    """치찰음 한 구간의 목표 대 합성 성적표.

    돌려주는 것 중 `*_err` 는 **오차**(작을수록 좋다), `match_*` 는 **일치율 %**.
    """
    a = sibilant_features(target, fs)
    b = sibilant_features(synth, fs)
    nb = min(len(a["bands"]), len(b["bands"]))
    return dict(
        target=a, synth=b,
        centroid_err=b["centroid"] - a["centroid"],
        sd_err=b["sd"] - a["sd"],
        skew_err=b["skew"] - a["skew"],
        kurt_err=b["kurt"] - a["kurt"],
        peak_err=b["peak"] - a["peak"],
        sibilance_err=b["sibilance"] - a["sibilance"],
        band_mae_db=float(np.nanmean(np.abs(b["bands"][:nb] - a["bands"][:nb]))),
        mod_mae_pct=(_finite_mean(np.abs(b["mod"] - a["mod"]))
                     if np.isfinite(a["mod"]).any() and np.isfinite(b["mod"]).any()
                     else float("nan")),
        kurtosis_err=b["kurtosis_t"] - a["kurtosis_t"],
        match_spectrum=spectrum_match(target, synth, fs),
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def target_regions(target: np.ndarray, fs: float, voiced: np.ndarray,
                   frame_ms: float = 1.0) -> dict:
    """Target-only, heuristic masks, reusable unchanged for every render/condition.

    Voicing comes from target analysis. Frication is measured independently of
    voicing: analyzer ``fricative`` is explicitly unvoiced-only and cannot identify
    overlap. No phonetic annotation or calibrated classification confidence is
    claimed. A clip beginning in a vowel has no observed vowel onset.
    """
    x = np.asarray(target, dtype=float)
    if fs <= 0 or frame_ms <= 0 or x.ndim != 1:
        raise ValueError("positive sample rate/frame duration and mono target required")
    step = fs * frame_ms / 1000.0
    nf = min(len(voiced), int(np.ceil(len(x) / step)))
    v = np.asarray(voiced[:nf], dtype=bool)
    rms, ratio = np.zeros(nf), np.full(nf, -np.inf)
    width = max(32, int(round(0.020 * fs)))
    freq = np.fft.rfftfreq(width, 1 / fs)
    low = (freq >= 300) & (freq < 1000)
    high = (freq >= 3000) & (freq < min(12000, fs / 2))
    for i in range(nf):
        center = int(round((i + 0.5) * step))
        a, b = max(0, center - width // 2), min(len(x), center + width // 2)
        z = x[a:b]
        if not len(z):
            continue
        rms[i] = np.sqrt(np.mean(z ** 2))
        if high.sum() >= 4 and low.sum() >= 4:
            p = np.abs(np.fft.rfft(z * np.hanning(len(z)), n=width)) ** 2
            ratio[i] = 10 * np.log10((p[high].sum() + 1e-30) /
                                    (p[low].sum() + 1e-30))
    active = (rms > max(float(rms.max()) if nf else 0, 1e-12) * 10 ** (-25 / 20))
    active &= rms > 1e-10
    fric = active & (ratio > 0)
    core = fric & ~v
    trim = max(1, int(round(5 / frame_ms)))
    interior = np.zeros(nf, dtype=bool)
    for a, b in _runs(core):
        if b - a > 2 * trim:
            interior[a + trim:b - trim] = True
    near_core = np.zeros(nf, dtype=bool)
    vicinity = max(1, int(round(30 / frame_ms)))
    for a, b in _runs(core):
        near_core[max(0, a - vicinity):min(nf, b + vicinity)] = True
    overlap = fric & v & near_core
    onset = np.zeros(nf, dtype=bool)
    onset_frames = max(1, int(round(50 / frame_ms)))
    for a, b in _runs(v & active & ~fric):
        # Require observed preceding frication, not merely a segment boundary.
        if a > 0 and np.any(fric[max(0, a - vicinity):a]):
            onset[a:min(b, a + onset_frames)] = True
    masks = {"frication_core": interior, "voiced_overlap": overlap, "vowel_onset": onset}
    definitions = {
        "frication_core": "active AND high/low > 0 dB AND NOT target-voiced; trim 5 ms at each run edge",
        "voiced_overlap": "active AND high/low > 0 dB AND target-voiced; within 30 ms of unvoiced frication",
        "vowel_onset": "first <=50 ms of active target-voiced/nonfricative run after frication within 30 ms",
    }
    return {
        "source": "raw target only; fixed across denoise conditions and synthesis seeds",
        "frame_ms": frame_ms,
        "analysis_window_ms": 20,
        "frication_bands_hz": [[300, 1000], [3000, min(12000, fs / 2)]],
        "activity_gate_db": -25,
        "confidence": "heuristic, uncalibrated; voicing errors and high harmonics can misclassify",
        "regions": {
            name: {
                "definition": definitions[name],
                "confidence": "low" if high.sum() < 4 or not mask.any() else "heuristic",
                "spans": [[int(round(a * step)), min(len(x), int(round(b * step)))]
                          for a, b in _runs(mask)],
                "status": "available" if mask.any() else "unavailable",
            } for name, mask in masks.items()
        },
    }


def _finite_mean(values) -> float | None:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def diagnostic_metrics(target: np.ndarray, synth: np.ndarray, fs: float,
                       bandwidth: float | None = None) -> dict:
    """Metrics for ONE contiguous interval, never a concatenation of mask islands.

    Null metrics are unavailable (silence, insufficient duration or bandwidth).
    Modulation requires >=2 cycles at each band's lower edge, >=2 FFT bins,
    and an upper edge no greater than the carrier bandwidth.
    Envelope correlations use a 2 ms moving average and discard 5 ms edges.
    These objective diagnostics do not establish perceptual equivalence.
    """
    from scipy.ndimage import uniform_filter1d
    t, s = np.asarray(target, dtype=float), np.asarray(synth, dtype=float)
    n = min(len(t), len(s))
    t, s = t[:n], s[:n]
    result = {"status": "unavailable", "reason": None, "metrics": {}}
    if n < max(64, int(0.020 * fs)):
        result["reason"] = "less than 20 ms contiguous data"
        return result
    if not np.isfinite(t).all() or not np.isfinite(s).all():
        result["reason"] = "nonfinite waveform"
        return result
    rt, rs = float(np.sqrt(np.mean(t ** 2))), float(np.sqrt(np.mean(s ** 2)))
    if rt <= 1e-10:
        result["reason"] = "silent target"
        return result
    bw = min(fs / 2, bandwidth if bandwidth is not None else fs / 2)
    metrics = {"rms_error_db": float(20 * np.log10(max(rs, 1e-12) / rt))}
    result.update(status="available", metrics=metrics, bandwidth_hz=float(bw))
    f, pt = average_psd(t, fs)
    _, ps = average_psd(s, fs)
    keep = f <= bw
    f, pt, ps = f[keep], pt[keep], ps[keep]
    for name in ("centroid_err_hz", "band_mae_db", "spectrum_match"):
        metrics[name] = None
    if rs > 1e-10 and len(f) >= 8:
        ct, cs = spectral_moments(f, pt), spectral_moments(f, ps)
        err = cs["centroid"] - ct["centroid"]
        metrics["centroid_err_hz"] = float(err) if np.isfinite(err) else None
        edges = [e for e in LTAS_EDGES if e <= bw]
        if len(edges) >= 2:
            metrics["band_mae_db"] = _finite_mean(
                np.abs(band_levels_db(f, pt, edges) - band_levels_db(f, ps, edges)))
        a, b = 10 * np.log10(pt + 1e-30), 10 * np.log10(ps + 1e-30)
        keep = a > a.max() - 60
        a, b = a[keep], b[keep]
        if len(a) >= 8 and np.std(a) > 1e-6:
            a, b = a - a.mean(), b - b.mean()
            metrics["spectrum_match"] = float(100 * (1 - np.linalg.norm(a - b) /
                                                     np.linalg.norm(a)))
    for lo, hi in ((300, 1000), (1000, 3000), (4000, 12000)):
        label = f"{lo}_{hi}"
        corr_key = f"envelope_corr_{label}"
        metrics[corr_key] = None
        for a, b in zip(MOD_EDGES[:-1], MOD_EDGES[1:]):
            for prefix in ("mod_target_pct", "mod_synth_pct", "mod_error_pp"):
                metrics[f"{prefix}_{label}_{int(a)}_{int(b)}"] = None
        # Do not silently relabel a truncated carrier band as the full band.
        if hi > min(bw, 0.45 * fs) or rs <= 1e-10:
            continue
        bt, et = _band_envelope(t, fs, lo, hi)
        bs, es = _band_envelope(s, fs, lo, hi)
        if bt is None or min(np.std(bt) / rt, np.std(bs) / rs) < 1e-3:
            continue
        edge = max(1, int(0.005 * fs))
        window = max(1, int(0.002 * fs))
        et = uniform_filter1d(et, window)[edge:-edge]
        es = uniform_filter1d(es, window)[edge:-edge]
        if n >= int(0.030 * fs) and min(et.std(), es.std()) > 1e-10:
            metrics[corr_key] = float(np.corrcoef(et, es)[0, 1])
        mt, ms = modulation_bands(t, fs, lo, hi), modulation_bands(s, fs, lo, hi)
        fm = np.fft.rfftfreq(n, 1 / fs)
        for i, (a, b) in enumerate(zip(MOD_EDGES[:-1], MOD_EDGES[1:])):
            if b > hi - lo or n / fs < 2 / a or ((fm >= a) & (fm < b)).sum() < 2:
                continue
            for prefix, value in (("mod_target_pct", mt[i]), ("mod_synth_pct", ms[i]),
                                  ("mod_error_pp", abs(mt[i] - ms[i]))):
                if np.isfinite(value):
                    metrics[f"{prefix}_{label}_{int(a)}_{int(b)}"] = float(value)
    return result


def region_diagnostics(target: np.ndarray, synth: np.ndarray, fs: float,
                       regions: dict, bandwidth: float | None = None) -> dict:
    """Evaluate each fixed target run separately, then duration-weight scalar metrics."""
    out = {}
    for name, region in regions["regions"].items():
        runs = []
        for a, b in region["spans"]:
            end = min(b, len(target), len(synth))
            r = diagnostic_metrics(target[a:end], synth[a:end], fs, bandwidth)
            runs.append(dict(start_sample=a, end_sample=end, **r))
        keys = {key for r in runs for key in r["metrics"]}
        means, coverage = {}, {}
        for key in sorted(keys):
            valid = [(r["metrics"].get(key), r["end_sample"] - r["start_sample"])
                     for r in runs if r["metrics"].get(key) is not None]
            coverage[key] = sum(n for _, n in valid) / fs
            means[key] = (float(np.average([v for v, _ in valid],
                                          weights=[n for _, n in valid]))
                          if valid else None)
        available = any(r["status"] == "available" for r in runs)
        out[name] = dict(
            definition=region["definition"], confidence=region["confidence"],
            status="available" if available else "unavailable",
            reason=(None if available else "no eligible contiguous target runs" if runs
                    else "no target frames match this mask"),
            duration_s=sum(b - a for a, b in region["spans"]) / fs,
            runs=runs, metrics=means, metric_duration_s=coverage,
        )
    return out


def seed_summary(rows: list[dict]) -> dict:
    """Descriptive distributions over same-fit renders, NOT confidence intervals."""
    result = {}
    for key in sorted({key for row in rows for key in row}):
        values = [row.get(key) for row in rows]
        a = np.array([v for v in values if isinstance(v, (float, int, np.number))
                      and not isinstance(v, (bool, np.bool_)) and np.isfinite(v)])
        result[key] = dict(count=len(a), median=float(np.median(a)) if len(a) else None,
                           min=float(a.min()) if len(a) else None,
                           max=float(a.max()) if len(a) else None,
                           std=float(a.std()) if len(a) > 1 else None)
    return result


# ------------------------------------------------- 적합 성적표 (편향 보정)
def _stft_mag(x: np.ndarray, n: int) -> np.ndarray:
    """(F, T) 크기 스펙트로그램. fit.py 의 torch.stft 와 같은 규약(hop=n/4, hann)."""
    x = np.asarray(x, dtype=np.float64)
    hop = n // 4
    pad = n // 2
    xp = np.pad(x, pad, mode="reflect") if len(x) > pad else np.pad(x, pad)
    t = max(1, 1 + (len(xp) - n) // hop)
    w = np.hanning(n + 1)[:n]
    fr = np.stack([xp[i * hop:i * hop + n] * w for i in range(t)])
    return np.abs(np.fft.rfft(fr, axis=-1)).T


def spectral_fidelity(target: np.ndarray, synth: np.ndarray,
                      synth2: np.ndarray | None, fs: float,
                      sizes=(256, 512, 1024, 2048, 4096),
                      f_max: float | None = None) -> dict:
    """다해상도 정밀 일치율을 **실현 잡음을 빼고** 다시 잰다.

    `synth2` 는 같은 파라미터를 시드만 바꿔 합성한 것이다. 없으면 보정 없이 원래
    값만 돌려준다.

    돌려주는 값:
    * `fine`      — 지금까지 쓰던 값 (비교용으로 남긴다)
    * `fine_corr` — 분산 차감 후 클리핑된 편향 추정. 100 은 완벽함의 증거가 아니다.
    * `floor`     — 실현 잡음 기여량에서 환산한 기대 점수; 엄밀한 상한이 아니다.
    * `trust` / `resolved` — 잔여 거리 비율 / 휴리스틱 임계 판정. 통계적 신뢰도가
      아니다. 미분해 점수는 하한(`>=`)도 신뢰구간도 아니다.
    * `noise_ratio` — 합성/목표의 실현 분산 비. 1 이면 잡음량이 맞다. 실측으로
      진폭을 6 dB 낮춘 합성에서 0.24 가 나온다 (이론 0.25).

    그래서 치찰음 구간의 성적은 **세 값을 함께** 읽는다:
    `fine_corr` 이 높고 `noise_ratio` ≈ 1 이면 좋은 것이고, `fine_corr` 이 높은데
    `noise_ratio` 가 1 에서 멀면 "모양은 못 재겠고 잡음량은 틀렸다" 는 뜻이다.
    """
    n = min(len(target), len(synth))
    if synth2 is not None:
        n = min(n, len(synth2))
    t, p = target[:n], synth[:n]
    p2 = synth2[:n] if synth2 is not None else None
    raw, cor, flo, tru, nr = [], [], [], [], []
    for k in sizes:
        if n < k:
            continue
        bmax = int(np.ceil(f_max / (fs / k))) + 1 if f_max else None
        A = _stft_mag(t, k)[:bmax]
        B = _stft_mag(p, k)[:bmax]
        C = _stft_mag(p2, k)[:bmax] if p2 is not None else None
        m = min(A.shape[1], B.shape[1]) if C is None else min(A.shape[1], B.shape[1], C.shape[1])
        A, B = A[:, :m], B[:, :m]
        C = C[:, :m] if C is not None else None
        raw.append(corrected_sc(A, B, None)["sc"])
        d = corrected_sc(A, B, C)
        cor.append(d["sc"]); flo.append(d["floor"])
        tru.append(d["trust"]); nr.append(d["noise_ratio"])
    if not raw or not np.any(np.abs(t) > 1e-12):
        return dict(fine=float("nan"), fine_corr=float("nan"), floor=float("nan"),
                    trust=float("nan"), noise_ratio=float("nan"), resolved=False,
                    correction_status="unavailable")
    trust = float(np.mean(tru))
    return dict(fine=100.0 * (1.0 - float(np.mean(raw))),
                fine_corr=100.0 * (1.0 - float(np.mean(cor))),
                floor=100.0 * (1.0 - float(np.mean(flo))),
                trust=trust, noise_ratio=float(np.mean(nr)) if nr else float("nan"),
                resolved=bool(trust >= TRUST_MIN),
                correction_status=("uncorrected" if synth2 is None else
                                   "resolved" if trust >= TRUST_MIN else "unresolved"))


def effective_bandwidth(x: np.ndarray, fs: float, drop_db: float = 35.0,
                        floor_db: float = 25.0, cliff_khz: float = 1.5,
                        flat_db_per_khz: float = 0.2) -> float:
    """녹음의 **실제** 대역 상한 (Hz). 손실 압축의 로우패스를 찾아낸다.

    **진단용이다. 적합 손실의 상한으로 쓰지 않는다.** 녹음에서는 정확하지만(코퍼스
    20.1 kHz) 합성 신호의 자연스러운 고역 롤오프를 컷으로 오인한 전력이 있다
    (9.96 kHz → 조건을 두 번 조인 뒤에도 11.1 kHz). 잘못 자르면 진짜 신호를 버리므로
    `CopySynthFitter` 는 표본화율만 쓴다.

    왜 필요한가
    -----------
    적합 손실이 녹음의 대역 위를 보면, 적합기가 "저 위를 비워라" 를 물리 파라미터로
    달성하려 들고 그 왜곡이 가청 대역을 망친다 (실측: 남성 /사/ 에서 8~12 kHz 가
    10 dB 어두워졌다 — `CopySynthFitter.f_max` 주석).

    `fit.py` 는 표본화율로 그것을 막아 왔다. 그런데 **손실 압축을 거친 음원은 표본화율이
    맞아도 대역이 잘려 있다.** orphan 코퍼스 실측: 48 kHz 파일인데 전부 20 kHz 에서
    급락한다 (나이퀴스트 24 kHz). 그 차이 4 kHz 를 손실이 보고 있었다.

    **자연스러운 고역 롤오프와 구별해야 한다.** 음성은 소스 기울기(−12 dB/oct) 때문에
    고역이 원래 완만히 떨어진다. 레벨만 보면 그것도 "컷" 으로 읽히고, 그러면 진짜
    신호를 버린다 (실측: 합성 음성에서 9.96 kHz 를 상한이라고 답했다).

    가르는 것은 **기울기**다. 손실 압축의 컷오프는 절벽이라 `cliff_khz` 폭 안에서
    `drop_db` 가 통째로 떨어진다. 자연 롤오프는 그 폭에서 몇 dB 뿐이다.

    그리고 컷 **위가 평탄한 바닥**이어야 한다. 손실 압축은 그 위에 양자화·디더 잡음만
    남아 거의 수평이지만, 자연 롤오프는 계속 떨어진다 (실측: 코퍼스 −0.07 dB/kHz,
    합성 음성 −0.5 dB/kHz).

    방법: 중역(2~8 kHz) 레벨 대비 `drop_db` 아래로 떨어지고, **그 위로 다시 안
    올라오며**, 하락이 `cliff_khz` 안에서 일어나고, 그 위 기울기가 `flat_db_per_khz`
    보다 완만한 첫 주파수. 못 찾으면 나이퀴스트를 돌려준다 — **확실하지 않으면
    건드리지 않는다.** 잘못 자르면 진짜 신호를 버린다.
    """
    f, p = average_psd(np.asarray(x, dtype=np.float64), fs)
    if len(f) < 16:
        return 0.5 * fs
    db = 10.0 * np.log10(p + 1e-30)
    mid = (f >= 2000.0) & (f <= 8000.0)
    if not mid.any():
        return 0.5 * fs
    ref = float(np.median(db[mid]))
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    back = max(1, int(round(cliff_khz * 1000.0 / max(df, 1e-9))))
    hi = np.flatnonzero(f > max(4000.0, 0.25 * fs))
    for i in hi:
        if db[i] >= ref - drop_db:
            continue
        if not bool(np.all(db[i:] < ref - floor_db)):
            continue
        # 절벽인가 — 바로 앞 cliff_khz 안에 이 하락의 대부분이 들어 있어야 한다.
        j = max(0, i - back)
        if float(db[j] - db[i]) < 0.6 * drop_db:
            continue
        # 컷 위가 **평탄한 바닥**인가. 손실 압축은 그 위에 양자화·디더 잡음만 남아
        # 거의 수평이고, 자연 롤오프는 계속 떨어진다. 실측: 코퍼스 −0.07 dB/kHz,
        # 합성 음성 −0.5 dB/kHz. 이 조건이 없으면 합성의 고역 롤오프를 컷으로 오인한다.
        tail = db[i:]
        if len(tail) >= 8:
            slope = float(np.polyfit(f[i:] / 1000.0, tail, 1)[0])
            if slope < -flat_db_per_khz:
                continue
        return float(f[i])
    return 0.5 * fs
