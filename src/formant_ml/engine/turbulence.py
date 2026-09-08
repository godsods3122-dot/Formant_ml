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

즉 **완벽한 모형이라도 치찰음에서 정밀 34.5 % 가 상한**이다. 95 % 를 목표로 걸면
그건 달성 불가능한 목표가 아니라 **틀린 자**다.

무엇으로 바꾸는가
-----------------
1. **편향 보정 스펙트럼 수렴도** (`corrected_sc`). 같은 파라미터를 시드만 바꿔 한 번 더
   합성하면 그 둘의 거리가 곧 "완벽한 모형이라도 남는" 실현 잡음이다. 그 바닥을
   제곱 영역에서 빼면 남는 것이 **편향** — 모형이 실제로 틀린 만큼이다. 완벽한
   모형에서 100 % 로 수렴한다.

   **다만 이 자에는 신뢰도가 붙는다** (`trust` / `resolved`). 편향이 실현 잡음보다
   작으면 분해가 안 되고, 그때의 값은 "이 값이다" 가 아니라 **"적어도 이 값"** 이다.
   완벽한 모형이 거기 오고(편향이 실제로 0 이니까) 잡음이 과한 합성도 거기 온다 —
   **둘을 가르는 것은 `noise_ratio`** (합성/목표 실현 분산 비) 다. 실측 대조군:

    | 대조군            | 정밀  | 보정  | 신뢰도 | 분해 | 잡음비 |
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
# 보정 일치율이 **분해**됐다고 볼 최소 신뢰도. 목표-합성 거리 중 실현 잡음으로 설명되지
# 않는 몫이 이만큼은 되어야, 남은 편향을 "쟀다" 고 말할 수 있다. 그 아래면 값은
# 상한일 뿐이다 (편향 ≤ 그 값). 완벽한 모형도 여기 오므로 실패 표시가 아니다 —
# 잡음량이 맞는지는 `noise_ratio` 로 따로 본다.
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
    tot = E[(fm >= edges[0]) & (fm < edges[-1])].sum() + 1e-30
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

    이므로 **빼야 할 것은 V_t + V_p** 이고 그중 V_p 만 정확히 안다 (= D_pp'/2).

    **V_t ≈ V_p 를 가정하고 D_pp' 를 통째로 빼면 안 된다.** 합성의 실현 분산이 목표보다
    크면 (실제로 그렇다 — 녹음은 위너 차감을 거쳐 스펙트럼이 평활해져 있다) 과하게
    빼서 **잡음 투성이 합성이 100 % 를 받는다.** 실측: 치찰음 구간 넷 전부 100.00.

    그래서 V_t 를 따로 추정한다. 시간 평활 잔차는 절대 눈금이 안 맞지만, 합성 쪽에서는
    참값(D_pp'/2)을 아니까 그것으로 눈금을 교정해 목표에 옮긴다:

        V_t ≈ V_t^est · (D_pp'/2) / V_p^est

    V_p = 2·V_t 인 경우로 확인하면 정확히 ‖m_t−m_p‖² 가 남는다.

    돌려주는 값
    -----------
    * `sc`    — 보정 SC. 완벽한 모형에서 0 (= 일치 100 %).
    * `floor` — 실현 잡음이 만드는 바닥. 원래 SC 가 이보다 좋아질 수 없다.
    * `trust` — `(D_tp − 뺀 양) / D_tp`. **0 에 가까우면 보정값을 믿으면 안 된다** —
      실현 잡음이 거리를 통째로 설명해 버려 편향을 못 잰다. 그럴 때는
      `spectrum_match`(시간 평균 스펙트럼)를 대신 본다.
    * `noise_ratio` — V_p/V_t 추정. 1 보다 크면 합성이 목표보다 잡음이 심하다.
      **이건 보정으로 지워지는 양이 아니라 별개의 결함이다** (잡음량도 물리
      파라미터다). 모양이 맞는지와 잡음량이 맞는지를 갈라서 봐야 한다.

    성질:
    * 완벽한 모형 -> sc 0. 원래 SC 는 0.655 (일치 34.5 %).
    * 결정적 신호(하모닉) -> D_pp' ≈ 0 이므로 보정이 아무 일도 안 한다.
    """
    n_t = float((np.asarray(mag_t, dtype=np.float64) ** 2).sum())
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
    v_t = vt_est * (v_p / max(vp_est, 1e-30))    # 합성 쪽 참값으로 눈금을 교정해 옮긴다
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
        mod_mae_pct=float(np.nanmean(np.abs(b["mod"] - a["mod"]))),
        kurtosis_err=b["kurtosis_t"] - a["kurtosis_t"],
        match_spectrum=spectrum_match(target, synth, fs),
    )


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
    * `fine_corr` — 편향만 남긴 값. 완벽한 모형에서 100 % 로 수렴한다.
    * `floor`     — 실현 잡음이 만드는 바닥. 원래 값이 이보다 좋아질 수 없다.
    * `trust` / `resolved` — **`resolved` 가 False 면 `fine_corr` 을 "이 값이다" 로
      읽으면 안 되고 "적어도 이 값" 으로 읽어야 한다.** 편향이 실현 잡음보다 작아
      분해가 안 된 것이다. 완벽한 모형도 여기 오고 (편향이 실제로 0 이니까), 잡음이
      과한 합성도 여기 온다 — **둘을 가르는 것은 `noise_ratio` 다.**
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
    if not raw:
        return dict(fine=float("nan"), fine_corr=float("nan"), floor=float("nan"),
                    trust=float("nan"), noise_ratio=float("nan"))
    trust = float(np.mean(tru))
    return dict(fine=100.0 * (1.0 - float(np.mean(raw))),
                fine_corr=100.0 * (1.0 - float(np.mean(cor))),
                floor=100.0 * (1.0 - float(np.mean(flo))),
                trust=trust, noise_ratio=float(np.mean(nr)) if nr else float("nan"),
                resolved=bool(trust >= TRUST_MIN))
