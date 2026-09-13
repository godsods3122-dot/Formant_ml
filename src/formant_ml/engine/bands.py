"""성대 밴드 구조 — 음원의 분지(band)를 프로파일하고 발화 전체를 한 번에 걷는다.

사용자(2026-09-12): *"성대 밴드 구조를 만들어서 프로파일해. 그리고 한 샘플에서 항상 밴드 위를
연속적으로 걷도록 강제하고, 분기점이 나타나면 더 좋은 방향으로 가도록 해. 어지간하면 에너지적으로
최적인 길로 갈 수 있도록(즉, 물리적으로) 매커니즘을 짜봐. 기류가 바뀌는 등 조건이 뒤집어진다면
점핑을 해도 돼. 한 목소리에 대해 가능한 여러 밴드의 점들이 존재할 것이고, 그것들을 병렬로 돌리다가
더 잘 맞는 쪽만 남기도록 하면 결국엔 최적 결과가 나올 것이다."*

**밴드가 무엇인가.** 이 엔진의 성문 음원은 LF 모형이고, 그 모양을 정하는 것은 `Rd` 한 값이다
(`glottis.lf_table` 이 Rd 격자 24 개에 대해 하모닉 복소계수를 미리 만들어 둔다). 즉 **Rd 격자가 곧
밴드**이고, 각 밴드는 고유한 스펙트럼 지문 — 음원 기울기 H1–H2, H2–H4, H4–H8 — 을 갖는다. 압착
발성(낮은 Rd)은 기울기가 완만하고 기식 발성(높은 Rd)은 가파르다. 문헌(A 층)이 음원을 기울기 넷으로
기술하는 것과 같은 좌표다 (Kreiman·Garellek 2016).

**왜 필요한가.** 지금까지 `rd_offset` 의 분석 초기값은 **전 구간 0** 이었다 — 음원 모양에 대한 관측이
아예 없어서 적합기가 맨손으로 지어냈고, 그것이 모음 0~1 kHz(= H1·H2 대역, 포락 오차의 85~93 %,
MEASUREMENTS §51.2)를 흔드는 자유도가 됐다. 목표에서 기울기를 **재서** 밴드를 고르면 그 자유도가
관측에 묶인다.

**어떻게 고르는가.** 프레임마다 가장 잘 맞는 밴드를 따로 고르면 창마다 튄다(포먼트 배정에서 이미 겪었다,
§51.21). 그래서 **발화 전체를 놓고 비터비**로 푼다:

    값 = Σ_t [ 밴드 적합도(t, i) + 에너지(t, i) ] + Σ_t 연속성(i_{t-1} → i_t) · 문턱(t)

* **밴드 적합도** — 목표에서 잰 기울기와 그 밴드가 예측하는 기울기의 차 [dB]. 성도 전달함수는
  추적된 포먼트로 보정해서 뺀다 (Iseli & Alwan 식 보정).
* **연속성** — 이웃 프레임의 밴드 사이 거리. 이것이 "밴드 위를 연속적으로 걷는다" 이다.
* **문턱** — 기류 조건이 뒤집히면(협착 면적·유성 여부·폐압이 급변하면) 연속성 값을 낮춘다. 즉
  **조건이 바뀔 때만 점프가 싸다.**
* **에너지** — 같은 소리 크기를 더 작은 구동으로 내는 밴드를 선호한다 (물리적으로 자연스러운 길).

분기점에서는 두 길의 값이 비슷해지고, 비터비가 **뒤의 증거까지 보고** 더 나은 쪽을 남긴다 — 사용자가
말한 "병렬로 돌리다가 더 잘 맞는 쪽만 남긴다" 가 격자(lattice) 그 자체다.
"""
from __future__ import annotations

import numpy as np

#: 밴드의 지문으로 쓸 하모닉 묶음 — (H1–H2, H2–H4, H4–H8).
BAND_FEATS = ((1, 1, 2, 2), (2, 2, 3, 4), (3, 4, 5, 8))
#: 지문마다의 가중 (저역이 포락 오차를 지배하므로 H1–H2 를 무겁게).
BAND_FEAT_W = (1.0, 0.7, 0.4)
#: 연속성 값 — **Rd 한 단위를 움직이는 값 [dB]**. 격자 눈금이 아니라 Rd 에 대해 정의한다.
#:
#: 왜: 처음에는 "한 칸당 2 dB" 였는데, 사용자의 *"샘플을 딸 거면 훨씬 더 촘촘하게 해"* 에 따라 격자를
#: 24 → 512 로 늘리면서 이 값을 같이 고치지 않았다. 눈금이 0.104 → 0.0047 로 22 배 고와졌으므로
#: 한 칸당 2 dB 는 **Rd 한 단위당 426 dB** 가 되어 길이 얼어붙었다 — 실측으로 1.4 초 발화에서 고원이
#: `s040` 15 개, `s101` 30 개뿐이었고 지문 잔차가 4.11 / 4.09 dB 였다.
#:
#: 값은 **점프가 기류 뒤집힘에 묶여 있는지**로 골랐다 (그것이 사용자가 요구한 성질이다). ΔRd > 0.3 인
#: 도약이 일어난 자리의 `flip` 중앙값 / 지문 잔차 [dB] (`s040` · `s101`):
#:
#:     426 dB/Rd  flip 1.00 · 1.00   잔차 4.11 · 4.09   (얼어붙음)
#:     100        flip 1.00 · 1.00   잔차 3.84 · 3.59
#:  ** 50        flip 1.00 · 1.00   잔차 3.40 · 3.28 **
#:      19.2     flip 0.35 · 0.32   잔차 3.08 · 2.92   (점프가 조건과 무관해지기 시작)
#:       8       flip 0.11 · 0.07   잔차 2.84 · 2.77   (아무 데서나 뛴다)
#:
#: 50 dB/Rd 가 **모든 큰 도약이 flip = 1.00 에 붙어 있는 가장 느슨한 값**이다. 전체 프레임의 flip
#: 중앙값은 0.04~0.05 이므로 이것은 우연이 아니다.
BAND_W_CONT = 50.0
#: 점프가 가장 쌀 때의 바닥 [dB/Rd] — 조건이 완전히 뒤집혀도 이만큼은 든다.
BAND_G_FLOOR = 4.0
BAND_W_JUMP = 1.0
BAND_W_ENERGY = 0.15


#: 밴드 격자의 촘촘함. 사용자: *"샘플을 딸 거면 훨씬 더 촘촘하게 해."* 전이 값이 거리에 비례하므로
#: 비터비가 O(n_rd) 거리 변환으로 풀려, 격자를 키워도 시간이 늘지 않는다 (24 → 512 에서 측정상 차이 없음).
#: Rd 범위 0.3~2.7 을 512 로 나누면 눈금이 0.0047 — LF 모양의 변화가 눈금 사이에서 보이지 않는 수준이다.
BAND_N_RD = 512


def dense_profile(n_rd: int = BAND_N_RD, n_harm: int = 16):
    """촘촘한 Rd 격자의 밴드 지문을 만든다 -> (rds, feats, total)."""
    from .glottis import lf_table
    rds, coef = lf_table(n_rd, n_harm=n_harm)
    rds = rds.numpy().astype(float)
    feats, total = band_profile(rds, coef.numpy())
    return rds, feats, total


def band_profile(rds: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rd 격자의 하모닉 계수 -> 밴드 지문 (n_rd, n_feat) [dB] 와 전체 수준 (n_rd,) [dB].

    `coef` 는 `glottis.lf_table` 이 만든 (n_rd, K) 복소 계수다. 하모닉 크기는 |c_k| 에 비례한다.
    """
    mag = np.abs(np.asarray(coef))
    n_rd = mag.shape[0]
    feats = np.zeros((n_rd, len(BAND_FEATS)))
    for j, (a0, a1, b0, b1) in enumerate(BAND_FEATS):
        lo = 20.0 * np.log10(mag[:, a0 - 1:a1].mean(1) + 1e-20)
        hi = 20.0 * np.log10(mag[:, b0 - 1:b1].mean(1) + 1e-20)
        feats[:, j] = lo - hi
    total = 10.0 * np.log10((mag[:, :12] ** 2).sum(1) + 1e-20)
    return feats, total


#: 성도 보정에 쓸 **고차 극 꼬리**의 개수. 추적된 F1~F4 위로 균일관 간격 c/2L 의 극을 이만큼 더 세운다.
#: 빼먹으면 안 된다 — 고차 극은 2 kHz 근처에도 주파수에 따라 커지는 치마를 드리우므로, 없으면 높은
#: 하모닉이 실제보다 강해 보이고 음원이 평평해 **보인다** (실측: H4–H8 잔차가 11.3 → 아래 표).
BAND_N_TAIL = 24


def tract_level_db(f: np.ndarray, formants: np.ndarray, bws: np.ndarray,
                   spacing: float = 0.0, n_tail: int = BAND_N_TAIL) -> np.ndarray:
    """전극 성도의 크기응답 [dB] (DC 정규화). f, formants, bws 는 Hz.

    |H(f)| = Π_k |1 - (f/F_k)² + j f B_k / F_k²|⁻¹ — 아날로그 2 차 공진의 표준형이다.
    `spacing` 을 주면 마지막 포먼트 위로 그 간격의 고차 극을 `n_tail` 개 더 세운다 (엔진의
    `VocalTract._extra_cascade` 와 같은 구성). 관측 하모닉 수준에서 성도의 몫을 걷어내는 것이 목적이다.
    """
    f = np.asarray(f, float)[:, None]
    F = list(np.asarray(formants, float))
    B = list(np.asarray(bws, float))
    if spacing > 0.0 and n_tail > 0:
        fl = F[-1]
        for _ in range(n_tail):
            fl = fl + spacing
            F.append(fl)
            B.append(40.0 + 0.05 * fl)
    F = np.maximum(np.asarray(F, float)[None, :], 1.0)
    B = np.maximum(np.asarray(B, float)[None, :], 1.0)
    d = 1.0 - (f / F) ** 2 + 1j * f * B / (F ** 2)
    return -20.0 * np.log10(np.abs(d) + 1e-20).sum(1)


def measure_slopes(y: np.ndarray, fs: float, f0: np.ndarray, voiced: np.ndarray,
                   formants: np.ndarray, bws: np.ndarray, hop: int,
                   n_harm: int = 8, cycles: float = 3.0, spacing: float = 0.0):
    """프레임마다 목표의 **음원** 기울기를 잰다 -> (T, n_feat) [dB], 유효 마스크 (T,).

    하모닉 k·f0 의 크기를 읽고 성도 전달함수(추적된 포먼트)를 빼서 음원만 남긴다.
    창은 3 주기 해닝 — 하모닉이 분해되면서도 주기 간 변화를 뭉개지 않는다.
    """
    T = len(f0)
    obs = np.zeros((T, len(BAND_FEATS)))
    lev = np.zeros(T)
    ok = np.zeros(T, bool)
    n = len(y)
    for t in range(T):
        if not voiced[t] or not (50.0 < f0[t] < 800.0):
            continue
        w = int(cycles * fs / f0[t])
        w = int(min(max(w, 128), 4096))
        c = t * hop + hop // 2 - w // 2
        if c < 0 or c + w > n:
            continue
        seg = y[c:c + w] * np.hanning(w)
        nfft = 1 << int(np.ceil(np.log2(w * 4)))
        S = np.abs(np.fft.rfft(seg, nfft)) + 1e-20
        fr = np.fft.rfftfreq(nfft, 1.0 / fs)
        hk = np.arange(1, n_harm + 1) * f0[t]
        if hk[-1] > 0.45 * fs:
            continue
        amp = np.empty(n_harm)
        half = max(1, int(0.35 * f0[t] / (fr[1] - fr[0])))
        for k, fk in enumerate(hk):
            i = int(round(fk / (fr[1] - fr[0])))
            lo, hi = max(i - half, 0), min(i + half + 1, len(S))
            amp[k] = S[lo:hi].max()
        db = 20.0 * np.log10(amp) - tract_level_db(hk, formants[t], bws[t], spacing)
        wk = harmonic_weights(hk, formants[t], bws[t])
        for j, (a0, a1, b0, b1) in enumerate(BAND_FEATS):
            wa, wb = wk[a0 - 1:a1], wk[b0 - 1:b1]
            if wa.sum() < 1e-3 or wb.sum() < 1e-3:
                obs[t, j] = np.nan
                continue
            obs[t, j] = ((db[a0 - 1:a1] * wa).sum() / wa.sum()
                         - (db[b0 - 1:b1] * wb).sum() / wb.sum())
        lev[t] = 10.0 * np.log10((10.0 ** (db / 10.0)).sum() + 1e-20)
        ok[t] = bool(np.all(np.isfinite(obs[t])))
    return obs, lev, ok


def harmonic_weights(hk: np.ndarray, formants: np.ndarray, bws: np.ndarray) -> np.ndarray:
    """하모닉마다의 **신뢰도** 0~1 — 포먼트 봉우리에 가까울수록 낮다.

    왜: 성도 보정은 포먼트 위치의 오차에 민감하고, 그 민감도는 봉우리 근처에서 폭발한다. 이 화자는
    F0 가 290 Hz 라 **H2 가 F1 바로 위**에 놓여, F1 이 3 % 만 틀려도 H2 의 보정이 몇 dB 씩 어긋난다.
    봉우리에서 1.5 대역폭 안의 하모닉은 무게를 줄인다.
    """
    d = (np.asarray(hk, float)[:, None] - np.asarray(formants, float)[None, :])
    b = np.maximum(np.asarray(bws, float)[None, :], 40.0) * 1.5
    near = np.exp(-((d / b) ** 2)).max(1)
    return 1.0 - 0.85 * near


def speaker_offset(obs: np.ndarray, ok: np.ndarray, feats: np.ndarray):
    """화자 고유의 음원 모양 = 관측과 밴드 예측의 **상수 차** (n_feat,) [dB], 그리고 기준 밴드 번호.

    LF 한 모수(Rd)로는 H1–H2 와 H2–H8 을 동시에 못 맞춘다 (실측: 잔차 2.2 / 5.8 / 6.4 dB). 그 상수 몫은
    **화자 프로파일**(정적 음원 EQ)의 것이고, 밴드 추적이 따라야 할 것은 그 둘레의 **변화분**이다
    (docs/FOUNDATION.md §1 의 화자 프로파일 / 발화 제어 분리).
    """
    m = np.asarray(ok, bool)
    if m.sum() < 8:
        return np.zeros(obs.shape[1]), 0
    med = np.median(obs[m], axis=0)
    i0 = int(np.abs(feats - med[None, :]).sum(1).argmin())
    return med - feats[i0], i0


def track_bands(obs: np.ndarray, lev: np.ndarray, ok: np.ndarray,
                feats: np.ndarray, total: np.ndarray, flip: np.ndarray,
                w_cont: float = BAND_W_CONT, w_jump: float = BAND_W_JUMP,
                w_energy: float = BAND_W_ENERGY, d_rd: float = 1.0) -> np.ndarray:
    """밴드 격자 위의 비터비 -> 프레임마다의 밴드 번호 (T,).

    `flip` 은 0~1 의 "조건이 뒤집힌 정도" (기류·유성·폐압의 급변). 1 에 가까우면 점프가 싸다.

    `d_rd` 는 격자 눈금 (Rd 단위/칸) 이다. 전이 값은 `w_cont`(dB/Rd) × `d_rd` × |i−j| 라
    **격자를 촘촘히 해도 길의 뻣뻣함이 변하지 않는다** (`BAND_W_CONT` 참조).

    **격자는 촘촘해도 된다.** 전이 값이 밴드 사이 거리 |i−j| 에 비례하므로
    `min_j (cost[j] + g·|i−j|)` 는 앞뒤로 한 번씩 훑는 **거리 변환**으로 O(n_rd) 에 풀린다
    (naive 는 O(n_rd²)). 그래서 512 격자도 24 격자와 사실상 같은 시간에 돈다.
    """
    T, n_rd = len(obs), feats.shape[0]
    fw = np.asarray(BAND_FEAT_W)[None, :]
    local = np.zeros((T, n_rd))
    for t in range(T):
        if not ok[t]:
            continue
        local[t] = (np.abs(obs[t][None, :] - feats) * fw).sum(1) / fw.sum()
        # 에너지: 같은 관측 수준을 내는 데 필요한 구동. 작을수록(효율적일수록) 좋다.
        need = lev[t] - total
        local[t] += w_energy * (need - need.min())
    back = np.zeros((T, n_rd), np.int32)
    cost = local[0].copy()
    idx0 = np.arange(n_rd, dtype=np.int32)
    for t in range(1, T):
        # 값은 **Rd 단위**로 매기고 눈금을 곱한다 — 격자를 촘촘히 해도 뻣뻣함이 안 변한다.
        g = max(w_cont * (1.0 - w_jump * float(np.clip(flip[t], 0.0, 1.0))),
                BAND_G_FLOOR) * float(d_rd)
        m, src = cost.copy(), idx0.copy()
        for i in range(1, n_rd):                      # 앞으로 한 번
            if m[i - 1] + g < m[i]:
                m[i], src[i] = m[i - 1] + g, src[i - 1]
        for i in range(n_rd - 2, -1, -1):             # 뒤로 한 번
            if m[i + 1] + g < m[i]:
                m[i], src[i] = m[i + 1] + g, src[i + 1]
        back[t] = src
        cost = m + local[t]
    path = np.zeros(T, np.int64)
    path[-1] = int(cost.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def flow_flip(a_c: np.ndarray, p_sub: np.ndarray, voiced: np.ndarray,
              win: int = 5) -> np.ndarray:
    """기류 조건이 뒤집힌 정도 0~1 — 협착 면적·폐압의 상대 변화율과 유성 전환.

    조음이 바뀌는 순간에는 성대의 분지가 실제로 갈아타므로, 그때는 연속성을 풀어 준다.
    """
    def rate(x):
        x = np.asarray(x, float)
        d = np.abs(np.gradient(np.log(np.maximum(x, 1e-6))))
        k = np.ones(win) / win
        return np.convolve(d, k, mode="same")
    r = rate(a_c) / 0.05 + rate(p_sub) / 0.02
    v = np.abs(np.gradient(np.asarray(voiced, float)))
    v = np.convolve(v, np.ones(win) / win, mode="same") * 4.0
    return np.clip(r + v, 0.0, 1.0)


#: 비강 분기의 격자 탐색 범위 (MEASUREMENTS §51.28). 엔진의 `_nasal_branch` 와 같은 구성 —
#: 극 셋(130·300·500 Hz 대역폭 × damp)과 영점 하나.
NASAL_GRID = dict(f1=(200.0, 400.0, 9), f2=(700.0, 1700.0, 11), f3=(1800.0, 3200.0, 8),
                  z=(700.0, 2600.0, 13), damp=(0.3, 2.0, 8))


#: 비강의 **여분 극** — 인두+비강 20 cm 의 c/2L ≈ 875 Hz 간격, 2.4 kHz 위. `tract.VocalTract` 와 같다.
NASAL_EXTRA_F0, NASAL_EXTRA_SP, NASAL_EXTRA_MAX = 2400.0, 34000.0 / (2.0 * 20.0), 0.70 * 24000.0
#: 측지 영점의 대역폭 = fz × 이 값 × damp. `tract.VocalTract.nasal_zfrac` 와 같아야 한다.
NASAL_ZFRAC = 0.80


def nasal_extra_poles(f_max: float = NASAL_EXTRA_MAX) -> np.ndarray:
    """엔진의 `nasal_extra` 와 같은 극 열."""
    return np.array([NASAL_EXTRA_F0 + k * NASAL_EXTRA_SP for k in range(1, 64)
                     if NASAL_EXTRA_F0 + k * NASAL_EXTRA_SP < f_max], dtype=float)


def nasal_transfer_db(f, f1, f2, f3, z, damp, extra: np.ndarray | None = None,
                      zfrac: float = NASAL_ZFRAC, fs: float = 48000.0):
    """비강 분기의 크기응답 [dB] — **엔진의 계수 함수를 그대로 써서** 디지털 응답을 낸다.

    두 번 틀렸던 자리다.

    1. 극 셋 + 폭 400 Hz 영점만 두었는데, 엔진은 2.4 kHz 위에 875 Hz 간격의 여분 극 16 개를
       더 세우고 영점 폭도 `fz × 0.80 × damp` 를 쓴다 — 같은 모수에서 3~6.5 kHz 가 **39~48 dB**
       벌어졌다.
    2. 여분 극을 넣어도 **아날로그 근사식**(1 − (f/F)² + j f B/F²)은 감쇠가 낮을 때
       디지털 구현과 갈라진다 — `damp` = 0.30 에서 2~9 kHz 가 **12~21 dB** 벌어졌다.

    그래서 `tviir.resonator_coeffs`·`notch_coeffs` 가 낸 바이쿼드를 단위원에서 그대로 평가한다.
    적합하는 식과 렌더하는 식이 **구성상** 같아진다 (`out/_tmp/nasal_branch_check.py` 로 검산).
    """
    from .tviir import resonator_coeffs, notch_coeffs
    f = np.asarray(f, float)
    zz = np.exp(-2j * np.pi * f / fs)

    def bq(b0, b1, b2, a1, a2):
        num = b0 + b1 * zz + b2 * zz * zz
        den = 1.0 + a1 * zz + a2 * zz * zz
        return 20.0 * np.log10(np.abs(num / den) + 1e-20)

    out = np.zeros_like(f)
    ex = nasal_extra_poles() if extra is None else np.asarray(extra, float)
    for fp, bw in ([(f1, 130.0), (f2, 300.0), (f3, 500.0)]
                   + [(fk, 300.0 + 0.25 * fk) for fk in ex]):
        out += bq(*resonator_coeffs(float(fp), float(bw * damp), fs))
    out += bq(*notch_coeffs(float(z), float(z * zfrac * damp), fs))
    return out


def _nasal_pole_db(f, fp, bw, fs):
    from .tviir import resonator_coeffs
    zz = np.exp(-2j * np.pi * np.asarray(f, float) / fs)
    b0, b1, b2, a1, a2 = resonator_coeffs(float(fp), float(bw), fs)
    return 20.0 * np.log10(np.abs((b0 + b1 * zz + b2 * zz * zz)
                                  / (1.0 + a1 * zz + a2 * zz * zz)) + 1e-20)


def _nasal_zero_db(f, fz, bw, fs):
    from .tviir import notch_coeffs
    zz = np.exp(-2j * np.pi * np.asarray(f, float) / fs)
    b0, b1, b2, a1, a2 = notch_coeffs(float(fz), float(bw), fs)
    return 20.0 * np.log10(np.abs((b0 + b1 * zz + b2 * zz * zz)
                                  / (1.0 + a1 * zz + a2 * zz * zz)) + 1e-20)


def fit_nasal(freqs, level_db, w=None, fs: float = 48000.0):
    """머머의 (음원을 나눈) 스펙트럼에 비강 전달함수를 맞춘다 -> (f1, f2, f3, z, damp, 잔차 dB).

    **추측하지 않고 잰다.** 기본값으로 두면 비강 분기가 0.3~2 kHz 에서 10~20 dB 어둡다(실측).
    격자 탐색이라 국소 최소가 없고, 눈금이 촘촘하지 않아도 적합기가 뒤에서 다듬는다.

    응답은 `nasal_transfer_db` 와 같은 디지털 식이되, **극마다·damp 마다 미리 계산해 두고 더한다**
    — 격자를 넓히면 조합이 수십만이라 매번 19 개 바이쿼드를 평가하면 분 단위가 된다.
    """
    m = (freqs >= 100) & (freqs <= 5000)
    fr, y = np.asarray(freqs, float)[m], np.asarray(level_db, float)[m]
    y = y - y.max()
    ww = np.ones_like(y) if w is None else np.asarray(w)[m]
    g = {k: np.linspace(a, b, n) for k, (a, b, n) in NASAL_GRID.items()}
    ex = nasal_extra_poles()
    # damp 마다: 여분 극 합, 그리고 극·영점 후보별 응답을 미리 만든다.
    P1, P2, P3, Z, EX = {}, {}, {}, {}, {}
    for d in g["damp"]:
        EX[d] = sum(_nasal_pole_db(fr, fk, (300.0 + 0.25 * fk) * d, fs) for fk in ex)
        P1[d] = {v: _nasal_pole_db(fr, v, 130.0 * d, fs) for v in g["f1"]}
        P2[d] = {v: _nasal_pole_db(fr, v, 300.0 * d, fs) for v in g["f2"]}
        P3[d] = {v: _nasal_pole_db(fr, v, 500.0 * d, fs) for v in g["f3"]}
        Z[d] = {v: _nasal_zero_db(fr, v, v * NASAL_ZFRAC * d, fs) for v in g["z"]}
    best = None
    for d in g["damp"]:
        base = EX[d]
        for f1 in g["f1"]:
            h1 = base + P1[d][f1]
            for f2 in g["f2"]:
                if f2 <= f1: continue
                h2 = h1 + P2[d][f2]
                for f3 in g["f3"]:
                    if f3 <= f2: continue
                    h3 = h2 + P3[d][f3]
                    for z in g["z"]:
                        h = h3 + Z[d][z]
                        e = y - (h - h.max())
                        e = e - np.average(e, weights=ww)
                        c = float(np.sqrt(np.average(e * e, weights=ww)))
                        if best is None or c < best[0]:
                            best = (c, f1, f2, f3, z, d)
    c, f1, f2, f3, z, damp = best
    return f1, f2, f3, z, damp, c
