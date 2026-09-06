"""시변(time-varying) 2차 재귀 필터 원시연산.

세 가지 구현이 **같은 방정식**을 푼다 (테스트가 세 경로의 일치를 고정한다).

1. `tv_biquad`      : torch, 결합 스캔(associative scan). 길이 N 에 대해
                       O(N log N) 병렬, autograd 로 정확히 미분가능. **학습 경로.**
2. `tv_biquad_seq`  : torch, 샘플 루프. 느리지만 자명한 참조 구현.
3. `tv_biquad_np`   : numpy/numba, 샘플 루프. **실시간 경로.** numba 가 없으면
                       블록별 scipy.lfilter 로 떨어진다(계수를 블록 안에서 고정).

방정식 — 전치 직접형 II(TDF-II). scipy.signal.lfilter 와 같은 상태 규약이라
`zi/zf` 를 그대로 주고받을 수 있다.

    y[n]  = b0[n]·x[n] + s1[n-1]
    s1[n] = (b1[n] − a1[n]·b0[n])·x[n] − a1[n]·s1[n-1] + s2[n-1]
    s2[n] = (b2[n] − a2[n]·b0[n])·x[n] − a2[n]·s1[n-1]

상태 s = [s1, s2] 에 대해 이것은 선형 점화식 s[n] = A[n]·s[n-1] + v[n] 이고,

    A[n] = [[−a1[n], 1], [−a2[n], 0]],   v[n] = [(b1−a1·b0)·x, (b2−a2·b0)·x][n]

(A, v) 쌍의 합성 (A2,v2)∘(A1,v1) = (A2·A1, A2·v1 + v2) 은 결합법칙을 만족하므로
Hillis–Steele 스캔으로 log2(N) 단계에 푼다. 극이 단위원 안이면 A 의 곱은
수축이라 float32 로도 안정하다.

왜 이것이 v1 의 LTV-FIR 를 대체하는가
--------------------------------------
* 계수가 **샘플마다** 바뀐다. 프레임 경계가 없으므로 위상 계단이 없다.
* 상태가 이어지므로 경계조건이 급변할 때의 **과도응답**이 저절로 나온다.
* 인과 재귀라 프리에코가 구조적으로 불가능하다.
* 극 반지름 r = exp(−π·BW/fs) < 1 이면 설계상 항상 안정하다.
"""
from __future__ import annotations

import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------- 스캔
def _shift(x: torch.Tensor, k: int, fill: float) -> torch.Tensor:
    """x[..., n] <- x[..., n-k] (앞쪽 k 개는 fill). 시간축은 마지막 축."""
    pad = torch.full_like(x[..., :k], fill)
    return torch.cat([pad, x[..., :-k]], dim=-1)


def linear_recurrence_2x2(p11, p12, p21, p22, v1, v2):
    """s[n] = P[n]·s[n-1] + v[n] 을 결합 스캔으로 푼다. 모든 인자 (B, N).

    반환 s1, s2 : (B, N) — 각 n 에서의 상태 (s[-1] = 0 으로 시작).
    초기 상태가 있으면 v[0] 에 P[0]·s[-1] 을 더해서 넣으면 된다.
    """
    n = p11.shape[-1]
    k = 1
    while k < n:
        q11, q12 = _shift(p11, k, 1.0), _shift(p12, k, 0.0)
        q21, q22 = _shift(p21, k, 0.0), _shift(p22, k, 1.0)
        u1, u2 = _shift(v1, k, 0.0), _shift(v2, k, 0.0)
        # S[n] <- S[n] + P[n]·S[n-k]
        v1, v2 = v1 + p11 * u1 + p12 * u2, v2 + p21 * u1 + p22 * u2
        # P[n] <- P[n]·P[n-k]
        p11, p12, p21, p22 = (p11 * q11 + p12 * q21, p11 * q12 + p12 * q22,
                              p21 * q11 + p22 * q21, p21 * q12 + p22 * q22)
        k *= 2
    return v1, v2


def _bcast(c, like: torch.Tensor) -> torch.Tensor:
    """스칼라/(B,1)/(B,N) 계수를 (B,N) 으로 방송한다."""
    if not torch.is_tensor(c):
        c = torch.tensor(float(c), dtype=like.dtype, device=like.device)
    c = c.to(like.dtype)
    return c.expand_as(like) if c.shape != like.shape else c


SCAN_DTYPE = torch.float64     # 스캔은 float64 로. 아래 docstring 참조.


def tv_biquad(x: torch.Tensor, b0, b1, b2, a1, a2,
              zi: torch.Tensor | None = None):
    """시변 2차 필터 (결합 스캔). x: (B, N). 계수: 스칼라 | (B,1) | (B,N).

    반환 (y (B,N), zf (B,2)).  zi/zf 는 scipy.lfilter 의 TDF-II 상태와 같다.

    내부 연산은 float64 다. 결합 스캔은 log2(N) 단계의 행렬 곱 트리라 반올림 오차가
    고 Q 공명기의 이득 1/(1−r)² (r=0.996 이면 6×10⁴) 만큼 증폭된다 — float32 로는
    스트리밍과 오프라인이 0.3 % 어긋났다(측정). float64 면 1e-6 이하다. 입력 dtype 은 유지된다.
    """
    dt = x.dtype
    x = x.to(SCAN_DTYPE)
    b0, b1, b2 = _bcast(b0, x), _bcast(b1, x), _bcast(b2, x)
    a1, a2 = _bcast(a1, x), _bcast(a2, x)
    if zi is not None:
        zi = zi.to(SCAN_DTYPE)
    v1 = (b1 - a1 * b0) * x
    v2 = (b2 - a2 * b0) * x
    if zi is not None:
        # s[0] = A[0]·s[-1] + v[0]
        s1p, s2p = zi[:, 0:1], zi[:, 1:2]
        v1 = torch.cat([v1[:, :1] + (-a1[:, :1] * s1p + s2p), v1[:, 1:]], -1)
        v2 = torch.cat([v2[:, :1] + (-a2[:, :1] * s1p), v2[:, 1:]], -1)
    s1, s2 = linear_recurrence_2x2(-a1, torch.ones_like(a1), -a2,
                                   torch.zeros_like(a1), v1, v2)
    s1_prev = _shift(s1, 1, 0.0)
    if zi is not None:
        s1_prev = torch.cat([zi[:, 0:1], s1_prev[:, 1:]], -1)
    y = b0 * x + s1_prev
    zf = torch.stack([s1[:, -1], s2[:, -1]], dim=-1)
    return y.to(dt), zf


def tv_biquad_seq(x, b0, b1, b2, a1, a2, zi=None):
    """샘플 루프 참조 구현 (테스트용)."""
    b0, b1, b2 = _bcast(b0, x), _bcast(b1, x), _bcast(b2, x)
    a1, a2 = _bcast(a1, x), _bcast(a2, x)
    b, n = x.shape
    s1 = torch.zeros(b, dtype=x.dtype, device=x.device)
    s2 = torch.zeros_like(s1)
    if zi is not None:
        s1, s2 = zi[:, 0].clone(), zi[:, 1].clone()
    ys = []
    for i in range(n):
        y = b0[:, i] * x[:, i] + s1
        s1_new = b1[:, i] * x[:, i] - a1[:, i] * y + s2
        s2 = b2[:, i] * x[:, i] - a2[:, i] * y
        s1 = s1_new
        ys.append(y)
    return torch.stack(ys, -1), torch.stack([s1, s2], -1)


# ------------------------------------------------------------ numpy / numba
try:                                                  # 실시간 경로
    import numba as _nb

    @_nb.njit(cache=True, fastmath=True)
    def _biquad_kernel(x, b0, b1, b2, a1, a2, s1, s2, y):
        for i in range(x.shape[0]):
            yi = b0[i] * x[i] + s1
            s1n = b1[i] * x[i] - a1[i] * yi + s2
            s2 = b2[i] * x[i] - a2[i] * yi
            s1 = s1n
            y[i] = yi
        return s1, s2

    HAVE_NUMBA = True
except Exception:                                     # pragma: no cover
    HAVE_NUMBA = False


def tv_biquad_np(x: np.ndarray, b0, b1, b2, a1, a2, zi=None, block: int = 16):
    """numpy 실시간 경로. x: (N,). 계수: 스칼라 | (N,). 반환 (y, zf(2,))."""
    n = x.shape[0]
    x = np.ascontiguousarray(x, dtype=np.float64)
    co = [np.ascontiguousarray(np.broadcast_to(np.asarray(c, dtype=np.float64), (n,)))
          for c in (b0, b1, b2, a1, a2)]
    s1, s2 = (0.0, 0.0) if zi is None else (float(zi[0]), float(zi[1]))
    y = np.empty(n, dtype=np.float64)
    if HAVE_NUMBA:
        s1, s2 = _biquad_kernel(x, *co, s1, s2, y)
        return y, np.array([s1, s2])
    from scipy.signal import lfilter                  # pragma: no cover
    z = np.array([s1, s2])
    for i in range(0, n, block):
        j = min(n, i + block)
        m = (i + j) // 2
        b = [co[0][m], co[1][m], co[2][m]]
        a = [1.0, co[3][m], co[4][m]]
        y[i:j], z = lfilter(b, a, x[i:j], zi=z)
    return y, z


# -------------------------------------------------------------- 계수 생성기
def pole_radius(bw_hz, fs: float):
    """대역폭 [Hz] -> 극 반지름 r = exp(−π·BW/fs) (< 1 이면 항상 안정)."""
    if torch.is_tensor(bw_hz):
        return torch.exp(-math.pi * bw_hz / fs)
    return np.exp(-math.pi * np.asarray(bw_hz, dtype=np.float64) / fs)


def _cos(theta):
    return torch.cos(theta) if torch.is_tensor(theta) else np.cos(theta)


def resonator_coeffs(f_hz, bw_hz, fs: float, gain=1.0):
    """Klatt 식 공명기 (전극, DC 이득 = gain). 반환 (b0, b1, b2, a1, a2).

    H(z) = gain·(1 + a1 + a2) / (1 + a1·z⁻¹ + a2·z⁻²)
    """
    r = pole_radius(bw_hz, fs)
    a1 = -2.0 * r * _cos(TWO_PI * f_hz / fs)
    a2 = r * r
    b0 = gain * (1.0 + a1 + a2)
    return b0, 0.0 * b0, 0.0 * b0, a1, a2


def antiresonator_coeffs(f_hz, bw_hz, fs: float):
    """반공명기 (영점쌍, DC 이득 1). 비음 안티포먼트, 설측음 측지 영점."""
    r = pole_radius(bw_hz, fs)
    c1 = -2.0 * r * _cos(TWO_PI * f_hz / fs)
    c2 = r * r
    g = 1.0 / (1.0 + c1 + c2)
    return g + 0.0 * c1, g * c1, g * c2, 0.0 * c1, 0.0 * c1


def notch_coeffs(f_hz, bw_zero_hz, fs: float, pole_ratio: float = 4.0):
    """극-영점 쌍 노치 (측지/비강 안티포먼트).

    영점쌍만 DC 정규화하면 먼 대역이 (1+2r·cosθ+r²)/(1−2r·cosθ+r²) 배로 뜬다 —
    3.3 kHz 영점이 44.1 kHz 에서 고역을 +25 dB 들어올린다(측정). 같은 각도에 더 넓은
    극쌍을 두면 노치 바깥에서 두 인자가 상쇄돼 응답이 1 로 돌아온다. 깊이는
    (bw_zero/bw_pole) 로 정해진다 — 물리적으로도 측지 손실이 깊이를 정한다.
    """
    rz = pole_radius(bw_zero_hz, fs)
    rp = pole_radius(bw_zero_hz * pole_ratio, fs)
    cs = _cos(TWO_PI * f_hz / fs)
    b1, b2 = -2.0 * rz * cs, rz * rz
    a1, a2 = -2.0 * rp * cs, rp * rp
    g = (1.0 + a1 + a2) / (1.0 + b1 + b2)
    return g + 0.0 * b1, g * b1, g * b2, a1, a2


def allpass_coeffs(f_hz, r, fs: float):
    """2차 올패스: |H| ≡ 1, 군지연만 바꾼다(위상차 필터).

    H(z) = (r² − 2r·cosθ·z⁻¹ + z⁻²) / (1 − 2r·cosθ·z⁻¹ + r²·z⁻²)
    """
    a1 = -2.0 * r * _cos(TWO_PI * f_hz / fs)
    a2 = r * r
    return a2, a1, 1.0 + 0.0 * a1, a1, a2


def lowpass_coeffs(f_hz, q, fs: float):
    """RBJ 2차 저역통과 (쌍일차 변환). 큰 대역폭의 DC 정규화 공명기를 저역통과로 쓰면
    안 된다 — 나이퀴스트에서 −11 dB 밖에 안 떨어진다(측정). 이쪽은 −40 dB/dec 다."""
    if torch.is_tensor(f_hz):
        w0 = TWO_PI * f_hz / fs
        sn, cs = torch.sin(w0), torch.cos(w0)
    else:
        w0 = TWO_PI * np.asarray(f_hz, dtype=np.float64) / fs
        sn, cs = np.sin(w0), np.cos(w0)
    alpha = sn / (2.0 * q)
    a0 = 1.0 + alpha
    b0 = (1.0 - cs) / 2.0 / a0
    b1 = (1.0 - cs) / a0
    b2 = b0
    a1 = (-2.0 * cs) / a0
    a2 = (1.0 - alpha) / a0
    return b0, b1, b2, a1, a2


def highpass_coeffs(f_hz, q, fs: float):
    """RBJ 2차 고역통과."""
    if torch.is_tensor(f_hz):
        w0 = TWO_PI * f_hz / fs
        sn, cs = torch.sin(w0), torch.cos(w0)
    else:
        w0 = TWO_PI * np.asarray(f_hz, dtype=np.float64) / fs
        sn, cs = np.sin(w0), np.cos(w0)
    alpha = sn / (2.0 * q)
    a0 = 1.0 + alpha
    b0 = (1.0 + cs) / 2.0 / a0
    b1 = -(1.0 + cs) / a0
    b2 = b0
    a1 = (-2.0 * cs) / a0
    a2 = (1.0 - alpha) / a0
    return b0, b1, b2, a1, a2


def peaking_eq_coeffs(f_hz, q, gain_db, fs: float):
    """RBJ 피킹 EQ (최소위상 2차). 잔차 보정망의 유일한 스펙트럼 손잡이."""
    if torch.is_tensor(gain_db):
        A = torch.pow(10.0, gain_db / 40.0)
        w0 = TWO_PI * f_hz / fs
        sn, cs = torch.sin(w0), torch.cos(w0)
    else:
        A = 10.0 ** (np.asarray(gain_db) / 40.0)
        w0 = TWO_PI * np.asarray(f_hz) / fs
        sn, cs = np.sin(w0), np.cos(w0)
    alpha = sn / (2.0 * q)
    a0 = 1.0 + alpha / A
    b0 = (1.0 + alpha * A) / a0
    b1 = (-2.0 * cs) / a0
    b2 = (1.0 - alpha * A) / a0
    a1 = (-2.0 * cs) / a0
    a2 = (1.0 - alpha / A) / a0
    return b0, b1, b2, a1, a2


def first_difference_coeffs(alpha, like):
    """입술 방사 근사 y = x − α·x[n−1] (α≈1 이면 미분)."""
    one = torch.ones_like(like) if torch.is_tensor(like) else np.ones_like(like)
    return one, -alpha * one, 0.0 * one, 0.0 * one, 0.0 * one


def biquad_response(b0, b1, b2, a1, a2, fs: float, n_freq: int = 1025):
    """상수 계수의 주파수응답 (검증용). 반환 (f_hz, H 복소)."""
    f = np.linspace(0.0, fs / 2, n_freq)
    z = np.exp(-1j * TWO_PI * f / fs)
    num = b0 + b1 * z + b2 * z * z
    den = 1.0 + a1 * z + a2 * z * z
    return f, num / den
