"""녹음 경로(방 + 마이크 + 코덱)를 **녹음에서 추정해서** 순방향 모형에 넣는다.

왜 필요한가 (docs/MEASUREMENTS.md §19~§20, §23)
-----------------------------------------------
목표는 방에서 잡은 소리이고 엔진은 마르다. 방은 **정적 필터로 흉내 낼 수 없는 것**을
한다 — 시간 구조를 바꾼다. 그래서 적합기는 방의 크기 응답만 성도·소스 파라미터로
흡수하고(대역 MAE 0.25 dB), 시간 구조는 못 맞춘 채로 남긴다. 남은 것이 사용자가
"세로 얼룩" 과 "지글거림" 으로 듣는 양이다. 실측:

    세로 얼룩(대역 평균 |ΔdB| 의 p95, 목표 대비)   마른 합성 1.65  ->  방 걸침 1.11
    마찰 구간 변조 지수 (목표 대비)                마른 합성 4.12  ->  방 걸침 1.99

동시에 흡수한 만큼 **물리 파라미터가 왜곡**된다. 학습 데이터로 나갈 값이라 그냥 둘 수
없다 (`fric_gain` 41.8, `aspiration` 0.031 같은 값이 그 왜곡일 수 있다).

왜 합성 IR 은 안 되는가
-----------------------
측정한 RT60 으로 만든 **지수감쇠 잡음 IR** 을 걸면 포락선 통계는 맞아지는데 파형
상관이 단조 감소한다 (0.795 -> 0.769 -> 0.653, mix 0.25/0.45). 초기 반사의 위상이
실제 방과 다르기 때문이다. 이 프로젝트는 위상까지 맞추는 것이 전제이므로 그 손해를
받을 수 없다.

그래서 **추정한다**
-------------------
마른 합성이 이미 목표와 0.795 로 상관하므로, 그것을 알려진 입력으로 놓고
`target ≈ dry * h` 를 푼다 (주파수 영역 정규화 최소제곱 = 위너 역합성곱).

**과적합을 반드시 검증할 것.** 탭이 많으면 방이 아니라 소스의 오차까지 맞춘다.
앞 절반으로 추정하고 뒤 절반으로 시험한 결과 (yang_00000101, 유성 구간 파형 상관):

    기준(마른 합성)              앞 0.924   뒤 **0.672**
    탭  256 (5 ms)  λ=0.1      앞 0.911   뒤 **0.771**
    탭 1024 (21 ms) λ=0.1      앞 0.920   뒤 **0.779**
    탭 4096 (85 ms) λ=0.1      앞 0.920   뒤 **0.780**
    탭 12000 (250 ms) λ=0.1    앞 0.922   뒤 0.764

뒤 절반에서 0.672 -> 0.78 이면 **못 본 구간에서도 좋아진다** — 방을 잡은 것이지
잡음을 외운 것이 아니다. 그리고 이득의 대부분이 **첫 20 ms** 에 있다: 긴 꼬리가
아니라 초기 반사와 채널 착색이 주범이다. 탭 12000 은 오히려 나빠지므로(과적합)
기본값은 4096 으로 둔다.
"""
from __future__ import annotations

import numpy as np
import torch

#: 기본 탭 수. 85 ms — 위 표에서 시험 상관이 포화하는 지점이고, 그 위는 과적합이다.
DEFAULT_TAPS = 4096
#: 정규화 세기 (입력 전력 평균 대비). 0.1 이 시험 상관을 최대로 했다.
DEFAULT_LAMBDA = 0.1


def estimate_ir(dry: np.ndarray, target: np.ndarray, taps: int = DEFAULT_TAPS,
                lam: float = DEFAULT_LAMBDA) -> np.ndarray:
    """`target ≈ dry * h` 의 h 를 낸다. 주파수 영역 정규화 최소제곱.

    정규화 항은 **입력 전력의 평균**에 비례한다 — 절대값으로 두면 신호 크기에 따라
    세기가 달라져 파일마다 다른 필터가 나온다.
    """
    x = np.asarray(dry, float)
    y = np.asarray(target, float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    size = 1
    while size < n + taps:
        size *= 2
    X = np.fft.rfft(x, size)
    Y = np.fft.rfft(y, size)
    px = np.abs(X) ** 2
    H = (np.conj(X) * Y) / (px + lam * px.mean() + 1e-30)
    return np.fft.irfft(H, size)[:taps]


def apply_ir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """(B,N) 에 IR 을 건다. FFT 합성곱 — 시간영역이면 4096 탭 × 68k 샘플이 너무 느리다.

    미분 가능하다 (적합의 순방향에 들어간다). 길이는 입력과 같게 자른다.
    """
    n = x.shape[-1]
    size = 1
    while size < n + h.shape[-1]:
        size *= 2
    X = torch.fft.rfft(x, n=size, dim=-1)
    H = torch.fft.rfft(h.to(x.dtype), n=size)
    return torch.fft.irfft(X * H, n=size, dim=-1)[..., :n]


def direct_gain(h: np.ndarray, ms: float = 2.0, fs: float = 48000.0) -> float:
    """직접음 몫의 크기 — IR 이 이득으로 흡수한 양을 보고할 때."""
    k = max(1, int(ms * 1e-3 * fs))
    return float(np.sqrt((np.asarray(h, float)[:k] ** 2).sum()))
