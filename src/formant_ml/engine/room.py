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
#: 정규화 세기 (입력 전력 평균 대비). **사전이 0 이 아니라 δ 라는 점이 중요하다** —
#: 아래 `estimate_ir` 참조.
#:
#: 0.3 -> 0.1. 사전만 δ 로 바꾸면 λ=0.1 에서도 거친 응답이 이미 평평하다
#: (저역−고역 기울기 s101 −0.6 / s040 −0.2 dB). 0.3 은 홀드아웃 상관을 조금 깎고
#: (0.730 -> 0.720) **IR 을 거의 δ 로 만들어** 방의 시간 구조까지 지운다 — s040 에서
#: 직접음 몫이 0.954 가 되고, 그 IR 로 재적합하니 세로 얼룩이 1.532 로 합격선을
#: 크게 넘었다(λ 없이 마른 적합이 1.242 였는데도). 처음에 0.3 을 고른 근거는
#: **재적합 없이** IR 만 걸어 본 근사였고, 그 근사가 재적합 결과를 예측하지 못했다.
DEFAULT_LAMBDA = 0.1
#: 거친 EQ 를 빼는 평활 반복 횟수. 3 이면 두 파일 다 저역−고역 기울기가 1 dB 안이다.
FLATTEN_PASSES = 3


def flatten_response(h: np.ndarray, octaves: float = 0.5, size: int = 16384,
                     fs: float = 48000.0) -> np.ndarray:
    """IR 에서 **거친 이퀄라이저 몫만** 뺀다 — 잔결과 꼬리는 그대로.

    옥타브 평활한 크기응답으로 나눈다. 위상과 미세 구조는 손대지 않으므로 방의
    **시간 구조**는 남고, 역합성곱이 잡아 버린 거친 EQ 만 사라진다.

    왜 필요한가: 최소제곱 해는 방뿐 아니라 **합성의 계통적 스펙트럼 오차까지**
    흡수한다. 실측에서 저역−고역 기울기가 +17~18 dB 였는데, 그걸 순방향에 넣으면
    적합기가 그만큼 밝은 마른 소리를 내야 하고 파라미터 상한에 걸린다.

    **그리고 이 평탄화는 홀드아웃 상관을 올린다** — 즉 그 EQ 는 과적합이었다:

        s101 뒤 절반 상관   평탄 x0 0.780 → x1 **0.803** → x2 0.801 → x3 0.797
        s040 뒤 절반 상관   평탄 x0 0.758 → x1 0.838 → x2 0.864 → x3 **0.871**
        저역−고역 기울기    x0 +18.4/+17.0 dB → x3 **−0.1/+0.6 dB**
        직접음 몫          x0 0.114/0.176 → x3 0.838/0.931

    한 번으로는 0.5 옥타브 이동평균이 그 눈금의 구조를 다 못 없앤다. 세 번이면
    두 파일 다 기울기가 1 dB 안으로 들어온다.
    """
    H = np.fft.rfft(np.asarray(h, float), size)
    fr = np.fft.rfftfreq(size, 1.0 / fs)
    db = 20.0 * np.log10(np.abs(H) + 1e-12)
    # 로그 주파수 격자에서 상자 평활 — 빈마다 마스크를 만들면 O(N²) 이라 못 쓴다.
    lf = np.log2(np.maximum(fr, 20.0))
    grid = np.linspace(lf[0], lf[-1], 2048)
    g = np.interp(grid, lf, db)
    w = max(1, int(round(octaves / (grid[1] - grid[0]))))
    c = np.concatenate([[0.0], np.cumsum(g)])
    lo = np.clip(np.arange(len(g)) - w // 2, 0, len(g))
    hi = np.clip(np.arange(len(g)) + w // 2 + 1, 0, len(g))
    sm = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    corr = np.interp(lf, grid, sm)
    return np.fft.irfft(H / (10.0 ** (corr / 20.0) + 1e-12), size)[:len(h)]


def estimate_ir(dry: np.ndarray, target: np.ndarray, taps: int = DEFAULT_TAPS,
                lam: float = DEFAULT_LAMBDA, flatten_passes: int = FLATTEN_PASSES
                ) -> np.ndarray:
    """`target ≈ dry * h` 의 h 를 낸다 — 최소제곱 뒤 **거친 EQ 를 뺀다.**

        H = conj(X)·Y / (|X|² + λ·mean|X|²)   →   flatten_response ×3

    정규화 항은 **입력 전력의 평균**에 비례한다 — 절대값으로 두면 신호 크기에 따라
    세기가 달라져 파일마다 다른 필터가 나온다.

    .. note::
       사전을 δ 쪽으로 수축시키는 판(`+λ·P` 를 분자에 더하는 것)도 시험했다.
       기울기는 없어지지만 **방의 시간 구조까지 같이 없어진다** — s040 에서 직접음
       몫이 0.954 가 되고, 그 IR 로 재적합하니 세로 얼룩이 1.532 로 마른 적합
       (1.242)보다 나빠졌다. 시간 구조는 남기고 크기만 펴는 `flatten_response`
       쪽이 맞다 (docs/MEASUREMENTS.md §23.10).
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
    h = np.fft.irfft(H, size)[:taps]
    for _ in range(max(0, int(flatten_passes))):
        h = flatten_response(h)
    return h


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

def holdout_gain(dry: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None,
                 **kw) -> dict:
    """**앞 절반으로 추정하고 뒤 절반에서 시험한다.** 방을 쓸지 말지의 판단 근거.

    IR 이 항상 도움이 되는 것은 아니다. 마른 모형이 이미 잘 맞는 파일에서는 역합성곱이
    잡을 계통 성분이 별로 없고, 그때 IR 은 오히려 못 본 구간을 나빠지게 한다. 실측:

        s101  기준 0.672 → IR 0.798   (크게 좋아진다 — 방이 필요하다)
        s040  기준 0.902 → IR 0.869   (나빠진다 — 이 파일은 마른 모형으로 충분하다)

    그래서 파일마다 이 값을 보고 정한다. 문턱을 감으로 두지 말고 **못 본 절반에서
    실제로 좋아지는가**만 본다.

    반환: before(IR 없이), after(IR 걸고), improves(bool).
    """
    x = np.asarray(dry, float)
    y = np.asarray(target, float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    m = (np.ones(n, bool) if mask is None else np.asarray(mask, bool)[:n])
    half = n // 2
    h = estimate_ir(x[:half], y[:half], **kw)
    z = apply_ir(torch.as_tensor(x[None, half:]), torch.as_tensor(h))[0].numpy()

    def corr(a, b, mm):
        a, b = a[mm], b[mm]
        if a.size < 16:
            return float("nan")
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))

    mh = m[half:]
    before, after = corr(y[half:], x[half:], mh), corr(y[half:], z, mh)
    return dict(before=before, after=after, improves=bool(after > before))

