"""실제 음성에서 치찰음 지문(극-영점 파라미터)을 뽑아낸다.

절차
----
1. 마찰음 프레임을 고른다: 주기성이 낮고(무성), 에너지가 충분하고,
   스펙트럼 무게중심이 높은 프레임.
2. 그 프레임들의 평균 로그 스펙트럼을 구한다.
3. `sibilant_response(pole, zero, tilt)` 의 로그 크기를 **경사하강으로** 맞춘다.
   (Vocal Tract Area Estimation by Gradient Descent 와 같은 발상 — 우리 합성
   모듈이 미분가능하니 역추정도 같은 코드로 한다.)

결과는 6개 숫자다. 이것이 이 화자의 /s/ 를 정의하고, 스크립트에서 그대로
바꿔 쓸 수 있으며, 학습 시 인코더의 초기값/사전분포로도 쓸 수 있다.
"""
from __future__ import annotations

import math

import torch

from ..data.features import stft
from ..dsp.sibilant import SibilantParams, sibilant_response, spectral_moments


def _sigmoid_range(u, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(u)


def _inv_sigmoid_range(v, lo, hi):
    p = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def find_sibilant_frames(x: torch.Tensor, sample_rate: int = 24000, hop: int = 240,
                         n_fft: int = 1024, centroid_hz: float = 3000.0,
                         max_periodicity: float = 0.35,
                         energy_percentile: float = 0.35) -> torch.Tensor:
    """마찰음으로 보이는 프레임의 불리언 마스크 (T,).

    에너지 문턱은 **후보(고역·비주기) 프레임들 안에서** 잡는다. 녹음 전체의
    백분위로 자르면, 마찰음은 원래 모음보다 조용하기 때문에 모음이 섞인 녹음에서
    후보가 통째로 사라진다.
    """
    from ..models.losses import periodicity
    x = x.detach()
    if x.dim() == 1:
        x = x[None]
    X = stft(x, n_fft, hop)[0].abs().transpose(0, 1)          # (T, F)
    f = torch.linspace(0, sample_rate / 2, X.shape[-1])
    energy = X.sum(-1)
    ctr = (X * f).sum(-1) / energy.clamp_min(1e-9)
    per = periodicity(x, sample_rate, hop)[0][:X.shape[0]]
    shape = (ctr > centroid_hz) & (per < max_periodicity)
    if int(shape.sum()) == 0:
        return shape
    thr = torch.quantile(energy[shape], energy_percentile)
    # 완전한 무음까지 잡지 않도록 하한도 둔다. 최댓값 기준으로 잡으면 안 된다 —
    # 전이부의 클릭 한 프레임이 중앙값의 2000 배씩 나오면 문턱이 그리로 끌려가
    # 마찰음 프레임이 통째로 사라진다. 중앙값 기준이 안전하다.
    floor = float(torch.quantile(energy, 0.5)) * 0.1
    return shape & (energy > max(float(thr), floor))


def mean_log_spectrum(x, mask, hop: int = 240, n_fft: int = 1024, eps: float = 1e-9):
    """선택된 프레임의 평균 로그 크기 스펙트럼 (F,) dB."""
    x = x.detach()
    if x.dim() == 1:
        x = x[None]
    X = stft(x, n_fft, hop)[0].abs().transpose(0, 1)
    m = mask[:X.shape[0]]
    if int(m.sum()) == 0:
        raise ValueError("마찰음 프레임을 찾지 못했습니다 (임계값을 낮춰 보세요)")
    return 20.0 * torch.log10(X[m].clamp_min(eps)).mean(0)


def measure(x, sample_rate: int = 24000, hop: int = 240, mask=None,
            smooth_bins: int = 21) -> dict:
    """모멘트 기반 기술자 (센트로이드/폭/왜도/첨도/피크). 비교·검증용.

    피크는 평활한 포락선에서 읽는다. 난류의 한 실현(realization)에는 잔물결이
    많아서, 생스펙트럼의 argmax 는 평탄한 구간에서 아무 빈이나 고른다.
    """
    mask = find_sibilant_frames(x, sample_rate, hop) if mask is None else mask
    db = mean_log_spectrum(x, mask, hop)
    if smooth_bins > 1:
        k = torch.ones(1, 1, smooth_bins) / smooth_bins
        db = torch.nn.functional.conv1d(
            torch.nn.functional.pad(db.view(1, 1, -1),
                                    (smooth_bins // 2, smooth_bins // 2),
                                    mode="replicate"), k).view(-1)
    mag = 10 ** (db / 20.0)
    f = torch.linspace(0, sample_rate / 2, len(mag))
    band = f > 1000.0
    m1, sd, sk, ku = spectral_moments(mag[band], f[band])
    return {"centroid_hz": round(float(m1), 1), "spread_hz": round(float(sd), 1),
            "skew": round(float(sk), 3), "kurtosis": round(float(ku), 3),
            "peak_hz": round(float(f[band][mag[band].argmax()]), 1),
            "n_frames": int(mask.sum())}


def fit_sibilant(x, sample_rate: int = 24000, hop: int = 240, mask=None,
                 steps: int = 400, lr: float = 0.08, fmin: float = 1000.0,
                 fmax: float | None = None, smooth_bins: int = 21,
                 verbose: bool = False) -> dict:
    """치찰음 극-영점 파라미터를 경사하강으로 적합. 반환 dict (스칼라 float).

    적합 대상은 *모양*이지 절대 게인이 아니다(게인은 별도 자유변수로 흡수).
    극-영점 한 쌍은 성도 포먼트의 잔물결까지 표현할 수 없으므로 목표 스펙트럼을
    먼저 평활한다. 그래야 잔물결에 끌려 극이 엉뚱한 데로 가지 않는다.
    """
    fmax = fmax or sample_rate / 2 * 0.96
    mask = find_sibilant_frames(x, sample_rate, hop) if mask is None else mask
    target = mean_log_spectrum(x, mask, hop)                  # (F,) dB
    if smooth_bins > 1:
        k = torch.ones(1, 1, smooth_bins) / smooth_bins
        target = torch.nn.functional.conv1d(
            torch.nn.functional.pad(target.view(1, 1, -1),
                                    (smooth_bins // 2, smooth_bins // 2),
                                    mode="replicate"), k).view(-1)
    n_freq = len(target)
    f = torch.linspace(0, sample_rate / 2, n_freq)
    w = ((f >= fmin) & (f <= fmax)).float()

    # 합성 모형과 **같은** 자유도를 준다. 스커트 기울기를 빼고 적합하면 극이
    # 그 역할을 대신하려고 비정상적으로 좁아진다(측정: BW 336 Hz 로 수렴).
    ranges = {"pole_f": (1500.0, 11000.0), "pole_bw": (400.0, 4000.0),
              "zero_bw": (150.0, 4000.0), "tilt": (-8.0, 8.0),
              "slope_lo": (0.0, 45.0), "slope_hi": (-20.0, 0.0)}
    init = {"pole_f": 6000.0, "pole_bw": 2000.0, "zero_bw": 2000.0, "tilt": 0.0,
            "slope_lo": 14.0, "slope_hi": -4.0}
    u = {k: torch.tensor(_inv_sigmoid_range(v, *ranges[k]), requires_grad=True)
         for k, v in init.items()}
    u["zero_ratio"] = torch.tensor(0.0, requires_grad=True)   # zero_f = pole_f * ratio
    gain = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam(list(u.values()) + [gain], lr=lr)

    def build():
        pf = _sigmoid_range(u["pole_f"], *ranges["pole_f"]).view(1, 1, 1)
        pb = _sigmoid_range(u["pole_bw"], *ranges["pole_bw"]).view(1, 1, 1)
        zf = pf * (0.12 + 0.76 * torch.sigmoid(u["zero_ratio"]))
        zb = _sigmoid_range(u["zero_bw"], *ranges["zero_bw"]).view(1, 1, 1)
        ti = _sigmoid_range(u["tilt"], *ranges["tilt"]).view(1, 1, 1)
        lo = _sigmoid_range(u["slope_lo"], *ranges["slope_lo"]).view(1, 1, 1)
        hi = _sigmoid_range(u["slope_hi"], *ranges["slope_hi"]).view(1, 1, 1)
        return SibilantParams(pole_f=pf, pole_bw=pb, zero_f=zf, zero_bw=zb,
                              tilt=ti, mix=torch.ones(1, 1, 1),
                              slope_lo=lo, slope_hi=hi)

    last = float("nan")
    for i in range(steps):
        p = build()
        H = sibilant_response(p, sample_rate, n_freq)
        pred = 20.0 * torch.log10(H.abs().clamp_min(1e-9))[0, 0] + gain
        loss = ((pred - target).pow(2) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if verbose and i % 100 == 0:
            print(f"  step {i:4d} rmse={last ** 0.5:.2f} dB")

    with torch.no_grad():
        p = build()
        out = {"pole_f": round(float(p.pole_f), 1), "pole_bw": round(float(p.pole_bw), 1),
               "zero_f": round(float(p.zero_f), 1), "zero_bw": round(float(p.zero_bw), 1),
               "tilt": round(float(p.tilt), 3),
               "slope_lo": round(float(p.slope_lo), 2),
               "slope_hi": round(float(p.slope_hi), 2),
               "rmse_db": round(last ** 0.5, 3), "n_frames": int(mask.sum())}
    return out


def params_from_dict(d: dict, shape, device=None, dtype=torch.float32,
                     mix: float = 1.0, roughness: float = 0.3) -> SibilantParams:
    """`fit_sibilant` 결과 -> 합성용 SibilantParams."""
    return SibilantParams.constant(shape, d["pole_f"], d["pole_bw"], d["zero_f"],
                                   d["zero_bw"], d["tilt"], mix, roughness,
                                   device=device, dtype=dtype)
