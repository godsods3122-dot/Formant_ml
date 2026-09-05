"""손실 함수.

크기 스펙트럼만 맞추면 위상은 아무렇게나 되어도 되므로(= 흔한 '메탈릭/버즈' 잡음의
원인), 여기서는 위상의 *미분량* — 순시주파수(IF)와 군지연(GD) — 도 함께 맞춘다.
위상은 순환량이므로 언랩 대신 anti-wrapping 함수를 쓴다.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..data.features import log_band_energy, log_mel, stft


def _anti_wrap(x: torch.Tensor) -> torch.Tensor:
    """위상 차이를 (-pi, pi] 로 접어 넣는다."""
    return x - 2 * math.pi * torch.round(x / (2 * math.pi))


def spectral_convergence(x_mag, y_mag, eps=1e-7):
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro").clamp_min(eps)


def multi_resolution_stft_loss(x, y, ffts=(2048, 1024, 512, 256),
                               hops=(480, 240, 120, 60), eps=1e-7):
    """크기 스펙트럼: log-L1 + spectral convergence."""
    total = x.new_zeros(())
    for n_fft, hop in zip(ffts, hops):
        X = stft(x, n_fft, hop).abs().clamp_min(eps)
        Y = stft(y, n_fft, hop).abs().clamp_min(eps)
        total = total + F.l1_loss(torch.log(X), torch.log(Y)) + spectral_convergence(X, Y)
    return total / len(ffts)


def phase_derivative_loss(x, y, n_fft=1024, hop=240, weight_by_mag=True, eps=1e-7):
    """순시주파수(시간축 위상차) + 군지연(주파수축 위상차) 손실.

    크기가 작은 빈의 위상은 지각적으로 무의미하므로 크기로 가중한다.
    """
    X, Y = stft(x, n_fft, hop), stft(y, n_fft, hop)
    px, py = torch.angle(X), torch.angle(Y)
    w = (Y.abs() / Y.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(eps)) \
        if weight_by_mag else torch.ones_like(py)

    ifr = _anti_wrap((px[..., 1:] - px[..., :-1]) - (py[..., 1:] - py[..., :-1]))
    gd = _anti_wrap((px[..., 1:, :] - px[..., :-1, :])
                    - (py[..., 1:, :] - py[..., :-1, :]))
    return (ifr.abs() * w[..., 1:]).mean() + (gd.abs() * w[..., 1:, :]).mean()


# --------------------------------------------------------------------- 고역
def band_energy_loss(x, y, sample_rate: int = 24000, n_bands: int = 24,
                     fmin: float = 50.0, hf_hz: float = 4000.0,
                     hf_boost: float = 2.0) -> torch.Tensor:
    """로그 주파수축 대역별 로그에너지 L1 — **에너지 불변** 손실.

    핵심은 '고역에 가중치를 더 준다'가 아니라 **대역의 절대 에너지와 무관하게
    dB 오차를 잰다**는 것이다. `multi_resolution_stft_loss` 의 spectral
    convergence 항은 프로베니우스 노름이라 에너지가 큰 저역이 지배하고, 8 kHz
    대역(저역보다 40 dB 조용하다)에서 6 dB 를 틀려도 거의 움직이지 않는다.
    여기서는 조용한 대역의 6 dB 오차와 큰 대역의 6 dB 오차가 같은 값을 낸다.

    로그 간격이라 대역 수 자체는 저역에 더 많이 배정된다(옥타브당 균등).
    거기에 더해 고역을 명시적으로 강조하고 싶으면 `hf_boost` 를 쓴다.
    """
    bx = log_band_energy(x, sample_rate, n_bands=n_bands, fmin=fmin)
    by = log_band_energy(y, sample_rate, n_bands=n_bands, fmin=fmin)
    err = (bx - by).abs()
    edges = torch.logspace(math.log10(fmin), math.log10(sample_rate / 2 * 0.99),
                           n_bands + 1, device=err.device, dtype=err.dtype)
    ctr = (edges[:-1] * edges[1:]).sqrt()
    w = torch.where(ctr >= hf_hz, torch.full_like(ctr, hf_boost),
                    torch.ones_like(ctr))
    return (err * w).mean() / w.mean()


# ------------------------------------------------------------------- 주기성
def periodicity(x: torch.Tensor, sample_rate: int = 24000, hop: int = 240,
                frame_len: int = 1024, f0_min: float = 60.0,
                f0_max: float = 500.0, eps: float = 1e-8) -> torch.Tensor:
    """프레임별 정규화 자기상관 최대치 (B, T), 0..1. 미분가능.

    유성 구간에서 높고 마찰음 구간에서 낮아야 한다. 두 방향의 실패를 한 번에 잡는
    지표다: 값이 너무 낮으면 HNR 붕괴(유성음이 잡음으로 채워짐), 너무 높으면
    마찰음이 주기적 텍스처가 된다.
    """
    from ..data.features import frame_signal
    if x.dim() == 1:
        x = x[None]
    w = frame_signal(x, frame_len, hop)
    w = w - w.mean(-1, keepdim=True)
    nfft = 1
    while nfft < 2 * frame_len:
        nfft <<= 1
    S = torch.fft.rfft(w, nfft)
    ac = torch.fft.irfft(S.real ** 2 + S.imag ** 2, nfft)[..., :frame_len]
    ac = ac / ac[..., :1].clamp_min(eps)
    lo = max(int(sample_rate / f0_max), 2)
    hi = min(int(sample_rate / f0_min) + 1, frame_len - 1)
    return ac[..., lo:hi].amax(-1).clamp(0.0, 1.0)


def periodicity_match_loss(pred, target, sample_rate: int = 24000, hop: int = 240,
                           n_frames: int | None = None) -> torch.Tensor:
    """예측/목표의 프레임별 주기성을 맞춘다.

    * 유성 구간: 목표보다 주기성이 낮으면 = 노이즈 경로가 에너지를 훔친 것 -> 벌점.
      (PLAN §7-5 의 HNR 붕괴에 대한 직접적인 대응이다. 노이즈 게인에 거는
      일반 페널티와 달리, '얼마나' 노이즈여야 하는지를 데이터가 알려준다.)
    * 무성 구간: 목표보다 주기성이 높으면 = 마찰음이 주기적 텍스처가 된 것 -> 벌점.
      치찰음을 패턴으로 외워 반복하는 실패가 정확히 여기에 걸린다.
    """
    p = periodicity(pred, sample_rate, hop)
    t = periodicity(target, sample_rate, hop)
    n = n_frames or min(p.shape[1], t.shape[1])
    return F.l1_loss(p[:, :n], t[:, :n])


def excess_periodicity(pred, voicing, sample_rate: int = 24000, hop: int = 240,
                       margin: float = 0.15) -> torch.Tensor:
    """무성 구간에서 나타나는 초과 주기성만 벌한다 (목표 신호 없이도 쓸 수 있다)."""
    p = periodicity(pred, sample_rate, hop)
    n = min(p.shape[1], voicing.shape[1])
    unvoiced = (1.0 - voicing[:, :n]).clamp(0.0, 1.0)
    return (F.relu(p[:, :n] - margin) * unvoiced).mean()


# --------------------------------------------------------------- 하모닉 상대위상
def harmonic_phase(x, f0, sample_rate: int = 24000, hop: int = 240,
                   n_fft: int = 1024, n_harmonics: int = 10):
    """하모닉 위치의 복소 스펙트럼 값 (B, T, K). f0: (B, T) Hz."""
    X = stft(x, n_fft, hop)                              # (B, F, T)
    X = X.transpose(1, 2)                                # (B, T, F)
    t = min(X.shape[1], f0.shape[1])
    X, f0 = X[:, :t], f0[:, :t]
    df = sample_rate / n_fft
    k = torch.arange(1, n_harmonics + 1, device=x.device, dtype=f0.dtype)
    bins = (f0.unsqueeze(-1) * k / df).round().long().clamp(0, X.shape[-1] - 1)
    return torch.gather(X, -1, bins)


def relative_phase_loss(pred, target, f0, sample_rate: int = 24000, hop: int = 240,
                        n_harmonics: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """상대위상 RPS_k = ∠X_k − k·∠X_1 의 원형 거리. **위상차 파라미터의 학습 신호.**

    IF/GD 손실은 '위상이 시간/주파수축으로 매끄러운가'를 본다. 그것만으로는
    하모닉끼리의 *정렬 관계*(성문 폐쇄의 날카로움, 화자 개성)가 잘 잡히지 않는다.
    RPS 는 분석창 위치에 무관한 양이라 그 관계만 직접 비교할 수 있다.
    """
    from ..dsp.phase import relative_phase_shift
    Hp = harmonic_phase(pred, f0, sample_rate, hop, n_harmonics=n_harmonics)
    Ht = harmonic_phase(target, f0, sample_rate, hop, n_harmonics=n_harmonics)
    rp = relative_phase_shift(Hp, n_harmonics)
    rt = relative_phase_shift(Ht, n_harmonics)
    w = Ht.abs()
    w = w / w.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    # 1 - cos(Δ) 는 위상의 원형 거리(랩 안전, 어디서나 미분가능)
    return ((1.0 - torch.cos(rp - rt)) * w).mean()


def residual_energy_ratio(corrected, physics, eps: float = 1e-9) -> torch.Tensor:
    """잔차 에너지 / 물리모델 에너지. **물리모델이 얼마나 설명했는가**의 지표.

    이 값이 커지면 신경망이 물리모델을 대체하기 시작한 것이다(PLAN §4 의 붕괴).
    학습 중에 이 항에 페널티를 걸어 −20 dB 아래로 눌러 둔다.
    """
    n = min(corrected.shape[-1], physics.shape[-1])
    d = corrected[..., :n] - physics[..., :n]
    return d.pow(2).mean() / physics[..., :n].pow(2).mean().clamp_min(eps)


def residual_energy_db(corrected, physics) -> float:
    return float(10.0 * torch.log10(
        residual_energy_ratio(corrected, physics).clamp_min(1e-12)).detach())


def mel_loss(x, y, **kw):
    return F.l1_loss(log_mel(x, **kw), log_mel(y, **kw))


def smoothness(param: torch.Tensor, order: int = 1) -> torch.Tensor:
    """제어 파라미터의 시간 미분 페널티(떨림/삐걱임 억제)."""
    d = param
    for _ in range(order):
        d = d[:, 1:] - d[:, :-1]
    return d.pow(2).mean()


def area_smoothness(area: torch.Tensor) -> torch.Tensor:
    """성도 단면적의 공간 급변 페널티(해부학적으로 매끄러운 관)."""
    d = area[..., 1:] - area[..., :-1]
    return d.pow(2).mean()


def formant_ordering_penalty(freq: torch.Tensor, margin: float = 100.0):
    """F1 < F2 < ... 위반량. (인코더에서 구조적으로 보장하면 0 이 된다.)"""
    gap = freq[..., 1:] - freq[..., :-1]
    return F.relu(margin - gap).mean()


# ---------------------------------------------- 포락선/무게중심 (측정 전용)
#
# **주의: 아래 둘은 손실이 아니다.** 손실로 넣으려다 반증했다.
#
# 가설이었던 것: "기존 손실(STFT/mel/band)은 스펙트럼만 보므로 시간축의 모양 —
# 아치 포락선(§5.6), 고역 페이드인(§5.11), 무게중심 아치(§5.9) — 을 벌하지
# 못한다. 그래서 학습을 켜면 우리가 만든 물리가 우회당한다."
#
# 실측(고역이 중역보다 **40 dB 조용한** 상황에서 고역 온셋만 60 ms 틀리게 두고
# d(loss)/d(onset) 을 잰 것. 정답 0.36 s):
#
#     hf_on    band   d/d      stft   d/d      mel    d/d      env    d/d
#     0.30   0.1997 -2.883   0.4045 -5.709   0.1474 -2.067   0.0051 -0.086
#     0.33   0.1025 -3.140   0.2072 -5.905   0.0749 -2.164   0.0026 -0.083
#     0.36   0.0000  0.000   0.0000  0.000   0.0000  0.000   0.0000  0.000
#     0.39   0.0998 +2.987   0.2041 +6.101   0.0725 +2.104   0.0025 +0.079
#
# 무게중심 아치의 정점 위치를 틀린 경우도 같다(정답 0.50):
#
#     peak     band   d/d       mel   d/d   centroid   d/d
#     0.35   0.5404 -3.022   0.5483 -3.158   0.0386 -0.262
#     0.65   0.5394 +3.204   0.5513 +3.248   0.0388 +0.263
#
# 즉 기존 손실은 이미 **부호가 맞는 깨끗한 기울기**를 주고, 세기는 오히려
# 10~35배 크다. `band_energy_loss` 가 로그대역 등가중이라 40 dB 조용한 고역도
# 동등하게 본다 — 그게 원래 그 손실을 만든 이유였다(위 docstring).
#
# 그러므로 물리가 학습에서 우회당하는 원인은 "손실이 못 본다"가 **아니다**.
# 진짜 원인은 두 가지다: (1) 제어가 인코더에 노출되지 않은 것
# (`aspiration_bands`, `teeth_*` — 이번에 고쳤다), (2) 성도 역문제의 부정형성
# (`control_supervision_loss` 와 `tract_anchor_loss` 로 다룬다).
#
# 아래 둘은 **측정 도구로만** 남긴다(§8). 손실 합에 넣지 마라.
def band_envelope(x, sample_rate: int = 24000, n_fft: int = 1024, hop: int = 240,
                  n_bands: int = 24, fmin: float = 50.0) -> torch.Tensor:
    """(B, T, K) 대역별 **선형** 시간 포락선."""
    return torch.exp(log_band_energy(x, sample_rate, n_fft, hop, n_bands, fmin))


def envelope_moment_loss(x, y, sample_rate: int = 24000, n_fft: int = 1024,
                         hop: int = 240, n_bands: int = 24, fmin: float = 50.0,
                         eps: float = 1e-8) -> torch.Tensor:
    """대역별 포락선의 **시간 1·2차 모멘트** 손실.

    왜 필요한가
    -----------
    기존 손실(STFT/mel/band)은 전부 프레임을 독립적으로 본다. 프레임을 섞어도
    값이 같다. 그런데 이번 세션에 고친 것은 전부 *시간축의 모양*이었다:

    - 마찰음이 직사각형이 아니라 아치여야 한다 (min-jerk 조음, §5.6)
    - 고역이 중역보다 **늦게** 올라와야 한다 (Curle 의 U^6 다이폴, §5.11)

    둘 다 대역별 포락선의 무게중심(mu)과 폭(sd)이다. 대역마다 mu 를 맞추면
    대역 간 온셋 순서가 강제되고, sd 를 맞추면 직사각형이 벌을 받는다.
    문턱값(10 % 교차점) 대신 모멘트를 쓰는 이유는 **미분 가능**하기 때문이다 —
    측정 도구(§8)는 교차점을 쓰지만 손실은 그럴 수 없다.

    단위는 발화 길이로 나눈 무차원 값이라 길이/샘플레이트에 무관하다.
    """
    ex = band_envelope(x, sample_rate, n_fft, hop, n_bands, fmin)
    ey = band_envelope(y, sample_rate, n_fft, hop, n_bands, fmin)
    T = ex.shape[1]
    tau = torch.linspace(0.0, 1.0, T, device=ex.device, dtype=ex.dtype)[None, :, None]

    def moments(e):
        p = e / e.sum(dim=1, keepdim=True).clamp_min(eps)
        mu = (p * tau).sum(dim=1)                                   # (B, K)
        var = (p * (tau - mu[:, None, :]) ** 2).sum(dim=1)
        return mu, var.clamp_min(eps).sqrt()

    mx, sx = moments(ex)
    my, sy = moments(ey)
    return (mx - my).abs().mean() + (sx - sy).abs().mean()


def centroid_trajectory_loss(x, y, sample_rate: int = 24000, n_fft: int = 1024,
                             hop: int = 240, fmin: float = 200.0,
                             eps: float = 1e-8) -> torch.Tensor:
    """프레임별 스펙트럼 무게중심(log-Hz) 궤적 손실.

    무게중심이 마찰음 중앙에서 최고가 되고 양옆으로 완만히 떨어진다는 것이
    §5.9/§5.11 의 핵심 주장인데, 어떤 기존 손실도 이것을 직접 벌하지 않는다.
    조용한 프레임의 무게중심은 의미가 없으므로 **목표 신호의 에너지로 가중**한다.
    """
    X = stft(x, n_fft, hop).abs()
    Y = stft(y, n_fft, hop).abs()
    f = torch.linspace(0, sample_rate / 2, X.shape[-2], device=X.device, dtype=X.dtype)
    lf = torch.log(f.clamp_min(fmin))[None, :, None]
    cx = (X * lf).sum(-2) / X.sum(-2).clamp_min(eps)                # (B, T)
    cy = (Y * lf).sum(-2) / Y.sum(-2).clamp_min(eps)
    w = Y.sum(-2)
    w = w / w.sum(-1, keepdim=True).clamp_min(eps)
    return ((cx - cy).abs() * w).sum(-1).mean()


# ------------------------------------------------ 제어 지도학습 (물리 스크립트 = 교사)
#: 각 제어의 정규화 척도. "1 척도만큼 틀리면 손실 1" 이 되도록 잡는다.
CONTROL_SCALE = {
    "f0": 100.0, "harmonic_amp": 1.0, "rd": 1.0,
    "formant_freq": 1000.0, "formant_bw": 500.0, "formant_gain": 1.0,
    "noise_bands": 1.0, "aspiration_bands": 1.0, "noise_entry": 5.0,
    "noise_am": 1.0, "noise_rough": 1.0, "noise_back_leak": 1.0,
    "noise_bw_scale": 3.0, "tilt": 10.0,
}
SIB_SCALE = {
    "pole_f": 5000.0, "pole_bw": 1000.0, "zero_f": 5000.0, "zero_bw": 1000.0,
    "tilt": 6.0, "mix": 1.0, "slope_lo": 20.0, "slope_hi": 20.0,
    "teeth_f": 5000.0, "teeth_bw": 1000.0, "teeth_gain": 1.0,
}


def control_supervision_loss(pred, target, eps: float = 1e-8) -> torch.Tensor:
    """인코더 제어를 **물리 스크립트가 만든 제어**에 직접 맞춘다.

    성도 추정이 어려운 이유는 역문제가 부정형(ill-posed)이기 때문이다 — 같은
    스펙트럼을 만드는 성도 형상이 여러 개다. 오디오 손실만으로는 그중 어느
    것으로 갈지 정해지지 않는다.

    그런데 우리에게는 **정답 제어를 아는 데이터가 무한히 있다**: `score.py` 가
    물리에서 유도한 제어 궤적으로 렌더한 음성이다. (오디오, 제어) 쌍을 그대로
    쓰면 인코더는 "이 소리를 낸 조음은 이것이었다"를 지도학습으로 배운다.
    실제 녹음에는 정답이 없지만, 스크립트 렌더로 먼저 초기화해 두면 실제
    녹음에서의 오디오 손실이 엉뚱한 해로 가지 않는다.

    None 인 필드는 건너뛴다(교사가 안 만든 제어는 자유롭게 둔다).
    """
    total, n = None, 0
    for k, sc in CONTROL_SCALE.items():
        a, b = getattr(pred, k, None), getattr(target, k, None)
        if a is None or b is None:
            continue
        t = min(a.shape[1], b.shape[1])
        d = (a[:, :t] - b[:, :t]).abs().mean() / sc
        total, n = d if total is None else total + d, n + 1
    sa, sb = getattr(pred, "sib", None), getattr(target, "sib", None)
    if sa is not None and sb is not None:
        for k, sc in SIB_SCALE.items():
            a, b = getattr(sa, k, None), getattr(sb, k, None)
            if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b):
                continue
            t = min(a.shape[1], b.shape[1])
            d = (a[:, :t] - b[:, :t]).abs().mean() / sc
            total, n = d if total is None else total + d, n + 1
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total / max(n, 1)


def tract_anchor_loss(controls, anchor_hz, start: int = 3,
                      tol: float = 0.10) -> torch.Tensor:
    """상위 포먼트를 화자 프로파일의 **실측값**에 묶는다 (데드밴드 있는 힌지).

    F1~F3 은 모음마다 크게 움직이므로 묶으면 안 된다. 그러나 F4 이상은 성도
    *길이*가 정하는 값이라 같은 화자 안에서 거의 상수다(Fant 의 균일관 근사).
    성도 추정 모델이 따로 없어도, 이 앵커 하나로 "성도를 고정한다"는 요구를
    학습 안에서 표현할 수 있다.

    ``tol`` 안에서는 벌점이 0 이다 — 실측 자체가 ±10 % 는 흔들리기 때문에
    데드밴드 없이 묶으면 모델이 실측 잡음을 따라가게 된다.
    """
    f = controls.formant_freq
    if f is None or anchor_hz is None:
        return f.new_zeros(()) if f is not None else torch.zeros(())
    a = torch.as_tensor(anchor_hz, device=f.device, dtype=f.dtype).reshape(-1)
    k = min(f.shape[-1], a.shape[0])
    if start >= k:
        return f.new_zeros(())
    dev = (f[..., start:k] / a[start:k].clamp_min(1.0)).clamp_min(1e-3).log().abs()
    return F.relu(dev - math.log1p(tol)).mean()


class VoiceLoss(torch.nn.Module):
    """전체 손실 = 크기 + 고역 + 위상 + 주기성 + 정규화.

    가중치의 의도
    -------------
    stft/mel   : 기본 스펙트럼 일치.
    band       : 로그 대역별 동등 가중 -> 고역이 학습된다.
    phase      : IF/GD (위상 미분) — 시간축 연속성.
    rps        : 하모닉 상대위상 — 위상차 파라미터가 실제로 학습되게 하는 항.
    period     : 유성/무성 주기성 일치 — HNR 붕괴와 '주기적인 치찰음'을 동시에 막는다.
    noise      : 유성 구간 노이즈 게인 억제(보조). period 가 있으면 작게 둬도 된다.
    ctrl       : 물리 스크립트가 만든 제어에 대한 지도학습(교사). 성도 역문제의
                 부정형성을 없애는 항이라 **사전학습에서 가장 크게** 준다.
    anchor     : 상위 포먼트를 화자 실측에 묶는 힌지 — '성도 고정'의 학습 표현.
    """

    def __init__(self, w_stft=1.0, w_band=1.0, w_phase=0.2, w_rps=0.3, w_mel=1.0,
                 w_period=1.0, w_smooth=1e-3, w_area=1e-3, w_noise=5e-3,
                 w_residual=0.3, w_ctrl=0.0,
                 w_anchor=0.0, anchor_hz=None, anchor_start: int = 3,
                 sample_rate: int = 24000, hop: int = 240):
        super().__init__()
        self.w = dict(stft=w_stft, band=w_band, phase=w_phase, rps=w_rps, mel=w_mel,
                      period=w_period, smooth=w_smooth, area=w_area, noise=w_noise,
                      residual=w_residual, ctrl=w_ctrl, anchor=w_anchor)
        self.sample_rate = sample_rate
        self.hop = hop
        self.anchor_hz = anchor_hz
        self.anchor_start = anchor_start

    def forward(self, pred_audio, target_audio, controls=None,
                voicing: torch.Tensor | None = None,
                f0: torch.Tensor | None = None,
                physics_audio: torch.Tensor | None = None,
                teacher_controls=None) -> dict:
        sr, hop = self.sample_rate, self.hop
        out = {
            "stft": multi_resolution_stft_loss(pred_audio, target_audio),
            "band": band_energy_loss(pred_audio, target_audio, sr),
            "phase": phase_derivative_loss(pred_audio, target_audio),
            "mel": mel_loss(pred_audio, target_audio),
            "period": periodicity_match_loss(pred_audio, target_audio, sr, hop),
        }
        out["rps"] = (relative_phase_loss(pred_audio, target_audio, f0, sr, hop)
                      if f0 is not None else pred_audio.new_zeros(()))

        out["ctrl"] = (control_supervision_loss(controls, teacher_controls)
                       if (controls is not None and teacher_controls is not None)
                       else pred_audio.new_zeros(()))
        out["anchor"] = (tract_anchor_loss(controls, self.anchor_hz, self.anchor_start)
                         if (controls is not None and self.anchor_hz is not None)
                         else pred_audio.new_zeros(()))

        # 유성 구간에서 노이즈 경로가 에너지를 가져가는 것을 억제한다(보조 항).
        noise_pen = pred_audio.new_zeros(())
        if controls is not None:
            v = voicing[..., None] if voicing is not None else 1.0
            noise_pen = (controls.noise_bands * v).mean()
        out["noise"] = noise_pen

        reg = pred_audio.new_zeros(())
        if controls is not None:
            reg = reg + smoothness(controls.formant_freq / 1000.0)
            reg = reg + smoothness(controls.noise_bands)
            if controls.tilt is not None:
                reg = reg + smoothness(controls.tilt / 10.0)
            if controls.aspiration_bands is not None:
                reg = reg + smoothness(controls.aspiration_bands)
            if getattr(controls, "sib", None) is not None:
                reg = reg + smoothness(controls.sib.pole_f / 5000.0)
                if controls.sib.teeth_f is not None:
                    reg = reg + smoothness(controls.sib.teeth_f / 5000.0)
            if controls.area is not None:
                reg = reg + area_smoothness(controls.area)
        out["reg"] = reg

        # 잔차망을 쓸 때: 물리모델을 대체하지 못하게 누른다.
        out["residual"] = (residual_energy_ratio(pred_audio, physics_audio)
                           if physics_audio is not None else pred_audio.new_zeros(()))

        out["total"] = (self.w["stft"] * out["stft"] + self.w["band"] * out["band"]
                        + self.w["phase"] * out["phase"] + self.w["rps"] * out["rps"]
                        + self.w["mel"] * out["mel"] + self.w["period"] * out["period"]
                        + self.w["smooth"] * out["reg"] + self.w["noise"] * out["noise"]
                        + self.w["residual"] * out["residual"]
                        + self.w["ctrl"] * out["ctrl"]
                        + self.w["anchor"] * out["anchor"])
        return out
