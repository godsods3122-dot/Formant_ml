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


# --------------------------------------------------------------- 조음 제약
def articulatory_rate_loss(freq: torch.Tensor, frame_rate: float,
                           limits: torch.Tensor | None = None) -> torch.Tensor:
    """생리적 최대 조음 속도를 넘는 만큼만 벌점 (hinge).

    단순 L2 평활은 *모든* 움직임을 줄여 전이를 뭉갠다. 여기서는 상한 아래의
    빠른 전이는 전혀 벌하지 않고, 혀가 낼 수 없는 속도만 자른다.
    상한은 문헌의 F2 전이(10~25 Hz/ms)에서 잡았다 — `gestures_dynamics` 참고.
    """
    from ..dsp.gestures_dynamics import max_rate_limits
    rate = (freq[:, 1:] - freq[:, :-1]).abs() * frame_rate          # (B, T-1, K)
    lim = limits if limits is not None else max_rate_limits(
        freq.shape[-1], freq.device, freq.dtype)
    return (F.relu(rate - lim) / lim).mean()


def formant_subspace_loss(freq: torch.Tensor, n_modes: int = 3,
                          eps: float = 1e-9) -> torch.Tensor:
    """포먼트가 저차원 부분공간 밖으로 움직이는 성분에 벌점.

    포먼트는 독립적인 K 개의 자유도가 아니다. 성도 형상이 (턱, 혀몸통, 혀끝,
    입술) 정도의 소수 좌표로 결정되고, 그 결과 포먼트들은 함께 움직인다.
    Story & Titze 의 면적함수 실증 직교모드에서는 **2개 모드가 분산의 97% 이상**을
    설명한다. 여기서는 시간에 따른 포먼트 편차 행렬의 상위 n_modes 밖 특이값
    에너지 비율을 벌점으로 쓴다 (0 = 완전히 저차원, 1 = 완전히 독립).
    """
    d = freq - freq.mean(dim=1, keepdim=True)
    d = d / d.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)
    total = torch.zeros((), device=freq.device, dtype=freq.dtype)
    for b in range(d.shape[0]):
        sv = torch.linalg.svdvals(d[b])
        e = sv ** 2
        total = total + e[n_modes:].sum() / e.sum().clamp_min(eps)
    return total / d.shape[0]


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
    rate       : 생리적 조음 속도 상한 초과분(hinge). 상한 아래 전이는 안 벌한다.
    subspace   : 포먼트가 저차원 부분공간 밖으로 움직이는 성분.
    """

    def __init__(self, w_stft=1.0, w_band=1.0, w_phase=0.2, w_rps=0.3, w_mel=1.0,
                 w_period=1.0, w_smooth=1e-3, w_area=1e-3, w_noise=5e-3,
                 w_rate=0.5, w_subspace=0.1, n_modes=3,
                 sample_rate: int = 24000, hop: int = 240):
        super().__init__()
        self.w = dict(stft=w_stft, band=w_band, phase=w_phase, rps=w_rps, mel=w_mel,
                      period=w_period, smooth=w_smooth, area=w_area, noise=w_noise,
                      rate=w_rate, subspace=w_subspace)
        self.n_modes = n_modes
        self.sample_rate = sample_rate
        self.hop = hop

    def forward(self, pred_audio, target_audio, controls=None,
                voicing: torch.Tensor | None = None,
                f0: torch.Tensor | None = None) -> dict:
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
            if getattr(controls, "sib", None) is not None:
                reg = reg + smoothness(controls.sib.pole_f / 5000.0)
            if controls.area is not None:
                reg = reg + area_smoothness(controls.area)
        out["reg"] = reg

        # 조음 제약: 불가능한 속도와 불가능한 포먼트 조합
        zero = pred_audio.new_zeros(())
        out["rate"] = out["subspace"] = zero
        if controls is not None:
            fr = self.sample_rate / self.hop
            out["rate"] = articulatory_rate_loss(controls.formant_freq, fr)
            out["subspace"] = formant_subspace_loss(controls.formant_freq,
                                                    self.n_modes)

        out["total"] = (self.w["stft"] * out["stft"] + self.w["band"] * out["band"]
                        + self.w["phase"] * out["phase"] + self.w["rps"] * out["rps"]
                        + self.w["mel"] * out["mel"] + self.w["period"] * out["period"]
                        + self.w["smooth"] * out["reg"] + self.w["noise"] * out["noise"]
                        + self.w["rate"] * out["rate"]
                        + self.w["subspace"] * out["subspace"])
        return out
