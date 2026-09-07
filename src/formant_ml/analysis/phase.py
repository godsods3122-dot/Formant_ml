"""실제 음성에서 **위상차 파라미터**를 뽑아낸다.

무엇을 재는가
-------------
하모닉 상대위상 RPS_k = ∠X_k − k·∠X_1 은 분석창 위치에 무관한 양이다.
그 프레임 평균은 화자의 '성문 폐쇄가 얼마나 정렬되어 있는가'를 담고,
프레임 간 산포(PDD)는 거칠기/기식성을 담는다.

측정된 RPS 를 그대로 합성에 쓰면 안 된다. 그중 상당 부분은
(a) 우리 LF 소스가 이미 만들어내는 위상과 (b) 성도 필터의 위상이기 때문이다.
그래서 여기서는 **잔차만** 올패스 체인으로 적합한다:

    RPS_측정 − RPS_LF(Rd)  ≈  RPS_allpass(F_s, r_s)

이렇게 얻은 (F_s, r_s) 가 `Controls.disp_freq/disp_radius` 다. 올패스라서
크기 스펙트럼을 절대 바꾸지 않으므로, 위상을 자유롭게 예측할 때 생기는
phasiness 가 되돌아오지 않는다.
"""
from __future__ import annotations

import math

import torch

from ..data.features import yin_f0
from ..dsp.glottal import LFTableBank
from ..dsp.phase import (allpass_phase, circular_mean, phase_distortion_deviation,
                         relative_phase_shift, resonator_phase)
from .registers import harmonic_spectrum

TWO_PI = 2.0 * math.pi


def measure_rps(x: torch.Tensor, sample_rate: int = 24000, hop: int = 240,
                n_harmonics: int = 20, min_voicing: float = 0.5) -> dict:
    """유성 프레임의 상대위상 통계.

    반환: {"rps": (K,) 원형평균, "pdd": (K,) 원형표준편차, "n_frames", "f0_median"}
    """
    x = x.detach()
    if x.dim() == 1:
        x = x[None]
    f0, voicing = yin_f0(x, sample_rate, hop)
    f0f = torch.where(f0[0] > 0, f0[0], torch.full_like(f0[0], 120.0))
    H = harmonic_spectrum(x, f0f, sample_rate, hop, n_harmonics=n_harmonics)
    v = voicing[0][:H.shape[0]] > min_voicing
    if int(v.sum()) < 5:
        raise ValueError("유성 프레임이 너무 적습니다")
    rps = relative_phase_shift(H[v], n_harmonics)             # (Tv, K)
    mean, _ = circular_mean(rps, dim=0)
    return {"rps": mean, "pdd": phase_distortion_deviation(rps, dim=0),
            "n_frames": int(v.sum()), "f0_median": float(f0f[v].median())}


def cascade_rps(formant_f, formant_bw, f0: float, n_harmonics: int,
                sample_rate: int) -> torch.Tensor:
    """포먼트 캐스케이드(최소위상)가 만드는 상대위상 (K,).

    성도가 만든 위상을 빼 주지 않으면 소스의 위상차를 성도 위상으로 착각한다.
    """
    f = torch.as_tensor(formant_f, dtype=torch.float32).view(1, 1, -1)
    b = torch.as_tensor(formant_bw, dtype=torch.float32).view(1, 1, -1)
    k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
    fk = (f0 * k).view(1, 1, -1)
    psi = resonator_phase(fk, f, b, sample_rate)[0, 0]
    return psi - k * psi[0]


def lf_reference_rps(rd: float, n_harmonics: int = 20) -> torch.Tensor:
    """LF 소스 자체가 만드는 상대위상 (K,). 측정값에서 빼야 할 기준선."""
    bank = LFTableBank(n_tables=2, table_size=4096, rd_min=rd, rd_max=rd + 1e-3,
                       n_harmonics=n_harmonics)
    return relative_phase_shift(bank.spectra[0], n_harmonics)


def allpass_rps(ap_freq, ap_radius, f0: float, n_harmonics: int,
                sample_rate: int) -> torch.Tensor:
    """올패스 체인이 만드는 상대위상 (K,). ψ(k f0) − k·ψ(f0)."""
    k = torch.arange(1, n_harmonics + 1, dtype=ap_freq.dtype, device=ap_freq.device)
    fk = (f0 * k).view(1, 1, -1)
    psi = allpass_phase(fk, ap_freq, ap_radius, sample_rate)[0, 0]
    return psi - k * psi[0]


def fit_dispersion(target_rps: torch.Tensor, f0: float, rd: float = 1.0,
                   n_stages: int = 3, sample_rate: int = 24000, steps: int = 600,
                   lr: float = 0.05, weight: torch.Tensor | None = None,
                   reference_rps: torch.Tensor | None = None) -> dict:
    """측정 RPS -> 올패스 위상차 파라미터 (경사하강, 원형 손실).

    `reference_rps` 는 '모델이 이미 만들어내는 위상'이다. 보통
    LF 소스 위상 + 성도 캐스케이드 위상이며, 이걸 빼야 *남는* 위상차만 적합된다.
    주지 않으면 LF 만 기준으로 삼는다(성도 위상까지 올패스가 흡수하게 되므로,
    위상 중립적인 성도로 합성할 때만 그렇게 쓸 것).

    반환 {"freq": [Hz]*n_stages, "radius": [..], "residual_rad": float}
    """
    n_h = target_rps.numel()
    ref = (lf_reference_rps(rd, n_h) if reference_rps is None
           else reference_rps).to(target_rps.device)
    resid = target_rps - ref
    resid = resid - TWO_PI * torch.round(resid / TWO_PI)
    w = torch.ones_like(resid) if weight is None else weight

    uf = torch.linspace(-1.0, 1.0, n_stages).clone().requires_grad_(True)
    ur = torch.zeros(n_stages).requires_grad_(True)
    opt = torch.optim.Adam([uf, ur], lr=lr)
    nyq = sample_rate * 0.45

    def build():
        f = (200.0 + (nyq - 200.0) * torch.sigmoid(uf)).view(1, 1, -1)
        r = (0.95 * torch.sigmoid(ur)).view(1, 1, -1)
        return f, r

    last = float("nan")
    for _ in range(steps):
        f, r = build()
        pred = allpass_rps(f, r, f0, n_h, sample_rate)
        loss = ((1.0 - torch.cos(pred - resid)) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())

    with torch.no_grad():
        f, r = build()
        pred = allpass_rps(f, r, f0, n_h, sample_rate)
        d = pred - resid
        d = d - TWO_PI * torch.round(d / TWO_PI)
        return {"freq": [round(float(v), 1) for v in f.view(-1)],
                "radius": [round(float(v), 4) for v in r.view(-1)],
                "residual_rad": round(float(d.abs().mean()), 4),
                "circular_loss": round(last, 5)}


def model_reference_rps(f0: float, rd: float, formant_f, formant_bw,
                        n_harmonics: int = 20, cfg=None, n_frames: int = 120
                        ) -> torch.Tensor:
    """**모델이 실제로 내는** 상대위상 (K,) — 위상차가 0 인 프로브를 합성해서 측정.

    해석식(LF 위상 + 캐스케이드 위상)을 기준선으로 쓰면 LTV-FIR 절단/창가림 때문에
    0.5 rad 쯤의 계통 오차가 남고, 그 오차를 위상차 파라미터가 흡수해 버린다.
    같은 합성기로 프로브를 만들어 재면 그런 모델 아티팩트가 전부 상쇄된다.
    """
    from ..config import Config
    from ..models.synth import Controls, PhysicalVoiceSynth
    cfg = cfg or Config()
    K = len(formant_f)
    T = n_frames
    ones = torch.ones(1, T, 1)
    ff = torch.as_tensor(formant_f, dtype=torch.float32).view(1, 1, -1).expand(1, T, K)
    bw = torch.as_tensor(formant_bw, dtype=torch.float32).view(1, 1, -1).expand(1, T, K)
    c = Controls(
        f0=ones * f0, harmonic_amp=ones, rd=ones * rd,
        formant_freq=ff.contiguous(), formant_bw=bw.contiguous(),
        formant_gain=torch.ones(1, T, K),
        noise_bands=torch.full((1, T, cfg.noise.n_bands), 1e-6),
        noise_entry=torch.zeros(1, T, 1), noise_am=torch.zeros(1, T, 1))
    syn = PhysicalVoiceSynth(cfg)
    with torch.no_grad():
        y = syn(c)["audio"]
    return measure_rps(y, cfg.audio.sample_rate, cfg.audio.hop_size,
                       n_harmonics)["rps"]


def extract(x: torch.Tensor, rd: float = 1.0, sample_rate: int = 24000,
            hop: int = 240, n_harmonics: int = 20, n_stages: int = 3,
            formant_f=None, formant_bw=None, empirical_reference: bool = True
            ) -> dict:
    """한 번에: 측정 -> 모델 기준선 제거 -> 잔차를 올패스로 적합.

    `formant_f/bw` (그 화자의 평균 포먼트) 를 주면 성도가 만드는 위상까지 기준선에
    들어간다. 주지 않으면 소스(LF) 기준선만 쓰므로, 나온 파라미터에는 성도 위상이
    섞여 있다 — 위상 중립적인 성도로 합성할 때만 그렇게 쓸 것.
    """
    m = measure_rps(x, sample_rate, hop, n_harmonics)
    if formant_f is not None and formant_bw is not None:
        ref = (model_reference_rps(m["f0_median"], rd, formant_f, formant_bw,
                                   n_harmonics)
               if empirical_reference
               else lf_reference_rps(rd, n_harmonics)
               + cascade_rps(formant_f, formant_bw, m["f0_median"],
                             n_harmonics, sample_rate))
    else:
        ref = lf_reference_rps(rd, n_harmonics)
    fit = fit_dispersion(m["rps"], m["f0_median"], rd, n_stages, sample_rate,
                         reference_rps=ref)
    fit["measured_rps"] = [round(float(v), 4) for v in m["rps"]]
    fit["pdd"] = [round(float(v), 4) for v in m["pdd"]]
    fit["f0_median"] = round(m["f0_median"], 2)
    fit["n_frames"] = m["n_frames"]
    return fit
