"""목표 포먼트를 내는 성도 면적함수를 경사하강으로 찾는다.

왜 이게 필요한가
----------------
`presets.area_function` 은 시그모이드로 손으로 그린 근사다. 재 보니 /아/ 가
F1 820 / F2 1605 를 내는데 실측은 730 / 1220 이다 — F2 가 400 Hz 틀렸고,
성도를 여성 길이(20 단)로 줄이면 1928 로 더 벌어진다. 곡선을 손으로 고치면
다음 모음에서 다시 무너지므로, **면적함수를 목표 포먼트에 맞춰 푼다.**

방법은 이 레포의 LITERATURE.md §2 에 이미 적혀 있는 것이다 —
"Vocal Tract Area Estimation by Gradient Descent" (arXiv:2307.04702).
`tract.tract_response` 가 면적에 대해 미분가능하므로 그대로 쓴다.

목표 스펙트럼은 포먼트 캐스케이드로 만든다(그쪽은 목표치 ±3% 로 검증되어 있다).
즉 "캐스케이드가 내는 응답과 같은 응답을 내는 면적함수" 를 찾는 것이다.
"""
from __future__ import annotations

import torch

from .dsp.filters import resonator_stage_responses
from .dsp.tract import tract_response


def target_response(freqs, bws, sample_rate: int, n_freq: int) -> torch.Tensor:
    f = torch.tensor(freqs, dtype=torch.float32).reshape(1, 1, -1)
    b = torch.tensor(bws, dtype=torch.float32).reshape(1, 1, -1)
    g = torch.ones_like(f)
    return resonator_stage_responses(f, b, g, sample_rate, n_freq).prod(dim=2)


def fit_area(freqs, bws=None, n_sections: int = 24, sample_rate: int = 24000,
             n_freq: int = 513, steps: int = 1500, lr: float = 0.08,
             smooth: float = 0.15, fmax: float = 5000.0,
             a_min: float = 0.15, a_max: float = 11.0, rho: float = 0.99,
             seed: int = 0, verbose: bool = False) -> torch.Tensor:
    """목표 포먼트를 내는 면적함수 (n_sections,) [cm^2]. 성문 -> 입술.

    면적을 [a_min, a_max] 로 **묶는 것이 필수다.** 안 묶으면 F1/F2 는 맞추면서
    45 cm^2 짜리 단면이 나오는 퇴화해로 간다(측정 확인). 사람 성도는 대략
    0.15~11 cm^2 이고, 그 밖은 해가 아니라 수치적 요행이다.
    매끄러움 페널티도 같은 이유로 세게 건다.
    """
    bws = bws or [90.0 + 40.0 * i for i in range(len(freqs))]
    H_t = target_response(freqs, bws, sample_rate, n_freq).abs()[0, 0]
    f_grid = torch.linspace(0.0, sample_rate / 2, n_freq)
    band = f_grid <= fmax
    log_t = torch.log(H_t.clamp_min(1e-6))
    log_t = log_t - log_t[band].mean()

    torch.manual_seed(seed)
    z = torch.zeros(n_sections, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for i in range(steps):
        area = a_min + (a_max - a_min) * torch.sigmoid(z)
        H = tract_response(area.reshape(1, 1, -1), sample_rate, n_freq,
                           rho=rho).abs()[0, 0]
        log_h = torch.log(H.clamp_min(1e-6))
        log_h = log_h - log_h[band].mean()
        loss = (log_h[band] - log_t[band]).pow(2).mean()
        loss = loss + smooth * (z[1:] - z[:-1]).pow(2).mean() * n_sections
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and i % 300 == 0:
            print(f"    step {i:4d} loss {float(loss):.4f}")
    return (a_min + (a_max - a_min) * torch.sigmoid(z)).detach()


def peaks_of(area: torch.Tensor, sample_rate: int = 24000, n: int = 4,
             n_freq: int = 2049, rho: float = 0.99):
    """면적함수가 실제로 내는 포먼트 [Hz] — 맞았는지 재는 용도."""
    H = tract_response(area.reshape(1, 1, -1), sample_rate, n_freq,
                       rho=rho).abs()[0, 0]
    f = torch.linspace(0.0, sample_rate / 2, n_freq)
    out = [float(f[i]) for i in range(1, n_freq - 1)
           if H[i] > H[i - 1] and H[i] > H[i + 1]]
    return [round(v) for v in out[:n]]
