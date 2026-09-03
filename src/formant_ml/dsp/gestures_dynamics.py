"""조음 동역학(Task Dynamics): 목표값 -> 물리적으로 가능한 궤적.

혀는 순간이동하지 않는다. 그런데 지금까지 제어 파라미터는 스크립트가 그린
꺾은선을 그대로 썼고(경계 이동평균만), 학습 경로에서는 인코더가 프레임마다
자유롭게 아무 값이나 낼 수 있었다. 둘 다 **물리적으로 불가능한 조음 속도**를
허용하며, 그게 기계음의 전형적 원인이다.

Haskins 의 Task Dynamics (Saltzman & Munhall 1989) 는 각 조음 과제를
**임계감쇠 2차계**로 본다.

    x'' + 2w x' + w^2 (x - target) = 0

목표 -> 실제의 전달은 임펄스응답 h(t) = w^2 t e^{-wt} (적분 1) 인 선형필터다.
프레임률에서 FFT 컨볼루션 한 번이면 되고, 미분가능하며, **불가능한 속도가
표현 자체로 불가능해진다**(페널티가 아니라 구조적 제약).

- 계단 목표에 대한 10~90% 상승시간 t_rise = 3.357 / w
- 크기 dF 계단에 대한 최대 속도 = dF * w / e
  (t_rise 60 ms, dF 1000 Hz -> 20.6 Hz/ms. 문헌의 F2 전이 10~25 Hz/ms 와 일치)

시간상수는 조음기관마다 다르다. 아래 값은 문헌의 전이 지속시간(10~50 ms 급속
전이, 50~100 ms 모음간 이동)에서 역산한 기본값이고, 화자마다 조정 가능하다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# 10~90% 상승시간 [초]. 빠른 순: 성문 < 후두 < 혀끝 < 입술 < 혀몸통 < 턱 < 연구개
RISE_TIME = {
    "glottis": 0.015,     # 성문 개폐(유성/무성 전환), 공기역학은 빠르다
    "larynx": 0.030,      # F0, 성구
    "tongue_tip": 0.035,  # /s/ /t/ /n/ 의 혀끝
    "lips": 0.060,        # 원순, 폐쇄
    "tongue_body": 0.060,  # 모음 F1/F2
    "jaw": 0.080,         # 개구도
    "velum": 0.100,       # 연구개(비음) — 가장 느리다
}

# 제어 파라미터 -> 담당 조음기관
PARAM_ARTICULATOR = {
    "f0": "larynx", "rd": "larynx", "tilt": "larynx",
    "harmonic_amp": "glottis", "noise_am": "glottis", "aspiration": "glottis",
    "noise_bands": "tongue_tip", "noise_gain": "tongue_tip",
    "sib_pole_f": "tongue_tip", "sib_zero_f": "tongue_tip",
    "sib_pole_bw": "tongue_tip", "sib_zero_bw": "tongue_tip", "sib_tilt": "tongue_tip",
    "sib_mix": "tongue_tip",
    "formant_bw": "tongue_body", "formant_gain": "tongue_body",
    "area": "tongue_body", "velum_open": "velum",
    # 연구개는 가장 느린 조음기관이다 -> 비음화가 앞뒤 모음으로 번지는
    # (nasal coarticulation) 현상이 이 시간상수 하나에서 저절로 나온다.
}
# 포먼트별 담당 조음기관 (F1 은 턱/개구도, F2 는 혀몸통, F3 는 혀끝, 그 위는 고정 해부)
FORMANT_ARTICULATOR = ["jaw", "tongue_body", "tongue_tip", "lips"]

# 생리적 최대 변화율 [Hz/s]. 손실(hinge)에서 이 값을 넘는 만큼만 벌점을 준다.
FORMANT_MAX_RATE = [12_000.0, 30_000.0, 25_000.0, 12_000.0]
FORMANT_MAX_RATE_HIGH = 8_000.0        # F5 이상: 성도 길이가 정하는 거의 고정값


def rise_to_omega(t_rise: float) -> float:
    """10~90% 상승시간 -> 임계감쇠 고유각주파수."""
    return 3.3567 / max(t_rise, 1e-4)


def gesture_kernel(t_rise: float, frame_rate: float, device=None,
                   dtype=torch.float32, max_len: int | None = None) -> torch.Tensor:
    """임계감쇠 2차계의 임펄스응답 (프레임률, 합 = 1)."""
    w = rise_to_omega(t_rise)
    n = max_len or max(3, int(round(8.0 / w * frame_rate)))
    t = torch.arange(n, device=device, dtype=dtype) / frame_rate
    h = (w ** 2) * t * torch.exp(-w * t)
    return h / h.sum().clamp_min(1e-12)


def apply_dynamics(x: torch.Tensor, t_rise: float, frame_rate: float
                   ) -> torch.Tensor:
    """목표 궤적 (B, T, C) 에 조음 동역학을 적용. 인과적(미래를 안 본다).

    앞쪽은 replicate 패딩이라 첫 프레임에서 0 으로부터의 가짜 상승이 없다.
    """
    if x.shape[1] < 2:
        return x
    h = gesture_kernel(t_rise, frame_rate, x.device, x.dtype)
    n = h.shape[0]
    b, t, c = x.shape
    xp = F.pad(x.transpose(1, 2), (n - 1, 0), mode="replicate")     # (B, C, T+n-1)
    ker = h.flip(0).reshape(1, 1, n).expand(c, 1, n)
    return F.conv1d(xp, ker, groups=c).transpose(1, 2)


def apply_to_controls(ctrl: dict, frame_rate: float,
                      scale: float = 1.0, skip: tuple[str, ...] = ()) -> dict:
    """제어 dict 전체에 파라미터별 동역학을 적용한다.

    `scale` 로 전체 시간상수를 늘리거나(느린 말투) 줄인다(빠른 말투).
    `skip` 에 넣은 키는 건드리지 않는다 — 특히 `noise_entry` 는 연속량이 아니라
    이산적 조음 상태라서 보간하면 어떤 성도 형상에도 대응하지 않는 필터가 된다
    (score.py 의 주석 참고).
    """
    out = dict(ctrl)
    for key, val in ctrl.items():
        if key in skip or not torch.is_tensor(val) or val.dim() != 3:
            continue
        if key == "noise_entry":
            continue
        if key == "formant_freq":
            cols = []
            for k in range(val.shape[-1]):
                art = (FORMANT_ARTICULATOR[k] if k < len(FORMANT_ARTICULATOR)
                       else "tongue_body")
                cols.append(apply_dynamics(val[..., k:k + 1],
                                           RISE_TIME[art] * scale, frame_rate))
            out[key] = torch.cat(cols, dim=-1)
            continue
        art = PARAM_ARTICULATOR.get(key)
        if art is None:
            continue
        out[key] = apply_dynamics(val, RISE_TIME[art] * scale, frame_rate)
    return out


# ------------------------------------------------------------------- 진단
def formant_rates(freq: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """포먼트 변화율 [Hz/s]. (B, T-1, K)"""
    return (freq[:, 1:] - freq[:, :-1]).abs() * frame_rate


def max_rate_limits(k: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """포먼트별 생리적 상한 [Hz/s]. (K,)"""
    vals = [FORMANT_MAX_RATE[i] if i < len(FORMANT_MAX_RATE) else FORMANT_MAX_RATE_HIGH
            for i in range(k)]
    return torch.tensor(vals, device=device, dtype=dtype)


def formant_motion_rank(freq: torch.Tensor, energy: float = 0.97) -> float:
    """포먼트 궤적이 실제로 몇 차원에서 움직이는지 (특이값 에너지 기준).

    Story & Titze 의 면적함수 실증 직교모드에서는 **2개 모드가 분산의 97% 이상**을
    설명한다. 합성 결과가 5~6 차원으로 움직이고 있다면 조음적으로 불가능한
    포먼트 조합을 쓰고 있다는 뜻이다.
    """
    d = freq - freq.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(d.reshape(-1, d.shape[-1]) if d.dim() == 3 else d)
    c = torch.cumsum(s ** 2, 0) / (s ** 2).sum().clamp_min(1e-12)
    return float((c < energy).sum() + 1)
