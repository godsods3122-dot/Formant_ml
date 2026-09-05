"""유음 렌더링 — 혀끝 물리모델을 성도 도파관에 연결한다.

`dsp/tongue.py` 는 혀끝의 간극과 접촉까지만 낸다. 소리가 되려면 그 간극이
**성도 면적함수의 한 구간**이 되어 Kelly-Lochbaum 도파관을 지나야 한다.
이 모듈이 그 다리다.

    혀끝 다질량 사슬 (샘플률)  ->  면적함수의 전방 구간 (프레임률)  ->  도파관

이 연결이 다질량으로 간 실질적 이유다. 질량이 n 개면 협착의 **모양**이 n 개
마디를 가지므로 도파관이 먹을 면적함수가 실제로 움직인다. 2 질량으로는
계단 하나뿐이라 먹일 모양이 없다.

설측음은 다르게 붙는다. 중앙이 막혀도 옆이 열려 있으므로 (a) 협착 구간의 면적을
0 이 아니라 **측면 통로 면적**으로 두고, (b) 측지(side branch)가 만드는
**반공명 2 개**를 건다. 설상공 영점이 지배적이고 치간 통로 영점은 비대칭
조음에서 중요해진다(Charles & Lulich 2019). 한국어 여성 화자 5 명 중 3 명이
비대칭이었다(Hwang et al. 2019) — 그래서 둘 다 필요하다.
"""
from __future__ import annotations

import torch

from .config import Config, DEFAULT
from .dsp.tongue import TipParams, gesture, simulate_tip_chain

# 치경 협착의 위치: 입술에서 몇 번째 단인가.
# 24 kHz 에서 한 단은 c/(2fs) = 0.729 cm. 치경융기는 입술에서 약 2~3 cm 이므로
# 입술쪽 끝에서 3~4 단 안쪽이다.
LIP_MARGIN = 2

# 측지 기하 -> 반공명. 1/4 파장 관의 영점은 c/(4L).
SOUND_SPEED = 35000.0            # cm/s


def side_branch_zero(length_cm: float) -> float:
    """길이 L 인 측지가 만드는 반공진 주파수 [Hz]. Stevens(1998): L = 2~4 cm."""
    return SOUND_SPEED / (4.0 * max(length_cm, 0.2))


def lateral_antiformants(t: int, supra_cm: float = 3.0, inter_cm: float = 2.2,
                         bw_supra: float = 300.0, bw_inter: float = 500.0,
                         mix: float = 1.0):
    """설측음의 반공명 2 개 -> (freq (1,T,2), bw (1,T,2)).

    supra_cm : 설상공(supralingual). 지배적인 반공진.
    inter_cm : 치간 통로. 좌우 비대칭일 때 중요해진다.
    mix      : 0 이면 반공명을 끈다(설측음이 아닌 구간).

    기본값의 영점은 각각 2917 Hz, 3977 Hz — 문헌의 2000~5000 Hz 대역 안이다.
    """
    fz = torch.tensor([side_branch_zero(supra_cm), side_branch_zero(inter_cm)])
    bw = torch.tensor([bw_supra, bw_inter])
    # mix<1 이면 영점을 나이퀴스트 밖으로 밀어 효과를 없앤다(응답이 1 로 수렴).
    fz = fz.reshape(1, 1, 2).expand(1, t, 2).contiguous()
    bw = (bw.reshape(1, 1, 2).expand(1, t, 2).contiguous()
          / max(mix, 1e-3))
    return fz, bw


def tip_to_area(tip_gap: torch.Tensor, base_area: torch.Tensor,
                hop: int, n_masses: int, width_cm: float = 1.2,
                lip_margin: int = LIP_MARGIN,
                floor_cm2: float = 1e-3) -> torch.Tensor:
    """혀끝 마디별 간극 (N_samples, n_masses) -> 면적함수 (1, T, N_sections).

    base_area: (N_sections,) 모음의 바탕 면적함수 (성문 -> 입술).
    혀끝이 차지하는 전방 구간만 덮어쓰고 나머지는 모음 그대로 둔다.
    협착이므로 바탕과 혀끝 중 **좁은 쪽**을 취한다.
    """
    n_sec = base_area.shape[-1]
    t = tip_gap.shape[0] // hop
    # 샘플률 -> 프레임률 (프레임 평균이 아니라 **최솟값**: 협착은 순간의 최솟값이
    # 음향을 지배하고, 평균을 쓰면 접촉이 통째로 사라진다)
    g = tip_gap[: t * hop].reshape(t, hop, n_masses).amin(dim=1)   # (T, n_masses)
    tip_area = (width_cm * g).clamp_min(floor_cm2)                 # (T, n_masses)

    area = base_area.reshape(1, 1, n_sec).expand(1, t, n_sec).clone()
    start = n_sec - lip_margin - n_masses
    assert start >= 1, "혀끝 마디가 성도보다 길다"
    seg = area[0, :, start:start + n_masses]
    area[0, :, start:start + n_masses] = torch.minimum(seg, tip_area)
    return area.contiguous()


def vowel_area(n_sec: int, vowel: str = "a") -> torch.Tensor:
    """모음의 바탕 면적함수 (성문 -> 입술). presets 의 것을 그대로 쓴다."""
    from .presets import area_function
    return area_function(vowel, n_sec)


def liquid_area(seconds: float, h0_points, cfg: Config = DEFAULT,
                po: float = 8000.0, n_masses: int = 5, vowel: str = "a",
                lateral_area_cm2: float | None = None,
                tip: TipParams | None = None):
    """혀끝 제스처 -> (면적함수 (1,T,N), 접촉 (T,), 혀끝 궤적).

    lateral_area_cm2 가 주어지면 **설측음**: 중앙이 막혀도 옆이 열려 있으므로
    협착 구간의 면적이 0 으로 내려가지 않고 이 값에서 바닥을 친다.
    """
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    n_sec = cfg.filt.n_tract_sections
    n = int(seconds * sr / hop) * hop
    p = tip or TipParams()
    p = TipParams(**{**p.__dict__, "po": po})
    out = simulate_tip_chain(gesture(n, h0_points, sr), p, n_masses, sr,
                             oversample=4)
    traj = out["traj"]
    base = vowel_area(n_sec, vowel)
    area = tip_to_area(traj, base, hop, n_masses, p.width)
    if lateral_area_cm2 is not None:
        area = area.clamp_min(lateral_area_cm2)
    t = area.shape[1]
    contact = out["contact"][: t * hop].reshape(t, hop).amax(dim=1)
    return area, contact, traj
