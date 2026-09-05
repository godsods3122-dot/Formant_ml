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
                         mix=1.0):
    """설측음의 반공명 2 개 -> (freq (1,T,2), bw (1,T,2)).

    supra_cm : 설상공(supralingual). 지배적인 반공진.
    inter_cm : 치간 통로. 좌우 비대칭일 때 중요해진다.
    mix      : 스칼라 또는 (T,) 프레임별 0~1. **반드시 설측 구간에만 켜야 한다.**
               발화 전체에 걸어 두었더니 모음의 F3 까지 눌려 3050 -> 1890 Hz 로
               내려갔다(측정). 반공명은 옆 통로가 열려 있는 동안만 존재한다.

    기본값의 영점은 각각 2917 Hz, 3977 Hz — 문헌의 2000~5000 Hz 대역 안이다.
    끄는 방법은 대역폭을 키우는 것이다: BW -> 무한이면 r = exp(-pi*BW/fs) -> 0 이라
    D/Ddc -> 1 로 응답이 정확히 평탄해진다.
    """
    fz = torch.tensor([side_branch_zero(supra_cm), side_branch_zero(inter_cm)])
    bw0 = torch.tensor([bw_supra, bw_inter])
    m = (mix if torch.is_tensor(mix)
         else torch.full((t,), float(mix))).clamp(1e-3, 1.0).reshape(1, t, 1)
    fz = fz.reshape(1, 1, 2).expand(1, t, 2).contiguous()
    bw = (bw0.reshape(1, 1, 2) / m).expand(1, t, 2).contiguous()
    return fz, bw


def tip_to_area(tip_gap: torch.Tensor, base_area: torch.Tensor,
                hop: int, n_masses: int, gap_open_cm: float = 0.25,
                lip_margin: int = LIP_MARGIN,
                floor_cm2: float = 1e-3) -> torch.Tensor:
    """혀끝 마디별 간극 (N_samples, n_masses) -> 면적함수 (1, T, N_sections).

    **간극을 면적으로 바로 환산하면 안 된다.** 처음에 `면적 = 폭 x 간극` 으로
    두었더니, 혀끝이 '열린' 상태(간극 0.30 cm)에서도 앞쪽 단면이 0.36 cm^2 로
    묶여서 모음이 통째로 빨대를 통과한 소리가 됐다 — 측정: /아/ 목표
    F1 730 / F2 1220 인데 합성이 395 / 715 로 나왔다.

    실제로는 혀끝이 구개에서 2~3 mm 만 떨어져도 더 이상 최협착이 아니고,
    그 지점의 단면적은 **모음 자신의 기하**가 정한다. 그래서 혀끝은 면적을
    *만드는* 게 아니라 모음의 면적을 *깎는다*:

        a = base * smoothstep(gap / gap_open)

    간극 0 -> 완전 폐쇄, gap_open 이상 -> 모음 그대로.
    """
    n_sec = base_area.shape[-1]
    t = tip_gap.shape[0] // hop
    # 샘플률 -> 프레임률. 평균이 아니라 **최솟값**: 협착은 순간의 최솟값이
    # 음향을 지배하고, 평균을 쓰면 접촉이 통째로 사라진다.
    g = tip_gap[: t * hop].reshape(t, hop, n_masses).amin(dim=1)   # (T, n_masses)

    u = (g / gap_open_cm).clamp(0.0, 1.0)
    close = u * u * (3.0 - 2.0 * u)                                # smoothstep

    area = base_area.reshape(1, 1, n_sec).expand(1, t, n_sec).clone()
    start = n_sec - lip_margin - n_masses
    assert start >= 1, "혀끝 마디가 성도보다 길다"
    seg = area[0, :, start:start + n_masses]
    area[0, :, start:start + n_masses] = (seg * close).clamp_min(floor_cm2)
    return area.contiguous()


def vowel_area(n_sec: int, vowel: str = "a") -> torch.Tensor:
    """모음의 바탕 면적함수 (성문 -> 입술).

    20 단(여성)에는 목표 포먼트에 맞춰 푼 면적함수를 쓴다. 손으로 그린 쪽은
    /아/ 의 F2 가 실측보다 700 Hz 높아 한국어 모음으로 들리지 않았다.
    """
    from .presets import VOWEL_AREA_20, area_function
    if n_sec == 20 and vowel in VOWEL_AREA_20:
        return torch.tensor(VOWEL_AREA_20[vowel], dtype=torch.float32)
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
    area = tip_to_area(traj, base, hop, n_masses)
    if lateral_area_cm2 is not None:
        area = area.clamp_min(lateral_area_cm2)
    t = area.shape[1]
    contact = out["contact"][: t * hop].reshape(t, hop).amax(dim=1)
    return area, contact, traj


# ------------------------------------------------------------------ 음절 렌더
def posture_area(name: str, n_sec: int) -> torch.Tensor:
    """자세 이름 -> 면적함수. 모음이면 모음, 유음이면 유음 자세."""
    from .presets import LIQUID_AREA_20, VOWEL_AREA_20
    if n_sec == 20:
        if name in LIQUID_AREA_20:
            return torch.tensor(LIQUID_AREA_20[name], dtype=torch.float32)
        if name in VOWEL_AREA_20:
            return torch.tensor(VOWEL_AREA_20[name], dtype=torch.float32)
    return vowel_area(n_sec, name)


def posture_track(keyframes, t: int, n_sec: int) -> torch.Tensor:
    """[(위치 0~1, 자세이름), ...] -> 면적함수 궤적 (1, T, N).

    **로그 면적에서 보간한다.** 선형으로 섞으면 0.3 cm^2 와 6 cm^2 의 중간이
    3.15 로 나와 협착이 순식간에 풀린다. 면적은 기하적으로 변하는 양이다.

    이것이 축 3 이다 — 혀끝만이 아니라 **성도 전체 모양**이 자세 사이를
    움직인다. 이게 없으면 혀끝이 앞쪽 몇 단만 건드리므로 실측의
    F1 294 -> 761 Hz 같은 전이가 나오지 않는다.
    """
    pos = torch.tensor([p for p, _ in keyframes], dtype=torch.float32)
    mats = torch.stack([torch.log(posture_area(n, n_sec).clamp_min(1e-3))
                        for _, n in keyframes])                 # (K, N)
    x = torch.linspace(0, 1, t)
    i = torch.searchsorted(pos, x.clamp(pos[0], pos[-1])).clamp(1, len(pos) - 1)
    p0, p1 = pos[i - 1], pos[i]
    w = ((x - p0) / (p1 - p0).clamp_min(1e-6)).clamp(0, 1)
    # 자세 전이는 부드럽게 (근육은 계단으로 움직이지 않는다)
    w = (w * w * (3.0 - 2.0 * w)).unsqueeze(-1)
    return torch.exp(mats[i - 1] * (1 - w) + mats[i] * w).unsqueeze(0)


def liquid_syllable(seconds: float, keyframes, h0_points, cfg: Config = DEFAULT,
                    po: float = 2000.0, n_masses: int = 5,
                    lateral_area_cm2: float | None = None,
                    tip_overlay: bool = True,
                    tip: TipParams | None = None):
    """자세 궤적 + 혀끝 제스처 -> (면적 (1,T,N), 접촉 (T,), 혀끝 궤적).

    자세 궤적이 성도 전체를 움직이고, 그 위에 혀끝이 협착을 덧씌운다.
    두 층을 분리하는 이유는 Hwang et al.(2019) 이 잰 것 때문이다 — 혀몸과
    설근은 독립 손잡이가 아니라 관상 폐쇄 위치에서 따라 나온다. 그래서
    자세 하나를 고르면 혀몸까지 같이 정해진다.

    `tip_overlay` 를 조심해서 써야 한다. 자세 면적함수는 **실측 포먼트에 맞춰
    푼 것**이라 그 자세의 협착을 이미 담고 있다. 그 위에 혀끝 폐쇄를 또 곱하면
    이중계산이 되어 적합이 깨진다 — 측정: 설측음 자세는 316/1412/2795 를 내는데
    덧씌우고 렌더하니 222/676/1470 이 나왔다.

    * 설측음: `False`. 중앙 폐쇄와 측면 통로가 이미 자세에 들어 있다.
    * 탄음/전동음: `True`. 완전 폐쇄는 자세로 표현 못 한다(폐쇄 중에는
      방사음이 없으므로 포먼트를 잴 수도 없다). 혀끝이 그 구간을 만든다.
    """
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    n_sec = cfg.filt.n_tract_sections
    n = int(seconds * sr / hop) * hop
    p = tip or TipParams()
    p = TipParams(**{**p.__dict__, "po": po})
    out = simulate_tip_chain(gesture(n, h0_points, sr), p, n_masses, sr,
                             oversample=4)
    t = n // hop
    base = posture_track(keyframes, t, n_sec)                   # (1, T, N)
    traj = out["traj"]
    g = traj[: t * hop].reshape(t, hop, n_masses).amin(dim=1)
    area = base.clone()
    if tip_overlay:
        u = (g / 0.25).clamp(0.0, 1.0)
        close = u * u * (3.0 - 2.0 * u)                         # smoothstep
        start = n_sec - LIP_MARGIN - n_masses
        seg = area[0, :, start:start + n_masses]
        seg = (seg * close).clamp_min(1e-3)
        if lateral_area_cm2 is not None:
            seg = seg.clamp_min(lateral_area_cm2)
        area[0, :, start:start + n_masses] = seg
    contact = out["contact"][: t * hop].reshape(t, hop).amax(dim=1)
    return area.contiguous(), contact, traj
