"""유음 /ㄹ/ — 포먼트 궤적으로 짓는다.

근거는 전부 `docs/LIQUID.md` 다. 여기서는 그 §4 의 결정을 코드로 옮긴다.
숫자마다 그 문서의 절을 달아 두었으니, 값을 바꾸고 싶으면 **근거부터** 봐라.

왜 이 모양인가
--------------
앞선 구현은 유음 자세를 **면적함수 + 영점의 목록**으로 두고 그것들 사이를
오갔다. 그게 실패한 이유가 셋이다:

1. **영점의 근거가 없었다.** 켑스트럼 포락선을 전극 예측과 빼서 확인하면,
   가장 신뢰할 만한 구간(을라의 380 ms 설측음)에서 전극이 못 만드는 골이
   **없다**(LIQUID.md §2.4). 있다고 가정한 영점이 맞춰 둔 극을 부쉈다.
2. **설측음·탄음을 별도 경로로 두었다.** 측정은 둘이 **같은 제스처의 다른
   설정**이라고 말한다 — 혀끝 협착 하나이고, 깊게 짧으면 탄음(−14 dB, 40 ms),
   얕게 길면 설측음(−8 dB, 255 ms)이다(§2.2).
3. **혀 물리모델을 소리보다 먼저 붙였다.** 여기서는 궤적으로 소리를 먼저
   맞추고, 그 궤적을 물리모델이 낼 수 있는지는 나중에 본다(§4-6).

무엇이 파라미터인가
-------------------
* `hold_ms` — 협착을 **유지**하는 시간. 탄음 0, 설측음 215.
* `depth_db` — 협착의 깊이. 탄음이 **더 깊다**(−14 vs −8). 처음에 "깊이는 같고
  지속만 다르다" 고 봤다가 고쳤다 — 20 ms 분석창이 40 ms 짜리 탄음 골을
  뭉갠 것이었다(LIQUID.md §2.2). **하나로 뭉치는 파라미터는 음향에서 안 나온다.**
* `body_raise` — 설체 상승 [0, 1]. Lee(2015) 가 **범주적 차이**로 보고한 축이고
  (§1.1), F2 목표를 정한다. 탄음 0, 설측음 1.
* 나머지(F1/F3 목표, 세기 골, F3 선행)는 측정에서 고정한 상수다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 유음 구간의 포먼트 목표 [Hz]. 사용자 녹음 실측 (LIQUID.md §2.1).
#:
#: **F1 이 유음의 정체다.** 모음 700 에서 200~390 으로 떨어지고, 을라의 지속
#: 구간에서는 380 ms 동안 333 ± 2 Hz 로 붙박이다 — 뚜렷한 목표값이다.
#: 문헌(F1 250~500, F2 1250~1450, §1.5)과도 맞는다.
F1_TARGET = 333.0

#: F2 목표. `body_raise` 가 이 둘 사이를 간다.
#: 설체가 올라가면(설측음) F2 가 올라가고, 안 올라가면(탄음) 모음 값에 가깝다.
#: 실측: 설측음 1445, 탄음 1240~1290, 모음 1200 (§2.1).
F2_FLAT = 1265.0        # body_raise = 0 (탄음)
F2_RAISED = 1445.0      # body_raise = 1 (설측음)

#: F3 목표. 실측 2100~2480 의 가운데. 문헌의 2800(§1.5)보다 낮은데,
#: **문헌이 아니라 이 화자를 따른다**(§4-1).
F3_TARGET = 2300.0

#: 유음 구간의 세기 [dB, 이웃 모음 대비]. **탄음과 설측음이 다르다** (§2.2).
#:
#: 처음에 20 ms 창으로 재고 "둘이 비슷하다" 고 결론냈다가 고쳤다 — 그 창이
#: 40 ms 짜리 탄음 골을 뭉갠 것이었다. 10 ms 창(F0 108 Hz 의 한 주기 9.3 ms
#: 보다 길어야 포락선이 된다)으로 재면 탄음 −16.6 / −10.9, 설측음 −8.0 이다.
#:
#: 물리적으로도 그게 맞다: 탄음은 짧지만 **완전 폐쇄**에 가깝고, 설측음은
#: 혀 옆 통로가 계속 열려 있다.
DEPTH_TAP_DB = -14.0
DEPTH_LATERAL_DB = -8.0

#: F3 가 조음(=세기 골)보다 앞서는 시간 [ms].
#:
#: 실측 5 ms(아라#1), 25 ms(아라#2) (§2.3). Ying(2026) 의 15~30 ms 선행과
#: 부호·규모가 맞는다(§1.4). **F1/F2 는 반대로 뒤진다** — 그래서 F3 만 앞세운다.
F3_LEAD_MS = 15.0

#: 협착으로 들어가고 나오는 시간 [ms].
#:
#: 탄음의 세기 골 전체가 40 ms 인데 그 안에 유지 구간이 없으므로(§2.2),
#: 들어가고 나오는 데 각각 20 ms 다. Cathcart(2012) 의 "탄도(ballistic)" 도
#: 이 그림이다 — 올라갔다 바로 내려온다(§1.3).
TRANSIT_MS = 20.0

#: 이름이 붙은 자세. `hold_ms` 하나와 `body_raise` 하나로 전부 표현된다.
#: **별도 경로가 아니다** — 같은 제스처의 다른 지속이다(§4-2).
PRESETS = {
    # 모음사이 탄음 (아라): 유지 없음, 설체 안 올라감, 깊다
    "tap": dict(hold_ms=0.0, body_raise=0.0, depth_db=DEPTH_TAP_DB),
    # 긴 설측음 (을라): 오래 유지, 설체 올라감, 얕다
    "lateral": dict(hold_ms=215.0, body_raise=1.0, depth_db=DEPTH_LATERAL_DB),
    # 어두 설측음 (라): 설측음인데 짧다
    "onset": dict(hold_ms=45.0, body_raise=1.0, depth_db=DEPTH_LATERAL_DB),
}


@dataclass
class LiquidGesture:
    """유음 제스처 하나. 기본값은 어두 설측음."""
    hold_ms: float = 45.0
    body_raise: float = 1.0
    depth_db: float = DEPTH_LATERAL_DB
    f1: float = F1_TARGET
    f3: float = F3_TARGET
    f3_lead_ms: float = F3_LEAD_MS
    transit_ms: float = TRANSIT_MS

    @classmethod
    def preset(cls, name: str) -> "LiquidGesture":
        if name not in PRESETS:
            raise KeyError(f"{name!r} 은 없다. {sorted(PRESETS)} 중 하나여야 한다.")
        return cls(**PRESETS[name])

    @property
    def f2(self) -> float:
        r = float(np.clip(self.body_raise, 0.0, 1.0))
        return F2_FLAT + (F2_RAISED - F2_FLAT) * r

    @property
    def total_ms(self) -> float:
        """제스처 전체 길이 [ms] — 들어가기 + 유지 + 나오기."""
        return 2.0 * self.transit_ms + self.hold_ms


def constriction(t: int, hop: int, sample_rate: int, gesture: LiquidGesture,
                 center_s: float) -> np.ndarray:
    """협착 정도 c(t) in [0, 1]. 0 = 모음, 1 = 유음 목표.

    사다리꼴을 **매끄럽게** 만든 것이다(smoothstep 으로 들어가고 나온다).
    유지 구간이 0 이면 삼각형이 되고, 그게 탄음이다 — 별도 코드가 아니다.

    각진 사다리꼴을 쓰면 안 된다. 기울기가 꺾이는 지점이 포먼트 궤적의
    2 차 미분에 임펄스로 남고, 그게 소리에서 '툭' 하고 들린다.
    """
    ms = np.arange(t) * hop / sample_rate * 1000.0 - center_s * 1000.0
    half_hold = gesture.hold_ms / 2.0
    tr = max(gesture.transit_ms, 1e-3)
    d = np.abs(ms) - half_hold          # 유지 구간 밖으로 나간 거리 [ms]
    u = np.clip(1.0 - d / tr, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def liquid_track(vowel_f: np.ndarray, gesture: LiquidGesture, t: int,
                 hop: int = 240, sample_rate: int = 24000,
                 center_s: float | None = None):
    """모음 포먼트 궤적 -> 유음이 얹힌 궤적. 반환 (F, gain).

    `vowel_f` 는 (T, K) 모음의 포먼트 [Hz]. 반환 `F` 는 같은 모양이고,
    `gain` 은 (T,) 진폭 배율이다.

    **F3 만 앞세운다.** 실측과 Ying(2026) 둘 다 F3 가 조음보다 5~30 ms 먼저
    움직이고 F1/F2 는 뒤진다고 말한다(§2.3, §1.4). 그래서 F3 의 협착 곡선만
    시간을 당긴다 — 궤적 전체를 하나로 묶으면 이 구조가 사라진다.
    """
    vowel_f = np.asarray(vowel_f, dtype=np.float64)
    if center_s is None:
        center_s = t * hop / sample_rate / 2.0
    c = constriction(t, hop, sample_rate, gesture, center_s)
    lead = gesture.f3_lead_ms / 1000.0
    c3 = constriction(t, hop, sample_rate, gesture, center_s - lead)

    F = vowel_f.copy()
    tgt = (gesture.f1, gesture.f2, gesture.f3)
    for k, (target, cc) in enumerate(zip(tgt, (c, c, c3))):
        if k >= F.shape[1]:
            break
        F[:, k] = F[:, k] * (1.0 - cc) + target * cc
    gain = 10.0 ** (gesture.depth_db * c / 20.0)
    return F, gain
