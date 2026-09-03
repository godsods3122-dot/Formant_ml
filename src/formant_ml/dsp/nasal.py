"""비강 결합: 연구개(velum) 개도 하나로 조종되는 극-영점 쌍.

비강은 근육이 없다. 부비동을 포함한 형상이 화자마다 고정이고 발화 중에 변하지
않으므로 **전달함수가 사실상 상수**다. 변하는 것은 딱 하나, 연구개 포트의 열림뿐이다.
그래서 "화자마다 필터 하나 + 개도 하나"라는 모델링이 물리적으로 옳다.

다만 '구강 응답에 비강 응답을 더한다'는 아니다. 비강은 인두 분기점에서 **병렬
곁가지**로 붙으므로, 그 입력 임피던스가 구강 쪽 전달함수 자체를 바꾼다.

  1. 비강 극이 새로 생긴다 (250~300 Hz 부근, 화자 고정)
  2. 구강 출력에 **영점**이 생긴다 (곁가지가 단락처럼 보이는 주파수)
  3. F1 이 넓어지고 약해진다 (곁가지로 에너지가 새고 비강 벽이 흡수한다)

Klatt(1980) 의 방법을 따른다: 극과 영점을 같은 주파수에 두면 정확히 상쇄되므로
**닫힌 상태에서 응답이 항등이 된다**. 연구개가 열리면 둘을 벌리고 대역폭을 키운다.
파라미터 하나(개도)로 연속적으로 비음화가 들어오고 나간다.

비선형인가? 아니다 — 음향적으로는 선형계다. 다만 '독립적으로 더해지지' 않을 뿐,
곁가지의 임피던스가 병렬로 들어오는 정확히 계산 가능한 선형 결합이다.
도파관 모드(`tract.py`)에서는 3-포트 산란 접합으로 두면 위 세 효과가 근사 없이
전부 저절로 나온다.
"""
from __future__ import annotations

import torch

from .filters import antiresonator_response, resonator_response

# 화자 고정값(성인 남성 기준). 프로파일에서 덮어쓸 수 있다.
NASAL_POLE_HZ = 270.0
NASAL_POLE_BW = 100.0
# Klatt(1980) 의 공칭값: 비자음에서 FNP=270, FNZ=450.
# 영점을 700 Hz 이상으로 올리면 /a/ 의 F1(730) 위에 얹혀 F1 을 통째로 지운다.
NASAL_ZERO_MAX_HZ = 450.0      # 완전 개방 시 영점 위치
NASAL_ZERO_BW = 200.0
F1_DAMPING = 1.6               # 완전 개방 시 F1 대역폭 배수


def nasal_response(velum_open: torch.Tensor, sample_rate: float, n_freq: int,
                   pole_hz: float = NASAL_POLE_HZ, pole_bw: float = NASAL_POLE_BW,
                   zero_max_hz: float = NASAL_ZERO_MAX_HZ,
                   zero_bw: float = NASAL_ZERO_BW) -> torch.Tensor:
    """velum_open: (B, T, 1) 0~1 -> 복소응답 (B, T, n_freq).

    개도 0 에서 극과 영점이 정확히 겹쳐 응답이 항등(1)이 된다.
    """
    o = velum_open.clamp(0.0, 1.0)
    fp = torch.full_like(o, pole_hz)
    bp = pole_bw * (1.0 + o)
    fz = pole_hz + o * (zero_max_hz - pole_hz)
    bz = pole_bw + o * (zero_bw - pole_bw)
    h_pole = resonator_response(fp, bp, torch.ones_like(o), sample_rate, n_freq)
    h_zero = antiresonator_response(fz, bz, sample_rate, n_freq)
    return h_pole.squeeze(2) * h_zero if h_pole.dim() == 4 else h_pole * h_zero


def f1_bandwidth_factor(velum_open: torch.Tensor,
                        damping: float = F1_DAMPING) -> torch.Tensor:
    """연구개가 열리면 F1 대역폭이 넓어진다. (B, T, 1) -> 배수."""
    return 1.0 + (damping - 1.0) * velum_open.clamp(0.0, 1.0)
