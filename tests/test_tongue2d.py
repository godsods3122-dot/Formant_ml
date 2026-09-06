"""2D 혀 검증 — Hwang(2019)의 결합이 **정말로** 나오는가.

    PYTHONPATH=src python3 -m pytest tests/test_tongue2d.py -q

Hwang, Charles & Lulich (2019) 가 서울말 여성 5 명에게서 관찰한 것:

> "혀몸 위치와 설근 전진은 대체로 **관상 폐쇄의 위치에 의해 결정된다.**"
> 치음 폐쇄는 혀 전체가 앞으로 나가야 하고, 그 결과 혀몸이 다른 위치만큼
> 올라갈 수 없다.

두 가지 주장이 들어 있다:
  (A) 혀끝이 앞으로 가면 **설근이 전진한다**
  (B) 혀끝이 앞으로 가면 **설체 상승이 억제된다**

측정 결과 **(A) 는 견고하고 (B) 는 아니다.** 아래 검사가 그 경계를 지킨다.
견고하지 않은 것을 견고한 척 넣으면, 나중에 그 위에 상수를 고르게 된다
(docs/PLAN_TONGUE2D.md §0 의 실수 #1).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.tongue2d import (TongueShape, _edge_lengths, _second_difference,
                                 elastic_energy, polygon_area, solve_shape,
                                 tongue_metrics)


def _move_tip(shape: TongueShape, dx: float, dy: float, lock_area: bool = True):
    """혀끝만 (dx, dy) 만큼 옮기고 나머지를 푼다. 설근 노드는 고정."""
    p0 = shape.initial()
    a0 = polygon_area(p0)
    m0 = tongue_metrics(p0, shape)
    tip = shape.n_upper - 1
    handles = {0: p0[0].clone(),
               tip: p0[tip] + torch.tensor([dx, dy], dtype=shape.dtype)}
    if lock_area:
        p = solve_shape(shape, handles, p0=p0, area0=a0, steps=600)
    else:                                        # 대조군: 넓이를 안 지킨다
        rest, rb = _edge_lengths(p0), _second_difference(p0)
        q = p0.clone().requires_grad_(True)
        idx = torch.tensor(sorted(handles))
        tgt = torch.stack([handles[int(i)] for i in idx])
        opt = torch.optim.Adam([q], lr=0.02)
        for _ in range(600):
            e = elastic_energy(q, rest, rb, shape)
            opt.zero_grad(set_to_none=True)
            e.backward()
            opt.step()
            with torch.no_grad():
                q[idx] = tgt
        p = q.detach()
    m = tongue_metrics(p, shape)
    return {k: m[k] - m0[k] for k in ("body_y", "root_x")} | {"area": m["area"],
                                                              "area0": float(a0)}


def test_area_is_conserved_exactly():
    """비압축성은 **등식 구속**이지 벌점이 아니다.

    벌점으로 두면 '얼마나 안 지켜지는지' 가 가중치 속에 숨는다. 투영으로
    풀어서 상대오차가 기계 정밀도 수준이어야 한다.
    """
    sh = TongueShape()
    for dx, dy in ((0.0, 0.8), (0.5, 0.4), (1.0, 0.8)):
        d = _move_tip(sh, dx, dy)
        rel = abs(d["area"] - d["area0"]) / d["area0"]
        assert rel < 1e-5, f"넓이가 {rel:.2e} 어긋났다"


def test_not_moving_the_tip_changes_nothing():
    """혀끝을 안 움직이면 아무것도 안 변해야 한다 — 기준선의 온전성.

    한때 굽힘 항이 **곡률 자체**를 벌해서, 혀끝을 전혀 안 움직여도 설체가
    0.73 cm 내려앉았다. 그 상태에서는 관찰된 '결합' 이 비압축성 때문인지
    에너지 항 때문인지 못 가른다. 쉬는 형상이 에너지의 최소여야 한다.
    """
    d = _move_tip(TongueShape(), 0.0, 0.0)
    assert abs(d["body_y"]) < 1e-6 and abs(d["root_x"]) < 1e-6, d


def test_tip_advance_pulls_the_root_forward():
    """**(A) 혀끝이 앞으로 가면 설근이 전진한다** — 탄성 상수와 무관하게.

    Hwang(2019): "치음 폐쇄는 혀 전체가 앞으로 나가야 한다."
    강성을 9 가지로 바꿔도 부호가 유지되는 것을 확인했다(+0.013 ~ +0.087).
    """
    for sk in (0.3, 1.0, 3.0):
        for bk in (0.1, 0.35, 1.0):
            d = _move_tip(TongueShape(stretch_k=sk, bend_k=bk), 1.0, 0.8)
            assert d["root_x"] > 0.005, (
                f"stretch_k={sk} bend_k={bk}: 설근이 {d['root_x']:+.4f} 로 "
                "전진하지 않았다")


def test_body_height_coupling_is_NOT_established():
    """**(B) 설체 상승 억제는 아직 확립되지 않았다.** 그걸 여기 박아 둔다.

    측정: 혀끝을 앞·위로 밀 때 설체 높이 변화가 강성에 따라
    **부호까지 뒤집힌다** (+0.041 ~ −0.336 cm).

    | stretch_k | bend_k | 설체 높이 변화 |
    |---|---|---|
    | 0.3 | 1.0 | **+0.041** (오히려 올라간다) |
    | 1.0 | 0.35 | −0.139 |
    | 3.0 | 0.1 | −0.336 |

    즉 지금 모델에서 (B) 는 물리가 아니라 **내가 고른 강성**이 정한다.
    강성을 데이터로 정하기 전에는 (B) 를 근거로 아무것도 지어서는 안 된다.

    이 검사가 실패한다면 부호가 안정된 것이므로 좋은 소식이다 — 그때
    docs/PLAN_TONGUE2D.md §5 의 G2 항목과 이 설명을 같이 갱신하라.
    """
    signs = set()
    for sk, bk in ((0.3, 1.0), (1.0, 0.35), (3.0, 0.1)):
        d = _move_tip(TongueShape(stretch_k=sk, bend_k=bk), 1.0, 0.8)
        signs.add(d["body_y"] > 0)
    assert len(signs) == 2, (
        "설체 높이 변화의 부호가 강성에 무관해졌다 — (B) 가 확립된 것이라면 "
        "계획서 §5 의 G2 조건과 이 검사를 갱신하라")


def test_incompressibility_is_nearly_redundant_here():
    """**비압축성 구속이 결합의 원인이 아니다.** 그것도 박아 둔다.

    대조군(넓이를 안 지킴)이 거의 같은 결합을 낸다: 설체 −0.125 vs −0.139,
    설근 +0.065 vs +0.062. 넓이도 저절로 2.3 % 밖에 안 변한다.

    이유는 물리적으로 자연스럽다 — 변이 잘 안 늘어나면 넓이도 잘 안 변한다.
    즉 결합을 만드는 것은 **연결성**이지 넓이 보존이 아니다.
    처음에는 '비압축성이 Hwang 의 결합을 낳는다' 고 적으려 했는데, 대조군이
    그 주장을 지지하지 않았다.
    """
    sh = TongueShape()
    locked = _move_tip(sh, 1.0, 0.8, lock_area=True)
    free = _move_tip(sh, 1.0, 0.8, lock_area=False)
    assert abs(free["root_x"] - locked["root_x"]) < 0.02, (
        "대조군과 차이가 커졌다 — 비압축성의 역할이 달라졌다면 설명을 갱신하라")
    assert abs(free["area"] - free["area0"]) / free["area0"] < 0.06, (
        "구속 없이도 넓이가 거의 안 변한다는 전제가 깨졌다")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
