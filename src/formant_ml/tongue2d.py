"""2D 혀 — 비압축성 연속체. 혀몸·설근은 손잡이가 아니라 **결과**다.

왜 2D 인가
----------
Hwang, Charles & Lulich (2019), 서울말 여성 5 명 3D 초음파:

> "혀몸 위치와 설근 전진은 대체로 **관상 폐쇄의 위치에 의해 결정된다.**"

치음 폐쇄는 혀 전체가 앞으로 나가야 하고, 권설 폐쇄도 혀몸이 앞으로 가야
혀끝이 말릴 수 있다 — 그 결과 혀몸이 다른 위치만큼 올라갈 수 없다.
**TB·TR 은 자유 파라미터가 아니다.**

혀끝만 있는 1D 모델(`dsp/tongue.py`)은 이 구속을 표현할 수 없다. 혀몸이
없으니 TB·TR 을 넣으려면 별도 손잡이로 넣어야 하고, 그러면 문헌이 아니라고
한 것을 그대로 하게 된다.

여기서는 혀를 **비압축성 2D 몸**으로 둔다. 물이 든 주머니처럼, 한쪽을 밀면
다른 쪽이 반드시 물러난다. 근육 모델을 먼저 넣지 않는 이유는, 그러면
파라미터가 폭발하는데 그걸 검증할 데이터가 우리에게 없기 때문이다.

무엇이 재현되었고 무엇이 안 되었나 (측정 결과)
------------------------------------------------
Hwang 의 서술에는 사실 **두 개의 주장**이 들어 있고, 둘의 처지가 다르다.

  (A) 혀끝이 앞으로 가면 **설근이 전진한다** — **재현된다.**
      강성을 9 가지로 바꿔도 부호가 유지된다 (+0.013 ~ +0.087 cm).

  (B) 혀끝이 앞으로 가면 **설체 상승이 억제된다** — **재현되지 않았다.**
      같은 조작에서 설체 높이 변화가 강성에 따라 부호까지 뒤집힌다
      (+0.041 ~ −0.336 cm). 지금 모델에서 (B) 를 정하는 것은 물리가 아니라
      내가 고른 강성값이다.

그리고 **비압축성은 결합의 원인이 아니었다.** 넓이 구속을 끈 대조군이 거의
같은 결합을 낸다 (설체 −0.125 vs −0.139, 설근 +0.065 vs +0.062). 변이 잘 안
늘어나면 넓이도 저절로 거의 안 변하기(2.3 %) 때문이다. 결합을 만드는 것은
넓이 보존이 아니라 **표면의 연결성**이다.

처음에 이 파일은 "비압축성 하나로 Hwang 의 관찰이 재현된다" 고 적을 작정이
었다. 대조군이 그 주장을 지지하지 않았으므로 주장을 내린다. 세 결론 모두
`tests/test_tongue2d.py` 에 검사로 박혀 있어서, 나중에 사정이 바뀌면 검사가
먼저 깨진다.

따라서 지금 상위 모듈이 **써도 되는 것은 (A) 뿐이다.** (B) 를 근거로 무언가를
지으려면 먼저 강성을 화자 데이터로 정해야 한다 (계획서 §5 G2).

무엇이 정확하고 무엇이 아닌가
-----------------------------
* **면적 보존은 정확하다.** 다각형 넓이를 신발끈 공식으로 정확히 계산하고,
  등식 구속으로 건다(벌점이 아니다). 벌점은 "거의 보존" 이라 어디까지
  안 지켜지는지가 가중치에 숨는다.
* **탄성은 모형이고, 그 세부가 답을 바꾼다.** 변끼리의 스프링과 굽힘 저항으로
  둔다. 실제 혀는 비선형 초탄성체이고 근육이 능동적이다. 원래 의도는 '탄성의
  세부가 결론을 바꾸지 않는다' 였는데, 위 (B) 에서 **바뀐다.** 그래서 이
  모듈에서 나온 값 중 강성에 의존하는 것은 결론으로 쓰지 않는다.
* **면적함수 환산은 여기 없다.** 중시상면 거리에서 단면적으로 가는 것은
  구개 폭이 필요하고, 그건 해부 자료 문제다. 이 모듈은 **혀 형상까지**만
  낸다.

단위는 cm. 이 레포의 다른 모듈과 같다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TongueShape:
    """혀 윤곽의 초기 형상과 물성.

    윤곽은 **닫힌 다각형**이다: 위쪽(구강면)이 조음에 쓰이는 표면이고,
    아래쪽은 구강저·설골 쪽 경계다. 닫혀 있어야 넓이가 정의되고, 넓이가
    정의되어야 비압축성을 등식으로 걸 수 있다.
    """
    n_upper: int = 13           # 위쪽(구강면) 노드 수. 앞이 혀끝.
    n_lower: int = 7            # 아래쪽 경계 노드 수
    length_cm: float = 7.0      # 혀의 전후 길이
    height_cm: float = 3.2      # 혀의 최대 높이
    stretch_k: float = 1.0      # 변 길이 유지 강성
    bend_k: float = 0.35        # 굽힘 저항 (표면이 각지지 않게)
    dtype: torch.dtype = torch.float64

    def initial(self) -> torch.Tensor:
        """초기 노드 좌표 (M, 2). x 는 뒤->앞, y 는 아래->위.

        타원 반쪽을 위 표면으로, 완만한 곡선을 아래 경계로 둔 중립 형상이다.
        절대 좌표에 뜻은 없다 — 구개와의 관계는 상위 모듈이 정한다.
        """
        t = torch.linspace(0.0, 1.0, self.n_upper, dtype=self.dtype)
        ux = t * self.length_cm
        uy = self.height_cm * torch.sin(torch.pi * t) ** 0.7
        s = torch.linspace(1.0, 0.0, self.n_lower + 2, dtype=self.dtype)[1:-1]
        lx = s * self.length_cm
        ly = 0.25 * self.height_cm * torch.sin(torch.pi * s) * 0.4
        return torch.stack([torch.cat([ux, lx]), torch.cat([uy, ly])], dim=-1)


def polygon_area(p: torch.Tensor) -> torch.Tensor:
    """닫힌 다각형의 넓이 (신발끈 공식). p: (..., M, 2) -> (...)

    **정확한 식이다.** 비압축성을 이 값의 등식으로 걸기 때문에, 근사로 두면
    '얼마나 안 지켜지는지' 가 가중치 속에 숨는다.
    """
    x, y = p[..., 0], p[..., 1]
    return 0.5 * (x * y.roll(-1, dims=-1) - x.roll(-1, dims=-1) * y).sum(-1).abs()


def _edge_lengths(p: torch.Tensor) -> torch.Tensor:
    return (p.roll(-1, dims=-2) - p).pow(2).sum(-1).sqrt()


def _second_difference(p: torch.Tensor) -> torch.Tensor:
    d = p.roll(-1, dims=-2) - p
    return d.roll(-1, dims=-2) - d


def elastic_energy(p: torch.Tensor, rest_len: torch.Tensor,
                   rest_bend: torch.Tensor, shape: TongueShape) -> torch.Tensor:
    """변 길이와 굽힘의 탄성 에너지. 최소화 대상이다.

    비압축성만으로는 형상이 정해지지 않는다(넓이가 같은 모양은 무한히 많다).
    탄성이 그중 **자연스러운 것**을 고른다. 이 항이 없으면 노드가 뒤엉킨다.

    **굽힘은 곡률 자체가 아니라 쉬는 곡률로부터의 이탈을 벌해야 한다.**
    곡률 자체를 벌하면 에너지 최소화가 혀를 펴 버려서, 혀끝을 전혀 안 움직인
    경우에도 설체가 0.73 cm 내려앉는다(실제로 그렇게 나왔다). 그러면 관찰된
    '결합' 이 비압축성 때문인지 굽힘 항 때문인지 못 가른다.
    쉬는 형상이 에너지의 최소여야 기준선이 깨끗하다.
    """
    stretch = (_edge_lengths(p) - rest_len).pow(2).sum(-1)
    bend = (_second_difference(p) - rest_bend).pow(2).sum(-1).sum(-1)
    return shape.stretch_k * stretch + shape.bend_k * bend


def solve_shape(shape: TongueShape, handles: dict[int, torch.Tensor],
                p0: torch.Tensor | None = None, area0: torch.Tensor | None = None,
                steps: int = 400, lr: float = 0.02,
                area_tol: float = 1e-6) -> torch.Tensor:
    """손잡이 노드를 지정한 곳에 두고, 나머지를 **넓이를 지킨 채** 푼다.

    handles: {노드 번호: (2,) 목표 좌표}.  반환 (M, 2).

    방법은 벌점이 아니라 **투영**이다. 매 반복에서 탄성 에너지를 한 걸음
    내린 뒤, 넓이가 정확히 `area0` 이 되도록 형상을 법선 방향으로 되민다.
    벌점으로 하면 가중치에 따라 넓이가 조금씩 어긋나고, 그 '조금' 이
    Hwang 의 결합을 만드는 바로 그 양이라 결론이 가중치에 딸려 온다.
    """
    p = (shape.initial() if p0 is None else p0.clone()).to(shape.dtype)
    rest = _edge_lengths(p)
    rest_bend = _second_difference(p)
    if area0 is None:
        area0 = polygon_area(p)
    idx = torch.tensor(sorted(handles), dtype=torch.long)
    tgt = torch.stack([handles[int(i)].to(shape.dtype) for i in idx])

    q = p.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(steps):
        e = elastic_energy(q, rest, rest_bend, shape)
        # 손잡이는 '목표에 가깝게' 가 아니라 **그 자리에** 있어야 한다.
        # 아래에서 매 걸음 직접 덮어쓰므로 여기서는 에너지만 최소화한다.
        opt.zero_grad(set_to_none=True)
        e.backward()
        opt.step()
        with torch.no_grad():
            q[idx] = tgt                                  # 손잡이 고정 (하드)
            _project_area(q, area0, idx, area_tol)
    return q.detach()


def _project_area(q: torch.Tensor, area0: torch.Tensor, fixed: torch.Tensor,
                  tol: float, max_iter: int = 60) -> None:
    """넓이를 정확히 `area0` 으로 되미는 투영 (제자리 수정).

    넓이의 노드별 기울기 방향으로 최소한만 움직인다(가장 가까운 해).
    손잡이 노드는 움직이지 않는다 — 그게 '지정한 자리에 둔다' 의 뜻이다.
    """
    mask = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
    mask[fixed] = False
    for _ in range(max_iter):
        a = polygon_area(q)
        err = a - area0
        if err.abs() <= tol * area0:
            return
        x, y = q[:, 0], q[:, 1]
        gx = 0.5 * (y.roll(-1) - y.roll(1))
        gy = 0.5 * (x.roll(1) - x.roll(-1))
        g = torch.stack([gx, gy], dim=-1)
        sign = torch.sign(0.5 * (x * y.roll(-1) - x.roll(-1) * y).sum())
        g = g * sign
        g[~mask] = 0.0
        denom = (g * g).sum()
        if denom <= 1e-18:
            return
        q -= (err / denom) * g


def tongue_metrics(p: torch.Tensor, shape: TongueShape) -> dict[str, float]:
    """Hwang 이 잰 양들에 대응하는 형상 지표.

    * `tip_x`, `tip_y` — 혀끝(앞쪽 끝 노드)
    * `body_y` — 설체 높이. 위 표면 중간 1/3 의 평균 높이
    * `root_x` — 설근 전진. 뒤쪽 1/4 노드의 평균 x
    """
    up = p[:shape.n_upper]
    n = shape.n_upper
    return {
        "tip_x": float(up[-1, 0]), "tip_y": float(up[-1, 1]),
        "body_y": float(up[n // 3: 2 * n // 3, 1].mean()),
        "root_x": float(up[: max(1, n // 4), 0].mean()),
        "area": float(polygon_area(p)),
    }
