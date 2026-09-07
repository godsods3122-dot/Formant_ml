"""감정 토큰 레지스트리 — 학습 중 동적으로 생기고, 우리가 식별·통제·분리할 수 있다.

토큰이 무엇인가
---------------
토큰 = **물리 파라미터 공간의 델타** (P 차원) + 메타데이터.
파형도, 임베딩 블랙박스도 아니다. "화났을 때 한숨" 토큰이 생겼다면 그 내용은
`p_sub` 가 잠깐 오르고 `adduction` 이 떨어지고 `tilt` 가 내려가는 **숫자 열**이고,
그래서 (a) 사람이 읽을 수 있고, (b) 스크립트의 어느 구간에 적용됐는지 마커로 남고,
(c) 후처리에서 그 구간만 빼거나(제거 = 델타를 다시 빼면 된다) 다른 토큰으로
바꿀 수 있다.

생명주기
--------
    propose (학습 중 잔차가 큰 문맥의 델타를 군집) -> spawn (이름·감정·사유 기록)
    -> use (usage_count) -> tag / rename / freeze / exclude -> prune (지원 부족)

통제 손잡이
-----------
* `budget`      : 최대 토큰 수. 넘으면 spawn 이 거부된다.
* `frozen`      : 학습이 델타를 더 못 바꾼다.
* `excluded`    : 합성에서 적용되지 않는다 (있지만 쓰지 않는 상태).
* `strength`    : 적용 세기 0~1 (스크립트별).
* `allowed_params`: 토큰이 건드릴 수 있는 파라미터의 화이트리스트(예: 포먼트는 금지).

학습 측 (`TokenTable`)
----------------------
nn.Module. 델타 행렬이 파라미터이고 행을 동적으로 늘릴 수 있다. 순전파는
프레임별 토큰 가중치 (B,T,M) 와 델타 (M,P) 의 곱 — 그래서 델타에 대해 미분가능하고
**제어열에 더해지는 것 외에 아무것도 못 한다**. 이 구조적 제약이 곧 통제 가능성이다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .control import INDEX, PARAM_NAMES, PARAMS, ControlTrack


@dataclass
class Token:
    id: int
    name: str
    emotion: str
    delta: list[float]                       # P 차원, 파라미터 단위(로그 파라미터는 로그 비율)
    reason: str = ""                         # 어떤 문맥/손실에서 생겼나
    created_step: int = 0
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    frozen: bool = False
    excluded: bool = False
    tags: list[str] = field(default_factory=list)
    support: int = 0                         # 군집 크기 (증거의 양)

    def describe(self, top: int = 5) -> str:
        d = np.asarray(self.delta)
        order = np.argsort(-np.abs(d))[:top]
        parts = [f"{PARAM_NAMES[i]}{'x' if PARAMS[PARAM_NAMES[i]].log else ''}"
                 f"{np.exp(d[i]):.2f}" if PARAMS[PARAM_NAMES[i]].log else
                 f"{PARAM_NAMES[i]}{d[i]:+.3g}" for i in order if abs(d[i]) > 1e-9]
        return f"[{self.id}] {self.name} ({self.emotion}): " + ", ".join(parts)


class TokenRegistry:
    def __init__(self, budget: int = 64, allowed_params: list[str] | None = None):
        self.tokens: dict[str, Token] = {}
        self.budget = budget
        self.allowed = set(allowed_params) if allowed_params else set(PARAM_NAMES)
        self._next = 0

    # ----------------------------------------------------------- 생성/조회
    def spawn(self, delta, emotion: str, reason: str = "", name: str | None = None,
              step: int = 0, support: int = 0, tags=()) -> Token:
        if len(self.tokens) >= self.budget:
            raise RuntimeError(f"token budget {self.budget} exhausted")
        d = np.zeros(len(PARAM_NAMES))
        d[:] = np.asarray(delta, dtype=np.float64)
        for i, n in enumerate(PARAM_NAMES):           # 화이트리스트 밖은 0 으로
            if n not in self.allowed:
                d[i] = 0.0
        if name is None:
            kind = self._guess_kind(d)
            k = sum(1 for t in self.tokens.values() if t.emotion == emotion)
            name = f"{emotion}_{kind}_{k:02d}"
        if name in self.tokens:
            raise KeyError(f"token {name} exists")
        tok = Token(self._next, name, emotion, d.tolist(), reason, step,
                    support=support, tags=list(tags))
        self.tokens[name] = tok
        self._next += 1
        return tok

    @staticmethod
    def _guess_kind(d: np.ndarray) -> str:
        """델타의 주성분에서 사람이 읽을 이름 조각을 만든다."""
        g = lambda n: d[INDEX[n]]
        if g("p_sub") > 0.5 and g("adduction") < -0.1:
            return "sigh"
        if g("p_sub") < -0.5:
            return "soft"
        if g("adduction") < -0.15:
            return "breathy"
        if g("adduction") > 0.15:
            return "pressed"
        if g("f0_scale") > 0.1:
            return "high"
        if g("f0_scale") < -0.1:
            return "low"
        if g("tension") > 0.1:
            return "tense"
        return "misc"

    def get(self, name: str) -> Token:
        return self.tokens[name]

    def by_emotion(self, emotion: str) -> list[Token]:
        return [t for t in self.tokens.values() if t.emotion == emotion]

    def active(self) -> list[Token]:
        return [t for t in self.tokens.values() if not t.excluded]

    # ----------------------------------------------------------- 통제
    def freeze(self, name: str, on: bool = True) -> None:
        self.tokens[name].frozen = on

    def exclude(self, name: str, on: bool = True) -> None:
        self.tokens[name].excluded = on

    def rename(self, old: str, new: str) -> None:
        t = self.tokens.pop(old); t.name = new; self.tokens[new] = t

    def tag(self, name: str, *tags: str) -> None:
        self.tokens[name].tags.extend(t for t in tags if t not in self.tokens[name].tags)

    def prune(self, min_support: int = 0, min_usage: int = 0) -> list[str]:
        gone = [n for n, t in self.tokens.items()
                if not t.frozen and (t.support < min_support or t.usage_count < min_usage)]
        for n in gone:
            del self.tokens[n]
        return gone

    # ----------------------------------------------------------- 적용/분리
    def apply(self, track: ControlTrack, name: str, t0: float, t1: float,
              strength: float = 1.0, ramp_ms: float = 30.0) -> ControlTrack:
        """토큰 델타를 [t0, t1) 구간에 더하고 **마커를 남긴다**."""
        tok = self.tokens[name]
        if tok.excluded:
            return track
        env = self._envelope(track, t0, t1, ramp_ms) * strength
        d = np.asarray(tok.delta)
        for i, n in enumerate(PARAM_NAMES):
            if d[i] == 0.0:
                continue
            col = track.values[:, i]
            if PARAMS[n].log:
                on = col > 0
                col[on] = col[on] * np.exp(d[i] * env[on])
            else:
                col += d[i] * env
        track.events.append(dict(kind="token", name=name, t0=t0, t1=t1,
                                 strength=strength, ramp_ms=ramp_ms, token_id=tok.id))
        tok.usage_count += 1
        track.clamp()
        return track

    def strip(self, track: ControlTrack, names: list[str] | None = None) -> ControlTrack:
        """마커를 읽어 토큰 효과를 되돌린다(후처리 분리). names=None 이면 전부."""
        keep = []
        for ev in track.events:
            if ev.get("kind") != "token" or (names is not None and ev["name"] not in names):
                keep.append(ev); continue
            tok = self.tokens[ev["name"]]
            env = self._envelope(track, ev["t0"], ev["t1"], ev["ramp_ms"]) * ev["strength"]
            d = np.asarray(tok.delta)
            for i, n in enumerate(PARAM_NAMES):
                if d[i] == 0.0:
                    continue
                col = track.values[:, i]
                if PARAMS[n].log:
                    on = col > 0
                    col[on] = col[on] * np.exp(-d[i] * env[on])
                else:
                    col -= d[i] * env
        track.events = keep
        return track

    @staticmethod
    def _envelope(track: ControlTrack, t0: float, t1: float, ramp_ms: float) -> np.ndarray:
        t = (np.arange(track.n_frames) + 0.5) * track.frame_ms / 1000.0
        r = max(ramp_ms, 1e-3) / 1000.0
        up = np.clip((t - t0) / r, 0, 1); dn = np.clip((t1 - t) / r, 0, 1)
        e = np.minimum(up, dn)
        return e * e * (3 - 2 * e)

    def markers(self, track: ControlTrack) -> list[dict]:
        return [e for e in track.events if e.get("kind") == "token"]

    # ----------------------------------------------------------- 저장
    def to_json(self) -> str:
        return json.dumps(dict(budget=self.budget, allowed=sorted(self.allowed),
                               next=self._next,
                               tokens=[asdict(t) for t in self.tokens.values()]),
                          ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "TokenRegistry":
        d = json.loads(s)
        reg = cls(d["budget"], d["allowed"])
        reg._next = d["next"]
        for t in d["tokens"]:
            tok = Token(**t); reg.tokens[tok.name] = tok
        return reg

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "TokenRegistry":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


# ---------------------------------------------------------------- 학습 측
class TokenTable(nn.Module):
    """미분가능 토큰 델타 표. 행(토큰)을 동적으로 늘린다.

    forward(weights (B,T,M)) -> 제어 델타 (B,T,P). 제어열에 더하는 것 외의 경로가 없다.
    `mask` 가 frozen/excluded 를 구현한다: excluded 는 출력 0, frozen 은 기울기 0.
    """

    def __init__(self, registry: TokenRegistry):
        super().__init__()
        self.registry = registry
        P = len(PARAM_NAMES)
        M = len(registry.tokens)
        init = torch.tensor([t.delta for t in registry.tokens.values()], dtype=torch.float32) \
            if M else torch.zeros(0, P)
        self.delta = nn.Parameter(init)
        self.names = list(registry.tokens.keys())
        self.register_buffer("allowed", torch.tensor(
            [1.0 if n in registry.allowed else 0.0 for n in PARAM_NAMES]))

    @property
    def n_tokens(self) -> int:
        return self.delta.shape[0]

    def add_token(self, name: str, delta=None, **spawn_kw) -> int:
        """레지스트리에 spawn 하고 표에도 행을 붙인다. 반환 행 인덱스."""
        P = len(PARAM_NAMES)
        d = torch.zeros(P) if delta is None else torch.as_tensor(delta, dtype=torch.float32)
        self.registry.spawn(d.numpy(), name=name, **spawn_kw)
        with torch.no_grad():
            self.delta = nn.Parameter(torch.cat([self.delta.data, d.unsqueeze(0)], 0))
        self.names.append(name)
        return self.n_tokens - 1

    def _masks(self):
        fro = torch.tensor([1.0 if self.registry.tokens[n].frozen else 0.0 for n in self.names])
        exc = torch.tensor([0.0 if self.registry.tokens[n].excluded else 1.0 for n in self.names])
        return fro.to(self.delta.device), exc.to(self.delta.device)

    def effective_delta(self) -> torch.Tensor:
        fro, exc = self._masks()
        d = self.delta * self.allowed
        d = fro[:, None] * d.detach() + (1 - fro)[:, None] * d       # frozen: 기울기 차단
        return d * exc[:, None]

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        if self.n_tokens == 0:
            return torch.zeros(*weights.shape[:2], len(PARAM_NAMES), device=weights.device)
        return weights @ self.effective_delta()

    def sync_to_registry(self) -> None:
        for i, n in enumerate(self.names):
            self.registry.tokens[n].delta = self.delta[i].detach().cpu().numpy().tolist()


class TokenDiscovery:
    """학습 중 토큰 제안. 잔차(목표 − 예측)가 큰 프레임의 **제어 델타**를 모아 군집한다.

    `observe(delta, emotion, loss)` 로 프레임을 쌓고, `propose()` 가 감정별 k-평균
    (numpy) 으로 중심을 찾아 지원이 충분한 군집만 토큰 후보로 돌려준다.
    우리가 통제하는 손잡이: loss_quantile(어느 정도 잔차부터 후보인가),
    min_support(군집 최소 크기), max_per_emotion.
    """

    def __init__(self, loss_quantile: float = 0.8, min_support: int = 50,
                 max_per_emotion: int = 4, k: int = 4, seed: int = 0):
        self.q, self.min_support, self.max_per = loss_quantile, min_support, max_per_emotion
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.buf: list[tuple[np.ndarray, str, float]] = []

    def observe(self, delta: np.ndarray, emotion: str, loss: float) -> None:
        self.buf.append((np.asarray(delta, dtype=np.float64), emotion, float(loss)))

    def _kmeans(self, X: np.ndarray, k: int, iters: int = 30):
        k = min(k, len(X))
        C = X[self.rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
            for j in range(k):
                if (lab == j).any():
                    C[j] = X[lab == j].mean(0)
        return C, lab

    def propose(self) -> list[dict]:
        if not self.buf:
            return []
        losses = np.array([b[2] for b in self.buf])
        thr = np.quantile(losses, self.q)
        out = []
        for emo in sorted({b[1] for b in self.buf}):
            X = np.stack([b[0] for b in self.buf if b[1] == emo and b[2] >= thr]) \
                if any(b[1] == emo and b[2] >= thr for b in self.buf) else None
            if X is None or len(X) < self.min_support:
                continue
            C, lab = self._kmeans(X, self.k)
            cands = []
            for j in range(len(C)):
                sup = int((lab == j).sum())
                if sup >= self.min_support and np.abs(C[j]).max() > 1e-6:
                    cands.append(dict(delta=C[j], emotion=emo, support=sup,
                                      reason=f"residual>={thr:.3g} cluster {j}"))
            cands.sort(key=lambda c: -c["support"])
            out.extend(cands[: self.max_per])
        return out

    def clear(self) -> None:
        self.buf.clear()
