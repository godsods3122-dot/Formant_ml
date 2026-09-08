"""제어 트랙 — "ms 단위로 물리 factor 를 조정하는 스크립트" 의 자료구조.

LLM(또는 사람, 또는 음소 제스처 라이브러리)이 만드는 것은 **키프레임**이다:

    {"t": 0.120, "f1": 340, "f2": 1400, "p_sub": 7.5, ...}

엔진은 그것을 (1) 제어 프레임률(기본 1 ms)로 보간하고, (2) 필터 계수는 다시
샘플률로 선형 보간한다. 두 번째 보간이 v2 의 핵심이다 — 계수가 샘플마다 움직여야
위상 계단이 생기지 않는다.

파라미터 이름·단위·범위는 `PARAMS` 한 곳에 있다. 스크립트 스키마, 신경망 헤드의
Bounded 범위, 토큰 델타의 클램프가 전부 이 표를 참조한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class ParamSpec:
    name: str
    unit: str
    lo: float
    hi: float
    default: float
    log: bool = False          # 로그 영역에서 보간/학습하는 양 (주파수, 면적)
    doc: str = ""


N_FORMANTS = 8
N_ALLPASS = 3

_P: list[ParamSpec] = [
    # --- 성문 (압력 구동) -------------------------------------------------
    ParamSpec("p_sub", "cmH2O", 0.0, 20.0, 0.0, False, "성문하압. 0 이면 침묵."),
    ParamSpec("adduction", "0-1", 0.0, 1.0, 0.6, False, "내전. 0 벌림(속삭임/기식) ~ 1 압착"),
    ParamSpec("tension", "0-1", 0.0, 1.0, 0.5, False, "성대 긴장. F0 와 Rd 를 움직인다"),
    ParamSpec("f0_target", "Hz", 50.0, 800.0, 0.0, True, "0 이 아니면 tension 대신 F0 직접 지정"),
    ParamSpec("f0_scale", "ratio", 0.5, 2.0, 1.0, True, "F0 배율(억양). 토큰이 주로 만진다"),
    ParamSpec("rd_offset", "Rd", -1.5, 1.5, 0.0, False, "LF Rd 오프셋(음질: -압착 / +기식)"),
    ParamSpec("tilt", "dB/oct", -12.0, 12.0, 2.0, False, "소스 추가 기울기 @1 kHz (실측 적합 +2)"),
    ParamSpec("jitter", "ratio", 0.0, 0.05, 0.004, False, "주기 요동"),
    ParamSpec("shimmer", "ratio", 0.0, 0.2, 0.03, False, "진폭 요동"),
    ParamSpec("aspiration", "0-1", 0.0, 1.0, 1.0, False, "성문 난류 배율(물리량 위에 곱)"),
    # --- 성도: 포먼트 + 위상차 -------------------------------------------
    *[ParamSpec(f"f{k}", "Hz", 100.0 * k, 12000.0, 0.0, True, f"포먼트 {k} 주파수")
      for k in range(1, N_FORMANTS + 1)],
    *[ParamSpec(f"bw{k}", "Hz", 20.0, 3000.0, 0.0, True, f"포먼트 {k} 대역폭 (0 이면 물리 기본)")
      for k in range(1, N_FORMANTS + 1)],
    *[ParamSpec(f"ap{k}_f", "Hz", 100.0, 12000.0, 0.0, True, f"올패스 {k} 중심(0 이면 끔)")
      for k in range(1, N_ALLPASS + 1)],
    *[ParamSpec(f"ap{k}_r", "0-1", 0.0, 0.98, 0.0, False, f"올패스 {k} 반지름")
      for k in range(1, N_ALLPASS + 1)],
    ParamSpec("tract_gain", "ratio", 0.0, 4.0, 1.0, False, "성도 출력 배율(폐쇄 감쇠 등)"),
    # --- 협착 / 마찰 노이즈 (물리) ---------------------------------------
    ParamSpec("a_c", "cm2", 0.0, 8.0, 3.0, False, "구강 최협착 단면적. 작을수록 마찰"),
    ParamSpec("c_place", "0-1", 0.0, 1.0, 0.9, False, "협착 위치 (0 성문 ~ 1 입술). 앞공동 길이"),
    ParamSpec("front_len", "cm", 0.3, 6.0, 0.0, True, "앞공동 길이 직접 지정(0 이면 c_place 로). 0↔값은 보간하지 않는다"),
    ParamSpec("obstacle", "0-1", 0.0, 1.0, 0.0, False, "제트가 장애물(앞니)을 때리는 정도(다이폴)"),
    ParamSpec("fric_gain", "ratio", 0.0, 256.0, 1.0, False,
              "마찰 노이즈 배율(물리량 위에 곱). 상한 4 -> 파찰음 10 dB 잘림, "
              "64 -> 남성 /사/ 복사합성이 7 dB 모자람(둘 다 실측). 지금 256 = 48 dB"),
    ParamSpec("back_leak", "0-1", 0.0, 1.0, 0.3, False, "마찰음이 뒤공동/성도 전체로 새는 비율"),
    # --- 측지 / 비강 ---------------------------------------------------------
    ParamSpec("lat_z1", "Hz", 500.0, 8000.0, 0.0, True, "측지 영점 1 (0 이면 끔)"),
    ParamSpec("lat_z2", "Hz", 500.0, 8000.0, 0.0, True, "측지 영점 2"),
    ParamSpec("lat_bw", "Hz", 50.0, 3000.0, 300.0, True, "측지 영점 대역폭"),
    ParamSpec("lat_mix", "0-1", 0.0, 1.0, 0.0, False,
              "측지 영점의 깊이 0~1. **주파수를 0 으로 껐다 켜면 안 된다** — 로그 파라미터는"
              " 영차 유지라 한 프레임에 뛰고, 그 계수 도약이 클릭이 된다(측정: 8 배 스파이크)."),
    ParamSpec("velum", "0-1", 0.0, 1.0, 0.0, False, "연구개 개방(비강 분기의 출력 비중)"),
    ParamSpec("oral_open", "0-1", 0.0, 1.0, 1.0, False,
              "구강 방사 개방도. 0 = 완전 폐쇄(비음·파열음의 폐쇄 구간)"),
    ParamSpec("nasal_f", "Hz", 150.0, 600.0, 280.0, True, "비강 제 1 극"),
    ParamSpec("nasal_f2", "Hz", 600.0, 2000.0, 1000.0, True, "비강 제 2 극"),
    ParamSpec("nasal_f3", "Hz", 1500.0, 4000.0, 2200.0, True, "비강 제 3 극"),
    ParamSpec("nasal_z", "Hz", 400.0, 4000.0, 1400.0, True,
              "구강 측지 영점 — 폐쇄 위치가 정한다 (ㅁ 낮고 ㅇ 높다)"),
    ParamSpec("nasal_damp", "ratio", 0.3, 3.0, 1.0, False, "비강 손실 배율 (대역폭 곱)"),
    ParamSpec("nasal_gain", "ratio", 0.0, 8.0, 1.0, False, "비강 분기의 출력 배율(폐쇄 위치별 차이)"),
    # --- 과도음 이벤트 / 잔차 ---------------------------------------------------
    ParamSpec("transient", "0-1", 0.0, 1.0, 0.0, False, "과도음 템플릿 트리거 세기(임펄스로 쓴다)"),
    ParamSpec("transient_id", "index", 0.0, 63.0, 0.0, False, "템플릿 번호"),
    ParamSpec("residual_mix", "0-1", 0.0, 1.0, 1.0, False, "잔차 보정 적용 비율"),
]
PARAMS: dict[str, ParamSpec] = {p.name: p for p in _P}
PARAM_NAMES: list[str] = [p.name for p in _P]
INDEX: dict[str, int] = {n: i for i, n in enumerate(PARAM_NAMES)}


def default_vector() -> np.ndarray:
    return np.array([p.default for p in _P], dtype=np.float64)


@dataclass
class ControlTrack:
    """프레임률 제어열. values: (T, P) numpy. frame_ms: 프레임 간격."""
    values: np.ndarray
    frame_ms: float = 1.0
    events: list[dict] = field(default_factory=list)   # 샘플 정확도 과도음 이벤트
    # 성문 폐쇄 시각(초, 트랙 시작 기준). 분석이 채우고 복사합성 적합이 위상 고정에 쓴다.
    pulses: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 프레임별 유성 여부. 분석이 채운다 — 적합기가 마찰 이득을 따로 보정하는 데 쓴다.
    voiced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # 프레임별 마찰 여부(무성 + 고역 우세). 폐쇄와 마찰은 다르게 다뤄야 한다.
    fricative: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[:, INDEX[name]]

    def __setitem__(self, name: str, v) -> None:
        self.values[:, INDEX[name]] = v

    def seconds(self) -> float:
        return self.n_frames * self.frame_ms / 1000.0

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.values, dtype=dtype, device=device).unsqueeze(0)

    def clamp(self) -> "ControlTrack":
        lo = np.array([p.lo for p in _P]); hi = np.array([p.hi for p in _P])
        # 0 = "끔/기본" 인 로그 파라미터는 0 을 보존한다
        keep0 = (self.values == 0.0)
        self.values = np.clip(self.values, lo, hi)
        self.values[keep0] = 0.0
        return self


def _interp_keyframes(times: np.ndarray, vals: np.ndarray, t_grid: np.ndarray,
                      log: bool, smooth: str, base: float) -> np.ndarray:
    """키프레임 사이 보간. smooth: "linear" | "cosine" | "minjerk".

    규약:
      * 첫 키프레임 이전은 **기본값**(base). 마지막 이후는 마지막 값 유지.
      * 로그 파라미터에서 0 은 "끔/기본" 이다. 0 과 양수 사이는 보간하지 않고
        **다음 키프레임까지 이전 값을 유지**한다(영차 유지). 0 을 지나며 대역폭이
        0.5 Hz 가 되어 극이 단위원에 붙는 사고(첫 렌더에서 3e10 으로 폭주)를 막는다.
    """
    out = np.full_like(t_grid, base)
    if len(times) == 1:
        out[t_grid >= times[0]] = vals[0]
        return out
    idx = np.clip(np.searchsorted(times, t_grid, side="right") - 1, 0, len(times) - 2)
    t0, t1 = times[idx], times[idx + 1]
    w = np.clip((t_grid - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    if smooth == "cosine":
        w = 0.5 - 0.5 * np.cos(np.pi * w)
    elif smooth == "minjerk":
        w = w ** 3 * (10.0 - 15.0 * w + 6.0 * w * w)
    v0, v1 = vals[idx], vals[idx + 1]
    if log:
        ok = (v0 > 0) & (v1 > 0)
        seg = np.where(ok, np.exp(np.log(np.maximum(v0, 1e-9)) * (1 - w)
                                  + np.log(np.maximum(v1, 1e-9)) * w),
                       np.where(w >= 1.0, v1, v0))
    else:
        seg = v0 * (1 - w) + v1 * w
    inside = t_grid >= times[0]
    out[inside] = seg[inside]
    return out


def track_from_keyframes(keyframes: list[dict], seconds: float | None = None,
                         frame_ms: float = 1.0, smooth: str = "minjerk",
                         base: np.ndarray | None = None) -> ControlTrack:
    """키프레임 리스트 -> ControlTrack.

    각 키프레임은 {"t": 초, 이름: 값, ...}. 어떤 파라미터가 특정 키프레임에 없으면
    그 파라미터는 **자기 키프레임들 사이에서만** 보간된다(파라미터마다 독립
    시간축 — "F3 가 F1 보다 11 ms 먼저" 같은 대역별 시차를 그대로 쓸 수 있다).
    "event" 키가 있으면 과도음 이벤트로 별도 기록한다.
    """
    kf = sorted(keyframes, key=lambda k: k["t"])
    dur = seconds if seconds is not None else max(k["t"] for k in kf)
    n = max(1, int(round(dur * 1000.0 / frame_ms)))
    t_grid = (np.arange(n) + 0.5) * frame_ms / 1000.0
    vals = np.tile(default_vector() if base is None else base, (n, 1))
    events = []
    for name, spec in PARAMS.items():
        ts, vs = [], []
        for k in kf:
            if name in k:
                ts.append(k["t"]); vs.append(float(k[name]))
        if ts:
            vals[:, INDEX[name]] = _interp_keyframes(
                np.asarray(ts), np.asarray(vs), t_grid, spec.log, smooth,
                float(vals[0, INDEX[name]]))
    for k in kf:
        if "event" in k:
            ev = dict(k["event"]); ev["t"] = k["t"]; events.append(ev)
    return ControlTrack(vals, frame_ms, events).clamp()


def frames_to_samples(x: torch.Tensor, hop: int) -> torch.Tensor:
    """프레임률 (B, T, C) -> 샘플률 (B, T·hop, C) 선형 보간 (마지막 프레임은 유지).

    정수 인덱스 산술로 한다 — `F.interpolate(align_corners=True)` 는 좌표를
    s·(T)/(T·hop) 로 계산해 T 에 따라 반올림이 달라지고, 그 1e-4 의 좌표 오차가
    고 Q 공명기를 지나며 스트리밍/오프라인 불일치(1e-3)로 커졌다(측정).
    """
    b, t, c = x.shape
    n = t * hop
    s = torch.arange(n, device=x.device)
    i0 = torch.div(s, hop, rounding_mode="floor")
    w = ((s - i0 * hop).to(x.dtype) / hop).view(1, n, 1)
    i1 = (i0 + 1).clamp(max=t - 1)
    x0 = x[:, i0]                                   # (B, N, C)
    x1 = x[:, i1]
    return x0 + (x1 - x0) * w


def frames_to_samples_np(x: np.ndarray, hop: int) -> np.ndarray:
    """(T,) 또는 (T, C) -> (T·hop, ...) 선형 보간."""
    t = x.shape[0]
    src = np.arange(t + 1) * hop
    dst = np.arange(t * hop)
    xx = np.concatenate([x, x[-1:]], axis=0)
    if x.ndim == 1:
        return np.interp(dst, src, xx)
    return np.stack([np.interp(dst, src, xx[:, c]) for c in range(x.shape[1])], -1)
