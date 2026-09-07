"""복사합성 적합기 — 녹음을 정답으로 두고 물리 파라미터를 역추정한다.

    fitter = CopySynthFitter(eng, target, sr, init_track)
    rep = fitter.fit_staged()
    y   = fitter.render()

왜 이 방식인가
--------------
음절을 손으로 작곡하면 "지표는 맞는데 사람 소리가 아닌" 상태에서 빠져나올 길이 없다.
무엇이 틀렸는지 비교할 정답이 없기 때문이다. 복사합성에는 정답이 있다 — 원본 파형.
되합성이 원본과 다르면 **엔진이 틀린 것**이고, 어느 파라미터가 부족한지 바로 좁혀진다.

두 가지 일치율을 따로 보고한다
------------------------------
* **포락 일치**: 멜 대역에서 잰다. "포먼트·세기·잡음색이 맞는가". 조음 파라미터가
  옳은지를 재는 값이고, 사람이 듣는 음색에 대응한다.
* **정밀 일치**: 선형 다해상도 STFT 에서 잰다. 창이 길면 하모닉이 분해되므로 F0 가
  0.1 % 만 틀려도 고차 하모닉이 통째로 빗나가 값이 0 근처로 무너진다. 즉 이 값은
  **F0 궤적과 성문 펄스 위치까지 맞았는가**를 재는 훨씬 가혹한 값이다.

둘 다 `100 · (1 − ‖|S_t| − |S_p|‖_F / ‖|S_t|‖_F)` (스펙트럼 수렴도) 로 정의한다.

왜 성김에서 촘촘함으로 가는가
-----------------------------
긴 창부터 켜면 F0 에 대한 손실면이 하모닉 간격마다 골이 파인 톱니가 되어 Adam 이
가장 가까운 가짜 골에 갇힌다. 짧은 창(256)에서는 하모닉이 분해되지 않아 손실면이
매끈하다. 그래서 짧은 창으로 포락과 세기를 먼저 맞추고, 길이를 늘려 가며 하모닉을
잠근다. 각 단계는 앞 단계의 해에서 출발한다.

경계 조건
---------
* 잔차 보정(residual)은 끈다. 학습되지 않은 NN 이 물리 적합을 오염시킨다.
* 로그 파라미터에서 0 은 "끔" 이다. 시그모이드 재매개화로는 0 에 닿을 수 없으므로
  초기값이 0 인 로그 파라미터는 고정한다(켜고 끄기는 물리 결정이지 적합 대상이 아니다).
* 전역 오프셋을 먼저 맞춘다. 전체 수준이 20 dB 틀린 채로 프레임별 적합을 돌리면
  Adam 이 그 20 dB 를 프레임마다 따로 메우고, 제어열이 물리적으로 말이 안 되게 굳는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .control import INDEX, PARAMS, ControlTrack

FFT_SIZES = (256, 512, 1024, 2048, 4096)
MEL_FFT = 256              # 포락용 창 (5.3 ms @48 kHz) — 하모닉이 분해되지 않는다
DB_RANGE = 70.0            # 정점 아래 이만큼까지만 본다
# 성김 -> 촘촘함. 각 단계에서 켜는 창 크기.
STAGES = ((256, 512), (256, 512, 1024), (256, 512, 1024, 2048), FFT_SIZES)

# 전역 오프셋을 붙일 파라미터. **수준·기울기·잡음량만**이다.
# 포먼트에 전역 오프셋을 주면 최적화가 F4 를 F1 아래로 끌어내리는 식으로 스펙트럼을
# 맞춘다(실측: F1 887->248, F2 1550->3250, F4 5153->915). 대역 에너지는 맞지만
# 조음으로는 말이 안 되는 해다. 그런 자유도는 애초에 주지 않는다.
# `tract_gain` 은 여기 없다. 전역 수준은 `log_gain` 이 이미 닫힌 형태로 잡는데
# 둘 다 열어 두면 서로를 상쇄하며 떠돌고, 그 사이에서 `p_sub` 까지 끌려간다
# (실측: 보통 세기 모음인데 p_sub 가 17.6 cmH2O — 큰 소리로 외치는 값 — 로 갔다).
# `tract_gain` 은 폐쇄 감쇠 같은 **시간 변화**로만 쓴다.
GLOBAL_PARAMS = frozenset((
    "p_sub", "adduction", "rd_offset", "tilt", "aspiration",
    "fric_gain", "back_leak", "nasal_damp", "nasal_gain",
    "jitter", "shimmer", "bw1", "bw2", "bw3", "bw4",
))

# 사전(prior) 가중. 분석이 믿을 만한 양은 세게 묶고, 관측되지 않는 양은 기본값에
# 묶어 둔다. 약하게 푸는 것은 수준·음질 계열뿐이다.
PRIOR_W: dict[str, float] = {
    "f0_target": 40.0, "f1": 40.0, "f2": 40.0, "f3": 20.0, "f4": 10.0,
    "velum": 20.0, "oral_open": 20.0, "nasal_f": 20.0, "nasal_f2": 20.0,
    "nasal_f3": 20.0, "nasal_z": 20.0, "nasal_gain": 10.0, "nasal_damp": 10.0,
    "a_c": 10.0, "obstacle": 10.0, "front_len": 10.0, "back_leak": 5.0,
    "bw1": 2.0, "bw2": 2.0, "bw3": 2.0, "bw4": 2.0,
    # p_sub 는 세기의 물리량이다. 이득과 겹치는 방향으로 끌려가지 않게 세게 묶는다.
    "p_sub": 3.0, "adduction": 0.3, "tilt": 0.1, "aspiration": 0.1,
    "rd_offset": 0.3, "tract_gain": 0.3, "fric_gain": 0.3,
    # 지터는 하모닉 차수에 비례해 위상변조 지수가 커진다(β ∝ k). 그래서 고차 하모닉을
    # 통째로 뭉개 **고역 잡음 바닥을 혼자 결정한다** (실측: 지터 0.004 + 시머 0.03 이
    # 광대역 바닥을 33 dB 들어올렸고, 그게 7 kHz 위 +9~+20 dB 오차의 정체였다).
    # 적합 대상에서 빼 두면 다른 파라미터가 그 초과분을 메우려고 비틀린다.
    "jitter": 0.1, "shimmer": 0.1,
}
F_MARGIN_HZ = 120.0        # 인접 포먼트 최소 간격

# 프레임별 적합의 **한 걸음 크기**(raw 좌표). Adam 은 기울기 크기와 무관하게 lr 만큼
# 걷기 때문에, 모든 파라미터에 같은 lr 을 주면 F0 가 한 걸음에 11 % 씩 뛴다(실측:
# 그래서 2 단계가 52 % -> 8 % 로 무너졌다). 성문 위상은 누적합이라 F0 를 한 프레임
# 흔들면 그 뒤 전부가 흔들린다. 물리적으로 말이 되는 분해능으로 걸음을 잘라 준다.
STEP: dict[str, float] = {
    "f0_target": 0.15, "f1": 0.25, "f2": 0.25, "f3": 0.3, "f4": 0.3,
    "bw1": 0.5, "bw2": 0.5, "bw3": 0.5, "bw4": 0.5,
    "velum": 0.5, "oral_open": 0.5, "a_c": 0.5, "front_len": 0.3,
    "jitter": 0.5, "shimmer": 0.5,
    "nasal_f": 0.3, "nasal_f2": 0.3, "nasal_f3": 0.3, "nasal_z": 0.3,
}
STEP_DEFAULT = 1.0

# 제어 격자 성김 -> 촘촘함 (ms). 1 ms 부터 풀면 200×28 개의 자유도가 잡음을 좇는다.
GRID_MS = (20.0, 10.0, 5.0, 1.0)

DEFAULT_PARAMS = (
    "p_sub", "adduction", "f0_target", "rd_offset", "tilt", "aspiration",
    "jitter", "shimmer",
    "f1", "f2", "f3", "f4", "bw1", "bw2", "bw3", "bw4",
    "tract_gain", "a_c", "fric_gain", "obstacle", "back_leak", "front_len",
    "velum", "oral_open", "nasal_f", "nasal_f2", "nasal_f3", "nasal_z",
    "nasal_damp", "nasal_gain",
)


def _stft(x: torch.Tensor, n: int, win: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, n_fft=n, hop_length=n // 4, win_length=n, window=win,
                      center=True, return_complex=True, pad_mode="reflect")


def mel_bank(n_fft: int, fs: float, n_mels: int = 80,
             fmin: float = 60.0, fmax: float | None = None) -> torch.Tensor:
    """삼각 멜 필터뱅크 (n_mels, n_fft//2+1). 포락 일치율용."""
    fmax = fmax or fs / 2
    def hz2m(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def m2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m = np.linspace(hz2m(fmin), hz2m(fmax), n_mels + 2)
    f = m2hz(m)
    bins = np.fft.rfftfreq(n_fft, 1.0 / fs)
    w = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, ct, hi = f[i], f[i + 1], f[i + 2]
        left = (bins - lo) / max(ct - lo, 1e-9)
        right = (hi - bins) / max(hi - ct, 1e-9)
        w[i] = np.clip(np.minimum(left, right), 0.0, None)
        s = w[i].sum()
        if s > 0:
            w[i] /= s
    return torch.as_tensor(w, dtype=torch.float32)


@dataclass
class FitReport:
    env: float                         # 포락 일치율 % (멜 크기의 스펙트럼 수렴도)
    fine: float                        # 정밀 일치율 % (선형 다해상도)
    db: float                          # 멜 대역 평균 |오차| dB — 가장 읽기 쉬운 값
    per_size: dict[int, float]
    loss: float
    iters: int
    history: list[tuple[float, float]] = field(default_factory=list)

    def __str__(self) -> str:
        per = " ".join(f"{n}:{v:.1f}" for n, v in sorted(self.per_size.items()))
        return (f"포락 {self.env:5.2f}%  정밀 {self.fine:5.2f}%   평균오차 {self.db:.2f} dB  "
                f"(손실 {self.loss:.4f}, {self.iters} 회)  [{per}]")


class CopySynthFitter:
    def __init__(self, engine, target: np.ndarray, sr: int, init: ControlTrack,
                 params: tuple[str, ...] = DEFAULT_PARAMS,
                 lam_smooth: float = 3e-3, lam_prior: float = 1e-4,
                 phase_weight: float = 0.0, pulse_weight: float = 1.0,
                 n_mels: int = 48, device: str = "cpu"):
        self.eng = engine
        self.hop = engine.cfg.hop
        self.fs = engine.cfg.sample_rate
        self.device = device
        self.lam_smooth, self.lam_prior = lam_smooth, lam_prior
        self.phase_weight = phase_weight
        self.pulse_weight = pulse_weight
        self._last_pulse = float("nan")
        self.sizes = list(FFT_SIZES)

        if sr != self.fs:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), int(self.fs))
            target = resample_poly(target, self.fs // g, int(sr) // g)
        self.track = ControlTrack(init.values.copy(), init.frame_ms, list(init.events))
        self.track["residual_mix"] = 0.0
        n = self.track.n_frames * self.hop
        t = np.zeros(n)
        t[:min(n, len(target))] = target[:min(n, len(target))]
        self.target = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(0)

        base = torch.as_tensor(self.track.values, dtype=torch.float64, device=device)
        self.base = base
        self.names = [p for p in params
                      if not (PARAMS[p].log and float(base[:, INDEX[p]].abs().min()) == 0.0)]
        self.cols = torch.tensor([INDEX[p] for p in self.names], device=device)
        self.specs = [PARAMS[p] for p in self.names]
        u0 = torch.stack([self._to_raw(base[:, INDEX[p]], PARAMS[p]) for p in self.names], 1)
        self.u0 = u0.detach()
        self.scale = torch.tensor([STEP.get(n, STEP_DEFAULT) for n in self.names],
                                  dtype=torch.float64, device=device)
        self.n_frames = u0.shape[0]
        self.stride = 1
        self.w = torch.zeros_like(u0).requires_grad_(True)   # (Tc, P_fit) 격자 위 증분
        self.d = torch.zeros(len(self.names), dtype=torch.float64, device=device,
                             requires_grad=True)          # 파라미터별 전역 오프셋
        self.d_mask = torch.tensor([1.0 if n in GLOBAL_PARAMS else 0.0
                                    for n in self.names], dtype=torch.float64,
                                   device=device)
        self.prior_w = torch.tensor([PRIOR_W.get(n, 1.0) for n in self.names],
                                    dtype=torch.float64, device=device)
        self.f_idx = [i for i, n in enumerate(self.names)
                      if n in ("f1", "f2", "f3", "f4")]
        self.log_gain = torch.zeros(1, dtype=torch.float64, device=device,
                                    requires_grad=True)

        # 목표의 성문 폐쇄 시각 -> 샘플 색인 (있으면 위상 고정에 쓴다)
        pl = np.asarray(getattr(init, "pulses", np.zeros(0)), dtype=np.float64)
        pl = pl[(pl >= 0.0) & (pl * self.fs < n - 1)]
        self.pulse_idx = torch.as_tensor((pl * self.fs).astype(np.int64), device=device)
        self.pulse_phi0 = torch.zeros(1, dtype=torch.float64, device=device,
                                      requires_grad=True)
        self.wins = {k: torch.hann_window(k, device=device) for k in FFT_SIZES}
        # 포락은 **짧은 창**에서 잰다. 1024 (21 ms) 는 F0 240 Hz 의 하모닉을 분해하므로
        # 그 위의 멜은 포락이 아니라 하모닉 정렬을 재게 된다.
        # 256 (5.3 ms, 분해능 187 Hz) 이면 하모닉이 뭉개져 순수한 포락이 남는다.
        self.mel = mel_bank(MEL_FFT, self.fs, n_mels).to(device)
        with torch.no_grad():
            self.tgt_S = {k: _stft(self.target, k, self.wins[k]).abs() for k in FFT_SIZES}
            self.tgt_M = self.mel @ self.tgt_S[MEL_FFT]
            self.tgt_Mdb = self._db(self.tgt_M)
            self.db_floor = float(self.tgt_Mdb.max()) - DB_RANGE
        self.calibrate_gain()

    # ------------------------------------------------------- 재매개화
    @staticmethod
    def _to_raw(v: torch.Tensor, spec) -> torch.Tensor:
        lo, hi = spec.lo, spec.hi
        if spec.log:
            x = (torch.log(v.clamp_min(1e-9)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            x = (v - lo) / (hi - lo)
        return torch.log(x.clamp(1e-4, 1 - 1e-4) / (1 - x.clamp(1e-4, 1 - 1e-4)))

    @staticmethod
    def _to_val(u: torch.Tensor, spec) -> torch.Tensor:
        x = torch.sigmoid(u)
        lo, hi = spec.lo, spec.hi
        if spec.log:
            return torch.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
        return lo + x * (hi - lo)

    def _delta(self) -> torch.Tensor:
        """격자 위 증분 -> 프레임별 raw 증분 (선형 보간)."""
        w = self.w
        if w.shape[0] != self.n_frames:
            w = torch.nn.functional.interpolate(
                w.t().unsqueeze(0), size=self.n_frames, mode="linear",
                align_corners=True)[0].t()
        return w * self.scale

    def _u(self) -> torch.Tensor:
        return self.u0 + self._delta() + self.d * self.d_mask

    def set_grid(self, grid_ms: float) -> int:
        """제어 격자 간격을 바꾼다. 현재 해를 보간해서 옮기므로 이어서 최적화된다."""
        stride = max(1, int(round(grid_ms / self.track.frame_ms)))
        tc = max(2, int(math.ceil(self.n_frames / stride)))
        with torch.no_grad():
            old = self.w.detach()
            new = torch.nn.functional.interpolate(
                old.t().unsqueeze(0), size=tc, mode="linear", align_corners=True)[0].t()
        self.w = new.clone().requires_grad_(True)
        self.stride = stride
        return tc

    def control(self) -> torch.Tensor:
        u = self._u()
        cols = torch.stack([self._to_val(u[:, i], s) for i, s in enumerate(self.specs)], 1)
        return self.base.clone().index_copy(1, self.cols, cols).unsqueeze(0).to(torch.float32)

    def calibrate_gain(self) -> float:
        """초기 이득을 RMS 로 닫힌 형태로 준다.

        안 하면 1 회차 수렴도가 −18000 % 에서 시작하고(실측) Adam 이 그것만 수십 회 민다.
        """
        with torch.no_grad():
            self.eng.reset()
            y = self.eng(self.control(), self.track.events, 0.0)["audio"]
            a = float(y.pow(2).mean().sqrt()) + 1e-12
            b = float(self.target.pow(2).mean().sqrt()) + 1e-12
            self.log_gain.copy_(torch.tensor([math.log(b / a)], dtype=torch.float64,
                                             device=self.device))
        return self.gain_db()

    # ------------------------------------------------------------ 손실
    def synth(self, want_phase: bool = False):
        self.eng.reset()
        out = self.eng(self.control(), self.track.events, 0.0)
        y = out["audio"] * torch.exp(self.log_gain).to(torch.float32)
        return (y, out["phase"]) if want_phase else y

    def pulse_loss(self, phase: torch.Tensor) -> torch.Tensor:
        """성문 펄스 위치를 목표의 폐쇄 시각에 건다.

        크기 스펙트럼만 맞추면 위상은 자유롭게 흐른다. 이 엔진은 위상이 모형에서 나오므로
        (시변 IIR 이 샘플마다 이어진다) 소스 펄스만 제자리에 놓으면 나머지 위상은 물리가
        정한다. 그래서 **파형 위상이 아니라 펄스 시각**을 건다 — 손실면이 매끈하다.

        `1 − cos(φ(t_k) − φ0)` 를 쓴다. φ0 는 학습되는 상수 하나다 (LF 파형의 폐쇄
        시점과 Praat 의 폐쇄 시각 정의가 몇 도 다르므로, 그 차이는 상수로 흡수시킨다).
        """
        idx = self.pulse_idx
        if idx.numel() == 0:
            return torch.zeros((), device=phase.device)
        ph = phase[0].index_select(0, idx).double()
        return (1.0 - torch.cos(ph - self.pulse_phi0)).mean()

    @staticmethod
    def _db(x: torch.Tensor) -> torch.Tensor:
        return 20.0 * torch.log10(x + 1e-10)

    @staticmethod
    def _sc(at: torch.Tensor, ap: torch.Tensor) -> torch.Tensor:
        m = min(at.shape[-1], ap.shape[-1])
        at, ap = at[..., :m], ap[..., :m]
        return torch.linalg.norm(at - ap) / (torch.linalg.norm(at) + 1e-9)

    def spectral_loss(self, y: torch.Tensor):
        """(정밀 SC, 포락 dB 오차, 포락 SC, 창별 SC%).

        **선형 빈의 로그 크기 평균을 쓰면 안 된다.** 24 kHz 까지 선형이면 빈의 90 % 가
        2.4 kHz 위에 있고, 그 대역은 목표가 거의 무음이다. 그러면 손실이 무음 구간
        일치로 지배되어 전체 이득을 낮추는 쪽이 이긴다 (실측: 그렇게 해서 0~500 Hz 가
        27 dB 낮은 해의 손실이 더 낮게 나왔다). 그래서 포락 오차는 **멜 대역의 dB**
        로 재고 (로그 주파수 = 대역마다 같은 무게), 정점 −70 dB 아래는 잘라 낸다.
        """
        sc_sum, per = 0.0, {}
        Smel = None
        for k in self.sizes:
            Sp = _stft(y, k, self.wins[k]).abs()
            if k == MEL_FFT:
                Smel = Sp
            sc = self._sc(self.tgt_S[k], Sp)
            sc_sum = sc_sum + sc
            per[k] = float(100.0 * (1.0 - sc.detach()))
        if Smel is None:
            Smel = _stft(y, MEL_FFT, self.wins[MEL_FFT]).abs()
        Mp = self.mel @ Smel
        m = min(Mp.shape[-1], self.tgt_M.shape[-1])
        Mp, Mt = Mp[..., :m], self.tgt_M[..., :m]
        a = self._db(Mt).clamp_min(self.db_floor)
        b = self._db(Mp).clamp_min(self.db_floor)
        env_db = (a - b).abs().mean() / 20.0
        env_sc = self._sc(Mt, Mp)
        return sc_sum / len(self.sizes), env_db, env_sc, per

    def phase_loss(self, y: torch.Tensor) -> torch.Tensor:
        """복소 STFT 잔차. 크기가 이미 맞을 때만 의미가 있다."""
        k = 1024
        Sp, St = _stft(y, k, self.wins[k]), _stft(self.target, k, self.wins[k])
        m = min(Sp.shape[-1], St.shape[-1])
        return (St[..., :m] - Sp[..., :m]).abs().mean() / (St[..., :m].abs().mean() + 1e-9)

    def penalty(self) -> torch.Tensor:
        """물리적으로 성립하지 않는 해를 막는다.

        (1) 포먼트 순서. F4 가 F1 아래로 내려가는 해가 실제로 나왔다.
        (2) 대역폭이 중심 주파수를 넘지 않을 것 — 넘으면 그건 극이 아니라 기울기다.
        """
        u = self._u()
        pen = torch.zeros((), dtype=torch.float64, device=self.device)
        if len(self.f_idx) >= 2:
            f = torch.stack([self._to_val(u[:, i], self.specs[i]) for i in self.f_idx], 1)
            gap = torch.relu(f[:, :-1] + F_MARGIN_HZ - f[:, 1:]) / 1000.0
            pen = pen + 10.0 * (gap * gap).mean()
            for j, i in enumerate(self.f_idx):
                nb = self.names[i].replace("f", "bw")
                if nb in self.names:
                    k = self.names.index(nb)
                    bw = self._to_val(u[:, k], self.specs[k])
                    over = torch.relu(bw - 0.8 * f[:, j]) / 1000.0
                    pen = pen + 2.0 * (over * over).mean()
        return pen

    def loss(self):
        want = self.pulse_weight > 0 and self.pulse_idx.numel() > 0
        out = self.synth(want_phase=want)
        y, phase = out if want else (out, None)
        sc, env_db, env_sc, per = self.spectral_loss(y)
        l = 4.0 * env_db + env_sc + 0.5 * sc      # 포락(dB)이 주 목적, SC 는 보조
        self._last_db = float(env_db.detach()) * 20.0
        if want:
            pl = self.pulse_loss(phase)
            self._last_pulse = float(pl.detach())
            l = l + self.pulse_weight * pl
        if self.phase_weight > 0:
            l = l + self.phase_weight * self.phase_loss(y)
        if self.lam_smooth > 0 and self.w.shape[0] > 1:
            d = self.w[1:] - self.w[:-1]
            l = l + self.lam_smooth * (d * d).mean()
        if self.lam_prior > 0:
            d = (self._u() - self.u0) * self.prior_w
            l = l + self.lam_prior * (d * d).mean()
        return l + self.penalty(), sc, env_sc, per

    # ------------------------------------------------------------- 적합
    def fit(self, iters: int = 200, lr: float = 0.05, log_every: int = 25,
            verbose: bool = True, params: list | None = None,
            sizes: tuple[int, ...] | None = None) -> FitReport:
        if sizes is not None:
            self.sizes = list(sizes)
        opt = torch.optim.Adam(params or [self.w, self.d, self.log_gain,
                                          self.pulse_phi0], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(iters, 1), eta_min=lr * 0.05)
        best = (-1e18, None, None, None)
        hist: list[tuple[float, float]] = []
        bad_grads = 0
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            l, sc, env_sc, per = self.loss()
            if not torch.isfinite(l):
                if verbose:
                    print(f"    [{it}] 손실 비유한 — 중단")
                break
            env = float(100.0 * (1.0 - env_sc.detach()))
            fine = float(100.0 * (1.0 - sc.detach()))
            hist.append((env, fine))
            # **파라미터 사본은 걸음을 딛기 전에 뜬다.** step() 뒤에 뜨면 n 회차의 손실과
            # n+1 회차의 파라미터가 짝지어져, 복원해도 그 손실이 안 나온다(실측: 최선
            # 1.3185 로 기록해 놓고 복원하니 1.4828).
            score = -float(l.detach())
            if score > best[0]:
                best = (score, (self.w.detach().clone(), self.d.detach().clone()),
                        self.log_gain.detach().clone(),
                        (env, fine, self._last_db, float(l.detach()), per))
            l.backward()
            ps = [self.w, self.d, self.log_gain, self.pulse_phi0]
            # **비유한 기울기로 걸음을 딛으면 안 된다.** Adam 의 모멘트가 NaN 으로
            # 오염되면 그 뒤 모든 파라미터가 NaN 이 되고, 손실을 보기 전에 엔진 안에서
            # 터진다(실측: 탄음 구간에서 int(NaN)). 그런 회차는 건너뛴다.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in ps):
                bad_grads += 1
                for p in ps:
                    p.grad = None
                sch.step()
                continue
            torch.nn.utils.clip_grad_norm_(ps, 5.0)
            opt.step(); sch.step()
            if verbose and (it % log_every == 0 or it == iters - 1):
                print(f"    [{it:4d}] 포락 {env:6.2f}%  정밀 {fine:6.2f}%  "
                      f"오차 {self._last_db:5.2f} dB  펄스 {self._last_pulse:.3f}  "
                      f"손실 {float(l.detach()):.4f}")
        if verbose and bad_grads:
            print(f"    (기울기 비유한 {bad_grads} 회 건너뜀)")
        if best[1] is not None:
            with torch.no_grad():
                self.w.copy_(best[1][0]); self.d.copy_(best[1][1])
                self.log_gain.copy_(best[2])
            env, fine, db, lv, per = best[3]
        else:
            env = fine = db = lv = float("nan"); per = {}
        return FitReport(env, fine, db, per, lv, len(hist), hist)

    def fit_staged(self, global_iters: int = 200, stage_iters: int = 150,
                   lr_global: float = 0.05, lr_frame: float = 0.04,
                   verbose: bool = True, log_every: int = 50) -> FitReport:
        """전역 스칼라 -> 제어 격자를 성기게에서 촘촘하게, 창도 함께 늘려 가며."""
        if verbose:
            print(f"  1 단계 전역 {len(self.names)} 스칼라 (이득 {self.gain_db():+.1f} dB)")
        rep = self.fit(global_iters, lr_global, log_every, verbose,
                       params=[self.d, self.log_gain, self.pulse_phi0], sizes=STAGES[0])
        for si, (grid, sizes) in enumerate(zip(GRID_MS, STAGES)):
            tc = self.set_grid(grid)
            if verbose:
                print(f"  2.{si + 1} 단계  격자 {grid:g} ms ({tc} 점)  창 {sizes}")
            rep = self.fit(stage_iters, lr_frame * (0.75 ** si), log_every, verbose,
                           sizes=sizes)
        self.sizes = list(FFT_SIZES)
        return rep

    # ------------------------------------------------------------- 결과
    def result_track(self) -> ControlTrack:
        with torch.no_grad():
            v = self.control()[0].double().cpu().numpy()
        return ControlTrack(v, self.track.frame_ms, list(self.track.events))

    def render(self) -> np.ndarray:
        with torch.no_grad():
            return self.synth()[0].cpu().numpy()

    def gain_db(self) -> float:
        return float(20.0 * self.log_gain.detach().item() / math.log(10))

    def moved(self) -> list[tuple[str, float, float]]:
        """(이름, 초기 중앙값, 적합 중앙값). 어떤 파라미터가 얼마나 움직였는지."""
        out = []
        with torch.no_grad():
            u = self._u()
            for i, sp in enumerate(self.specs):
                a = float(self._to_val(self.u0[:, i], sp).median())
                b = float(self._to_val(u[:, i], sp).median())
                out.append((sp.name, a, b))
        return out
