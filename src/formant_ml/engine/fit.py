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
# **난류 우세 빈에서 기대 스펙트럼을 볼 시간 폭 (ms). 0 이면 끈다.**
#
# 마찰 잡음의 STFT 크기는 확률변수다 (레일리). 프레임별 실현을 맞추라고 하면 기울기의
# 대부분이 그 잡음이고, 최적해도 실현마다 다르다. 재현할 수 있는 것은 **기대값**뿐이니
# 그것을 비교한다. 목표와 합성에 똑같이 걸리므로 편향은 없고 분산만 줄어든다.
#
# 25 ms 는 마찰음 길이(이 화자 실측 70~110 ms, `profiles/yang_female.json`)의 1/4 이라
# 정상부를 뭉개지 않으면서 창 256 에서 19 프레임을 평균한다(분산 1/19). 조화 우세
# 빈에는 `1−w` 가중이 0 이라 아무 일도 일어나지 않는다.
NOISE_EXPECT_MS = 25.0
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
# **조음 속도 상한 (Hz/ms).** 성도는 근육이 움직이는 물건이라 포먼트가 임의로 빨리
# 뛸 수 없다. 이 값을 넘으면 그건 조음이 아니라 적합기의 프레임별 난동이다.
#
# 계측(`data/ref` 여성 3 개, 유성 프레임 쌍 10,333 개, Praat 1 ms):
#
#   |    | 중앙 | 80분위 | 95분위 | 99분위 |
#   |----|------|--------|--------|--------|
#   | F1 |  2.9 |    8.5 |   27.2 |  147.3 |
#   | F2 |  7.0 |   22.7 |   92.7 |  604.9 |
#   | F3 | 11.5 |   37.5 |  151.0 |  770.7 |
#   | F4 | 10.7 |   31.2 |  145.2 |  771.8 |
#
# 99 분위의 폭주(F2 605, 최대 2043)는 Praat 추적기가 포먼트를 맞바꾼 것이지 조음이
# 아니다. 그래서 **95 분위를 상한**으로 잡는다. 그 아래는 공짜, 넘으면 이차로 문다
# (평활 벌점처럼 전 구간을 뭉개면 설측 F2 의 실제 급전이 같이 죽는다).
#
# 이 제약이 왜 필요했나 — 없을 때 적합된 궤적을 실측과 대조하면:
#   지속 모음 F1 실제 중앙 1.1 Hz/ms  vs  적합 126.4 Hz/ms (×115)
#   설측    F2 실제 중앙 25.5        vs  적합 309.0      (×12)
# 즉 `lam_smooth` 만으로는 네 자릿수 모자랐다. 해가 심하게 비-식별이 되어, 크기만
# 맞고 위상은 틀린 해로 굴러가도 밀어낼 힘이 없었다.
VEL_LIMIT_HZ_PER_MS: dict[str, float] = {
    "f1": 27.0, "f2": 93.0, "f3": 151.0, "f4": 145.0,
}

# **힌지는 틀린 형태였다.** 95 분위를 문턱으로 두면 그 아래의 균일한 난동을 그냥
# 통과시킨다. 실측: 지속 모음의 실제 F2 속도는 1.1 Hz/ms 인데 문턱은 93 이라, 모음에서
# 적합값이 57 Hz/ms 로 널뛰어도 벌점이 0 이다 (A/B 실측: 속도 77.9→57.3 으로 26 % 만
# 줄고 조화 SNR 은 4.6 dB 잃었다 — 나쁜 거래).
#
# 조음 속도의 실측 분포는 꼬리가 중앙의 200~300 배인 희소 신호다:
#
#   |    | 중앙 | 95분위 |   최대 | 꼬리/중앙 |
#   |----|------|--------|--------|-----------|
#   | F1 |  2.9 |   27.2 |  877.8 |      303× |
#   | F2 |  7.0 |   92.7 | 2043.0 |      292× |
#   | F3 | 11.5 |  151.0 | 2540.5 |      221× |
#
# "대부분 정지, 가끔 크게 이동" — 이런 분포의 사전은 이차(가우시안)가 아니라 L1
# (라플라스) 계열이다. 그래서 후버(smooth L1)를 쓴다: 무릎 아래는 이차라 미분이
# 살아 있고, 위는 선형이라 드문 급전에 이차 벌점처럼 가혹하지 않다.
#
# 무릎은 **실측 중앙값**이다 — "여기가 보통 속도" 라는 뜻이지 상한이 아니다.
VEL_KNEE_HZ_PER_MS: dict[str, float] = {
    "f1": 2.9, "f2": 7.0, "f3": 11.5, "f4": 10.7,
}
# **가속도 무릎 (Hz/ms²).** 1 차 차분(속도)으로는 난동과 실제 조음을 못 가른다 —
# 동적 구간에서 둘은 크기가 비슷하고 **시간 구조**만 다르다. 난동은 부호가 매 프레임
# 뒤집히고, 실제 급전은 한 방향으로 지속된다. 그래서 2 차 차분을 본다: 빠르게
# *움직이는* 것은 허용하고 빠르게 *방향을 바꾸는* 것만 문다.
#
# 실측 (data/ref 여성 3 개, 유성 프레임 삼중항 10,305 개, Praat 1 ms):
#
#   |    | 속도 중앙 | 가속 중앙 | 가속 80% | 가속 95% |
#   |----|-----------|-----------|----------|----------|
#   | F1 |       2.9 |       0.7 |      2.4 |     12.6 |
#   | F2 |       7.0 |       1.9 |      8.0 |     69.4 |
#   | F3 |      11.5 |       3.4 |     15.4 |    117.4 |
#   | F4 |      10.7 |       3.5 |     12.6 |    132.5 |
#
# 가속도가 더 나은 판별자인 이유가 이 표에 있다. 우리 적합의 난동은 F2 기준
# 300~500 Hz/ms 로 부호가 매 프레임 뒤집히므로 가속도는 약 600~1000 Hz/ms² 다.
#
#   속도로 보면   실제 7.0  vs 난동 ~300   ->  43 배
#   가속도로 보면 실제 1.9  vs 난동 ~600   -> 300 배
#
# 판별 여유가 7 배 넓다. 속도로는 실제 급전(최대 628 Hz/ms)이 난동 영역과 겹쳐서
# 못 갈랐는데, 가속도로는 실제 급전이 한 방향으로 지속되므로 겹치지 않는다.
ACC_KNEE_HZ_PER_MS2: dict[str, float] = {
    "f1": 0.7, "f2": 1.9, "f3": 3.4, "f4": 3.5,
}
# "off" | "hinge"(95 분위 상한) | "huber"(속도 중앙값 무릎) | "accel"(가속도 무릎)
#
# 앞의 둘은 기각됐다 (docs/MEASUREMENTS.md §8.6). 성긴 격자도 기각됐다 (§8.9) —
# 균일한 격자든 균일한 벌점이든 모든 시점에 똑같이 자유도를 주거나 뺏는데, 조음은
# "대부분 정지, 짧은 순간에 급전" 이라 그 틀로는 두 요구를 동시에 못 맞춘다.
VEL_MODE = "accel"
# 기본값 0 — **아직 켜지 않는다.** 기구와 계측 상수는 여기 있지만, 세기(VEL_W)가
# 포락 일치를 얼마나 깎는지 A/B 로 확인하기 전에는 모든 적합의 거동을 바꿀 수 없다.
# 힌지 자체는 검증했다: F2 이동 10/50/93 Hz/ms 는 벌점 0, 150 은 0.97, 309 는 14.0.
VEL_W = 0.0

PRIOR_W: dict[str, float] = {
    "f0_target": 40.0, "f1": 40.0, "f2": 40.0, "f3": 20.0, "f4": 10.0,
    # 곁가지는 분석이 못 재는 양이다. 세게 묶으면 적합기가 열지를 못한다 —
    # 남 /나/ 머머가 "이득 −24 dB 내린 모음" 으로 굳던 원인.
    "velum": 3.0, "oral_open": 3.0, "lat_mix": 1.0, "lat_bw": 2.0,
    "nasal_f": 20.0, "nasal_f2": 20.0,
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
    "lat_mix": 0.5, "lat_bw": 0.3,
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
    # 곁가지(side branch)를 적합 대상에 넣는다. 이게 빠져 있으면 설측·비음처럼
    # 곁가지가 정의적 특징인 조음을 **원리적으로 못 맞춘다** (실측 §8.4).
    "lat_mix", "lat_bw",
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

        # 녹음의 원래 나이퀴스트. **손실에서 그 위를 보면 안 된다.**
        # 44.1 kHz 녹음을 48 kHz 로 올리면 22.05~24 kHz 가 정확히 비어 있고, 게다가
        # 그 아래 16~22 kHz 도 녹음 장비의 안티에일리어싱이 만든 값이다. 그걸 목표로
        # 두면 적합기가 "저 위를 비워라" 를 물리 파라미터로 달성하려 들고, 그 왜곡이
        # 8~12 kHz 를 10 dB 어둡게 만들었다(실측: 남성 /사/).
        self.f_max = min(0.5 * float(sr), 0.5 * self.fs) * 0.90
        if sr != self.fs:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), int(self.fs))
            target = resample_poly(target, self.fs // g, int(sr) // g)
        # **분석이 붙여 준 부가 정보(성문 펄스·유성 마스크)를 같이 옮긴다.** 새로
        # ControlTrack 을 만들면서 빠뜨리면 조용히 기본값(빈 배열)이 되고, 마찰 이득
        # 보정이 아무 일도 안 하게 된다 (실측: 남 /사/ 무성부가 −40.8 dB 인 채로 시작).
        self.track = ControlTrack(init.values.copy(), init.frame_ms, list(init.events),
                                  pulses=np.asarray(getattr(init, "pulses", np.zeros(0))).copy(),
                                  voiced=np.asarray(getattr(init, "voiced",
                                                            np.zeros(0, dtype=bool))).copy(),
                                  fricative=np.asarray(getattr(init, "fricative",
                                                               np.zeros(0, dtype=bool))).copy())
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
        self.mel = mel_bank(MEL_FFT, self.fs, n_mels, fmax=self.f_max).to(device)
        # 선형 SC 도 같은 상한을 쓴다.
        self.bin_max = {k: int(math.ceil(self.f_max / (self.fs / k))) + 1 for k in FFT_SIZES}
        # 목표의 조화/잔차 분해. 위상 항(조화부만 본다)과 크기 항(난류부는 기대
        # 스펙트럼으로 본다)이 같은 것을 쓰므로 한 번만 만든다.
        self._har_res = self._decompose_target()
        with torch.no_grad():
            self._harm_w = {k: self._harmonic_weight(k) for k in FFT_SIZES}
            self._noise_w = {k: (1.0 - self._harm_w[k]) for k in FFT_SIZES}
            self._sm_frames = {k: max(1, int(round(NOISE_EXPECT_MS
                                                   / (1000.0 * (k // 4) / self.fs))))
                               for k in FFT_SIZES}
            raw = {k: _stft(self.target, k, self.wins[k]).abs()[:, :self.bin_max[k]]
                   for k in FFT_SIZES}
            self.tgt_S = {k: self._expect(v, k) for k, v in raw.items()}
            self.tgt_M = self.mel[:, :self.tgt_S[MEL_FFT].shape[1]] @ self.tgt_S[MEL_FFT]
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

    def _rms(self, x: torch.Tensor, mask: np.ndarray | None) -> float:
        if mask is None or not mask.any():
            return float(x.pow(2).mean().sqrt()) + 1e-12
        i = np.flatnonzero(np.repeat(mask, self.hop)[:x.shape[-1]])
        return float(x[..., i].pow(2).mean().sqrt()) + 1e-12

    def calibrate_gain(self) -> float:
        """초기 이득을 닫힌 형태로 준다. **유성/무성을 따로 맞춘다.**

        하나의 이득만 맞추면 유성과 무성 중 한쪽이 반드시 크게 어긋난다. 마찰 세기는
        `fric_gain` 이 정하는데 그 초기값이 1.0 이고, 실측과의 차이가 40 dB 를 넘으면
        경사 하강으로는 못 간다 (한 걸음이 상한 64 = 36 dB 안에서 움직인다). 실제로
        남 /사/ 는 마찰부가 −42.8 dB 인 채로 시작해 끝까지 못 따라잡았다.
        그래서 전체 이득은 **유성 프레임**에서, `fric_gain` 은 **무성 프레임**에서
        각각 닫힌 형태로 준다.
        """
        voi = np.asarray(getattr(self.track, "voiced", np.zeros(0, dtype=bool)))
        if voi.shape[0] != self.n_frames:
            voi = np.zeros(0, dtype=bool)
        # `fric_gain` 은 **마찰 프레임**에서만 잰다. 폐쇄(비음 머머 등)를 섞으면
        # 조용한 구간까지 마찰로 메우려 든다.
        fri = np.asarray(getattr(self.track, "fricative", np.zeros(0, dtype=bool)))
        if fri.shape[0] != self.n_frames:
            fri = ~voi if voi.size else np.zeros(0, dtype=bool)
        unv = fri
        with torch.no_grad():
            self.eng.reset()
            y = self.eng(self.control(), self.track.events, 0.0)["audio"]
            m = voi if voi.any() else None
            r = self._rms(self.target, m) / self._rms(y, m)
            self.log_gain.copy_(torch.tensor([math.log(r)], dtype=torch.float64,
                                             device=self.device))
            if unv.any() and "fric_gain" in self.names:
                self.eng.reset()
                y = self.eng(self.control(), self.track.events, 0.0)["audio"] * r
                need = self._rms(self.target, unv) / self._rms(y, unv)
                j = self.names.index("fric_gain")
                sp = self.specs[j]
                g0 = float(self._to_val(self.u0[:, j], sp).median())
                tgt = float(np.clip(g0 * need, sp.lo + 1e-3, sp.hi - 1e-3))
                # raw 좌표의 전역 오프셋으로 넣는다 (프레임별 모양은 보존)
                d = float(self._to_raw(torch.tensor([tgt], dtype=torch.float64), sp)[0]
                          - self._to_raw(torch.tensor([g0], dtype=torch.float64), sp)[0])
                self.d[j] = d / max(float(self.d_mask[j]), 1.0) if self.d_mask[j] else 0.0
                if not self.d_mask[j]:      # fric_gain 은 전역 대상이어야 한다
                    self.d[j] = d
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
            Sp = self._expect(_stft(y, k, self.wins[k]).abs()[:, :self.bin_max[k]], k)
            if k == MEL_FFT:
                Smel = Sp
            sc = self._sc(self.tgt_S[k], Sp)
            sc_sum = sc_sum + sc
            per[k] = float(100.0 * (1.0 - sc.detach()))
        if Smel is None:
            Smel = self._expect(
                _stft(y, MEL_FFT, self.wins[MEL_FFT]).abs()[:, :self.bin_max[MEL_FFT]], MEL_FFT)
        Mp = self.mel[:, :Smel.shape[1]] @ Smel
        m = min(Mp.shape[-1], self.tgt_M.shape[-1])
        Mp, Mt = Mp[..., :m], self.tgt_M[..., :m]
        # **clamp 도 abs 도 꺾임이다.** 멜 빈 수천 개마다 꺾이면 손실이 C² 가 아니게
        # 되고 헤시안 기반 진단이 통째로 막힌다 (docs/MEASUREMENTS.md §8.7). 폭은
        # dB 단위 0.5 — 계측 잡음보다 작아 거동은 사실상 그대로다.
        a = self._soft_floor(self._db(Mt), self.db_floor, 0.5)
        b = self._soft_floor(self._db(Mp), self.db_floor, 0.5)
        env_db = self._soft_abs(a - b, 0.5).mean() / 20.0
        env_sc = self._sc(Mt, Mp)
        return sc_sum / len(self.sizes), env_db, env_sc, per

    def phase_loss(self, y: torch.Tensor) -> torch.Tensor:
        """**단위 크기** 복소 잔차 — 크기와 직교한 순수 위상 거리, 조화 우세부에만.

        왜 복소 잔차를 그냥 쓰면 안 되는가
        ----------------------------------
        예전 형태는 `|S_t − S_p|` 였다. 위상이 상관된 곳(하모닉)에서는 옳게 작동하지만,
        **위상이 무상관인 곳(모든 난류)에서는 |S_p| = 0 이 최소다.** 두 위상이 독립이면
        E|S_t − S_p|² = |S_t|² + |S_p|² 이므로 합성을 **끄는 것**이 가장 좋은 해가 된다.

        실측 (같은 스펙트럼의 두 독립 실현, 합성 진폭 g 를 훑는다):

            | g   | 예전 위상항 | 크기항 SC |
            |-----|-------------|-----------|
            | 0.0 |  0.999 ←최소 |     1.000 |
            | 0.8 |  1.277      |     0.627 |
            | 1.0 |  1.411      |     0.665 |

        `fit_staged` 는 이 항을 기본으로 800 회 돌린다. 즉 마찰 구간은 800 회 내내
        "소리를 꺼라" 는 기울기를 받고 있었다.

        고친 형태
        ---------
        두 복소수를 단위 크기로 정규화한 뒤 거리를 재고, 목표 크기로 가중한다:

            d(t,f) = w(t,f) · |S_t/|S_t| − S_p/|S_p||

        `S_p` 의 크기에 대한 편미분이 항등적으로 0 이므로 **어떤 구간에서도 소리를
        끄지 않는다.** 크기는 크기 항이, 위상은 위상 항이 맡는 깨끗한 분해다.

        가중 w 는 **목표의 조화 우세도**다 (`_harmonic_weight`). 난류 빈에서는 0 이라
        아예 안 본다 — 재도 뜻이 없는 양이기 때문이다.

        여러 창을 쓰는 이유: 창이 길수록 하모닉이 분해되어 위상이 정밀해지지만 손실면이
        톱니가 된다. 짧은 창이 큰 틀을 잡고 긴 창이 다듬도록 겹친다.
        """
        out = 0.0
        for k in (256, 512, 1024):
            Sp, St = _stft(y, k, self.wins[k]), _stft(self.target, k, self.wins[k])
            m = min(Sp.shape[-1], St.shape[-1])
            b = self.bin_max[k]
            St, Sp = St[:, :b, :m], Sp[:, :b, :m]
            at = St.abs()
            w = self._harm_w[k][:, :, :m] * at
            # 단위 크기로 정규화. δ 는 크기가 0 에 가까운 빈에서 방향이 폭주하는 것을
            # 막는다 (그런 빈은 w 도 작아 어차피 기여가 없다).
            dt = 1e-4 * at.mean().detach() + 1e-12
            ut = St / torch.sqrt(at ** 2 + dt ** 2)
            up = Sp / torch.sqrt(Sp.abs() ** 2 + dt ** 2)
            d = ut - up
            # √(|z|²+δ²)−δ — |z| 는 0 에서 곡률이 발산한다. 적합이 좋아질수록 그리로
            # 가므로 바닥을 깔아 둔다 (relu 제거와 같은 부류, C² 유지).
            e = torch.sqrt(d.real ** 2 + d.imag ** 2 + 1e-6) - 1e-3
            out = out + (w * e).sum() / (w.sum() + 1e-9)
        return out / 3.0

    def _decompose_target(self) -> tuple[np.ndarray, np.ndarray] | None:
        """목표를 조화 성분과 잔차로 가른다 (`waveform.decompose`).

        성문 펄스에서 나온 국소 F0 로 3 주기 창에서 최소제곱을 푼다. 무성 구간에서는
        F0 가 없어 조화 성분이 0 이 되고 전부 잔차로 간다 — 따로 마스크가 필요 없다.
        """
        from .waveform import decompose
        tgt = self.target[0].detach().cpu().numpy().astype(np.float64)
        f0 = np.asarray(self.track["f0_target"], dtype=np.float64)
        voi = np.asarray(getattr(self.track, "voiced", np.zeros(0, dtype=bool)))
        if voi.shape[0] == f0.shape[0]:
            f0 = np.where(voi, f0, 0.0)
        try:
            _, har, res = decompose(tgt, self.fs, f0, self.hop)
        except Exception:
            return None
        return har, res

    def _harmonic_weight(self, k: int) -> torch.Tensor:
        """(1, F, T) 조화 우세도 — 이 시간·주파수 빈이 얼마나 '주기적'인가.

            w = |STFT(조화)| / (|STFT(조화)| + |STFT(잔차)|)

        모음의 낮은 하모닉에서 1 에 가깝고, 마찰 잡음과 모음 고역의 기식에서 0 에
        가깝다. 목표에서 한 번만 계산하는 상수다.

        두 곳이 이 값을 쓴다. 위상 항은 **w 로 가중**해 조화부만 보고(난류의 위상은
        재도 뜻이 없다), 크기 항은 **1−w 로 가중**해 난류부에서 기대 스펙트럼을
        비교한다(실현을 맞추라고 하면 기울기가 잡음이다).
        """
        b = self.bin_max[k]
        if self._har_res is None:
            return torch.ones(1, b, 1, device=self.device)
        har, res = self._har_res
        H = _stft(torch.as_tensor(har, dtype=torch.float32,
                                  device=self.device).unsqueeze(0), k, self.wins[k]).abs()[:, :b]
        R = _stft(torch.as_tensor(res, dtype=torch.float32,
                                  device=self.device).unsqueeze(0), k, self.wins[k]).abs()[:, :b]
        return (H / (H + R + 1e-9)).detach()

    def _time_smooth(self, S: torch.Tensor, k: int) -> torch.Tensor:
        """파워 영역에서 시간축 이동평균 -> 크기. 난류의 **기대** 스펙트럼을 만든다.

        파워가 가법적이므로 평균은 파워에서 낸다. 폭은 `NOISE_EXPECT_MS` 이고 창
        크기별로 프레임 수가 다르다 (긴 창은 이미 시간 폭이 커서 평활이 거의 없다).
        """
        n = self._sm_frames.get(k, 1)
        if n <= 1 or S.shape[-1] < 2:
            return S
        n = min(n, S.shape[-1] // 2 * 2 + 1)
        pad = n // 2
        p = torch.nn.functional.pad(S ** 2, (pad, n - 1 - pad), mode="replicate")
        return torch.sqrt(torch.nn.functional.avg_pool1d(p, n, stride=1) + 1e-20)

    def _expect(self, S: torch.Tensor, k: int) -> torch.Tensor:
        """난류 우세 빈만 기대 스펙트럼으로 갈아 끼운다.

        `(1−w)` 로 섞으므로 조화부는 프레임별 실현 그대로 남는다. 목표와 합성에
        **똑같이** 걸리므로 편향은 생기지 않고 기울기의 분산만 줄어든다.
        """
        if NOISE_EXPECT_MS <= 0 or self._har_res is None:
            return S
        wn = self._noise_w.get(k)
        if wn is None:
            return S
        m = min(S.shape[-1], wn.shape[-1])
        w = wn[..., :m]
        s = S[..., :m]
        return s + w * (self._time_smooth(s, k) - s)

    @staticmethod
    def _soft_over(x: torch.Tensor, width: float) -> torch.Tensor:
        """relu 의 부드러운 대체. width 만큼의 폭으로 무릎을 뭉갠다.

        **relu 를 쓰면 안 된다.** 꺾임점에서 2 차 도함수가 정의되지 않아 손실이 C² 가
        아니게 되고, 그러면 헤시안 기반 진단(유효 파라미터 수 γ 등)이 통째로 막힌다.
        실측으로 확인했다: 유한차분 헤시안의 대칭성(uᵀHv = vᵀHu)이 ε 을 4 자릿수
        쓸어도 41.9 / 47.5 / 178.2 / 153.5 % 로 깨졌다 (docs/MEASUREMENTS.md §8.7).
        손실은 결정적이었으므로 원인은 무작위성이 아니라 꺾임이다.

        width → 0 이면 relu 로 수렴한다. 벌점으로서의 성질(문턱 아래는 거의 0, 위는
        선형)은 그대로 두고 미분만 매끄럽게 만든다.
        """
        return width * torch.nn.functional.softplus(x / width)

    @staticmethod
    def _soft_abs(x: torch.Tensor, width: float) -> torch.Tensor:
        """|x| 의 부드러운 대체 (의사 후버). 작은 x 에서 x²/2width, 큰 x 에서 |x|.

        포락 오차의 L1 이 손실의 최대 꺾임원이었다 — 멜 빈 수천 개마다 a=b 에서
        꺾인다. width 는 "이 정도 차이는 오차로 안 친다" 는 뜻이고, dB 단위에서
        0.5 dB 면 계측 잡음보다 작아 거동이 사실상 안 바뀐다.
        """
        return width * (torch.sqrt(1.0 + (x / width) ** 2) - 1.0)

    @staticmethod
    def _soft_floor(x: torch.Tensor, floor: float, width: float) -> torch.Tensor:
        """clamp_min 의 부드러운 대체. floor + softplus(x − floor)."""
        return floor + width * torch.nn.functional.softplus((x - floor) / width)

    @staticmethod
    def _pseudo_huber(r: torch.Tensor) -> torch.Tensor:
        """후버의 매끄러운 형태. torch.where 는 1 차 도함수만 이어져 C² 가 아니다.

        √(1+r²) − 1 은 작은 r 에서 r²/2, 큰 r 에서 r−1 로 후버와 같은 모양이면서
        무한히 미분 가능하다.
        """
        return torch.sqrt(1.0 + r * r) - 1.0

    def penalty(self) -> torch.Tensor:
        """물리적으로 성립하지 않는 해를 막는다.

        (1) 포먼트 순서. F4 가 F1 아래로 내려가는 해가 실제로 나왔다.
        (2) 대역폭이 중심 주파수를 넘지 않을 것 — 넘으면 그건 극이 아니라 기울기다.
        (3) 조음 속도. 계측된 생리적 상한(`VEL_LIMIT_HZ_PER_MS`)을 넘는 이동은 문다.
        """
        u = self._u()
        pen = torch.zeros((), dtype=torch.float64, device=self.device)
        if len(self.f_idx) >= 2:
            f = torch.stack([self._to_val(u[:, i], self.specs[i]) for i in self.f_idx], 1)
            gap = self._soft_over(f[:, :-1] + F_MARGIN_HZ - f[:, 1:], 30.0) / 1000.0
            pen = pen + 10.0 * (gap * gap).mean()
            for j, i in enumerate(self.f_idx):
                nb = self.names[i].replace("f", "bw")
                if nb in self.names:
                    k = self.names.index(nb)
                    bw = self._to_val(u[:, k], self.specs[k])
                    over = self._soft_over(bw - 0.8 * f[:, j], 30.0) / 1000.0
                    pen = pen + 2.0 * (over * over).mean()
        if VEL_W > 0 and VEL_MODE != "off" and u.shape[0] > 1:
            dt = max(self.track.frame_ms, 1e-6)
            tab = {"hinge": VEL_LIMIT_HZ_PER_MS, "huber": VEL_KNEE_HZ_PER_MS,
                   "accel": ACC_KNEE_HZ_PER_MS2}[VEL_MODE]
            for nm, lim in tab.items():
                if nm not in self.names or u.shape[0] < 3:
                    continue
                k = self.names.index(nm)
                v = self._to_val(u[:, k], self.specs[k])
                if VEL_MODE == "accel":
                    # 2 차 차분. 한 방향으로 꾸준히 가는 급전은 여기서 0 에 가깝고,
                    # 부호가 뒤집히는 난동만 크게 나온다.
                    rate = (v[2:] - 2.0 * v[1:-1] + v[:-2]).abs() / (dt * dt)
                else:
                    rate = (v[1:] - v[:-1]).abs() / dt
                if VEL_MODE == "hinge":
                    over = self._soft_over(rate - lim, 0.05 * lim) / 100.0
                    pen = pen + VEL_W * (over * over).mean()
                else:
                    # 후버. lim 으로 나눠 무차원으로 만든다 (파라미터끼리 같은 저울).
                    r = rate / lim
                    h = self._pseudo_huber(r)
                    pen = pen + VEL_W * h.mean()
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
            sizes: tuple[int, ...] | None = None,
            patience: int = 0, tol: float = 3e-4) -> FitReport:
        """`patience` 회 동안 손실이 `tol`(상대) 만큼도 안 줄면 멈춘다.

        **코퍼스 규모로 가려면 수렴 판정이 필요하다.** 지금까지는 고정 반복 뒤 최저손실
        스냅샷을 취할 뿐이라 이미 수렴한 구간에서도 예산을 다 썼다 (docs/HANDOFF.md §5:
        "구간당 5 -> 25 분이 됐다"). 45 분 코퍼스를 200 ms 구간으로 쪼개면 수천 개다.

        `patience=0` 이면 끈다 — 예전 거동 그대로다. 코사인 스케줄러가 lr 을 줄이는
        중이므로 문턱을 너무 크게 잡으면 아직 내려갈 수 있는데 멈춘다. 3e-4 는 실측에서
        마지막 20 % 구간의 회차당 개선폭보다 작다.
        """
        if sizes is not None:
            self.sizes = list(sizes)
        opt = torch.optim.Adam(params or [self.w, self.d, self.log_gain,
                                          self.pulse_phi0], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(iters, 1), eta_min=lr * 0.05)
        best = (-1e18, None, None, None)
        hist: list[tuple[float, float]] = []
        bad_grads = 0
        stall = 0
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
            # 스냅샷은 조금이라도 나아지면 뜬다. 멈춤 판정만 **의미 있는** 개선을 센다.
            gain = score - best[0]
            if score > best[0]:
                best = (score, (self.w.detach().clone(), self.d.detach().clone()),
                        self.log_gain.detach().clone(),
                        (env, fine, self._last_db, float(l.detach()), per))
            if patience > 0:
                stall = 0 if gain > tol * abs(score) else stall + 1
                if stall >= patience:
                    if verbose:
                        print(f"    [{it}] 수렴 ({patience} 회 정체) — 조기 종료")
                    break
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
                      f"손실 {float(l.detach()):.4f}", flush=True)
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

    def _snapshot(self):
        return (self.w.detach().clone(), self.d.detach().clone(),
                self.log_gain.detach().clone(), self.pulse_phi0.detach().clone())

    def _restore(self, snap) -> None:
        with torch.no_grad():
            self.w.copy_(snap[0]); self.d.copy_(snap[1])
            self.log_gain.copy_(snap[2]); self.pulse_phi0.copy_(snap[3])

    def pick_lr_global(self, candidates, iters: int, verbose: bool = True) -> float:
        """짧은 탐침으로 전역 단계의 lr 을 고른다.

        **구간마다 맞는 걸음 크기가 다르다.** 40 ms 짜리 탄음은 lr 0.05 에서 포락
        60.3 % 인데 0.02 면 90.9 % 다 (실측, lr 에 단조). 반대로 소스 기울기가 7 dB/oct
        틀린 합성 모음은 0.02 로는 못 돌아오고 0.05 가 필요하다. 하나로 못 정하므로
        짧게 재 보고 손실이 가장 낮은 것을 쓴다.
        """
        start = self._snapshot()
        best = (float("inf"), float(candidates[0]))
        for lr in candidates:
            self._restore(start)
            rep = self.fit(iters, float(lr), 10 ** 9, False,
                           params=[self.d, self.log_gain, self.pulse_phi0], sizes=STAGES[0])
            if np.isfinite(rep.loss) and rep.loss < best[0]:
                best = (rep.loss, float(lr))
            if verbose:
                print(f"      lr {float(lr):5.3f} -> 손실 {rep.loss:.4f}")
        self._restore(start)
        return best[1]

    def pick_lr_phase(self, candidates, iters: int, verbose: bool = True) -> float:
        """위상 단계의 lr 도 짧은 탐침으로 고른다.

        전역 단계와 같은 이유이고, 실측으로도 구간마다 최적이 다르다 (위상 800 반복):
        정상 모음 0.05~0.12, 설측 0.12 이상, 탄음 0.05. 하나로 못 정한다.
        """
        start = self._snapshot()
        best = (float("inf"), float(candidates[0]))
        for lr in candidates:
            self._restore(start)
            rep = self.fit(iters, float(lr), 10 ** 9, False, sizes=FFT_SIZES)
            if np.isfinite(rep.loss) and rep.loss < best[0]:
                best = (rep.loss, float(lr))
            if verbose:
                print(f"      lr {float(lr):5.3f} -> 손실 {rep.loss:.4f}")
        self._restore(start)
        return best[1]

    def fit_staged(self, global_iters: int = 200, stage_iters: int = 150,
                   lr_global=(0.015, 0.03, 0.05, 0.09), lr_frame: float = 0.04,
                   phase_iters: int = 800, phase_w: float = 3.0,
                   lr_phase=(0.05, 0.12, 0.25),
                   verbose: bool = True, log_every: int = 50,
                   patience: int = 0) -> FitReport:
        """전역 스칼라 -> 제어 격자를 성기게에서 촘촘하게, 창도 함께 늘려 가며.

        `patience` 는 각 단계의 수렴 판정에 그대로 넘어간다. 코퍼스를 통째로 돌릴 때
        쓴다 (`scripts/parametrize_corpus.py`). 기본 0 = 예전 거동.
        """
        if verbose:
            print(f"  1 단계 전역 {len(self.names)} 스칼라 (이득 {self.gain_db():+.1f} dB)")
        if isinstance(lr_global, (int, float)):
            lr_global = (float(lr_global),)
        if len(lr_global) > 1:
            probe = max(20, global_iters // 4)
            lr_g = self.pick_lr_global(lr_global, probe, verbose)
            if verbose:
                print(f"      -> lr {lr_g:.3f} 선택")
        else:
            lr_g = float(lr_global[0])
        rep = self.fit(global_iters, lr_g, log_every, verbose,
                       params=[self.d, self.log_gain, self.pulse_phi0], sizes=STAGES[0],
                       patience=patience)
        for si, (grid, sizes) in enumerate(zip(GRID_MS, STAGES)):
            tc = self.set_grid(grid)
            if verbose:
                print(f"  2.{si + 1} 단계  격자 {grid:g} ms ({tc} 점)  창 {sizes}", flush=True)
            rep = self.fit(stage_iters, lr_frame * (0.75 ** si), log_every, verbose,
                           sizes=sizes, patience=patience)
        # **위상 단계는 기본이다.** 크기만 맞추면 위상은 물리가 강제하는 곳에서만 맞는다.
        # 실측(코퍼스 150 ms 창 4 개): 조화 SNR +3.3/+1.0/+3.5/−2.6 -> +24.7/+16.5/+12.9/+16.2,
        # 위상 모양 오차 7.6/29.9/65.5/25.4° -> 3.4/17.8/32.6/16.4°. 포락은 안 나빠졌다.
        # **위상 단계는 예산이 모자랐다.** 예전 값(lr = lr_frame·0.4 = 0.016, 200 반복)은
        # 어느 구간에서도 바닥에 못 닿았다. 학습률과 반복만 올린 실측(무제약):
        #
        #   | 구간      | 예전 (lr 0.016 / 200) | lr 0.12 / 800 |
        #   |-----------|-----------------------|---------------|
        #   | 정상 모음 |                  3.4° |      **0.1°** |
        #   | 설측 닐   |                 96.5° |     **36.4°** |
        #   | 탄음 이리 |                 50.9° |     **18.3°** |
        #
        # 기구를 하나도 안 더하고 설측 2.6 배, 탄음 2.8 배다. 크기 적합 단계는 설측의
        # 위상을 아예 못 건드렸고(94.8 -> 96.5°) 오직 이 단계만 움직였다.
        if phase_iters > 0:
            self.phase_weight = phase_w
            if verbose:
                print(f"  3 단계  위상 (가중 {phase_w}, {phase_iters} 반복)", flush=True)
            if isinstance(lr_phase, (int, float)):
                lr_p = float(lr_phase)
            else:
                lr_p = self.pick_lr_phase(lr_phase, max(30, phase_iters // 8), verbose)
                if verbose:
                    print(f"      -> lr {lr_p:.3f} 선택")
            rep = self.fit(phase_iters, lr_p, log_every, verbose, sizes=FFT_SIZES,
                           patience=patience)
            self.phase_weight = 0.0
        self.sizes = list(FFT_SIZES)
        return rep

    # ------------------------------------------------------------- 결과
    def result_track(self) -> ControlTrack:
        with torch.no_grad():
            v = self.control()[0].double().cpu().numpy()
        return ControlTrack(v, self.track.frame_ms, list(self.track.events))

    def render(self, seed: int | None = None) -> np.ndarray:
        """적합된 파라미터로 합성. `seed` 를 주면 **난류의 실현만** 바꾼다.

        같은 파라미터를 시드만 바꿔 두 번 합성하면, 그 둘의 거리가 곧 "완벽한 모형이라도
        남는" 실현 잡음이다. 일치율에서 그 바닥을 빼면 편향만 남는다
        (`turbulence.corrected_sc`). 치찰음을 정직하게 채점하는 유일한 길이다.

        시드는 **`cfg.seed` 로** 바꾼다. `eng.noise` 에 직접 대입하면 조용히 무시된다 —
        `synth()` 가 부르는 `VoiceEngine.reset()` 이 `NoiseBank(self.cfg.seed)` 로
        덮어쓰기 때문이다.
        """
        with torch.no_grad():
            if seed is None:
                return self.synth()[0].cpu().numpy()
            old = self.eng.cfg.seed
            try:
                self.eng.cfg.seed = int(seed)
                return self.synth()[0].cpu().numpy()
            finally:
                self.eng.cfg.seed = old

    def fidelity(self, seed_b: int = 991) -> dict:
        """실현 잡음을 뺀 성적표. 치찰음을 포함한 구간에서 유일하게 정직한 값이다.

        `fine` 은 지금까지 쓰던 값(비교용), `fine_corr` 이 편향만 남긴 값이다.
        `floor` 는 원래 자가 원리적으로 넘을 수 없는 상한 — `fine` 이 이 근처면
        **모형이 나쁜 게 아니라 자가 바닥에 닿은 것**이다.
        """
        from . import turbulence as tb
        tgt = self.target[0].detach().cpu().numpy().astype(np.float64)
        a = self.render()
        b = self.render(seed_b)
        out = tb.spectral_fidelity(tgt, a, b, self.fs, tuple(FFT_SIZES), self.f_max)
        out["spectrum_match"] = tb.spectrum_match(tgt, a, self.fs)
        return out

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
