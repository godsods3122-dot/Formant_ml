"""VoiceEngine — 스크립트(ms 단위 물리 factor) -> 오디오. 오프라인과 스트리밍이 같은 코드다.

    eng = VoiceEngine()                                  # 여성 화자 기본값, 44.1 kHz
    track = track_from_keyframes([...])                  # 또는 phones.syllables("아라")
    y = eng.render(track)                                # (N,) numpy
    for chunk in eng.stream(track, chunk_ms=20): ...     # 상태를 이어 가며 청크 출력

파이프라인 (사용자 명세 순서):

    GlottalSource ─ du ─────────────────────────────┐
    FricationNoise ─ 마찰 소스 (협착 레이놀즈) ─────┤
    AspirationNoise ─ 기식 (성문 포락선 × 색) ──────┼─> VocalTract (포먼트+올패스 -> 비강 -> 측지) ─> ResidualCorrector -> y
    TransientTemplateBank ─ 이벤트 과도음 ──────────┘

상태(스트리밍): 성문 위상, 모든 biquad 의 zi, 프레임 카운터. 청크 경계는 존재하지 않는 것과
같다 — 필터가 샘플마다 이어지므로 오프라인 결과와 부동소수 오차 안에서 같다(테스트).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn as nn

from .control import INDEX, PARAM_NAMES, ControlTrack, frames_to_samples

#: 보이스 바 벽 공진 (Hanna et al. 2016) — `vbar_gain` 이 있을 때만 쓴다 (§52.433).
HCORR_GATED = False
HCORR_AMP_REF = 0.4
#: **입의 닫힘을 공기역학도 보게 한다** (MEASUREMENTS §52.449). 성도 음향은 `oral_open`(구강 가지 게이트)으로 입을 닫고, 공기역학
#: (구강압 방정식·마찰 음원·기식의 압력 몫)은 `a_c` 만 본다 — 같은 기관을 두 손잡이가 나눠 가졌다. 적합된 파열 폐쇄는 게이트만 닫고
#: 협착은 열어 두어(K1a ㅍ 폐쇄 a_c 6.3~6.7 cm², oral_open 0.03; P3a 5.8~6.4) 구강압이 안 차고 개방 버스트가 약했다(K1 ㅍ 개방 저·중역
#: −15/−13 dB). 켜면 공기역학 모듈에 **직렬 오리피스의 유효 면적** A_eff = (a_c⁻² + A_open⁻²)^(−1/2) 를 넘긴다(성문-협착 직렬 유량과 같은 식),
#: A_open = `AERO_OPEN_MAX` × ((oral_open − 0.02)/0.98)² — 게이트 하한 0.02 에서 0, 열린 모음에서 입술 면적. 성도 음향은 그대로.
ORAL_AERO_GATE = False
AERO_OPEN_MAX = 4.0        # 열린 모음의 입술 면적 [cm²] (Fant 1960: /a/ 4~5 cm²)
#: **파열 개방의 면적 증가 속도** (MEASUREMENTS §52.456). 조음기는 질량이 있어 접촉이 떨어진 뒤 협착 면적이 유한한 속도로 커진다 —
#: Stevens (Acoustic Phonetics, 1998): 개방 순간 양순 ≈100, 치조 ≈50, 연구개 ≈25 cm²/s (Maddieson 2001 이 인용). 한국어 연구개 파열 (Song, Kim &
#: Cho 2023, EMA, 여성): 혀 등 개방 최고 속도 ㄱ 13.1±2.7, ㅋ 12.0±1.9, ㄲ 14.6±3.3 cm/s — 폭 ≈2 cm 면 ≈25 cm²/s 로 Stevens 와 맞는다. 버스트는 이 느린
#: 개방 동안 좁은 틈을 지나는 난류(+구강압 방전)라, 면적이 1 ms 에 수 cm² 로 뛰면(적합기 K3fa ㄲ: 0.0012 → 3.58 cm², 3600 cm²/s) 구강압이 3 ms 안에
#: 빠지고 마찰 구동이 /ㅅ/ 보다 16 dB 약해 **버스트가 안 난다**. 켜면 검출된 파열(`release_frames`, 절대 틀)마다 [r − PRE, r + POST) 창 안에서
#: 공기역학 유효 면적의 **증가**를 파열별 속도 `release_rate` [cm²/s] 로 묶는다(줄어드는 쪽은 그대로). 속도 값은 적합기가 고르고 문헌 분포가 사전이다
#: (fit.RELEASE_PRIOR_W) — 하드 상한이 아니다. 성도 음향(포먼트·게이트)은 건드리지 않는다.
RELEASE_RATE = False
RELEASE_PRE_MS = 10.0
RELEASE_POST_MS = 80.0
VOICE_BAR_HZ = 200.0
VOICE_BAR_BW = 100.0
from .glottis import GlottalSource
from .noise import AspirationNoise, FricationNoise, TransientTemplateBank
from .profile import SpeakerProfile
from .residual import ResidualCorrector
from .rng import NoiseBank
from .tract import VocalTract
from .tviir import tv_biquad

LOADED_SOURCE_VERSION = 1


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    """1 차원 최대 팽창. 마찰 게이트의 경계를 k 샘플만큼 넓힌다."""
    if k < 2:
        return x
    return torch.nn.functional.max_pool1d(x.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)


@dataclass
class EngineConfig:
    sample_rate: int = 48000          # 1 ms 프레임 = 정확히 48 샘플 (44.1 kHz 는 44.1 이라 시간축이 어긋난다)
    frame_ms: float = 1.0
    speaker: str = "female"
    tract_length_cm: float = 14.6
    n_extra_formants: int | None = None      # None = 기본 6 개. F_K 위로 c/(2L) 간격 (고차 극 보정)
    residual: bool = True
    seed: int = 0
    # Experimental A/B only; keep legacy until held-out pronunciation evidence.
    # Internal noise parameters are not fitted by the default control fitter.
    noise_modulation: str = "legacy"    # "lf": shared Rd-linked flow AM
    glottal_source: str = "lf"          # "loaded": prescribed-flow impedance proxy
    load_coupling: float = 1.0         # zero is an exact legacy bypass

    def __post_init__(self):
        if self.glottal_source not in ("lf", "loaded"):
            raise ValueError("glottal_source must be 'lf' or 'loaded'")
        if not math.isfinite(self.load_coupling) or not 0.0 <= self.load_coupling <= 1.0:
            raise ValueError("load_coupling must be finite and in [0, 1]")

    @property
    def loaded_source_enabled(self) -> bool:
        return self.glottal_source == "loaded" and self.load_coupling > 0.0

    @property
    def hop(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))


# **성문 펄스의 하모닉 위상 분산 (rad @ 나이퀴스트).**
#
# `glottis.forward` 는 하모닉마다 위상 오프셋 `rps` 를 받도록 되어 있는데 **아무도
# 넘기지 않고 있었다** (v1 에서 검증된 기능인데 v2 재구축에서 배선이 빠졌다).
# 그래서 모든 하모닉이 LF 스펙트럼의 위상 그대로 정렬되고, 시간영역에서 날카로운
# 펄스가 된다 — 실측: 고역 4~12 kHz 포락 첨도가 목표 27.2 인데 합성 42.3.
#
# 실제 성대는 딱딱하지 않고 점막파로 상하연이 시간차를 두고 닫히므로, 고역일수록
# 지연이 커지는 **분산**이 생긴다. 그것을 이차 위상 `−D·(f/f_nyq)²` 로 준다
# (= 주파수에 선형인 그룹 지연 = 시간축으로 퍼지는 chirp).
#
# 0 이면 항이 통째로 빠지므로 **거동이 예전과 완전히 같다.** 기본값은 실측으로 정한다.
GLOTTAL_DISPERSION = 0.0
#: **시간 영역 관 경로** (MEASUREMENTS §52.476, `voice_td.TDPath`, `tube_td`). 켜면 성문 면적 → 비선형 유량 → 임피던스 관(벽·방사·비강) →
#: 레이놀즈 난류를 한 풀이기로 낸다. LF 조화 음원·공명기 직렬·따로 더하던 잡음은 안 쓰인다. 통짜 렌더 전용(스트리밍 상태 없음).
TD_TUBE = False
#: 조음 모형(`voice_td.ARTIC` = "w02"·"maeda")에서 구강압 상미분방정식과 치찰 협착 사전이 볼 협착 면적을 **관의 실제 구강 최소 면적**으로
#: (`TDPath.oral_min_area`, §52.487). 끄면 예전처럼 제어 `a_c` — 그 모드에서는 관 모양에 안 쓰이는 값이라 관과 무관한 구강압 손잡이가 된다
#: (WPB 의 ㄲ 자리에서 적합기가 `a_c` 를 0 으로 내려 성문 진폭을 줄였다).
TUBE_AC = False

#: **외부 음원 시간 늘이기** (§52.59). 적합기의 f0_target / 음원 자신의 F0 비율을 이 폭 안에 가두고,
#: (비율 − 1) 을 시상수 EXT_WARP_TAU 의 누설 적분으로 표본 어긋남에 쌓는다 — 어긋남은 EXT_WARP_MAX·τ·fs (5 %·50 ms = 2.5 ms) 를
#: 넘지 않아 음원이 기식·마찰 경로와 멀어지지 않는다. 짧은 구간의 F0·위상은 LF 의 f0_target 처럼 고칠 수 있다.
EXT_WARP_MAX = 0.05
EXT_WARP_TAU = 0.05

class VoiceEngine(nn.Module):
    def __init__(self, cfg: EngineConfig | None = None, profile: SpeakerProfile | None = None):
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self.profile = profile
        fs, hop = self.cfg.sample_rate, self.cfg.hop
        f0r = (profile.f0_lo, profile.f0_hi, profile.f0_nominal) if profile else None
        if profile:
            self.cfg.tract_length_cm = profile.tract_length_cm
        self.loaded_source = None
        self.source_add = None      # 성문 음원에 **더하는** 보정 (fit.HCORR_K, §52.268)
        #: 구강 과도음 이벤트의 **이벤트별 이득** (fit.TRANSIENT_FIT, §52.307). `(E,)` 텐서를 넣으면
        #: `events` 순서대로 곱해진다 — 크기를 손으로 훑는 대신 적합기가 고른다.
        self.transient_gain = None
        #: 파열 개방 속도 (`RELEASE_RATE`, §52.456): 절대 틀 색인 목록과 파열별 속도 (K,) [cm²/s]
        self.release_frames = None
        self.release_rate = None
        # **외부 성문 음원** (1, N) — 물리 시뮬레이션이 만든 dU/dt (docs/MEASUREMENTS §52.31). None 이면 LF 음원 그대로.
        self.external_du = None
        # 외부 음원의 **성문 개방 곡선** (1, N) 0~1 — 물리 성문 면적에서 만든다. 있으면 성도의 `OPEN_DAMP` 가 LF 위상 대신 이것을 본다.
        self.external_open = None
        self.external_env = None
        self.external_f0 = None
        if self.cfg.loaded_source_enabled:
            from .loaded_source import ImpedanceLoadedSource
            self._check_source_options()
            self.loaded_source = ImpedanceLoadedSource(
                fs, self.cfg.tract_length_cm, self.cfg.load_coupling)
        self.glottis = GlottalSource(fs, hop, speaker=self.cfg.speaker, f0_range=f0r,
                                     noise_modulation=self.cfg.noise_modulation,
                                     flow_reference=self.cfg.loaded_source_enabled)
        self.frication = FricationNoise(fs, hop)
        self.aspiration = AspirationNoise(fs, hop)
        self.transients = TransientTemplateBank(fs)
        fbw = float(profile.sibilant.get("front_bw_slope", 0.20)) if profile else 0.20
        from .voice_td import TDPath
        self.td = TDPath(fs, hop)
        self.tract = VocalTract(fs, hop, length_cm=self.cfg.tract_length_cm,
                                n_extra=self.cfg.n_extra_formants,
                                front_bw_slope=fbw,
                                piriform_hz=(profile.piriform_hz if profile else 0.0),
                                trough_hz=(getattr(profile, "trough_hz", 0.0) if profile else 0.0))
        self.residual = ResidualCorrector(fs, hop) if self.cfg.residual else None
        self.reset()


    # ------------------------------------------------------------ 상태
    @staticmethod
    def _check_source_options():
        from . import glottis, tract
        if tract.OPEN_DAMP:
            raise ValueError("Loaded source and OPEN_DAMP cannot be combined")
        if glottis.HJIT_CYCLE and glottis.HARM_PHASE_JIT > 0:
            raise ValueError("Loaded source does not support non-streaming HJIT_CYCLE")

    def _pulse_rate(self, phase, sample0: int, n: int, batch: int):
        """Rate of the full, unwrapped external phase; never modulo differences.

        The very first sample has no predecessor, so use the first available
        interval there. Later chunks use the preceding sample of the same
        caller-supplied absolute phase trajectory.
        """
        if (phase.ndim != 2 or phase.shape[0] != batch
                or phase.shape[1] < max(2, sample0 + n)):
            raise ValueError("Loaded pulse_phase must supply the full phase trajectory "
                             "through emitted samples, with at least two samples")
        phase = phase.double()
        if sample0:
            previous = phase[:, sample0 - 1:sample0 + n - 1]
        else:
            previous = torch.cat((2 * phase[:, :1] - phase[:, 1:2], phase[:, :n - 1]), -1)
        rate = (phase[:, sample0:sample0 + n] - previous) * (self.cfg.sample_rate / (2 * math.pi))
        if not bool(torch.isfinite(rate).all()) or bool((rate <= 0).any()):
            raise ValueError("Loaded pulse_phase must be finite, unwrapped and strictly increasing")
        return rate

    def set_external_source(self, du, g_open=None, f0_src=None) -> None:
        """물리 음원 dU/dt (N,) 또는 (1, N) 을 건다. None 이면 떼어 LF 음원으로 돌아간다. 샘플률은 엔진과 같아야 한다.

        `g_open` 은 같은 길이의 0~1 성문 개방 곡선이다 (없으면 성도의 개방기 감쇠는 LF 위상을 본다). 기식 잡음의 주기 변조는
        아직 LF 위상을 따른다 — 알려진 한계 (§52.38).
        """
        if du is None:
            self.external_du = None
            self.external_open = None
            self.external_env = None
            self.external_f0 = None
            return
        t = torch.as_tensor(du, dtype=torch.float32)
        t = t.reshape(1, -1) if t.ndim == 1 else t
        if t.ndim != 2 or t.shape[0] != 1 or not bool(torch.isfinite(t).all()):
            raise ValueError("external source must be a finite (N,) or (1, N) array")
        self.external_du = t
        self.external_open = None
        # 무성 벌점이 쓸 **음원 포락** (1, N) — 프레임 rms 를 5 프레임 평균으로 펴고, 음원이 서 있는 프레임(최대의 1 % 위) 의
        # 중앙값을 1 로 둔다 (§52.46). 적합기는 이것 × voice_gain 을 무성 구간에서 문다.
        hop = int(self.cfg.hop)
        nf = int(t.shape[-1]) // hop
        if nf == 0:
            self.external_env = torch.ones_like(t)
        else:
            fr_rms = t[0, :nf * hop].double().reshape(nf, hop).pow(2).mean(1).sqrt()
            if nf >= 5:
                fr_rms = torch.nn.functional.avg_pool1d(fr_rms.view(1, 1, -1), 5, 1, 2,
                                                        count_include_pad=False).view(-1)
            live = fr_rms > fr_rms.max() * 1e-2
            ref = fr_rms[live].median() if bool(live.any()) else torch.ones((), dtype=fr_rms.dtype)
            env = (fr_rms / ref.clamp_min(1e-12)).float().repeat_interleave(hop)
            if env.shape[0] < t.shape[-1]:
                env = torch.cat([env, env[-1:].expand(int(t.shape[-1]) - env.shape[0])])
            self.external_env = env.view(1, -1)
        if g_open is not None:
            o = torch.as_tensor(g_open, dtype=torch.float32)
            o = o.reshape(1, -1) if o.ndim == 1 else o
            if o.shape != t.shape or not bool(torch.isfinite(o).all()) or bool((o < 0).any()) or bool((o > 1).any()):
                raise ValueError("g_open must be a finite 0..1 array with the same shape as du")
            self.external_open = o
        # 음원 자신의 프레임 F0 [Hz] — 있으면 적합기의 f0_target 과의 비율로 음원을 시간 축에서 늘인다 (§52.59). 0 이하는 비율 1.
        self.external_f0 = None
        if f0_src is not None:
            f = torch.as_tensor(np.nan_to_num(np.asarray(f0_src, dtype=np.float64), nan=0.0), dtype=torch.float64)
            self.external_f0 = f.reshape(1, -1)

    def set_voicing_gate(self, gate) -> None:
        """**목표에 성문 펄스가 없는 구간에서 배음 음원을 0 으로 막는다** (사용자 지시).

        > "음원에서 펄스를 인지하지 못했다면, 그 구간에 펄스를 아예 못 쓰게 하도록 해.
        >  자꾸 미세한 펄스가 생겨버리네."

        벌점이 아니라 **차단**이다 — 적합기가 무성 구간에 미세한 배음을 흘려 포락을 맞추는 길을
        아예 막는다. 기식·마찰 경로는 건드리지 않는다 (무성음은 그쪽이 만든다).
        `gate` 는 표본률 1 차원 배열(0~1), None 이면 끈다. 구간 밖은 1 로 본다.
        """
        self.voicing_gate = (None if gate is None else
                             torch.as_tensor(gate, dtype=torch.float32).reshape(1, -1))

    def _voicing_gate_slice(self, s0: int, n: int, dtype, device):
        g = getattr(self, "voicing_gate", None)
        if g is None:
            return None
        seg = g[:, s0:s0 + n]
        if seg.shape[-1] < n:
            seg = torch.nn.functional.pad(seg, (0, n - seg.shape[-1]), value=1.0)
        return seg.to(device=device, dtype=dtype)

    def set_noise_bank(self, bank) -> None:
        """난수 은행을 갈아 끼운다 (`rng.LockedNoiseBank` 등). **`reset()` 이 덮어쓰지 않는다.**

        `eng.noise = ...` 로 직접 넣으면 `reset()` 이 새 `NoiseBank` 로 갈아 끼워 조용히 무시된다 —
        실제로 그렇게 넣은 `--noise-lock` 판이 난수 판과 **바이트 단위로 같은** 소리를 냈다.
        None 을 주면 기본 난수 은행으로 돌아간다.
        """
        self._noise_override = bank
        self.noise = bank if bank is not None else NoiseBank(self.cfg.seed)

    def _limit_release(self, a: torch.Tensor, f0i: int, t_emit: int, st: dict) -> torch.Tensor:
        """`RELEASE_RATE`: 파열 창 안에서 유효 면적 a (B, T_all) 의 증가를 A(k) ≤ A(k−1) + R·Δt 로 묶는다 (절대 틀 기준, 스트리밍 상태 유지)."""
        dt = self.cfg.hop / float(self.cfg.sample_rate)
        pre = int(round(RELEASE_PRE_MS * 1e-3 / dt))
        post = int(round(RELEASE_POST_MS * 1e-3 / dt))
        t_all = a.shape[1]
        cols = list(a.unbind(1))
        rs = st.setdefault("release", {})
        rate = self.release_rate
        for k, r in enumerate(self.release_frames):
            lo, hi = int(r) - pre, int(r) + post
            s_, e_ = max(lo, f0i), min(hi, f0i + t_all)
            if s_ >= e_:
                continue
            step = rate[k].to(a.dtype) * dt if torch.is_tensor(rate) else float(rate[k]) * dt
            prev = rs.get(k) if s_ > lo else None
            for f in range(s_, e_):
                i = f - f0i
                cur = cols[i] if prev is None else torch.minimum(cols[i], prev + step)
                cols[i] = cur
                prev = cur
                if f == f0i + t_emit - 1:
                    rs[k] = cur.detach()
        return torch.stack(cols, 1)

    def reset(self) -> None:
        self.voicing_gate = getattr(self, "voicing_gate", None)
        self.state = dict(phase=None, amp=None, tract={}, residual={}, glottis={},
                          fric={}, asp={}, frame=0)
        override = getattr(self, "_noise_override", None)
        self.noise = override if override is not None else NoiseBank(self.cfg.seed)

    # ------------------------------------------------------------ 핵심
    def forward(self, ctrl: torch.Tensor, events: list[dict] | None = None,
                t_offset_s: float = 0.0, state: dict | None = None,
                emit: int | None = None,
                pulse_phase: torch.Tensor | None = None) -> dict:
        """ctrl: (B, T, P) 프레임률 제어 텐서. events: 과도음 이벤트(절대 시각).

        emit: 내보낼 프레임 수. 스트리밍은 T+1 프레임을 주고 emit=T 로 불러 마지막
        프레임의 샘플 보간이 다음 프레임을 보게 한다(선행 1 프레임 = 1 ms 지연).
        """
        st = state if state is not None else self.state
        hop = self.cfg.hop
        b, t_all, _ = ctrl.shape
        t = t_all if emit is None else int(emit)
        n = t * hop
        c = {name: ctrl[..., INDEX[name]] for name in PARAM_NAMES}
        c_aero = c
        if ORAL_AERO_GATE:
            _g = ((c["oral_open"] - 0.02) / 0.98).clamp(0.0, 1.0) ** 2
            _a_open = AERO_OPEN_MAX * _g
            _a_eff = torch.rsqrt(c["a_c"].clamp_min(1e-4) ** -2 + (_a_open + 1e-6) ** -2)
            c_aero = dict(c)
            c_aero["a_c"] = _a_eff
        f0i, s0 = st["frame"], st["frame"] * hop
        if RELEASE_RATE and self.release_frames is not None and len(self.release_frames):
            c_aero = dict(c_aero)
            c_aero["a_c"] = self._limit_release(c_aero["a_c"], f0i, t, st)
        if TD_TUBE:
            a_c = c["a_c"]
            if RELEASE_RATE and self.release_frames is not None and len(self.release_frames):
                a_c = self._limit_release(a_c, f0i, t, st)
            c_td = dict(c)
            c_td["a_c"] = a_c
            self.last_tube_ac = None
            from . import voice_td as _vtd
            if TUBE_AC and _vtd.ARTIC in ("w02", "maeda"):
                c_td["a_c"] = self.td.oral_min_area(c_td).to(a_c.dtype)
                self.last_tube_ac = c_td["a_c"]
            stg = self.glottis.physiology(c_td)
            if pulse_phase is None:
                f0s = frames_to_samples(stg["f0"].unsqueeze(-1), hop)[:, :n, 0]
                pp = torch.cumsum(2.0 * math.pi * f0s.double() / float(self.cfg.sample_rate), -1)
            else:
                pp = pulse_phase[:, s0:s0 + n]
            r = self.td(c_td, stg, pp, n)
            y = r["audio"]
            up = lambda v: frames_to_samples(v.unsqueeze(-1), hop)[:, :n, 0]
            z = torch.zeros_like(y)
            st["frame"] += t
            return dict(audio=y, du=z, fric=z, asp=z, transient=z, glottal_path=y, front_path=z,
                        f0=up(stg["f0"]), amp=up(stg["amp"]), phase=torch.remainder(pp, 2 * math.pi).to(y.dtype),
                        reynolds=z, glottal_area=r["glottal_area"], state=st)
        source_rate = None
        if self.loaded_source is not None:
            self._check_source_options()
            if ctrl.device.type != "cpu":
                raise ValueError("The loaded glottal source currently supports CPU only")
            if pulse_phase is not None:
                source_rate = self._pulse_rate(pulse_phase, s0, n, b)
        pp = None if pulse_phase is None else pulse_phase[:, s0:s0 + n]
        g = self.glottis(c_aero, phase0=st["phase"], noise=self.noise, frame0=f0i,
                         dispersion=GLOTTAL_DISPERSION,
                         amp0=st["amp"], state=st["glottis"], emit=t, pulse_phase=pp)
        prepared_tracks = None
        loaded = None
        if self.loaded_source is not None:
            if source_rate is None:
                source_rate = g["f0"]
            prepared_tracks = self.tract.prepare_tracks(c, n, st["tract"])
            ps = frames_to_samples(c["p_sub"].unsqueeze(-1), hop)[:, :n, 0]
            loaded = self.loaded_source(
                g["reference_flow"], g["flow_scale"], source_rate, g["ag_dc"],
                ps, prepared_tracks, st.get("loaded", {}), legacy_du=g["du"])
            g["du"] = loaded["du"]
            st["loaded"] = loaded["state"]
        if self.external_du is not None:
            # 외부 음원으로 성문 배음을 **갈아 끼운다.** 진폭·닫힘·지터는 물리가 정하고, 적합기는 성문 배음의 빠른 이득
            # `voice_gain` 만 곱한다. 기식·마찰 경로(asp_env, ag_dc)는 그대로 LF 생리에서 온다.
            ext = self.external_du
            if ext.shape[-1] < s0 + n:
                raise ValueError(f"external source has {ext.shape[-1]} samples, needs {s0 + n}")
            vg = frames_to_samples(torch.pow(10.0, c["voice_gain"] / 20.0).unsqueeze(-1), hop)[:, :n, 0]
            if self.external_f0 is None:
                g["du"] = ext[:, s0:s0 + n].to(device=vg.device, dtype=vg.dtype).expand(b, -1) * vg
                if self.external_env is not None:
                    g["amp_ext"] = (self.external_env[:, s0:s0 + n].to(device=vg.device, dtype=vg.dtype)
                                    .expand(b, -1) * vg)
                if self.external_open is not None:
                    g["open_phase"] = self.external_open[:, s0:s0 + n].to(device=vg.device, dtype=vg.dtype).expand(b, -1)
            else:
                # **시간 늘이기** (§52.59): 비율 r = f0_target / 음원 F0 (±EXT_WARP_MAX), 어긋남 D[k] = (r[k] − 1) + a·D[k−1] [표본].
                t_all_f = c["f0_target"].shape[-1]
                sf0 = self.external_f0[:, f0i:f0i + t_all_f].to(device=vg.device)
                if sf0.shape[-1] < t_all_f:
                    sf0 = torch.cat([sf0, torch.zeros(1, t_all_f - sf0.shape[-1], dtype=sf0.dtype, device=sf0.device)], -1)
                ft = c["f0_target"].to(torch.float64)
                ok = (sf0 > 0) & (ft > 0)
                ratio = torch.where(ok, ft / sf0.clamp_min(1e-6), torch.ones_like(ft))
                ratio = ratio.clamp(1.0 - EXT_WARP_MAX, 1.0 + EXT_WARP_MAX)
                rr = frames_to_samples((ratio - 1.0).unsqueeze(-1), hop)[:, :n, 0]
                a_w = math.exp(-1.0 / (EXT_WARP_TAU * float(self.cfg.sample_rate)))
                off, zf = tv_biquad(rr, 1.0, 0.0, 0.0, -a_w, 0.0, zi=st.get("ext_warp"))
                st["ext_warp"] = zf.detach()
                pos = (torch.arange(s0, s0 + n, device=vg.device, dtype=torch.float64).unsqueeze(0)
                       + off.to(torch.float64)).clamp(0.0, float(ext.shape[-1] - 2))
                i0 = pos.detach().floor()
                w = pos - i0
                i0 = i0.long()

                def _read(sig):
                    e = sig[0].to(device=vg.device, dtype=torch.float64)
                    return e[i0] * (1.0 - w) + e[i0 + 1] * w

                g["du"] = _read(ext).to(vg.dtype) * vg
                if self.external_env is not None:
                    g["amp_ext"] = _read(self.external_env).to(vg.dtype) * vg
                if self.external_open is not None:
                    g["open_phase"] = _read(self.external_open).to(vg.dtype).detach()
        fr = self.frication(c_aero, g["ag_dc_frames"], g["phase"], g["voiced"], noise=self.noise,
                            frame0=f0i, state=st["fric"], emit=t, noise_am=g["noise_am"],
                            po_frames=g.get("po_frames"), uc_frames=g.get("uc_frames"))
        _ct = None
        from . import noise as _nz
        if _nz.ASP_STROUHAL:
            # 셸프 꺾임 = 성문 제트의 스트로할 주파수 (noise.ASP_STROUHAL, §52.432)
            from . import glottis as _gl
            _up = lambda v: frames_to_samples(v.unsqueeze(-1), hop)[:, :n, 0]
            _A = _up(g["ag_dc_frames"]) + 0.5 * _gl.GLOT_A_PEAK * g["amp"][:, :n] * g["voiced"][:, :n]
            _ac = _up(c_aero["a_c"].clamp_min(1e-3))
            _dp = _up(c["p_sub"].clamp_min(0.0)) * _ac ** 2 / (_ac ** 2 + _A ** 2) * _gl.CMH2O
            _v = torch.sqrt(2.0 * _dp / _gl.RHO + 1e-6)
            _ct = (_nz.ASP_ST * _v / (2.0 * _A / _gl.GLOT_LEN_CM)).clamp(400.0, 8000.0)
            self._last_asp_corner = _ct.detach()
        asp = self.aspiration(g["asp_env"], noise=self.noise, sample0=s0, state=st["asp"],
                              voiced=g["voiced"], corner_t=_ct)
        ev = [dict(e, t=e["t"] - t_offset_s) for e in (events or [])
              if -0.1 <= e["t"] - t_offset_s < n / self.cfg.sample_rate]
        if ev and self.transient_gain is not None:
            # **이벤트마다 따로 그려 이득을 곱한다.** 한꺼번에 그리면 이득이 갈리지 않는다.
            # 잘린 이벤트가 있어도 색인이 어긋나지 않게 원래 자리를 들고 다닌다.
            keep = [k for k, e in enumerate(events or [])
                    if -0.1 <= e["t"] - t_offset_s < n / self.cfg.sample_rate]
            gg = self.transient_gain.to(ctrl.device)
            tr = torch.zeros(b, n, device=ctrl.device, dtype=gg.dtype)
            for k, e1 in zip(keep, ev):
                one = self.transients.render([e1], n, b, noise=self.noise,
                                             device=ctrl.device, sample0=s0)
                tr = tr + one.to(gg.dtype) * gg[k]
        else:
            tr = self.transients.render(ev, n, b, noise=self.noise, device=ctrl.device,
                                        sample0=s0) if ev else torch.zeros(b, n, device=ctrl.device)
        if self.source_add is not None:
            # **성문 음원에 더하는 보정** (fit.HCORR_K, §52.268). 성도 앞이므로 성도 응답을 그대로 탄다 —
            # 물리적으로 "주기마다 달라지는 성문 유량 성분" 이다. 유성 게이트도 함께 받는다.
            add = self.source_add
            if add.shape[-1] < s0 + n:
                raise ValueError(f"source_add has {add.shape[-1]} samples, needs {s0 + n}")
            add_s = add[:, s0:s0 + n].to(device=g["du"].device, dtype=g["du"].dtype)
            if HCORR_GATED:
                # **보정은 성대가 떨 때만 소리를 낸다** (MEASUREMENTS §52.437). 더하기 보정은 성문 음원이 꺼져 있어도
                # 목소리를 지어낼 수 있어, 최근 판들은 유성 틀의 52~63 % 에서 내전 < 0.10 (성대가 안 떪) 으로 두고
                # 보정만으로 목소리를 냈다 — "파형이 분리돼서 들려". 성문 떨림 진폭(게이트 포함)을 기준 진폭으로 나눠 곱한다.
                add_s = add_s * (g["amp"][:, :n] / HCORR_AMP_REF).clamp(0.0, 2.0)
            g["du"] = g["du"] + add_s
        vgate = self._voicing_gate_slice(s0, n, g["du"].dtype, g["du"].device)
        if vgate is not None:
            g["du"] = g["du"] * vgate
        out = self.tract(g["du"], fr["source"], asp.get("source_v", asp["source"]), tr, c,
                         state=st["tract"], asp_u=asp.get("source_u"), g_open=g.get("open_phase"),
                         prepared_tracks=prepared_tracks)
        y = out["audio"]
        vbg = getattr(self, "vbar_gain", None)
        if vbg is not None:
            # **보이스 바 — 닫힌 성도의 벽 방사** (MEASUREMENTS §52.433). 입·코가 닫혀도 성대가 떨면 볼·목 벽이 울려 저역이 새어 나온다.
            # 벽 질량과 성도 안 공기의 순응도가 ≈ 200 Hz 에서 공진한다 (Hanna et al. 2016, JASA: 여성 유효 질량 0.15 kg, 강성 2.1 kN/m).
            # 성문 음원을 그 공진기로 거르고 닫힌 정도 (1 − oral_open)(1 − velum) 만큼 더한다. 세기는 적합기의 전역 스칼라.
            from .tviir import resonator_coeffs
            _cl = frames_to_samples(((1.0 - c["oral_open"].clamp(0, 1)) * (1.0 - c["velum"].clamp(0, 1))).unsqueeze(-1),
                                    hop)[:, :n, 0]
            b0, b1, b2, a1, a2 = resonator_coeffs(VOICE_BAR_HZ, VOICE_BAR_BW, float(self.cfg.sample_rate))
            vb, st["vbar"] = tv_biquad(g["du"], b0, b1, b2, a1, a2, zi=st.get("vbar"))
            y = y + vbg.to(y.dtype) * vb * _cl
        if self.residual is not None:
            heads = self.residual(ctrl, state=st["residual"], emit=t)
            # **마찰 구간에서는 잔차를 끈다.** 사용자 규칙: 치찰음은 학습에 맡기지 말고
            # 직접 구현할 것. 잔차 EQ 는 전 대역·전 구간에 걸리므로, 막지 않으면 치찰음
            # 물리가 틀렸을 때 신경망이 ±6 dB EQ 로 덮어 버린다. 그러면 (1) 물리는 계속
            # 틀린 채로 남고 (2) 학습 데이터에 없는 새 발화에서 치찰음이 무너진다.
            # 기준은 레이놀즈 게이트를 지난 **실제 마찰 유량**이다 — 제어값이 아니라
            # 물리량이라 "마찰이 실제로 일어나는 곳" 과 정확히 일치한다.
            fr_on = (fr["source"].detach().abs() > 0).to(y.dtype)
            fr_on = _dilate(fr_on, hop)                     # 경계에서 새지 않게 넓힌다
            mix = frames_to_samples(c["residual_mix"].unsqueeze(-1), hop)[..., 0][:, :n]
            mix = mix * (1.0 - fr_on[:, :n])
            r = self.residual.apply(y, heads, mix, state=st["residual"])
            y, st["residual"] = r["audio"], r["state"]
        st["phase"] = g["phase_last"]
        st["amp"] = g["amp_last"]
        st["glottis"], st["fric"], st["asp"] = g["state"], fr["state"], asp["state"]
        st["tract"] = out["state"]
        st["frame"] += t
        result = dict(audio=y, du=g["du"], fric=fr["source"], asp=asp["source"], transient=tr,
                    glottal_path=out["glottal_path"], front_path=out["front_path"],
                    f0=g["f0"], amp=g["amp"], phase=g["phase"], reynolds=fr["reynolds"],
                    state=st)
        if loaded is not None:
            result.update(flow=loaded["flow"], load_pressure=loaded["load_pressure"],
                          glottal_flow=loaded["glottal_flow"],
                          source_rate=source_rate,
                          reference_flow=g["reference_flow"])
        if "amp_ext" in g:
            result["amp_ext"] = g["amp_ext"]
        return result

    # ------------------------------------------------------------ 편의 API
    @torch.no_grad()
    def render(self, track: ControlTrack, return_parts: bool = False):
        self.reset()
        ctrl = track.to_tensor()
        out = self.forward(ctrl, track.events, 0.0)
        y = out["audio"][0].numpy()
        return (y, {k: v[0].numpy() for k, v in out.items()
                    if torch.is_tensor(v) and v.dim() == 2}) if return_parts else y

    @torch.no_grad()
    def stream(self, track: ControlTrack, chunk_ms: float = 20.0):
        """청크 단위 생성기. 상태는 self.state 에 이어진다."""
        self.reset()
        ctrl = track.to_tensor()
        step = max(1, int(round(chunk_ms / self.cfg.frame_ms)))
        T = ctrl.shape[1]
        for i in range(0, T, step):
            j = min(T, i + step)
            t0 = i * self.cfg.hop / self.cfg.sample_rate      # 프레임 시각은 샘플 기준으로
            out = self.forward(ctrl[:, i:min(T, j + 1)], track.events, t0, emit=j - i)
            yield out["audio"][0].numpy()

    def latency_ms(self) -> float:
        return self.cfg.frame_ms      # 1 프레임: 필터는 인과이고 선행 프레임이 필요 없다
