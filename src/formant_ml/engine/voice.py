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
        self.tract = VocalTract(fs, hop, length_cm=self.cfg.tract_length_cm,
                                n_extra=self.cfg.n_extra_formants,
                                front_bw_slope=fbw)
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

    def reset(self) -> None:
        self.state = dict(phase=None, amp=None, tract={}, residual={}, glottis={},
                          fric={}, asp={}, frame=0)
        self.noise = NoiseBank(self.cfg.seed)

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
        f0i, s0 = st["frame"], st["frame"] * hop
        source_rate = None
        if self.loaded_source is not None:
            self._check_source_options()
            if ctrl.device.type != "cpu":
                raise ValueError("The loaded glottal source currently supports CPU only")
            if pulse_phase is not None:
                source_rate = self._pulse_rate(pulse_phase, s0, n, b)
        pp = None if pulse_phase is None else pulse_phase[:, s0:s0 + n]
        g = self.glottis(c, phase0=st["phase"], noise=self.noise, frame0=f0i,
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
        fr = self.frication(c, g["ag_dc_frames"], g["phase"], g["voiced"], noise=self.noise,
                            frame0=f0i, state=st["fric"], emit=t, noise_am=g["noise_am"])
        asp = self.aspiration(g["asp_env"], noise=self.noise, sample0=s0, state=st["asp"],
                              voiced=g["voiced"])
        ev = [dict(e, t=e["t"] - t_offset_s) for e in (events or [])
              if -0.1 <= e["t"] - t_offset_s < n / self.cfg.sample_rate]
        tr = self.transients.render(ev, n, b, noise=self.noise, device=ctrl.device, sample0=s0) \
            if ev else torch.zeros(b, n, device=ctrl.device)
        out = self.tract(g["du"], fr["source"], asp.get("source_v", asp["source"]), tr, c,
                         state=st["tract"], asp_u=asp.get("source_u"), g_open=g.get("open_phase"),
                         prepared_tracks=prepared_tracks)
        y = out["audio"]
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
