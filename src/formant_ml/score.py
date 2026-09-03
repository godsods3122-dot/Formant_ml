"""스크립트(YAML/JSON) -> 제어 파라미터 -> 음성.

설계 목표는 하나다: **물리모델의 모든 손잡이를 텍스트로 지정할 수 있게 한다.**
음소든 웃음이든 특별 취급이 없고, 전부 같은 파라미터를 다르게 흔든 것이다.

    name: 데모
    voice: profiles/me.json        # 없으면 기본 프로파일
    timeline:
      - {type: syllable, onset: s, vowel: a, dur: 0.45, f0: [150, 120]}
      - {type: laugh, dur: 1.1, rate_hz: 5.5, voiced: 0.85, tilt: 3}
      - {type: vowel, vowel: i, dur: 0.4, rd: [[0, 0.5], [1, 2.2]]}

값 쓰는 법 (모든 파라미터 공통)
-------------------------------
    rd: 1.2                     상수
    rd: [0.5, 2.2]              시작 -> 끝 선형
    rd: [[0, 0.5], [0.7, 0.5], [1, 2.2]]   구간별 브레이크포인트 (위치는 0~1)

주소가 있는 파라미터는 `PARAM_HELP` 에 전부 있다 (`python -m formant_ml.render
--list-params`). 세그먼트 안에서 쓰면 그 구간에만, `params:` 아래에 쓰면
전체 발화에 적용된다.
"""
from __future__ import annotations

import json
import os

import torch

from .config import Config
from .dsp.sibilant import PRESETS as SIB_PRESETS
from .dsp.sibilant import SibilantParams
from .gestures import GESTURES, base
from .models.synth import Controls, PhysicalVoiceSynth
from .presets import (FRICATIVES, LOCUS, SIB_POLE_RELEASE_HZ, VOWELS,
                      fricative_gain, vowel_table)
from .utils import band_bump, band_shelf, ramp
from .voice import VoiceProfile, extend_formants

# 프레임률 스칼라 제어 (전부 (1, T, 1))
SCALAR_PARAMS = {
    "f0": "기본주파수 [Hz]",
    "harmonic_amp": "유성 성분 세기 (0 = 무성)",
    "rd": "성문파 형상 0.3 pressed ~ 2.7 breathy (개방지수)",
    "tilt": "소스 스펙트럼 기울기 [dB/oct @1kHz] — 고역을 살리거나 죽인다",
    "jitter": "주기 요동 비율 (0.004 = 0.4%)",
    "shimmer": "진폭 요동 비율",
    "noise_entry": "난류 주입 위치 0=성문 … K=입술 … K+6=성도 완전 우회",
    "noise_am": "성문동기 노이즈 변조 깊이 (기식성)",
    "aspiration": "성문 기식 노이즈 (성도 전체를 통과). 0.01≈-25dB, 0.05≈-11dB",
    "level_db": "마찰음 레벨 [dB, 모음 기준]. /s/ 기본 -12, /f/ -26",
    "velum_open": "연구개 개도 0~1 (비음). 0 이면 비강 극-영점이 정확히 상쇄된다",
    "noise_rough": "가우시안 잡음 위의 느린(<10Hz) 진폭 흔들림. 0=순수 가우시안",
    "noise_gain": "난류 전체 세기",
    "noise_center": "난류 대역 중심 [Hz]",
    "noise_bw": "난류 대역폭 [Hz]",
    "sib_pole_f": "치찰음 앞공동 공진 [Hz]  (/s/ 5~8k, /ʃ/ 2.5~4k)",
    "sib_pole_bw": "그 대역폭 [Hz] (좁을수록 쨍하다)",
    "sib_zero_f": "치찰음 반공진 [Hz]",
    "sib_zero_bw": "반공진 대역폭 [Hz]",
    "sib_tilt": "치찰음 기울기 [dB/oct]",
    "sib_mix": "치찰음 필터 적용량 0~1",
}
FORMANT_PARAMS = {f"f{k + 1}": f"제{k + 1} 포먼트 [Hz]" for k in range(12)}
FORMANT_BW_PARAMS = {f"bw{k + 1}": f"제{k + 1} 포먼트 대역폭 [Hz]" for k in range(12)}
VECTOR_PARAMS = {
    "formant_freq": "포먼트 전체 [Hz] 리스트(상수)",
    "formant_bw": "포먼트 대역폭 전체 리스트(상수)",
    "formant_gain": "포먼트 게인 전체 리스트(상수)",
    "disp_freq": "위상차 올패스 중심주파수 리스트 [Hz]",
    "disp_radius": "위상차 올패스 극반지름 리스트 (0~0.95)",
    "area": "도파관 단면적 함수 [cm^2] (tract: waveguide 일 때)",
}
PARAM_HELP = {**SCALAR_PARAMS, **FORMANT_PARAMS, **FORMANT_BW_PARAMS,
              **VECTOR_PARAMS}
SEGMENT_TYPES = ["vowel", "glide", "fricative", "syllable", "silence",
                 *sorted(GESTURES)]


# ------------------------------------------------------------------ 값 해석
def curve(spec, t: int) -> torch.Tensor:
    """스칼라 / [시작, 끝] / [[위치, 값], ...] -> (1, T, 1)."""
    if isinstance(spec, (int, float)):
        return torch.full((1, t, 1), float(spec))
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
        return ramp(t, [(float(p), float(v)) for p, v in spec])
    vals = [float(v) for v in spec]
    if len(vals) == 1:
        return torch.full((1, t, 1), vals[0])
    pos = [i / (len(vals) - 1) for i in range(len(vals))]
    return ramp(t, list(zip(pos, vals)))


def _vowel_formants(name: str, prof: VoiceProfile, n: int) -> list[float]:
    """모음 프리셋을 화자 성도 규모로 스케일.

    성별에 따라 아예 다른 표를 쓴다. 남성 표에 상수를 곱하면 안 된다 —
    여성 성도는 인두가 구강보다 더 많이 짧아져서 모음마다 배율이 다르다.
    """
    tbl = vowel_table(getattr(prof, "vowel_set", "male"))
    scale = (prof.formants[0] / tbl["a"][0]) if prof.formants else 1.0
    f = [v * scale for v in tbl.get(name, tbl["a"])]
    return extend_formants(f + list(prof.formants[len(f):]), n)


def _locus_track(phone: str, vowel: str, prof: VoiceProfile, k: int, t: int,
                 split: float) -> torch.Tensor:
    """자음 locus -> 모음 목표의 포먼트 궤적 (1, T, K).

    실제 전이 곡선은 여기서 그리지 않는다. 계단 목표만 주고, 곡선은
    `gestures_dynamics` 의 임계감쇠 2차계가 만든다 — 혀의 관성이 곧 전이다.
    """
    vf = _vowel_formants(vowel, prof, k)
    loc = LOCUS.get(phone)
    if loc is None:
        return torch.tensor(vf).reshape(1, 1, -1).expand(1, t, k).contiguous()
    scale = (prof.formants[0] / VOWELS["a"][0]) if prof.formants else 1.0
    cf = [v * scale for v in loc]
    tracks = []
    for i in range(k):
        start = cf[i] if i < len(cf) else vf[i]
        tracks.append(ramp(t, [(0.0, start), (split, start),
                              (min(split + 1e-3, 1.0), vf[i]), (1.0, vf[i])]))
    return torch.cat(tracks, dim=-1)


# ------------------------------------------------------------- 세그먼트 만들기
def build_segment(seg: dict, prof: VoiceProfile, cfg: Config) -> dict:
    """세그먼트 하나 -> 프레임률 제어 dict (+ 'sib' 하위 dict)."""
    a = cfg.audio
    # 포먼트 개수는 화자의 성도 길이가 정한다(간격 = c/2L). 설정값은 상한.
    K = min(cfg.filt.n_formants, prof.n_formants(a.sample_rate))
    NB = cfg.noise.n_bands
    t = max(1, int(round(float(seg.get("dur", 0.3)) * a.frame_rate)))
    kind = seg.get("type", "vowel")

    if kind in GESTURES:
        opts = {k: v for k, v in seg.items()
                if k not in PARAM_HELP and k not in ("type", "dur", "sib")}
        opts.pop("vowel", None) if kind not in ("laugh", "whisper") else None
        fn = GESTURES[kind]
        try:
            c = fn(t, prof, n_formants=K, n_bands=NB, **opts)
        except TypeError:
            c = fn(t, prof, n_formants=K, n_bands=NB)
    elif kind == "silence":
        c = base(t, prof, K, NB)
        c["harmonic_amp"] = torch.zeros(1, t, 1)
        c["noise_bands"] = torch.full((1, t, NB), 1e-6)
    elif kind == "vowel":
        c = base(t, prof, K, NB, seg.get("vowel", "a"))
        c["formant_freq"] = torch.tensor(
            _vowel_formants(seg.get("vowel", "a"), prof, K)
        ).reshape(1, 1, -1).expand(1, t, K).contiguous()
    elif kind == "glide":
        names = seg.get("vowels", ["a", "i"])
        tracks = []
        for k in range(K):
            pts = [(i / max(len(names) - 1, 1), _vowel_formants(v, prof, K)[k])
                   for i, v in enumerate(names)]
            tracks.append(ramp(t, pts))
        c = base(t, prof, K, NB, names[0])
        c["formant_freq"] = torch.cat(tracks, dim=-1)
    elif kind in ("fricative", "syllable"):
        phone = seg.get("onset", seg.get("phone", "s"))
        vowel = seg.get("vowel", "a")
        c = base(t, prof, K, NB, vowel)
        cf, bw, pos, _ = FRICATIVES.get(phone, FRICATIVES["s"])
        # 레벨은 모음 대비 dB 로 지정한다 (presets.FRICATIVE_LEVEL_DB).
        g = fricative_gain(phone, seg.get("level_db"))
        # 소스는 광대역, 모양은 치찰음 필터가 만든다(둘이 겹치면 손잡이가 죽는다).
        nb = band_shelf(NB, 500.0, g, a.sample_rate).reshape(1, 1, -1)
        sp = SIB_PRESETS.get(phone)
        sib = prof.sibilant_params((1, t, 1))
        if sp is not None and seg.get("use_profile_sibilant", True) is False:
            sib = SibilantParams.constant((1, t, 1), *sp, 1.0, prof.roughness)

        if kind == "fricative":
            c["harmonic_amp"] = torch.zeros(1, t, 1)
            c["noise_bands"] = nb.expand(1, t, NB).contiguous()
            env = torch.ones(1, t, 1)
            c["formant_freq"] = _locus_track(phone, vowel, prof, K, t, split=1.0)
        else:
            # --- 자음 -> 모음 전이. 세 가지가 동시에 일어나야 '이어져' 들린다.
            #   (1) 포먼트가 자음의 조음 위치(locus)에서 모음으로 미끄러진다.
            #   (2) 협착이 열리며 압력이 성문으로 옮겨가 **기식 노이즈**가 난다.
            #       이 성분만이 성도 전체를 통과하므로, 마찰음과 모음이 같은
            #       공명기를 공유한다는 증거가 된다. 이게 없으면 두 스펙트럼을
            #       그냥 크로스페이드한 소리가 되어 따로 논다.
            #   (3) 앞공동이 길어져 치찰음 극이 내려간다.
            split = float(seg.get("onset_ratio", 0.42))
            vot = float(seg.get("vot", 0.05))          # 전이(성대 진동 시작)까지
            rel = split + vot / max(float(seg.get("dur", 0.5)), 1e-3)
            c["harmonic_amp"] = ramp(t, [(0.0, 0.0), (split, 0.0),
                                         (rel, 1.0), (1.0, 0.85)])
            env = ramp(t, [(0.0, 1.0), (split, 1.0), (rel, 0.02), (1.0, 0.01)])
            c["noise_bands"] = (nb * env).contiguous()
            c["formant_freq"] = _locus_track(phone, vowel, prof, K, t, split)
            c["aspiration"] = ramp(t, [
                (0.0, 0.0), (split, 0.0),
                (min(split + 0.03, 1.0), float(seg.get("aspiration_peak", 0.08))),
                (min(rel + 0.10, 1.0), 0.004), (1.0, 0.0)])
            # 앞공동이 열리며 치찰음 극이 내려간다
            p0 = float(sib.pole_f.reshape(-1)[0]) if torch.is_tensor(sib.pole_f) \
                else float(sib.pole_f)
            sib.pole_f = ramp(t, [(0.0, p0), (split, p0),
                                  (min(rel + 0.05, 1.0), SIB_POLE_RELEASE_HZ),
                                  (1.0, SIB_POLE_RELEASE_HZ)])
        # 치찰음 필터를 쓰는 동안에는 노이즈를 포먼트 캐스케이드에 통과시키지
        # 않는다. 앞공동 공진을 '캐스케이드의 마지막 포먼트' 와 '치찰음 극' 두
        # 군데서 모델링하면 이중 계산이 되어, 치찰음 파라미터를 돌려도 소리가
        # 안 바뀐다(캐스케이드의 고정된 극이 이긴다).
        # 성도 전체를 통과하는 성분은 위의 `aspiration` 이 따로 담당한다.
        c["noise_entry"] = torch.full((1, t, 1), float(K) + 6.0)
        sib.mix = env.clamp(0.0, 1.0) if kind == "syllable" else torch.ones(1, t, 1)
        c["sib"] = sib
    else:
        raise ValueError(f"모르는 세그먼트 type: {kind!r}. "
                         f"가능한 값: {', '.join(SEGMENT_TYPES)}")

    c.setdefault("sib", prof.sibilant_params((1, t, 1), mix=0.0))
    apply_overrides(c, seg, t, cfg, K)
    return c


def apply_overrides(c: dict, spec: dict, t: int, cfg: Config,
                    n_formants: int | None = None) -> None:
    """스크립트에 적힌 파라미터로 제어 dict 를 덮어쓴다."""
    K = n_formants or cfg.filt.n_formants
    NB = cfg.noise.n_bands
    noise_shape = {}
    for key, val in spec.items():
        if key in ("type", "dur"):
            continue
        if key in ("noise_gain", "noise_center", "noise_bw"):
            noise_shape[key] = val
        elif key.startswith("sib_"):
            field = key[4:]
            setattr(c["sib"], field, curve(val, t))
        elif key in SCALAR_PARAMS:
            c[key] = curve(val, t)
        elif key in FORMANT_PARAMS:
            k = int(key[1:]) - 1
            if k < K:
                c["formant_freq"] = c["formant_freq"].clone()
                c["formant_freq"][..., k: k + 1] = curve(val, t)
        elif key in FORMANT_BW_PARAMS:
            k = int(key[2:]) - 1
            if k < K:
                c["formant_bw"] = c["formant_bw"].clone()
                c["formant_bw"][..., k: k + 1] = curve(val, t)
        elif key in ("formant_freq", "formant_bw", "formant_gain", "area"):
            v = torch.tensor([float(x) for x in val], dtype=torch.float32)
            c[key] = v.reshape(1, 1, -1).expand(1, t, len(v)).contiguous()
        elif key in ("disp_freq", "disp_radius"):
            v = torch.tensor([float(x) for x in val], dtype=torch.float32)
            c[key] = v.reshape(1, 1, -1).expand(1, t, len(v)).contiguous()
    if noise_shape:
        sr = cfg.audio.sample_rate
        g = noise_shape.get("noise_gain", 1.0)
        ctr = noise_shape.get("noise_center", 5000.0)
        bw = noise_shape.get("noise_bw", 4000.0)
        base_nb = band_bump(NB, float(ctr) if not isinstance(ctr, list) else
                            float(ctr[0]),
                            float(bw) if not isinstance(bw, list) else float(bw[0]),
                            1.0, sr).reshape(1, 1, -1)
        c["noise_bands"] = (base_nb * curve(g, t)).contiguous()


# ---------------------------------------------------------------- 이어붙이기
def enforce_formant_spacing(freq: torch.Tensor, min_gap: float = 60.0
                            ) -> torch.Tensor:
    """F1 < F2 < … 와 최소 간격을 구조적으로 보장 (인코더와 같은 규약).

    인코더는 cumsum(softplus) 로 이걸 보장하지만 스크립트는 아무 값이나 쓸 수
    있다. 극이 겹치면 캐스케이드 이득이 폭발하므로 여기서 한 번 막는다.
    """
    out = [freq[..., :1]]
    for k in range(1, freq.shape[-1]):
        out.append(torch.maximum(freq[..., k:k + 1], out[-1] + min_gap))
    return torch.cat(out, dim=-1)


def _smooth(x: torch.Tensor, n: int) -> torch.Tensor:
    """프레임축 이동평균 (세그먼트 경계의 계단을 없앤다)."""
    if n < 2:
        return x
    k = torch.ones(x.shape[-1], 1, n, dtype=x.dtype) / n
    pad = torch.nn.functional.pad(x.transpose(1, 2), (n // 2, n - 1 - n // 2),
                                  mode="replicate")
    return torch.nn.functional.conv1d(pad, k, groups=x.shape[-1]).transpose(1, 2)


def build_controls(score: dict, prof: VoiceProfile, cfg: Config) -> Controls:
    """스크립트 전체 -> 하나의 Controls (위상은 발화 전체에서 연속이다)."""
    K = min(cfg.filt.n_formants, prof.n_formants(cfg.audio.sample_rate))
    segs = [build_segment(s, prof, cfg) for s in score.get("timeline", [])]
    if not segs:
        raise ValueError("timeline 이 비어 있습니다")
    keys = set().union(*[set(s) for s in segs]) - {"sib"}
    # 어떤 세그먼트에만 있는 키(예: aspiration)는 나머지 구간을 0 으로 채운다.
    ref = {k: next(s[k] for s in segs if k in s) for k in keys}
    merged = {}
    for k in keys:
        parts = []
        for seg in segs:
            if k in seg:
                parts.append(seg[k])
            else:
                n = seg["f0"].shape[1]
                parts.append(torch.zeros(1, n, ref[k].shape[-1],
                                         dtype=ref[k].dtype))
        merged[k] = torch.cat(parts, dim=1)
    sib_fields = {}
    for f in ("pole_f", "pole_bw", "zero_f", "zero_bw", "tilt", "mix", "roughness"):
        sib_fields[f] = torch.cat([getattr(s["sib"], f) for s in segs], dim=1)
    t = merged["f0"].shape[1]

    # 전역 오버라이드
    if score.get("params"):
        apply_overrides(merged | {"sib": SibilantParams(**sib_fields)},
                        score["params"], t, cfg, K)

    # 경계 평활. `noise_entry` 는 **일부러 빼 둔다**: 협착 위치는 연속적으로
    # 미끄러지는 양이 아니라 조음 상태다. 무음(성문 주입)에서 /s/(입술쪽 주입)로
    # 보간하면 중간 프레임이 '캐스케이드의 앞부분만 우회한 반쪽 필터' 가 되는데,
    # 그건 어떤 성도 형상에도 대응하지 않는 응답이고 국소 이득이 100 배까지
    # 솟아 전이부에 클릭을 만든다. 들리는 전이는 noise_bands 포락선이 만든다.
    n = int(score.get("smooth_frames", 2))
    for k in ("f0", "rd", "tilt", "formant_freq", "formant_bw", "harmonic_amp",
              "noise_bands", "noise_am", "noise_rough"):
        if k in merged:
            merged[k] = _smooth(merged[k], n)

    # 조음 동역학. 여기까지의 궤적은 '목표'이고, 실제 궤적은 조음기관의 관성이
    # 만든다. 파라미터마다 담당 조음기관의 시간상수가 다르다(혀끝 35 ms,
    # 혀몸통 60 ms, 턱 80 ms, 연구개 100 ms). 이걸 켜면 물리적으로 불가능한
    # 조음 속도가 표현 자체로 불가능해진다.  score 에 `coarticulation: false`
    # 를 쓰면 끄고, `speech_rate` 로 전체 속도를 조절한다(>1 이면 빠른 말투).
    if score.get("coarticulation", True):
        from .dsp.gestures_dynamics import apply_to_controls
        scale = 1.0 / max(float(score.get("speech_rate", 1.0)), 0.2)
        sib_in = {f"sib_{f}": v for f, v in sib_fields.items()}
        moved = apply_to_controls(merged | sib_in, cfg.audio.frame_rate, scale)
        merged = {k: v for k, v in moved.items() if not k.startswith("sib_")}
        sib_fields = {k[4:]: v for k, v in moved.items() if k.startswith("sib_")}

    merged["formant_freq"] = enforce_formant_spacing(merged["formant_freq"])

    # 이상와(piriform fossa): 성도의 곁가지라 화자 고정. 4~5 kHz 에 골을 만든다.
    pf = getattr(prof, "piriform", None)
    if pf and merged.get("notch_freq") is None:
        merged["notch_freq"] = torch.full((1, t, 1), float(pf["freq"]))
        merged["notch_bw"] = torch.full((1, t, 1), float(pf["bw"]))

    df, dr = prof.dispersion_tensors(1, t)
    merged.setdefault("disp_freq", df)
    merged.setdefault("disp_radius", dr)
    if merged.get("disp_freq") is None or merged.get("disp_radius") is None:
        merged["disp_freq"] = merged["disp_radius"] = None

    merged["sib"] = SibilantParams(**sib_fields)
    merged["noise_rough"] = merged.get("noise_rough")
    valid = {f for f in Controls.__dataclass_fields__}
    return Controls(**{k: v for k, v in merged.items() if k in valid})


def render(score: dict, prof: VoiceProfile | None = None, cfg: Config | None = None,
           seed: int | None = None) -> torch.Tensor:
    """스크립트 -> 파형 (1, N)."""
    cfg = cfg or Config()
    prof = prof or VoiceProfile()
    ctrl = build_controls(score, prof, cfg)
    synth = PhysicalVoiceSynth(cfg, tract_mode=score.get("tract", "formant"))
    gen = None
    if seed is not None or "seed" in score:
        gen = torch.Generator().manual_seed(int(score.get("seed", seed or 0)))
    with torch.no_grad():
        return synth(ctrl, generator=gen)["audio"]


def load_score(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:                               # pragma: no cover
            raise ImportError("YAML 스크립트를 읽으려면 pyyaml 이 필요합니다 "
                              "(pip install pyyaml). JSON 으로 써도 됩니다.") from e
        return yaml.safe_load(text)
    return json.loads(text)


PRESET_PROFILES = {"female": VoiceProfile.female, "male": VoiceProfile}


def load_profile(score: dict, base_dir: str = ".") -> VoiceProfile:
    """`voice:` 는 프로파일 JSON 경로이거나 프리셋 이름("female" / "male")."""
    p = score.get("voice")
    if not p:
        return VoiceProfile()
    if p in PRESET_PROFILES:
        return PRESET_PROFILES[p]()
    path = p if os.path.isabs(p) else os.path.join(base_dir, p)
    return VoiceProfile.load(path if os.path.exists(path) else p)
