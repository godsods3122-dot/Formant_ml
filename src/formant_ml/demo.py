"""학습 없이, 물리 모델만으로 음성을 합성하는 데모.

    python -m formant_ml.demo --out out

여기서 나오는 소리는 신경망이 만든 게 아니라 전부 방정식이 만든 것이다.
'AI 특유의 잡음'이 생길 여지가 구조적으로 없다는 걸 눈/귀로 확인하는 것이 목적.
"""
from __future__ import annotations

import argparse

import torch

from .config import Config
from .models.synth import Controls, PhysicalVoiceSynth
from .presets import BANDWIDTHS, FRICATIVES, VOWELS
from .utils import band_bump, n_frames, ramp, save_wav, vibrato


def _formant_track(t: int, segments, n_formants: int) -> torch.Tensor:
    """[(위치, 모음이름), ...] -> 시간에 따라 미끄러지는 포먼트 궤적 (1, T, K)."""
    tracks = []
    for k in range(n_formants):
        pts = []
        for pos, name in segments:
            f = VOWELS[name]
            pts.append((pos, float(f[k]) if k < len(f) else 4500.0 + 900.0 * (k - 4)))
        tracks.append(ramp(t, pts))
    return torch.cat(tracks, dim=-1)


def base_controls(cfg: Config, t: int, n_formants: int) -> dict:
    nb = cfg.noise.n_bands
    return dict(
        f0=torch.full((1, t, 1), 120.0),
        harmonic_amp=torch.ones(1, t, 1),
        rd=torch.full((1, t, 1), 1.2),
        formant_bw=torch.tensor(
            [BANDWIDTHS[min(k, len(BANDWIDTHS) - 1)] * (1 + 0.4 * k)
             for k in range(n_formants)]).reshape(1, 1, -1).expand(1, t, n_formants
                                                                  ).contiguous(),
        formant_gain=torch.ones(1, t, n_formants),
        noise_bands=torch.full((1, t, nb), 1e-4),
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.zeros(1, t, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seconds", type=float, default=1.2)
    args = ap.parse_args()

    cfg = Config()
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    K = cfg.filt.n_formants
    synth = PhysicalVoiceSynth(cfg)
    t = n_frames(args.seconds, sr, hop)

    # 1) 지속 모음 /아/ — 비브라토 + Rd 를 pressed -> breathy 로 스윕
    c = base_controls(cfg, t, K)
    c["f0"] = vibrato(t, 120.0, 5.0, 35.0, cfg.audio.frame_rate)
    c["rd"] = ramp(t, [(0.0, 0.5), (1.0, 2.4)])
    c["formant_freq"] = _formant_track(t, [(0.0, "a"), (1.0, "a")], K)
    c["noise_bands"] = band_bump(cfg.noise.n_bands, 4000, 8000, 0.02, sr
                                 ).reshape(1, 1, -1).expand(1, t, -1).contiguous()
    c["noise_am"] = ramp(t, [(0.0, 0.0), (1.0, 1.0)])     # 기식성 증가
    out = synth(Controls(**c))
    save_wav(f"{args.out}/01_vowel_a_pressed_to_breathy.wav", out["audio"], sr)

    # 2) 이중모음 활음 /아이우/ — 포먼트만 움직인다
    c = base_controls(cfg, t, K)
    c["f0"] = ramp(t, [(0.0, 130.0), (0.5, 145.0), (1.0, 100.0)])
    c["formant_freq"] = _formant_track(
        t, [(0.0, "a"), (0.5, "i"), (1.0, "u")], K)
    save_wav(f"{args.out}/02_diphthong_a_i_u.wav", synth(Controls(**c))["audio"], sr)

    # 3) 치찰음 /ㅅ/ — 성대 진동 없이 협착부 난류만
    c = base_controls(cfg, t, K)
    c["harmonic_amp"] = torch.zeros(1, t, 1)
    cf, bw, pos, g = FRICATIVES["s"]
    c["noise_bands"] = band_bump(cfg.noise.n_bands, cf, bw, g, sr
                                 ).reshape(1, 1, -1).expand(1, t, -1).contiguous()
    c["noise_entry"] = torch.full((1, t, 1), pos * K)
    c["formant_freq"] = _formant_track(t, [(0.0, "i"), (1.0, "i")], K)
    save_wav(f"{args.out}/03_fricative_s.wav", synth(Controls(**c))["audio"], sr)

    # 4) 음절 /사/ — 무성 마찰 -> 유성 모음 전이 (경계에서 위상 연속)
    t2 = n_frames(1.0, sr, hop)
    c = base_controls(cfg, t2, K)
    c["harmonic_amp"] = ramp(t2, [(0.0, 0.0), (0.42, 0.0), (0.5, 1.0), (1.0, 0.8)])
    c["f0"] = ramp(t2, [(0.0, 135.0), (1.0, 105.0)])
    c["formant_freq"] = _formant_track(t2, [(0.0, "i"), (0.45, "i"), (0.6, "a"),
                                            (1.0, "a")], K)
    ns = band_bump(cfg.noise.n_bands, 6500, 3000, 1.0, sr).reshape(1, 1, -1)
    env = ramp(t2, [(0.0, 1.0), (0.42, 1.0), (0.52, 0.02), (1.0, 0.01)])
    c["noise_bands"] = (ns * env).contiguous()
    c["noise_entry"] = torch.full((1, t2, 1), 0.92 * K)
    c["noise_am"] = ramp(t2, [(0.0, 0.0), (0.5, 0.3), (1.0, 0.3)])
    save_wav(f"{args.out}/04_syllable_sa.wav", synth(Controls(**c))["audio"], sr)

    # 5) 도파관(Kelly-Lochbaum) 모드 — 포먼트가 아니라 '단면적 함수'로 제어
    wg = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    n_sec = cfg.filt.n_tract_sections
    x = torch.linspace(0, 1, n_sec)
    area_a = 0.4 + 3.2 * torch.sigmoid((x - 0.45) * 12)          # 인두 좁고 구강 넓음
    c = base_controls(cfg, t, K)
    c["formant_freq"] = _formant_track(t, [(0.0, "a"), (1.0, "a")], K)  # 미사용
    c["area"] = area_a.reshape(1, 1, -1).expand(1, t, -1).contiguous()
    c["f0"] = vibrato(t, 110.0, 5.5, 25.0, cfg.audio.frame_rate)
    save_wav(f"{args.out}/05_waveguide_area_function.wav",
             wg(Controls(**c))["audio"], sr)

    # 6) 성대 물리모델(2질량 자가진동) 소스를 그대로 성도에 통과
    from .dsp.vocalfold import FoldParams, flow_to_excitation, simulate
    from .dsp.core import ltv_filter
    from .dsp.filters import resonator_stage_responses
    n = t * hop
    for tag, q, asym in [("modal", 1.0, 1.0), ("tense", 1.9, 1.0),
                         ("diplophonic", 1.0, 0.55)]:
        flow, _, _ = simulate(FoldParams(q=q, asym=asym, a01=0.02, a02=0.02),
                              n, sr, oversample=4)
        exc = flow_to_excitation(flow).to(torch.float32).unsqueeze(0)
        ff = _formant_track(t, [(0.0, "a"), (1.0, "a")], K)
        H = resonator_stage_responses(
            ff, torch.full((1, t, K), 90.0), torch.ones(1, t, K), sr,
            cfg.audio.n_fft // 2 + 1).prod(dim=2)
        y = ltv_filter(exc, H, hop, cfg.filt.ir_size)
        save_wav(f"{args.out}/06_vocalfold_{tag}.wav", y, sr)

    # ---------------------------------------------------------------- 새 손잡이들
    # 아래는 전부 score.render 를 거친다 = 스크립트로 낼 수 있는 소리와 동일하다.
    from .score import render
    from .voice import VoiceProfile
    prof = VoiceProfile()

    def R(name, timeline, **kw):
        y = render({"timeline": timeline, "seed": 7, **kw}, prof, cfg)
        save_wav(f"{args.out}/{name}.wav", y, sr)

    # 7) 스펙트럼 기울기(tilt) 스윕 — 같은 모음의 '고역' 만 -6 -> +8 dB/oct
    R("07_tilt_sweep_dark_to_bright",
      [{"type": "vowel", "vowel": "a", "dur": 2.4, "tilt": [-6, 8], "f0": 120}])

    # 8) 웃음 3종 — 같은 함수, 파라미터만 다르다
    R("08_laugh_belly", [{"type": "laugh", "dur": 1.6, "rate_hz": 4.2,
                          "voiced": 0.95, "pitch_lift": 1.5, "tilt": 2.0}])
    R("08_laugh_giggle", [{"type": "laugh", "dur": 1.4, "rate_hz": 8.0,
                           "voiced": 0.35, "breathiness": 0.7, "f0": 200}])
    R("08_laugh_breathy", [{"type": "laugh", "dur": 1.4, "rate_hz": 6.0,
                            "voiced": 0.0, "breathiness": 0.9}])

    # 9) 치찰음 지문: 극(앞공동 공진)만 옮겨서 /ㅅ/ -> /ㅅㅑ/ 로
    R("09_sibilant_pole_sweep",
      [{"type": "fricative", "phone": "s", "dur": 2.0,
        "sib_pole_f": [[0, 8000], [1, 3000]], "sib_pole_bw": 700, "sib_tilt": 0}])

    # 10) 위상차 파라미터 A/B — 크기 스펙트럼은 같고 하모닉 상대위상만 다르다
    R("10_phase_dispersion_off",
      [{"type": "vowel", "vowel": "a", "dur": 1.2, "f0": 110, "rd": 0.6}])
    R("10_phase_dispersion_on",
      [{"type": "vowel", "vowel": "a", "dur": 1.2, "f0": 110, "rd": 0.6,
        "disp_freq": [900, 2600, 5200], "disp_radius": [0.9, 0.9, 0.88]}])

    # 11) 성구 전환(파사지오) — 글리산도에서 Rd/tilt 가 계단처럼 꺾인다.
    #     analysis.registers.passaggio_candidates 가 이걸 되찾아낼 수 있어야 한다.
    R("11_glissando_with_passaggio",
      [{"type": "vowel", "vowel": "a", "dur": 3.0,
        "f0": [[0, 100], [1, 330]],
        "rd": [[0, 0.7], [0.54, 0.8], [0.60, 1.9], [1, 2.1]],
        "tilt": [[0, 2], [0.54, 2], [0.60, -3], [1, -4]]}])

    # 12) 비언어 발성 모음 — 전부 같은 물리 손잡이의 다른 조합
    R("12_nonverbal_suite",
      [{"type": "sigh", "dur": 0.9}, {"type": "silence", "dur": 0.15},
       {"type": "breath", "dur": 0.5, "inhale": True},
       {"type": "silence", "dur": 0.15},
       {"type": "throat_clear", "dur": 0.4},
       {"type": "silence", "dur": 0.15},
       {"type": "creak", "dur": 0.6, "rate_hz": 40},
       {"type": "silence", "dur": 0.15},
       {"type": "whisper", "dur": 0.7, "vowel": "i"},
       {"type": "silence", "dur": 0.15},
       {"type": "sob", "dur": 1.2}])

    print(f"WAV 파일을 {args.out}/ 에 저장했습니다.")


if __name__ == "__main__":
    main()
