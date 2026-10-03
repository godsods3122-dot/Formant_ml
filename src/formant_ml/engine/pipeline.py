"""복사합성 적합의 단계 함수 — `scripts/copyfit.py` 의 `main()` 에서 옮겼다 (MEASUREMENTS §52.470).

사용자: *"문제가 생기면 테스트 몇 번으로 바로 버그를 추적할 수 있어야 하는데, 버그가 자꾸 종속적으로 여기 저기 붙는 느낌"*. 단계마다 입력과 출력이 분명한
함수로 떼어 계약 시험(`tests/engine/test_pipeline_contract.py`)을 붙인다. 순서는 이렇다:

    configure (copyfit, 깃발 → 모듈 설정)  →  recipe.snapshot (조리법 고정)
    load_clean  →  prepare.Segment.cut / prepare.observe  →  vowel_onset_filter  →  prepare.burst_observations
    → 적합기 생성  →  restore_state  →  fit_staged  →  출력

옮기면서 동작은 바꾸지 않았다 — 기준 판 셋(되살리기·클릭 자동 가림·모음 시작을 거치는 판)의 합성이 옮기기 전과 비트 단위로 같다.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

#: 저장·복원하는 적합기 전역 스칼라 (보이스 바 세기, 방 잔향 RT·세기 — §52.439).
FIT_SCALARS = ("vb_log_g", "rv_log_rt", "rv_g_db")
#: 전사를 찾는 파일 (작업 디렉토리 기준). `wavs/…|전사|화자`.
METADATA_FILES = ("metadata_train.csv", "metadata_eval.csv")


# ------------------------------------------------------------------ 읽기·정리
@dataclass
class Clean:
    y: np.ndarray            # 클릭 메우기·잡음 제거까지 거친 신호 (파일 전체, float64, 단채널)
    y_raw: np.ndarray        # 잡음 제거 전 (클릭 메우기는 거침) — 코덱 차단 분석용
    sr: int
    clicks: list             # 메운 클릭 (s0, s1, dB)


def load_clean(path: str, profile=None, declick_at: str | None = None, declick_auto: bool = False,
               no_denoise: bool = False, log=print) -> Clean:
    """파일을 읽어 녹음 속 클릭을 메우고(`declick_at`·`declick_auto`, §52.461·464) 잡음을 뺀다. **표본 자리와 길이는 바뀌지 않는다.**"""
    from .denoise import denoise, noise_profile, snr_report
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    clicks = []
    if declick_at or declick_auto:
        # **클릭은 목소리가 아니다** (§52.461) — 적합기가 목소리 기구로 흉내 내며 뒤 발화까지 오염시켰다. 파일 전체에서, 표본 자리를 안 바꾸고 메운다.
        from . import declick as _dk
        from .analyze import glottal_pulses as _gp
        from .profile import DEFAULT_PROFILE
        prof = profile if profile is not None else DEFAULT_PROFILE
        at = [float(v) for v in str(declick_at or "").split(",") if v.strip()]
        if declick_auto:
            from .transients import auto_click_times
            auto = auto_click_times(y, sr)
            log(f"  클릭 자동 후보 {len(auto)} 곳" + (": " + ", ".join(f"{t:.4f}" for t in auto) if auto else ""))
            at = sorted(set(at) | set(auto))
        y, clicks = _dk.declick(y, sr, _gp(y, sr, prof), at=at, log=log)
        miss = [t for t in at if not any(abs(0.5 * (c0 + c1) / sr - t) <= 0.005 for c0, c1, _ in clicks)]
        if miss:
            log(f"  클릭 제거: 준 시각 중 못 찾은 곳 {miss}")
    y_raw = y
    if not no_denoise:
        noise = noise_profile(y, sr)
        yd = denoise(y, sr, noise)
        r = snr_report(y, yd, sr, noise)
        log(f"잡음 제거: 바닥 {r['noise_db_in']:.1f} -> {r['noise_db_out']:.1f} dB "
            f"(−{r['removed_db']:.1f} dB), 정점 {r['peak_db_in']:.1f} -> "
            f"{r['peak_db_out']:.1f} dB")
        y = yd
    return Clean(y=y, y_raw=y_raw, sr=int(sr), clicks=list(clicks))


# ------------------------------------------------------------------ 모음 시작
def transcript_for(wav_path: str, files=METADATA_FILES) -> str | None:
    """`metadata_*.csv` 에서 이 파일의 전사."""
    for mf in files:
        if os.path.exists(mf):
            for ln in open(mf, encoding="utf-8"):
                p = ln.strip().split("|")
                if len(p) >= 2 and os.path.basename(p[0]) == os.path.basename(wav_path):
                    return p[1]
    return None


def first_syllable_is_vowel(text: str | None) -> tuple[bool, str | None]:
    """(첫 한글 음절의 초성이 ㅇ 인가, 그 음절)."""
    syl = next((c for c in (text or "") if 0xAC00 <= ord(c) <= 0xD7A3), None)
    return (syl is not None and (ord(syl) - 0xAC00) // 588 == 11), syl


def vowel_onset_filter(track, segment, y_full: np.ndarray, sr: int, text: str | None, t0: float, log=print) -> int | None:
    """**모음으로 시작하는 발화의 첫머리 순간음은 구강 파열이 아니다** (§52.467). 빼면 그 파열 틀 번호, 아니면 None.

    038 "어느게" 는 0.051 s 에 단단한 성문 시작 [ʔ] 의 광대역 순간음과 함께 곧바로 떨기 시작하는데, 분석이 그것을 파열로 읽어 앞을 폐쇄로 닫았다
    (입 열림 0.02, 8~50 ms) — 합성의 첫 15~20 ms 가 10~20 dB 모자라 끊김으로 들렸다. 음향만으로는 짧은 VOT 의 된소리 파열과 못 가르므로 **전사**를
    쓴다: 첫 음절 초성이 ㅇ 이고, 구간이 발화 시작을 담고, 첫 파열 앞에 **말 수준**(파일 최대 −35 dB 안)의 유성 틀이 없으면 그 파열을 뺀다.
    """
    vowel, syl = first_syllable_is_vowel(text)
    b = np.asarray(getattr(track, "bursts", np.zeros(0, dtype=int)), int)
    e10 = np.convolve(y_full ** 2, np.ones(int(0.01 * sr)) / int(0.01 * sr), "same")
    on = int(np.argmax(10 * np.log10(e10 + 1e-20) > 10 * np.log10(e10.max()) - 40.0)) / sr   # 파일의 말 시작
    voi = np.asarray(getattr(track, "voiced", np.zeros(0, bool)), bool)
    if not (vowel and b.size and t0 <= on + 0.02):
        log(f"  모음 시작: 해당 없음 (전사 첫 음절 {syl!r}, 구간 시작 {t0:g} s, 말 시작 {on:.3f} s)")
        return None
    b0 = int(b.min())
    # 말 수준의 유성만 센다 — 038 은 첫머리 앞 0~50 ms 에 −57~−65 dB 의 약한 주기 성분(배경·앞 소리 꼬리)이 있어 분석이 유성으로 표시했다.
    fw = max(1, int(round(track.frame_ms * 1e-3 * sr)))
    sa = np.asarray(segment.audio, float)
    fl = np.array([10 * np.log10(np.mean(sa[i * fw:(i + 1) * fw] ** 2) + 1e-20) for i in range(b0)])
    loud = fl > 10 * np.log10(e10.max()) - 35.0
    if (voi[:b0] & loud[:len(voi[:b0])]).any():
        log("  모음 시작: 첫 파열 앞에 말 수준의 유성 틀이 있어 그대로 둔다")
        return None
    track.bursts = b[b != b0]
    log(f"  모음 시작: 전사 '{(text or '')[:12]}' 의 첫 음절 '{syl}' 이 초성 ㅇ — 첫머리 파열 {t0 + b0 * track.frame_ms / 1000:.3f} s 를 "
        f"성문 시작으로 보고 구강 파열에서 뺀다 (남은 파열 {track.bursts.size} 곳)")
    return b0


# ------------------------------------------------------------------ 되살리기
def restore_fit_scalars(fit, z, init_stem: str) -> str:
    """`_track.npz` 의 `fit_scalars` 를 적합기에 되살린다. 옛 판(필드 없음)은 그 판 로그의 마지막 진행 줄에서 읽는다."""
    from . import fit as _fm
    vals, src = {}, "npz"
    if "fit_scalars" in z.files:
        vals = json.loads(str(z["fit_scalars"]))
    else:
        src = "로그"
        try:
            lines = [ln for ln in open(init_stem + ".log", encoding="utf-8", errors="ignore")
                     if re.search(r"\[ *\d+\] 포락", ln)]
        except OSError:
            lines = []
        if lines:
            last = lines[-1]
            m = re.search(r"보이스바 ([-+0-9.]+)dB", last)
            if m:
                vals["vb_log_g"] = float(m.group(1)) * math.log(10.0) / 20.0
            m = re.search(r"잔향 RT (\d+)ms ([-+0-9.]+)dB", last)
            if m:
                vals["rv_log_rt"] = math.log(float(m.group(1)) / 1000.0)
                vals["rv_g_db"] = float(m.group(2)) * math.log(10.0) / 20.0
    done = []
    with torch.no_grad():
        for k, v in vals.items():
            p = getattr(fit, k, None)
            if p is not None and k in FIT_SCALARS:
                if k.startswith("rv_") and getattr(_fm, "LR_FIXED", False):
                    continue
                if k == "vb_log_g" and getattr(_fm, "VB_FIXED", False):
                    continue                             # 잔향이 녹음 상수로 고정되면 옛 판의 적합값을 쓰지 않는다
                p.fill_(float(v))
                done.append(f"{k} {v:+.3f}")
    return f"적합기 전역 스칼라 복원 ({src}): " + (", ".join(done) if done else "없음")


def restore_state(fit, init_stem: str, log=print) -> None:
    """앞 판(`init_stem`_track.npz)의 적합기 상태를 되살린다 — 펄스 잠금 → 주기별 보정 → 전역 스칼라 → 파열 개방 속도 (이 순서가 계약이다)."""
    from .fit import restore_hcorr, restore_pulse_state
    with np.load(init_stem + "_track.npz", allow_pickle=False) as z:
        log("  " + restore_pulse_state(fit, z))
        # 펄스 위상을 되살린 **뒤에** 부른다 — `_hc_c0` 가 그 위상에서 나온다 (§52.295).
        log("  " + restore_hcorr(fit, z))
        # **적합기 전역 스칼라** (보이스 바 세기, 방 잔향 RT·세기 — MEASUREMENTS §52.439). 예전에는 저장하지 않아
        # 다듬기(`--init`)가 이 값들을 초기값에서 새로 시작했고, `--no-fit` 재렌더도 원판과 달랐다.
        log("  " + restore_fit_scalars(fit, z, init_stem))
        # 결정적 방 FIR (§52.479) — 켠 판이면 되살린다
        if "room_fir" in z.files:
            from . import fit as _F
            hf = np.asarray(z["room_fir"], float).reshape(-1)
            if hf.size and _F.ROOM_FIR_MS > 0.0:
                K = int(_F.ROOM_FIR_MS * fit.fs / 1000.0)
                th = np.zeros(K)
                th[:min(K, hf.size)] = hf[:K] / _F.ROOM_FIR_SCALE
                fit.room_theta = torch.as_tensor(th, dtype=torch.float64, device=fit.target.device).requires_grad_(True)
                log(f"  방 FIR 복원: {hf.size} 탭 → {K} 탭, 에너지 {10 * np.log10(np.sum(hf ** 2) + 1e-20):+.1f} dB")
        # 파열별 개방 속도 (§52.456) — 파열 개수가 같을 때만
        if getattr(fit, "rel_log_r", None) is not None and "release_log_r" in z.files:
            rr = np.asarray(z["release_log_r"], float).reshape(-1)
            if rr.size == fit.rel_log_r.numel():
                with torch.no_grad():
                    fit.rel_log_r.copy_(torch.as_tensor(rr, dtype=fit.rel_log_r.dtype))
                log("  파열 개방 속도 복원: " + "/".join(f"{np.exp(v):.0f}" for v in rr) + " cm²/s")
            else:
                log(f"  파열 개방 속도 복원 안 함: 저장 {rr.size} 곳, 지금 {fit.rel_log_r.numel()} 곳")
        # 음소 목표·경계 (§52.487) — 구간 수가 같을 때만
        from . import fit as _F2
        if _F2.ART_SEGMENTS is not None and "art_z" in z.files and np.asarray(z["art_z"]).size:
            fit._art_setup()
            az, ad = np.asarray(z["art_z"], float), np.asarray(z["art_d"], float)
            if az.shape == tuple(fit.art_z.shape) and ad.shape == tuple(fit.art_d.shape):
                with torch.no_grad():
                    fit.art_z.copy_(torch.as_tensor(az, dtype=fit.art_z.dtype))
                    fit.art_d.copy_(torch.as_tensor(ad, dtype=fit.art_d.dtype))
                log(f"  음소 목표 복원: 구간 {az.shape[0]} · 경계 {ad.size}")
                # 자세 대응을 고친 음소 (§52.519, `fit.ART_RESET_LABELS`) — 저장된 목표 대신 새 MRI 자세에서 다시 출발한다
                if _F2.ART_RESET_LABELS and _F2.ART_LABELS is not None:
                    from .articulation import W02_FIT, W02_SHAPES
                    zs = np.load(W02_SHAPES, allow_pickle=True)
                    pn = list(zs["param_names"]); shapes = {str(n): s for n, s in zip(zs["names"], zs["shapes"])}
                    nr = 0
                    with torch.no_grad():
                        for k, (t0_, t1_, shp) in enumerate(_F2.ART_SEGMENTS):
                            lab = str(_F2.ART_LABELS[k]) if k < len(_F2.ART_LABELS) else ""
                            if lab in _F2.ART_RESET_LABELS and shp in shapes:
                                m = len(W02_FIT)                           # 목표는 정규화 좌표 (x − 기본) / 폭 (`_art_setup`)
                                x = np.array([shapes[shp][pn.index(n)] for n in W02_FIT])
                                fit.art_z[k, :m] = torch.as_tensor((x - fit._art_ne[:m]) / fit._art_sp[:m], dtype=fit.art_z.dtype)
                                nr += 1
                    log(f"  음소 목표 다시 놓음: {'·'.join(_F2.ART_RESET_LABELS)} {nr} 구간 (MRI 자세)")
            else:
                log(f"  음소 목표 복원 안 함: 저장 {az.shape}, 지금 {tuple(fit.art_z.shape)}")
        # 자기 진동 성대의 음높이 맞춤 (§52.488) — 틀 수가 같을 때만
        if "vf_pscorr" in z.files and np.asarray(z["vf_pscorr"]).size and getattr(fit.eng, "td", None) is not None:
            pc = np.asarray(z["vf_pscorr"], float).reshape(-1)
            if pc.size == int(fit.control().shape[1]):
                fit.eng.td.vf_pscorr = torch.as_tensor(pc[None, :], dtype=torch.float64)
                fit._vf_level_done = True
                log(f"  음량 맞춤 복원: 폐압 배율 {pc.min():.2f}–{pc.max():.2f}")
        from . import fit as _fitm
        if getattr(_fitm, "VF_QCORR_RESET", False) and "vf_qcorr" in z.files and np.asarray(z["vf_qcorr"]).size:
            log("  음높이 고리 배율을 복원하지 않는다 (--no-vf-lock, §52.530)")
        elif "vf_qcorr" in z.files and np.asarray(z["vf_qcorr"]).size and getattr(fit.eng, "td", None) is not None:
            qc = np.asarray(z["vf_qcorr"], float).reshape(-1)
            if qc.size == int(fit.control().shape[1]):
                fit.eng.td.vf_qcorr = torch.as_tensor(qc[None, :], dtype=torch.float64)
                log(f"  음높이 맞춤 복원: 긴장 배율 {qc.min():.3f}–{qc.max():.3f}")
        if "vf_qcyc" in z.files and np.asarray(z["vf_qcyc"]).size and getattr(fit.eng, "td", None) is not None:
            qy = np.asarray(z["vf_qcyc"], float).reshape(-1)
            from . import voice_td as _vtdq                                          # 표본마다의 배율 — 관 해상도가 다르면 되풀이·솎기 (§52.508)
            _osq = float(z["vf_pll_os"]) if "vf_pll_os" in z.files else 2.0
            if _vtdq.OS != _osq:
                qy = np.repeat(qy, int(_vtdq.OS // _osq)) if _vtdq.OS > _osq else qy[::int(_osq // _vtdq.OS)]
            if qy.size == int(fit.target.shape[-1]) * 2:
                fit.eng.td.vf_qcyc = qy
                log(f"  주기 시각 맞춤 복원: 표본 {qy.size}, 배율 {qy.min():.3f}–{qy.max():.3f}")
        if "vf_pll_off" in z.files and np.asarray(z["vf_pll_off"]).size:
            fit._pll_off = np.asarray(z["vf_pll_off"], float).reshape(-1)       # 고리 시각표 오프셋 (§52.493) — 표시 수가 같으면 쓰인다
            # 이어 적합에서는 첫 음높이 맞춤을 건너뛴다 (§52.498): 앞 판 동안 고리의 적분이 기본 긴장의 어긋남(중앙 60–100 cents)을 메우고
            # 있었는데, 고리를 끈 채 틀 배율을 다시 맞추면 앞 판이 맞춘 상태가 흔들렸다 (VQC 앞 판 끝 68.1 → 이 판 처음 55.8 %).
            fit._vf_pitch_done = True
            log(f"  고리 오프셋 복원: {fit._pll_off.size} 주기, 중앙 {np.median(fit._pll_off) / 48.0:.3f} ms")
        if "vf_pll_t" in z.files and np.asarray(z["vf_pll_t"]).size and getattr(fit.eng, "td", None) is not None:
            fit.eng.td.vf_pll_t = np.asarray(z["vf_pll_t"], float).reshape(-1)    # 고리 시각표 그대로 (적합 없이 렌더해도 같은 소리, §52.498)
            from . import voice_td as _vtdo                                          # 표본 단위를 지금 관 해상도로 (§52.508; 옛 트랙은 OS 2)
            _os_saved = float(z["vf_pll_os"]) if "vf_pll_os" in z.files else 2.0
            if _vtdo.OS != _os_saved:
                fit.eng.td.vf_pll_t = fit.eng.td.vf_pll_t * (_vtdo.OS / _os_saved)
            log(f"  고리 시각표 복원: 목표 닫힘 {fit.eng.td.vf_pll_t.size} 개")
