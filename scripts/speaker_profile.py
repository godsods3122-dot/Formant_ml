"""여러 발화에서 화자 공통 보정값만 합친다. 녹음 경로는 별도 scope 로 저장한다.

MVF·성문 위상 분산·음원 EQ·레벨 정규화는 발화에 남긴다.
이 보정값은 모델의 경험적 눈금이지 해부학의 직접 측정값은 아니다.

    python scripts/speaker_profile.py out/L86/s040 out/L86/s101 -o profiles/yang_female_speaker.json

쓸 때는 `copyfit --speaker-lock <파일>` — 값을 넣고 **옵티마이저에서 뺀다**.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formant_ml.engine.calibration import combine_calibrations, load_calibration


def _load(stem: str) -> dict:
    return load_calibration(stem if stem.endswith((".npz", ".json")) else stem + "_track.npz")


def combine(stems: list[str], scope: str = "speaker") -> dict:
    out = combine_calibrations([_load(s) for s in stems], scope)
    out["sources"] = list(stems)
    return out


#: **섭동 이론이 정하는 포먼트 구역.** 균일관의 k 번째 공명은 (2k−1)·c/4L 이고, 면적 함수를 흔들면
#: 그 둘레로 움직인다 — 움직일 수 있는 폭은 k 가 커질수록 좁아진다(마디가 많아 섭동이 서로 상쇄된다.
#: Chiba & Kajiyama 1941, Fant 1960, Stevens 1998 의 섭동 이론).
#:
#: 사용자: *"f1이 무슨 수로 12kHz까지 가? 그게 말이 돼? 올바른 이론은 없나?"* 지금 `ParamSpec` 은
#: `f1`~`f8` 이 전부 12 kHz 까지 갈 수 있게 두었고, 그래서 적합된 `f4` 가 균일관 예측의 **×1.79**
#: (7287 Hz) 까지 갔다. 같은 화자의 목표에서 실제로 관측된 배율은:
#:
#:     F1 ×0.56~1.93   F2 ×0.57~1.90   F3 ×0.83~1.53   F4 ×0.95~1.33
#:
#: k 가 커질수록 좁아지는 것이 이론 그대로다. 아래 값은 그 관측에 여유를 둔 것이고, F5~F8 은
#: 하인두가 모음에 거의 무관하다는 실측 문헌(Kitamura et al. 2005)에 따라 더 좁게 둔다.
TUBE_TOL = {1: (0.50, 2.10), 2: (0.52, 2.05), 3: (0.75, 1.65), 4: (0.85, 1.45),
            5: (0.80, 1.25), 6: (0.80, 1.25), 7: (0.80, 1.25), 8: (0.80, 1.25)}
C_SOUND = 34000.0


def formant_bands(stems: list[str], prof_path: str) -> dict:
    """섭동 이론의 구역 {이름: [하한, 상한]}. 목표에서 관측한 구간과 견줘 보고한다."""
    import soundfile as sf
    sys_path = __import__("sys").path
    if "src" not in sys_path:
        sys_path.insert(0, "src")
    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.segment import fricative_mask
    prof = SpeakerProfile.load(prof_path)
    f1t = C_SOUND / (4.0 * prof.tract_length_cm)
    acc: dict[str, list] = {}
    qacc: dict[int, list] = {}
    for st in stems:
        y, fs = sf.read(st + "_target.wav")
        y = np.asarray(y, float)
        tr = analyze(y, fs, prof, 48, t0=0.0, full=y)
        fm = fricative_mask(y, fs, prof, 48)
        n = min(tr.n_frames, len(fm))
        m = np.asarray(tr.voiced, bool)[:n] & ~fm[:n]
        for k in range(1, 5):
            v = np.asarray(tr[f"f{k}"], float)[:n][m]
            acc.setdefault(k, []).append(v[v > 0])
            b = np.asarray(tr[f"bw{k}"], float)[:n][m]
            ok_ = (v > 0) & (b > 0)
            qacc.setdefault(k, []).append(v[ok_] / b[ok_])
    out = {}
    for k in range(1, 9):
        c = (2 * k - 1) * f1t
        lo_r, hi_r = TUBE_TOL[k]
        out[f"f{k}"] = [float(c * lo_r), float(c * hi_r)]
        if k in acc:
            v = np.concatenate(acc[k])
            o1, o99 = np.percentile(v, 1), np.percentile(v, 99)
            print(f"    F{k}: 균일관 {c:6.0f} Hz | 관측 ×{o1/c:.2f}~×{o99/c:.2f} "
                  f"| 구역 ×{lo_r:.2f}~×{hi_r:.2f} = {c*lo_r:6.0f}~{c*hi_r:6.0f} Hz"
                  f"{'  <-- 관측이 구역 밖' if (o1 < c*lo_r or o99 > c*hi_r) else ''}")
        else:
            print(f"    F{k}: 균일관 {c:6.0f} Hz | 관측 없음(LPC 가 따로 못 잡는다) "
                  f"| 구역 {c*lo_r:6.0f}~{c*hi_r:6.0f} Hz")
    # **Q 구역은 목표에서 잰 중앙값에 매단다.** 두 가지를 다 피해야 한다 — 너무 뭉개지면
    # 포먼트 사이 골이 메워지고(사용자: "f1과 f2가 확고히 분리되는데 우리는 퉁쳐져 있다"),
    # 너무 날카로우면 "고역의 평평한 줄"(좁은 고정 공진)이 돌아온다. 생리 법칙(Fant·Klatt 의
    # B1 50~80 … B4 175~280 Hz = 엔진의 `default_bw`)은 이 화자의 F2~F4 에서 관측 중앙값보다
    # **더 날카로운** 값을 요구해서 그대로 쓰면 위험하다. 그래서 관측 중앙값의 0.85 배를 하한,
    # 3 배를 상한으로 둔다. LPC 의 5 % 분위(Q 1.5)는 전이·비음 프레임의 추정치라 쓰지 않는다.
    Q_LO_REL, Q_HI_REL = 0.85, 3.0
    qb = {}
    for k in range(1, 5):
        if k not in qacc: continue
        q = np.concatenate(qacc[k])
        med = float(np.median(q))
        qb[f"f{k}"] = [float(med * Q_LO_REL), float(med * Q_HI_REL)]
        fmed = float(np.median(np.concatenate(acc[k])))
        law = fmed / (40.0 + 0.05 * fmed)
        print(f"    F{k} Q: 관측 중앙 {med:5.1f} (법칙 {law:5.1f})  ->  구역 {med*Q_LO_REL:5.1f}~{med*Q_HI_REL:5.1f}")
    out["_q"] = qb
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="out/<태그>/<파일> (…_track.npz 를 읽는다)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--scope", choices=("speaker", "recording"), default="speaker",
                    help="화자 보정 또는 동일 녹음 환경의 출력 EQ·IR. 발화값은 합치지 않는다")
    ap.add_argument("--bands", action="store_true",
                    help="포먼트마다의 제 구역(F1~F4 는 목표 관측, F5~F8 은 균일관 추정)을 같이 넣는다")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    a = ap.parse_args()
    if a.bands and a.scope != "speaker":
        ap.error("--bands belongs to speaker calibration")
    prof = combine(a.stems, a.scope)
    if a.bands:
        fb = formant_bands(a.stems, a.profile)
        # **Q 구역은 넣지 않는다** (§51.39 의 반증). 모음 프레임에서 낸 Q 를 전 구간에 걸면
        # 출발점(분석 궤적)의 26~65 % 가 구역 밖이 되어 시그모이드가 포화하고, 1 단계부터
        # 6.5 점이 무너진다(`L97`: 2.1 단계 80.27 대 `L91` 86.78). 유성 마스크를 따로 두거나
        # 딱딱한 상자 대신 벌점으로 다시 만들어야 한다.
        fb.pop("_q", None)
        prof["formant_band"] = fb
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"{a.scope} 상수 {len(prof[a.scope])} 항목을 {len(a.stems)} 발화에서 합쳐 {a.out} 에 썼다")
    per = [_load(s)[a.scope] for s in a.stems]
    for k, v in sorted(prof[a.scope].items()):
        v = np.atleast_1d(np.asarray(v, float))
        spread = [np.atleast_1d(np.asarray(p[k], float)) for p in per]
        rng = np.ptp(np.stack(spread), 0)
        print(f"  {k:<22} 중앙 {np.round(v[:4], 3)}  발화 간 폭 최대 {np.max(rng):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
