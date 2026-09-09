"""치찰음의 **on/off 가 유성 구간을 오염시키는지** 잰다 (MEASUREMENTS §24).

    python scripts/probe_fric_onoff.py out/room2/s040 out/sw/r010

사용자 지적: "치찰음의 on,off 가 잘 안 돼서 그게 음성을 오염시킨 전적이 있었으니
체크해."

무엇을 재나
-----------
마찰 소스는 레이놀즈 게이트를 지난 **실제 유량**이다. 그것이 **유성 프레임 안에서**
켜지면, 에너지가 작아도 1 ms 짜리 광대역 스위칭이라 스펙트로그램에 세로 줄로 남는다.

1. `유성 켜짐 %`  — 마찰 소스가 0 이 아닌 유성 프레임의 비율.
2. `사건`         — 유성 구간 안에서 꺼짐 -> 켜짐으로 바뀐 횟수 (100 ms 당).
3. `상승 ms`      — 그 사건들이 0 에서 정점까지 걸린 시간의 중앙값. 조음기가
                    낼 수 있는 값은 수십 ms 다. 1 ms 면 조음이 아니라 스위칭이다.
4. `유성 중 마찰 dB` — 유성 구간 총에너지에서 마찰이 차지하는 몫. 작아도 (1)~(3)
                    이 나쁘면 소리는 거칠다 — 그래서 dB 만 보면 안 된다.
5. `a_c 속도`     — 협착 면적의 **로그** 변화율 [neper/ms]. 가장 빠른 실제 제스처가
                    0.172 다 (§24.2). 그 위는 물리적으로 불가능하다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.engine.control import INDEX, ControlTrack
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine


def measure(stem: str, prof: SpeakerProfile) -> dict:
    d = np.load(stem + "_track.npz")
    vals = np.asarray(d["values"], float)
    fm = float(d["frame_ms"])
    tr = ControlTrack(vals, frame_ms=fm)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=fm,
                                   speaker="female" if prof.f0_nominal > 165 else "male",
                                   residual=False), prof)
    eng.reset()
    with torch.no_grad():
        out = eng(tr.to_tensor(), getattr(tr, "events", None) or [], 0.0)
    hop = eng.cfg.hop
    fric = out["fric"].squeeze().numpy()
    audio = out["audio"].squeeze().numpy()
    n = len(fric) // hop

    def per_frame(x):
        return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))

    fr_f, au_f = per_frame(fric), per_frame(audio)
    ps = vals[:n, INDEX["p_sub"]]
    voiced = (ps > 2.0) & (vals[:n, INDEX["adduction"]] > 0.2)
    # **"켜짐" 을 fr > 0 으로 재면 안 된다.** 부동소수 잔여 때문에 마찰 소스가 정확히
    # 0 이 되는 일은 없고, 그러면 모든 프레임이 켜짐으로 잡힌다 (실측: 전부 100 %).
    # 실제 마찰 구간의 정점 대비 **−40 dB** 를 문턱으로 쓴다 — 그 아래는 귀에도
    # 스펙트로그램에도 안 남는다.
    peak = fr_f.max() if fr_f.size else 0.0
    thr = peak * 10.0 ** (-40.0 / 20.0)
    hot = fr_f > thr
    # 마찰 구간 자체는 뺀다 — 거기서 켜지는 것은 정상이다.
    strong = fr_f > peak * 10.0 ** (-20.0 / 20.0)
    voi = voiced & ~strong
    on = hot & voi
    res = {"frames": int(voi.sum()), "on_pct": float(100.0 * on[voi].mean()) if voi.any() else 0.0}
    # 켜짐 사건과 상승 시간
    idx = np.where(np.diff(hot.astype(int)) == 1)[0] + 1
    idx = np.array([i for i in idx if i < n and voi[i]])
    res["events_per_100ms"] = float(len(idx) / max(voi.sum() * fm / 100.0, 1e-9))
    rises = []
    for i in idx:
        j = min(i + int(80.0 / fm), n - 1)
        seg = fr_f[i:j + 1]
        if len(seg) < 2 or seg.max() <= thr:
            continue
        rises.append((int(np.argmax(seg)) + 1) * fm)
    res["rise_ms"] = float(np.median(rises)) if rises else float("nan")
    e_f = float((fr_f[voi] ** 2).sum()) if voi.any() else 0.0
    res["_thr_db"] = -40.0
    e_a = float((au_f[voi] ** 2).sum()) if voi.any() else 1e-30
    res["fric_in_voiced_db"] = 10.0 * np.log10(e_f / max(e_a, 1e-30) + 1e-30)
    ac = np.clip(vals[:, INDEX["a_c"]], 1e-4, None)
    rate = np.abs(np.diff(np.log(ac))) / fm
    res["ac_rate_p95"] = float(np.percentile(rate, 95))
    res["ac_rate_max"] = float(rate.max())
    res["ac_over_pct"] = float(100.0 * np.mean(rate > 0.20))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile) if os.path.exists(a.profile) else DEFAULT_PROFILE
    hdr = f"{'':26s} {'유성켜짐%':>9s} {'사건/100ms':>10s} {'상승ms':>7s} " \
          f"{'유성중마찰dB':>12s} {'a_c95':>7s} {'a_c최대':>8s} {'한계초과%':>9s}"
    print(hdr)
    for s in a.stems:
        r = measure(s, prof)
        print(f"{os.path.basename(s):26s} {r['on_pct']:9.2f} {r['events_per_100ms']:10.2f} "
              f"{r['rise_ms']:7.1f} {r['fric_in_voiced_db']:12.1f} "
              f"{r['ac_rate_p95']:7.3f} {r['ac_rate_max']:8.3f} {r['ac_over_pct']:9.2f}")
    print("\n(가장 빠른 실제 제스처의 a_c 속도는 0.172 neper/ms — §24.2)")


if __name__ == "__main__":
    main()
