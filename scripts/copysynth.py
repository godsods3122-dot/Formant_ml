"""복사합성 — 녹음을 분석해 그 파라미터로 물리모델을 구동한다.

    PYTHONPATH=src python3 scripts/copysynth.py IN.wav -o out/recon.wav [--from S --to S]

왜 이걸 먼저 하는가
-------------------
음절을 손으로 작곡하면 "스펙트럼 지표는 맞는데 사람 소리가 아닌" 상태에서
빠져나올 방법이 없다. 무엇이 틀렸는지 비교할 정답이 없기 때문이다.
복사합성은 **정답이 있다** — 원본 파형. 되합성이 원본처럼 안 들리면 엔진이
틀린 것이고, 어느 파라미터가 부족한지 바로 좁혀진다.
PLAN.md 의 Phase 1 이고, 이걸 건너뛰고 Phase 3(작곡)부터 한 것이 실패의 원인이다.

경로는 **포먼트 캐스케이드**를 쓴다. 도파관은 이득 정규화가 아직 안 풀렸고
(방사 임피던스 부재), 캐스케이드는 이 레포가 목표치 ±3 % 로 검증해 두었다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.analysis.acoustic import load as load16
from formant_ml.analysis.acoustic import band_db
from formant_ml.analysis.acoustic import voicing as periodicity
from formant_ml.analysis.track import track_formants
from formant_ml.config import Config
from formant_ml.data.features import yin_f0
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.utils import save_wav


def frame_rms(y: np.ndarray, hop: int, win: int) -> np.ndarray:
    t = max(0, 1 + (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(t)[:, None]
    return np.sqrt((y[idx] ** 2).mean(1))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """중앙값 + 이동평균. 한 프레임짜리 검출 실패를 지운다."""
    if len(x) < k or k < 2:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    med = np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)
    mp = np.pad(med, pad, mode="edge")
    return np.mean(np.stack([mp[i:i + len(x)] for i in range(k)]), axis=0)


def analyse(y: np.ndarray, cfg: Config):
    """파형 -> 복사합성에 필요한 프레임별 파라미터."""
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    win = 1024
    F, B = track_formants(y, sr, hop, win, n=cfg.filt.n_formants)
    t = len(F)
    yt = torch.from_numpy(y).float()[None]
    f0, voi = yin_f0(yt, sr, hop)
    f0, voi = f0[0, :t].numpy(), voi[0, :t].numpy()
    rms = frame_rms(y, hop, win)[:t]

    # **이진 유성 판정을 게이트로 쓰면 안 된다.** 유음처럼 진폭이 잠깐 꺼지고
    # 포먼트가 빨리 움직이는 구간에서 검출기가 자신을 잃고 0 을 내는데, 그
    # 순간 (a) 하모닉 진폭이 0 이 되어 소리가 파이고 (b) 무성으로 보고 마찰
    # 노이즈가 쏟아진다. 스펙트로그램에서 4~7 kHz 세로 뭉치로 나타난다.
    # 실제로 "라" 는 전 구간 유성이다 — 검출기가 틀린 것이지 소리가 무성인
    # 것이 아니다. 그래서 **연속 주기성**을 쓰고 시간축으로 평활한다.
    per, _ = periodicity(y, sr, hop_ms=hop / sr * 1000.0,
                         win_ms=win / sr * 1000.0)
    per = _smooth(per[:t], 7)
    per = np.clip((per - 0.25) / 0.45, 0.0, 1.0)

    # 비주기성만으로 노이즈를 켜면 안 된다. 모음 꼬리처럼 **조용한 유성 구간**
    # 에서는 SNR 이 떨어져 주기성 측정이 무너지는데, 그걸 무성으로 보고 노이즈를
    # 부으면 발화 끝에 쉬익 소리가 붙는다(스펙트로그램에서 확인).
    # 실제 마찰/기식은 **고역 에너지를 동반한다.** 그것을 조건으로 건다.
    hb = band_db(y, sr, 2500.0, 8000.0, hop / sr * 1000.0, win / sr * 1000.0)
    lb = band_db(y, sr, 100.0, 2500.0, hop / sr * 1000.0, win / sr * 1000.0)
    hf = _smooth(np.clip((hb - lb + 25.0) / 20.0, 0.0, 1.0)[:t], 7)

    # F0 는 신뢰 구간만 쓰고 나머지는 이어 붙인다 (0 이 들어가면 위상이 끊긴다)
    med = float(np.median(f0[voi > 0.5])) if (voi > 0.5).any() else 120.0
    fv = np.where(voi > 0.5, f0, np.nan)
    ok = ~np.isnan(fv)
    f0 = (np.interp(np.arange(t), np.arange(t)[ok], fv[ok]) if ok.any()
          else np.full(t, med))
    f0 = _smooth(np.clip(f0, cfg.source.f0_min, cfg.source.f0_max), 3)
    return dict(freq=F, bw=B, f0=f0, periodicity=per, hf=hf, rms=rms, t=t)


def build_controls(a: dict, cfg: Config, jitter: float, shimmer: float,
                   tilt: float, rd: float):
    t, K = a["t"], cfg.filt.n_formants
    nb = cfg.noise.n_bands

    def T(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    freq = T(a["freq"]).clamp(cfg.filt.f_min, cfg.filt.f_max)[None]
    bw = T(a["bw"]).clamp(cfg.filt.bw_min, cfg.filt.bw_max)[None]
    rms = a["rms"] / max(a["rms"].max(), 1e-9)
    per = a["periodicity"]
    hf = a["hf"]
    # 난류 소스의 스펙트럼 기울기. **평탄한 백색으로 두면 안 된다** — 노이즈
    # 경로를 재니 +13.4 dB/oct 로 치솟아(캐스케이드의 고역 극들을 그대로 통과)
    # 유성 구간에 섞인 2 % 만으로도 고역 전체를 덮어 "쨍한" 소리를 만들었다.
    # 실제 기식/마찰 소스는 고역으로 갈수록 떨어진다.
    fb = np.linspace(0.0, cfg.audio.sample_rate / 2, nb)
    shape = (1.0 / (1.0 + (fb / 700.0) ** 2)) ** 0.9
    shape = shape / shape.max()
    return Controls(
        f0=T(a["f0"]).reshape(1, t, 1),
        # 진폭은 **측정된 세기**를 그대로 따른다. 유성도로 곱해서 끄면
        # 검출 실패가 그대로 진폭 구멍이 된다.
        harmonic_amp=T(rms).reshape(1, t, 1),
        rd=torch.full((1, t, 1), rd),
        formant_freq=freq, formant_bw=bw,
        formant_gain=torch.ones(1, t, K),
        # 무성 구간의 마찰 성분. 유성 구간에는 소량의 기식만 남긴다.
        # 노이즈는 **연속 비주기성**에 비례한다. 이진 판정에 묶으면 검출이
        # 흔들릴 때마다 마찰음 버스트가 터진다.
        noise_bands=(T(rms * (0.12 * (1.0 - per) ** 2 * hf + 0.002)
                       ).reshape(1, t, 1) * T(shape).reshape(1, 1, nb)
                     ).contiguous(),
        # 기식은 **성문**에서 난다 -> 성도 전체를 지난다. 여기를 K*0.7 로 두면
        # 노이즈가 상위 포먼트(5~11 kHz)만 통과해 +33 dB/oct 로 치솟고,
        # 유성 구간에 섞인 2 % 만으로도 고역을 덮어 "쨍한" 소리가 된다.
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.full((1, t, 1), 0.5),
        tilt=torch.full((1, t, 1), tilt),
        jitter=torch.full((1, t, 1), jitter),
        shimmer=torch.full((1, t, 1), shimmer),
    )


def match_envelope(y: np.ndarray, target_rms: np.ndarray, hop: int,
                   floor: float = 1e-5) -> np.ndarray:
    """합성 파형의 프레임 세기를 원본의 세기 곡선에 맞춘다.

    성도 캐스케이드의 전체 이득은 포먼트 배치에 따라 달라지므로, 제어에 넣은
    진폭이 그대로 출력 세기가 되지 않는다. 무음 구간에서 특히 나빴다 —
    원본이 -39 dB 인데 합성이 -25 dB 로 나와 배경이 쉬익 소리로 들렸다.
    복사합성에서 세기 곡선은 **분석된 파라미터**이므로 맞추는 것이 옳다.
    """
    t = len(target_rms)
    n = t * hop
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    fr = y[:n].reshape(t, hop)
    cur = np.sqrt((fr ** 2).mean(1))
    tgt = target_rms / max(target_rms.max(), 1e-12)
    cur = cur / max(cur.max(), 1e-12)
    g = np.where(cur > floor, tgt / np.maximum(cur, floor), 0.0)
    g = np.clip(g, 0.0, 3.0)
    # 프레임 경계 클릭을 피해 샘플 단위로 선형보간
    gs = np.interp(np.arange(n), np.arange(t) * hop + hop / 2, g)
    return y[:n] * gs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", default="out/recon.wav")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    # 실측 정상 음성 범위(지터 0.2~0.5 %, 시머 2~4 %). 이전 0.006/0.04 는
    # 그보다 높았다. 고역 하모닉을 뭉갠다고 의심했으나 지표로는 기각됐다
    # (원본의 고역도 하모닉이 안 잡힌다) — 생리적 값으로만 낮춘 것이다.
    ap.add_argument("--jitter", type=float, default=0.002)
    ap.add_argument("--shimmer", type=float, default=0.02)
    # 측정으로 고른 값. tilt=0, Rd=1.2 로 두면 소스가 어두워 4~7 kHz 가
    # 원본보다 13 dB 낮았다(포락선 오차 10.6 dB). 이 조합에서 3.5 dB.
    ap.add_argument("--tilt", type=float, default=-12.0)
    ap.add_argument("--rd", type=float, default=0.6)
    args = ap.parse_args()

    cfg = Config()
    y, sr = load16(args.wav, cfg.audio.sample_rate)
    if args.t0 is not None:
        y = y[int(args.t0 * sr): int((args.t1 or len(y) / sr) * sr)]
    y = y / max(np.abs(y).max(), 1e-9)

    a = analyse(y, cfg)
    syn = PhysicalVoiceSynth(cfg, tract_mode="formant")
    c = build_controls(a, cfg, args.jitter, args.shimmer, args.tilt, args.rd)
    with torch.no_grad():
        out = syn(c)["audio"]
    out = match_envelope(out[0].numpy().astype(np.float64), a["rms"],
                         cfg.audio.hop_size)
    save_wav(args.out, torch.from_numpy(out).float(), sr)
    print(f"{args.out}  ({a['t'] * cfg.audio.hop_size / sr:.2f}s, "
          f"{a['t']} 프레임, 주기성 평균 {float(a['periodicity'].mean()):.2f})")


if __name__ == "__main__":
    main()
