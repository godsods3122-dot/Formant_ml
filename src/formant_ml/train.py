"""복사합성(copy-synthesis) 학습 루프.

목표: 입력 음성의 멜/F0 만 보고, 물리모델의 제어 파라미터를 예측해
원 파형을 재현한다. 여기서 잘 되면 그 다음 단계(텍스트->제어 파라미터)는
훨씬 저차원 문제가 된다.

    python -m formant_ml.train --data data/wavs --steps 2000 --out runs/exp1
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import AudioFolder, compute_features
from .models.encoder import ControlEncoder
from .models.losses import VoiceLoss
from .models.synth import PhysicalVoiceSynth
from .utils import save_wav


def build(cfg: Config, tract_mode: str, device: str):
    enc = ControlEncoder(cfg, tract_mode=tract_mode).to(device)
    syn = PhysicalVoiceSynth(cfg, tract_mode=tract_mode).to(device)
    return enc, syn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="wav 폴더 (24 kHz 모노)")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tract", default="formant", choices=["formant", "waveguide"])
    ap.add_argument("--w-phase", type=float, default=0.2)
    ap.add_argument("--w-noise", type=float, default=2e-2,
                    help="유성 구간 노이즈 억제 (0 이면 HNR 붕괴가 잘 일어남)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()

    cfg = Config()
    os.makedirs(args.out, exist_ok=True)
    n_files = len(AudioFolder(args.data, cfg, args.seconds).files)
    repeat = max(1, -(-(args.batch * 8) // n_files))   # 작은 데이터셋 보호
    ds = AudioFolder(args.data, cfg, args.seconds, repeat=repeat)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.workers,
                    persistent_workers=args.workers > 0)
    enc, syn = build(cfg, args.tract, args.device)
    loss_fn = VoiceLoss(w_phase=args.w_phase, w_noise=args.w_noise)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, args.lr * 0.05)

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            x = batch["audio"].to(args.device)
            with torch.no_grad():
                feat = compute_features(x, cfg)
            ctrl = enc(feat["mel"], feat["f0"], feat["voicing"])
            y = syn(ctrl)["audio"]
            n = min(y.shape[-1], x.shape[-1])
            losses = loss_fn(y[..., :n], x[..., :n], ctrl, feat["voicing"])

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                msg = "  ".join(f"{k}={float(v.detach()):.3f}" for k, v in losses.items())
                print(f"[{step:6d}/{args.steps}] {msg}  "
                      f"({(time.time() - t0) / step:.2f}s/step)", flush=True)
            if step % args.ckpt_every == 0:
                torch.save({"encoder": enc.state_dict(), "cfg": cfg, "step": step,
                            "tract_mode": args.tract},
                           os.path.join(args.out, "encoder.pt"))
                save_wav(os.path.join(args.out, f"recon_{step}.wav"),
                         y[0, :n], cfg.audio.sample_rate)
                save_wav(os.path.join(args.out, f"target_{step}.wav"),
                         x[0, :n], cfg.audio.sample_rate)
    print("완료:", args.out)


if __name__ == "__main__":
    main()
