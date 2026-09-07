"""스크립트 -> wav.

    PYTHONPATH=src python -m formant_ml.render script.yaml -o out/line.wav
    PYTHONPATH=src python -m formant_ml.render --list-params
"""
from __future__ import annotations

import argparse
import os

from .config import Config
from .score import PARAM_HELP, SEGMENT_TYPES, load_profile, load_score, render
from .utils import save_wav


def main() -> None:
    ap = argparse.ArgumentParser(description="물리 파라미터 스크립트 -> 음성")
    ap.add_argument("script", nargs="?", help="YAML 또는 JSON 스크립트")
    ap.add_argument("-o", "--out", default="out/render.wav")
    ap.add_argument("--voice", help="VoiceProfile JSON (스크립트 값보다 우선)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-params", action="store_true",
                    help="조종 가능한 모든 파라미터를 출력한다")
    args = ap.parse_args()

    if args.list_params:
        print("세그먼트 type:", ", ".join(SEGMENT_TYPES))
        print("\n파라미터 (세그먼트 안 또는 params: 아래에서 지정):")
        for k, v in PARAM_HELP.items():
            print(f"  {k:16s} {v}")
        print("\n값 형식:  1.2  |  [시작, 끝]  |  [[0, a], [0.5, b], [1, c]]")
        return

    if not args.script:
        ap.error("스크립트 경로가 필요합니다 (또는 --list-params)")
    score = load_score(args.script)
    prof = load_profile(score, os.path.dirname(os.path.abspath(args.script)))
    if args.voice:
        from .voice import VoiceProfile
        prof = VoiceProfile.load(args.voice)
    audio = render(score, prof, Config(), seed=args.seed)
    save_wav(args.out, audio, Config().audio.sample_rate)
    print(f"{args.out}  ({audio.shape[-1] / Config().audio.sample_rate:.2f} 초, "
          f"목소리: {prof.name})")


if __name__ == "__main__":
    main()
