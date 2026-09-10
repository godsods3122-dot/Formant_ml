"""적합 결과를 **들을 수 있고 볼 수 있는 꼴**로 모은다.

    python scripts/deliver.py out/fin/s040 out/fin/s101 --out out/deliver

만드는 것 (stem 마다):
  <이름>_1_목표.wav      잡음 제거를 거친 원본 (적합이 본 그대로)
  <이름>_2_합성.wav      적합 결과
  <이름>_3_합성_마른.wav 방 IR 을 쓴 경우의 마른 소리 (없으면 건너뛴다)
  <이름>_비교.png        파형·스펙트로그램·평균 스펙트럼 비교
그리고 `합격표.txt` 에 네 조건 판정을 시드 여러 벌로 적는다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--out", default="out/deliver")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for stem in a.stems:
        name = os.path.basename(stem)
        pairs = [("_target.wav", "_1_목표.wav"), ("_fit.wav", "_2_합성.wav"),
                 ("_dry.wav", "_3_합성_마른.wav")]
        for src_sfx, dst_sfx in pairs:
            src = stem + src_sfx
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(a.out, name + dst_sfx))
        png = os.path.join(a.out, f"{name}_비교.png")
        r = subprocess.run([sys.executable, os.path.join(HERE, "compare_fit.py"),
                            stem, "--out", png], capture_output=True, text=True)
        print(r.stdout[-1200:] if r.returncode == 0 else r.stderr[-1200:])

    rep = subprocess.run([sys.executable, os.path.join(HERE, "acceptance.py"),
                          *a.stems, "--seeds", str(a.seeds)],
                         capture_output=True, text=True)
    txt = rep.stdout or rep.stderr
    with open(os.path.join(a.out, "합격표.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
