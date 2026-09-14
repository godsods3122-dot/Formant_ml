"""적합 로그 나란히 보기 — 같은 단계·같은 반복에서 포락/정밀/오차를 줄 세운다 (2.1~2.2 단계 판정용).

    python scripts/diag/fit_race.py out/M/M18/s101.log out/M/M19/s101.log out/M/M14/s101.log

사용자 규칙: 2.1~2.2 단계가 끝나는 지점에서 대조군보다 뚜렷하게(약 2 점) 뒤지면 바로 세운다.
"""
from __future__ import annotations

import os
import re
import sys

STAGE = re.compile(r"^\s*(1 단계|2\.\d 단계|3 단계)")
ITER = re.compile(r"^\s*\[\s*(\d+)\]\s*포락\s*([-\d.]+)%\s*정밀\s*([-\d.]+)%\s*오차\s*([-\d.]+) dB")


def parse(path):
    """반환 {(단계, 반복): (포락, 정밀, 오차)} 와 단계 순서."""
    out, order, stage = {}, [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STAGE.match(line)
            if m:
                stage = m.group(1).replace(" 단계", "")
                order.append(stage)
                continue
            m = ITER.match(line)
            if m and stage is not None:
                out[(stage, int(m.group(1)))] = tuple(float(m.group(k)) for k in (2, 3, 4))
    return out, order


def main() -> int:
    paths = sys.argv[1:]
    runs = [(os.path.basename(os.path.dirname(p)) or p, *parse(p)) for p in paths]
    stages = []
    for _n, _d, order in runs:
        for s in order:
            if s not in stages:
                stages.append(s)
    head = "단계  반복 | " + " | ".join(f"{n:^24s}" for n, _d, _o in runs)
    print(head)
    print("-" * len(head))
    for s in stages:
        its = sorted({i for _n, d, _o in runs for (st, i) in d if st == s})
        for i in its:
            cells = []
            for _n, d, _o in runs:
                v = d.get((s, i))
                cells.append(f"{v[0]:6.2f} / {v[1]:6.2f} / {v[2]:5.2f}" if v else f"{'—':^24s}")
            print(f"{s:>4s} {i:5d} | " + " | ".join(cells))
    print("\n칸 = 포락 % / 정밀 % / 오차 dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
