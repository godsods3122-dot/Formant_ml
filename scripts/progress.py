"""적합 진척도 표시기 — 돌고 있는 판을 한눈에 (사용자: *"일일이 너한테 안 물어보게"*).

`out/**/<판>.log` 와 `<판>.argv` 를 읽는다. 적합 코드는 건드리지 않으므로 이미 돌고 있는 판에도 쓴다.

    python scripts/progress.py                 # 한 번 찍기
    python scripts/progress.py --watch         # 10 초마다 새로 그림 (Ctrl+C 로 끝)
    python scripts/progress.py --watch --html out/O/progress.html   # 브라우저로도 (15 초마다 스스로 새로고침)

판단: 끝 줄("결과:")이 있으면 '끝'. 없으면 **그 판을 쓰는 프로세스가 살아 있는지**(`--out <판>`)를 보고 살아 있으면 '진행', 아니면 '멈춤'
(죽었거나 중단). 프로세스 목록을 못 읽으면 예전처럼 '로그가 5 분 안에 바뀌었나' 로 짐작한다 — 그 방식은 학습률 탐침 한 번이 5 분을
넘는 긴 판을 '멈춤' 으로, 막 죽인 판을 5 분 동안 '진행' 으로 보였다.
남은 시간 = 경과 × (남은 회차 / 끝낸 회차). 회차는 전역 단계 + 격자 단계마다 `--stage-iters` — 고운 격자일수록 한 회차가 느리므로
앞쪽에서는 남은 시간을 짧게 잡는 편이다.
"""
import argparse
import glob
import html
import os
import re
import time

ROOTS = ("out/C", "out/CA", "out/_tmp/abl")
RECENT_S = 300
STAGE_RE = re.compile(r"^\s*(1 단계 전역|2\.\d+ 단계)")
IT_RE = re.compile(r"\[\s*(\d+)\]\s*포락\s+([\d.]+)%")
FINAL_RE = re.compile(r"^포락 ([\d.]+)%\s+정밀 ([\d.]+)%")


_LIVE = None


def live_outs():
    """살아 있는 적합 프로세스의 `--out` 값 집합. 못 읽으면 None (그러면 로그 시각으로 짐작한다)."""
    global _LIVE
    if _LIVE is not None:
        return _LIVE or None
    import subprocess
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { $_.CommandLine }"]
        else:
            cmd = ["ps", "-eo", "args"]
        txt = subprocess.run(cmd, capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace").stdout
        _LIVE = {os.path.normpath(m) for m in re.findall(r"--out\s+(\S+)", txt)}
    except Exception:
        _LIVE = set()
        return None
    return _LIVE


def _argv(stem):
    p = stem + ".argv"
    if not os.path.exists(p):
        return {}
    a = [x.strip() for x in open(p, encoding="utf-8").read().split("\n") if x.strip()]
    get = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    return {"grids": get("--grids", "20,8,3,1"), "stage": int(get("--stage-iters", "250")),
            "glob": int(get("--global-iters", "200")), "init": get("--init"), "no_fit": "--no-fit" in a}


def status(log):
    stem = log[:-4]
    name = os.path.basename(stem)
    txt = open(log, encoding="utf-8", errors="replace").read().splitlines()
    cfg = _argv(stem)
    n_grid = len(str(cfg.get("grids", "")).split(",")) if cfg else 0
    total = (cfg.get("glob", 0) if cfg else 0) + n_grid * (cfg.get("stage", 0) if cfg else 0)
    stage, it_in, env, done_before, stage_i = "준비", 0, None, 0, -1
    final = None
    for ln in txt:
        m = STAGE_RE.match(ln)
        if m:
            if stage_i >= 0:
                done_before += cfg.get("glob", 0) if stage_i == 0 and stage.startswith("1") else cfg.get("stage", 0)
            stage_i += 1
            stage, it_in = m.group(1), 0
        m = IT_RE.search(ln)
        if m:
            it_in, env = int(m.group(1)), float(m.group(2))
        m = FINAL_RE.match(ln)
        if m:
            final = (float(m.group(1)), float(m.group(2)))
    finished = any(ln.startswith("결과:") for ln in txt[-5:])
    crashed = any("Traceback" in ln for ln in txt)
    mt = os.path.getmtime(log)
    t0 = os.path.getmtime(stem + ".argv") if os.path.exists(stem + ".argv") else os.path.getctime(log)
    now = time.time()
    live = live_outs()
    alive = (os.path.normpath(stem) in live) if live is not None else (now - mt < RECENT_S)
    state = "끝" if finished else ("오류" if crashed else ("진행" if alive else "멈춤"))
    done = min(done_before + it_in, total) if total else 0
    frac = (1.0 if finished else (done / total if total else 0.0))
    elapsed = (mt if state != "진행" else now) - t0
    eta = elapsed * (1 - frac) / frac if (state == "진행" and frac > 0.02) else None
    return dict(name=name, state=state, stage=stage, it=it_in, env=env, final=final, frac=frac,
                elapsed=elapsed, eta=eta, mtime=mt, init=cfg.get("init"))


def collect(max_age_h):
    global _LIVE
    _LIVE = None                                         # 회차마다 프로세스 목록을 새로 읽는다 (--watch)
    rows = []
    for root in ROOTS:
        for log in glob.glob(os.path.join(root, "*.log")):
            if time.time() - os.path.getmtime(log) > max_age_h * 3600:
                continue
            try:
                rows.append(status(log))
            except Exception as e:                       # 표시기는 죽지 않는다
                rows.append(dict(name=os.path.basename(log), state=f"읽기 실패 {e}", stage="", it=0, env=None,
                                 final=None, frac=0, elapsed=0, eta=None, mtime=os.path.getmtime(log), init=None))
    order = {"진행": 0, "오류": 1, "멈춤": 2, "끝": 3}
    rows.sort(key=lambda r: (order.get(r["state"], 4), -r["mtime"]))
    return rows


def _hm(s):
    if s is None:
        return "  -  "
    s = int(max(s, 0))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}"


def render_text(rows):
    out = [time.strftime("%H:%M:%S") + "  적합 진척도  (진행 > 오류 > 멈춤 > 끝)", ""]
    out.append(f"{'판':12s} {'상태':4s} {'단계':12s} {'진척':24s} {'포락':>7s} {'경과':>6s} {'남음':>6s}  {'끝 점수'}")
    for r in rows:
        bar = "█" * int(r["frac"] * 20) + "·" * (20 - int(r["frac"] * 20))
        env = f"{r['env']:.2f}" if r["env"] is not None else "-"
        fin = f"포락 {r['final'][0]:.2f} 정밀 {r['final'][1]:.2f}" if r["final"] else ""
        out.append(f"{r['name'][:12]:12s} {r['state']:4s} {r['stage'][:12]:12s} {bar} {r['frac'] * 100:3.0f}% {env:>7s} "
                   f"{_hm(r['elapsed']):>6s} {_hm(r['eta']):>6s}  {fin}")
    return "\n".join(out)


def render_html(rows):
    cells = []
    for r in rows:
        col = {"진행": "#2a7", "오류": "#c33", "멈춤": "#c90", "끝": "#678"}.get(r["state"], "#999")
        fin = f"포락 {r['final'][0]:.2f} / 정밀 {r['final'][1]:.2f}" if r["final"] else ""
        env = f"{r['env']:.2f}" if r["env"] is not None else "-"
        cells.append(f"<tr><td>{html.escape(r['name'])}</td><td style='color:{col}'><b>{r['state']}</b></td>"
                     f"<td>{html.escape(r['stage'])}</td><td><div class=bar><div style='width:{r['frac'] * 100:.0f}%;background:{col}'>"
                     f"</div></div> {r['frac'] * 100:.0f}%</td><td>{env}</td><td>{_hm(r['elapsed'])}</td><td>{_hm(r['eta'])}</td>"
                     f"<td>{fin}</td><td>{html.escape(str(r['init'] or ''))}</td></tr>")
    return ("<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=15><title>적합 진척도</title>"
            "<style>body{font:14px sans-serif;margin:20px}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}"
            ".bar{display:inline-block;width:160px;height:10px;background:#eee;vertical-align:middle}.bar div{height:10px}</style>"
            f"<h3>적합 진척도 — {time.strftime('%H:%M:%S')} (15 초마다 새로고침)</h3><table><tr><th>판</th><th>상태</th><th>단계</th>"
            "<th>진척</th><th>포락 %</th><th>경과</th><th>남음</th><th>끝 점수</th><th>출발점</th></tr>" + "".join(cells) + "</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--html", default=None)
    ap.add_argument("--hours", type=float, default=12.0, help="이 시간 안에 바뀐 로그만")
    a = ap.parse_args()
    while True:
        rows = collect(a.hours)
        txt = render_text(rows)
        if a.html:
            open(a.html, "w", encoding="utf-8").write(render_html(rows))
        if a.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(txt, flush=True)
        if not a.watch:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
