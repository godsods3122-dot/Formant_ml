p="src/formant_ml/engine/fit.py"; s=open(p,encoding="utf-8").read()
def rep(a,b,cnt=1):
    global s
    n=s.count(a); assert n==cnt,(a[:70],n); s=s.replace(a,b)
rep("REFINE_SLIP_ITERS = 0\n",'''REFINE_SLIP_ITERS = 0
#: **위상 보정의 변화율 상한** (§52.525). 창별 보정·미끄러짐 수리는 잠금 위상을 시간축으로 당기거나 민다 (n → n + d(n)). d 가 급하게 변하면 그 대응이 거꾸로 가고,
#: 단조로 만들려던 `maximum.accumulate` 가 위상을 평평하게 만들어 **위상이 멈췄다** — m7 의 펄스 위상에 이웃 대비 +609 % · +538 % 주기 (목소리가 ~15 ms 끊김)와
#: −93 % 주기가 27 개 (원본 Praat 2 개), 사용자: *"진폭이 굉장히 왔다갔다 해서 마치 렉 걸린 소리"*. 보정량을 표본마다 ±`PHASE_WARP_RMAX` 로 슬루 제한한다 — 대응의
#: 기울기가 1 ± r 에 머물러 위상이 멈추지 않고, 보정 때문에 주기 길이가 r 넘게 바뀌지 않는다 (자연 지터 1–3 %).
PHASE_WARP_RMAX = 0.05


def _warp_phase(ph: np.ndarray, d: np.ndarray, rmax: float = None) -> np.ndarray:
    """잠금 위상 ph 를 n → n + d(n) [표본] 으로 당긴다 — d 를 표본마다 ±rmax 로 슬루 제한해 대응을 단조·완만하게 (§52.525)."""
    r = PHASE_WARP_RMAX if rmax is None else rmax
    n = len(ph)
    dc = np.empty(n)
    acc = float(d[0])
    for i in range(n):
        acc += min(max(float(d[i]) - acc, -r), r)
        dc[i] = acc
    src = np.clip(np.arange(n) + dc, 0.0, n - 1.0)
    return np.maximum.accumulate(np.interp(src, np.arange(n), ph))
''')
rep('''        src = np.clip(np.arange(n) + d, 0.0, n - 1.0)          # 합성이 늦으면 위상을 앞당긴다
        new = np.interp(src, np.arange(n), ph)
        new = np.maximum.accumulate(new)                        # 위상은 단조여야 한다 (엔진이 검사한다)''','''        new = _warp_phase(ph, d)                                 # 합성이 늦으면 위상을 앞당긴다 — 변화율 제한 (§52.525)''')
rep('''            new = np.interp(np.clip(np.arange(n) + dsmp, 0.0, n - 1.0), np.arange(n), ph)
            self._pulse_phase = torch.as_tensor(np.maximum.accumulate(new)[None, :], dtype=torch.float64,''','''            new = _warp_phase(ph, dsmp)                          # 변화율 제한 (§52.525)
            self._pulse_phase = torch.as_tensor(new[None, :], dtype=torch.float64,''')
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
