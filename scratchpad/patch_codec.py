p="src/formant_ml/engine/fit.py"; s=open(p,encoding="utf-8").read()
a="CODEC_TRACK = False\n"
b=r'''CODEC_TRACK = False
#: **실제 AAC 코덱을 적합 고리에** (MEASUREMENTS §52.523, `copyfit --codec-aac 64k`). 038 원본은 방송 코덱을 거친 녹음이다 — 코덱의 지각 양자화가
#: 고역 미세 구조를 지운다 (코덱 없는 여성 녹음 EARS 줄 0.08–0.18 → AAC 64k 0.0–0.04; 합성 q1 0.20 · 0.24 · 0.11 → 0.08 · 0.05 · 0.04, 원본
#: 0.06 · 0.09 · 0.02). `CODEC_TRACK` 은 시변 대역 차단만 흉내 내 이것을 못 한다. 합성 끝에서 ffmpeg AAC 로 부호화·복호화한 소리로 손실을 재고,
#: 기울기는 코덱을 항등으로 보고 흘린다 (y + (AAC(y) − y).detach()). 사용자: *"당장은 코덱을 넣고 진행하고, 적합이 끝나면 코덱을 제거하자"* —
#: `render()` (결과 파일)는 코덱 없이, `render_codec()` 는 코덱을 거쳐 낸다. 빈 문자열이면 끈다. 값은 비트율.
CODEC_AAC = ""
CODEC_FFMPEG = r"C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''        if CODEC_TRACK:                          # 코덱 — 녹음 사슬의 맨 끝 (§52.206)
            y = self._codec(y)
        return (y, out["phase"]) if want_phase else y
'''
b=r'''        if CODEC_TRACK:                          # 코덱 — 녹음 사슬의 맨 끝 (§52.206)
            y = self._codec(y)
        if CODEC_AAC and not getattr(self, "_codec_aac_off", False):    # 실제 AAC, 직통 기울기 (§52.523)
            y = y + (self._aac(y.detach()) - y.detach())
        return (y, out["phase"]) if want_phase else y

    def _aac(self, y: torch.Tensor) -> torch.Tensor:
        """ffmpeg AAC (`CODEC_AAC` 비트율) 부호화 → 복호화. 지연은 처음 한 번 상호상관으로 잰다 (보통 0). 창 없이 (CREATE_NO_WINDOW)."""
        import os, shutil, subprocess, tempfile
        import soundfile as _sf
        x = y.reshape(-1).double().cpu().numpy()
        d = tempfile.mkdtemp(prefix="aac_")
        src, enc, dec = (os.path.join(d, n) for n in ("a.wav", "a.m4a", "b.wav"))
        try:
            _sf.write(src, x, int(self.fs), subtype="FLOAT")
            fl = 0x08000000 if os.name == "nt" else 0
            subprocess.run([CODEC_FFMPEG, "-y", "-loglevel", "error", "-i", src, "-c:a", "aac", "-b:a", CODEC_AAC, enc],
                           check=True, creationflags=fl, stdin=subprocess.DEVNULL)
            subprocess.run([CODEC_FFMPEG, "-y", "-loglevel", "error", "-i", enc, "-ar", str(int(self.fs)), "-ac", "1", dec],
                           check=True, creationflags=fl, stdin=subprocess.DEVNULL)
            z, _ = _sf.read(dec)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if not hasattr(self, "_aac_lag"):
            m = min(len(x), len(z) - 2048)
            c = np.correlate(z[:m + 2048], x[:m], "valid")
            self._aac_lag = int(np.argmax(c))
        z = z[self._aac_lag:self._aac_lag + len(x)]
        z = np.pad(z, (0, len(x) - len(z)))
        return torch.as_tensor(z, dtype=y.dtype, device=y.device).reshape(y.shape)

    def render_codec(self) -> np.ndarray | None:
        """코덱(`CODEC_AAC`)을 거친 결과 — 원본과 같은 조건의 비교용. 코덱이 꺼져 있으면 None."""
        if not CODEC_AAC:
            return None
        with torch.no_grad():
            return self.synth()[0].cpu().numpy()
'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''        with torch.no_grad():
            if seed is None:
                return self.synth()[0].cpu().numpy()'''
b='''        with torch.no_grad():
            if seed is None:
                self._codec_aac_off = True                       # 결과 파일은 코덱 없이 (§52.523)
                try:
                    return self.synth()[0].cpu().numpy()
                finally:
                    self._codec_aac_off = False'''
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s)
p="scripts/copyfit.py"; s=open(p,encoding="utf-8").read()
a='    ap.add_argument("--vf-body-drive", action="store_true",'
b='''    ap.add_argument("--codec-aac", default="", metavar="BITRATE",
                    help="실제 AAC 코덱을 적합 고리에 (fit.CODEC_AAC, §52.523) — 원본이 거친 방송 코덱. 손실은 코덱을 거친 소리로, 기울기는 직통. "
                         "결과 _fit.wav 는 코덱 없이, _fit_codec.wav 는 코덱을 거쳐 쓴다. 예: 64k")
'''+a
assert s.count(a)==1; s=s.replace(a,b,1)
a='    sf.write(a.out + "_fit.wav", out, 48000)\n'
b='''    sf.write(a.out + "_fit.wav", out, 48000)
    if _fit.CODEC_AAC:                                  # 코덱을 거친 것 (원본과 같은 조건, §52.523)
        sf.write(a.out + "_fit_codec.wav", fit.render_codec(), 48000)
'''
assert s.count(a)==1; s=s.replace(a,b,1)
a="    if a.harm_fmax is not None:\n"
b='''    if a.codec_aac:
        _fit.CODEC_AAC = a.codec_aac
        print(f"  녹음 코덱: AAC {a.codec_aac} 를 적합 고리에 (직통 기울기, 결과 파일은 코덱 없이, §52.523)", flush=True)
'''+a
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
