p="src/formant_ml/engine/voice_td.py"; s=open(p,encoding="utf-8").read()
a="VF_JIT_SIGMA = 0.0\n"
b='''VF_JIT_SIGMA = 0.0
#: **신경 작용 — 후두근 운동 단위 요동** (§52.523, `physics/neural.py`, Titze 1991). 켜면 윤상갑상근(CT) 긴장의 운동 단위 합(평균 1)을 표본률로 덮개 긴장
#: Q 에, 갑상피열근(TA) 긴장을 프레임률로 갑상피열근 활성 `vf_ta` 에 곱한다. 단위 수 `VF_NEURAL_NMU` (200 이면 긴장 σ CT 1.6 · TA 2.6 %, 그 36–44 % 가
#: 4–12 Hz 떨림). 실현은 판의 씨앗으로 고정. 모드 3 (보–막) 에서만.
VF_NEURAL = False
VF_NEURAL_NMU = 200
VF_NEURAL_TREMOR = (6.0, 0.03)          # 생리적 떨림 [Hz], 공통 구동 깊이
'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''            ta = c["vf_ta"].clamp(0.0, 1.0) if "vf_ta" in c else torch.full_like(r1, 0.3)
            if VF_TA_SMOOTH_MS > 0.0:
                ta = _gauss_t(ta, VF_TA_SMOOTH_MS / (1000.0 * self.hop / self.fs))
            if SLEW_ON:
                ta = _slew(ta, VF_VMAX["vf_ta"] * self.hop / self.fs)
            eps = vf_ns_eps(f0)'''
b='''            ta = c["vf_ta"].clamp(0.0, 1.0) if "vf_ta" in c else torch.full_like(r1, 0.3)
            if VF_TA_SMOOTH_MS > 0.0:
                ta = _gauss_t(ta, VF_TA_SMOOTH_MS / (1000.0 * self.hop / self.fs))
            if SLEW_ON:
                ta = _slew(ta, VF_VMAX["vf_ta"] * self.hop / self.fs)
            if VF_NEURAL:                                   # 갑상피열근 긴장의 운동 단위 요동 (프레임률, 연축 ~15 ms)
                ta = (ta * self._neural("ta", ta.shape[-1], self.fs / self.hop, ta.dtype, ta.device)).clamp(0.0, 1.0)
            eps = vf_ns_eps(f0)'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''            q = u[..., 0] * self._vf_jitter(n_sim, u.dtype, u.device)
            if VF_BODY_DRIVE and pulse_phase is not None:'''
b='''            q = u[..., 0] * self._vf_jitter(n_sim, u.dtype, u.device)
            if VF_NEURAL:                                   # 윤상갑상근 긴장의 운동 단위 요동 (표본률) → 덮개 긴장
                q = q * self._neural("ct", n_sim, td.FS_SIM, q.dtype, q.device)
            if VF_BODY_DRIVE and pulse_phase is not None:'''
assert s.count(a)==1; s=s.replace(a,b,1)
a='''    def _vf_jitter(self, n: int, dtype, device) -> torch.Tensor | float:'''
b='''    def _neural(self, muscle: str, n: int, fs: float, dtype, device) -> torch.Tensor:
        """후두근 상대 긴장 (n,) — 운동 단위 모형 (`VF_NEURAL`), 씨앗 고정, 같은 길이면 다시 쓰지 않는다."""
        key = (muscle, n, fs, VF_NEURAL_NMU, VF_NEURAL_TREMOR, self.seed)
        cache = self.__dict__.setdefault("_neural_cache", {})
        if key not in cache:
            from ..physics import neural as _nr
            cache[key] = _nr.tension(n, fs, muscle, n_mu=VF_NEURAL_NMU, tremor_hz=VF_NEURAL_TREMOR[0],
                                     tremor_depth=VF_NEURAL_TREMOR[1], seed=int(self.seed))
        return torch.as_tensor(cache[key], dtype=dtype, device=device)

    def _vf_jitter(self, n: int, dtype, device) -> torch.Tensor | float:'''
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s)
p="scripts/copyfit.py"; s=open(p,encoding="utf-8").read()
a='    ap.add_argument("--vf-body-drive", action="store_true",'
b='''    ap.add_argument("--vf-neural", action="store_true",
                    help="신경 작용 (voice_td.VF_NEURAL, §52.523): 후두근(CT·TA) 긴장의 운동 단위 요동·생리적 떨림 (Titze 1991) 을 보–막 성대에")
'''+a
assert s.count(a)==1; s=s.replace(a,b,1)
a='''            if getattr(a, "vf_body_drive", False):'''
b='''            if getattr(a, "vf_neural", False):
                _vtdg.VF_NEURAL = True
                print(f"  신경 작용: 후두근 운동 단위 {_vtdg.VF_NEURAL_NMU} 개 · 떨림 {_vtdg.VF_NEURAL_TREMOR[0]:g} Hz (§52.523)", flush=True)
'''+a
assert s.count(a)==1; s=s.replace(a,b,1)
open(p,"w",encoding="utf-8",newline="\n").write(s); print("ok")
