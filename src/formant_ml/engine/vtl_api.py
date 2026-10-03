"""VocalTractLab API (Birkholz) 감싸개 — 여성 화자 W02 의 3 차원 기하 성도 모형 (MEASUREMENTS §52.486).

VocalTractLab 2.4 (`third_party/VTL2.4`, 로컬 전용·커밋하지 않는다)의 `VocalTractLabApi.dll` 을 ctypes 로 부른다. 쓰는 것은 기하뿐이다:
해부학 변수 19 개 (설골 HX·HY, 턱 JX·JA, 입술 LP·LD, 연구개 VS·VO, 혀 몸통 중심 TCX·TCY, 혀끝 TTX·TTY, 혀날 TBX·TBY, 혀뿌리 TRX·TRY,
혀 옆면 TS1–3) → 면적 함수 50 칸 (가변 길이) + 앞니 자리 + 연구개 열림. 음향은 우리 시간 영역 관(`tube_td`)이 푼다.
DLL 은 기울기를 주지 않으므로 적합기는 이것을 흉내 낸 대리 신경망(`articulation.W02Surrogate`)을 쓰고, 이 감싸개는 학습 자료와 검증에만 쓴다.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parents[3] / "third_party" / "VTL2.4" / "API"
#: 해부 적응 함수를 더해 다시 빌드한 API (§52.501, `third_party/VTL2.4_anat` — VTL 2.4 원본 + `vtlGetAnatomyParamInfo`·`vtlSetAnatomyParams`).
ANAT_API_DIR = Path(__file__).resolve().parents[3] / "third_party" / "VTL2.4_anat" / "API"
#: VTL `AnatomyParams` 의 차례 (단위 cm, 마지막은 도).
ANAT_NAMES = ("lip_width", "mandible_height", "lower_molars_height", "upper_molars_height", "palate_height", "palate_depth",
              "hard_palate_length", "soft_palate_length", "pharynx_length", "larynx_length", "larynx_width", "vocal_fold_length",
              "oral_pharyngeal_angle")


class VTL:
    def __init__(self, speaker: str = "W02.speaker", api_dir: Path = API_DIR):
        cwd = os.getcwd()
        self.api_dir = api_dir
        os.chdir(api_dir)
        try:
            self.lib = ctypes.cdll.LoadLibrary(str(api_dir / "VocalTractLabApi.dll"))
            if self.lib.vtlInitialize(speaker.encode()) != 0:
                raise RuntimeError(f"vtlInitialize 실패: {speaker}")
        finally:
            os.chdir(cwd)
        sr, nt, ntp, ngp = (ctypes.c_int(0) for _ in range(4))
        self.lib.vtlGetConstants(ctypes.byref(sr), ctypes.byref(nt), ctypes.byref(ntp), ctypes.byref(ngp))
        self.n_tube, self.n_par = nt.value, ntp.value
        names = ctypes.create_string_buffer(self.n_par * 32)
        lo, hi, ne = ((ctypes.c_double * self.n_par)() for _ in range(3))
        self.lib.vtlGetTractParamInfo(names, lo, hi, ne)
        self.names = names.value.decode().split()
        self.lo, self.hi, self.neutral = np.array(lo), np.array(hi), np.array(ne)

    def anatomy_info(self):
        """해부 변수 (이름, 최소, 최대, 지금 값) — VTL AnatomyParams: 입술 폭, 하악 높이, 아래·위 어금니 높이, 구개 높이·깊이, 경·연구개 길이,
        인두 길이, 후두 길이·폭, 성대 길이, 구강–인두 각 [cm, 도]. `ANAT_API_DIR` 의 DLL 에만 있다."""
        n = 16
        names = ctypes.create_string_buffer(n * 32)
        lo, hi, x = ((ctypes.c_double * n)() for _ in range(3))
        if self.lib.vtlGetAnatomyParamInfo(names, lo, hi, x) != 0:
            raise RuntimeError("vtlGetAnatomyParamInfo 실패")
        k = len(ANAT_NAMES)                  # 약어에 빈칸이 섞여("W1 minus incisor width") 이름은 VTL 순서의 고정 목록으로
        return list(ANAT_NAMES), np.array(lo)[:k], np.array(hi)[:k], np.array(x)[:k]

    def set_anatomy(self, x, reference: str = "W02.speaker") -> None:
        """해부 변수를 바꾼다 — 기준 화자(처음 한 번 읽는다)에 대한 변형으로 기하·조음 변수 범위·음소 자세를 옮긴다 (VTL GUI 해부 대화상자와 같은
        절차). 조음 변수의 범위(`lo`·`hi`)도 다시 읽는다."""
        x = np.asarray(x, float)
        buf = (ctypes.c_double * len(x))(*x)
        cwd = os.getcwd()
        os.chdir(self.api_dir)
        try:
            if self.lib.vtlSetAnatomyParams(reference.encode(), buf) != 0:
                raise RuntimeError("vtlSetAnatomyParams 실패")
        finally:
            os.chdir(cwd)
        names = ctypes.create_string_buffer(self.n_par * 32)
        lo, hi, ne = ((ctypes.c_double * self.n_par)() for _ in range(3))
        self.lib.vtlGetTractParamInfo(names, lo, hi, ne)
        self.lo, self.hi, self.neutral = np.array(lo), np.array(hi), np.array(ne)

    def shape(self, name: str) -> np.ndarray:
        """화자 파일에 저장된 음소 자세 (MRI 에서 만든 것)."""
        p = (ctypes.c_double * self.n_par)()
        if self.lib.vtlGetTractParams(name.encode(), p) != 0:
            raise KeyError(name)
        return np.array(p)

    def tube(self, p):
        """(칸 길이 (50,) [cm], 면적 (50,) [cm²], 조음기 번호 (50,), 앞니 자리 [cm], 혀끝 옆 높이, 연구개 열림 [cm²])."""
        tp = (ctypes.c_double * self.n_par)(*np.asarray(p, float))
        L, A = (ctypes.c_double * self.n_tube)(), (ctypes.c_double * self.n_tube)()
        art = (ctypes.c_int * self.n_tube)()
        inc, tts, vel = ctypes.c_double(0), ctypes.c_double(0), ctypes.c_double(0)
        self.lib.vtlTractToTube(tp, L, A, art, ctypes.byref(inc), ctypes.byref(tts), ctypes.byref(vel))
        return np.array(L), np.array(A), np.array(art), inc.value, tts.value, vel.value

    def shape_names(self, speaker_file: str = "W02.speaker") -> list[str]:
        import re
        txt = (API_DIR / speaker_file).read_text(encoding="latin-1")
        out = []
        for n in re.findall(r'<shape name="([^"]+)"', txt):          # 성대 자세(stop, modal …)도 같은 태그라 성도 자세로 읽히는 것만
            if n.endswith("-raw"):
                continue
            try:
                self.shape(n)
                out.append(n)
            except KeyError:
                pass
        return out

    #: VTL 단면 윤곽 (`ANAT_API_DIR` DLL 에만 있다, §52.526): 중심선 단면 129 개 × 좌우 표본 96 개, 너비 7 cm.
    N_SLICE = 129
    N_PROF = 96
    PROF_LEN = 7.0
    #: VTL `VocalTract::SurfaceIndex` 이름 (단면 윤곽의 벽 기관 번호).
    SURFACES = ("UPPER_TEETH", "LOWER_TEETH", "UPPER_COVER", "LOWER_COVER", "UPPER_LIP", "LOWER_LIP", "PALATE", "MANDIBLE",
                "LOWER_TEETH_ORIGINAL", "LOW_VELUM", "MID_VELUM", "HIGH_VELUM", "NARROW_LARYNX_FRONT", "NARROW_LARYNX_BACK",
                "WIDE_LARYNX_FRONT", "WIDE_LARYNX_BACK", "TONGUE", "UPPER_COVER_TWOSIDE", "LOWER_COVER_TWOSIDE",
                "UPPER_TEETH_TWOSIDE", "LOWER_TEETH_TWOSIDE", "UPPER_LIP_TWOSIDE", "LOWER_LIP_TWOSIDE", "LEFT_COVER",
                "RIGHT_COVER", "UVULA_ORIGINAL", "UVULA", "UVULA_TWOSIDE", "EPIGLOTTIS_ORIGINAL", "EPIGLOTTIS",
                "EPIGLOTTIS_TWOSIDE", "RADIATION")

    def cross_profiles(self, p) -> dict:
        """3 차원 성도의 단면 윤곽 (§52.526) — 중심선 단면 i 의 점 c_i (정중 시상면 x 앞, y 위 [cm]), 법선 n_i, 좌우 표본 k 의 옆 자리
        z_k = (k + ½ − 48)·7/96 cm 에서 공기는 c_i + s·n_i (s ∈ [lower, upper]) 사이. 벽 기관 번호(`SURFACES`)는 표본마다.
        반환: center (129, 2), normal (129, 2), pos (129,) [cm], upper/lower (129, 96) [cm, nan = 없음], usurf/lsurf (129, 96) int."""
        n, m = self.N_SLICE, self.N_PROF
        tp = (ctypes.c_double * self.n_par)(*np.asarray(p, float))
        cxy, cn, pos = (ctypes.c_double * (2 * n))(), (ctypes.c_double * (2 * n))(), (ctypes.c_double * n)()
        up, lo = (ctypes.c_double * (n * m))(), (ctypes.c_double * (n * m))()
        us, ls = (ctypes.c_int * (n * m))(), (ctypes.c_int * (n * m))()
        if self.lib.vtlGetCrossProfiles(tp, cxy, cn, pos, up, lo, us, ls) != 0:
            raise RuntimeError("vtlGetCrossProfiles 실패 (해부 DLL 인가?)")
        U = np.array(up).reshape(n, m)
        L = np.array(lo).reshape(n, m)
        bad = (np.abs(U) > 1e5) | (np.abs(L) > 1e5)
        U[bad] = np.nan
        L[bad] = np.nan
        return dict(center=np.array(cxy).reshape(n, 2), normal=np.array(cn).reshape(n, 2), pos=np.array(pos),
                    upper=U, lower=L, usurf=np.array(us).reshape(n, m), lsurf=np.array(ls).reshape(n, m),
                    z=(np.arange(m) + 0.5 - m / 2) * self.PROF_LEN / m)

    def export_obj(self, p, path, both_sides: bool = True) -> None:
        """3 차원 성도 표면 (혀·치아·입술·윗벽·아랫벽·옆벽·후두개·목젖) 을 Wavefront .obj 로 (§52.526)."""
        tp = (ctypes.c_double * self.n_par)(*np.asarray(p, float))
        if self.lib.vtlExportTractObj(tp, str(path).encode(), int(both_sides)) != 0:
            raise RuntimeError("vtlExportTractObj 실패")
