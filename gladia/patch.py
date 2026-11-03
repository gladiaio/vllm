"""Gladia's changes to an upstream vLLM tag before a CUDA 12 build (build.sh runs it in the sources).

    python3 patch.py <vllm source dir> <nvcc version, e.g. 12.6>

Each change says why, applies only when it is needed (by the toolkit), and fails the build when
its upstream text is gone and the toolkit still needs it: a change upstream made must be looked
at, never skipped silently.

1. cooperative_topk (csrc/cooperative_topk.cuh, the DSA indexer's top-k of DeepSeek-V3.2 / V4 and
   Qwen4-Exp) calls the cuda::ptx overloads with `sem_relaxed` (mbarrier_try_wait_parity,
   mbarrier_arrive_expect_tx) and cp_async_bulk(space_shared, ...), which the CCCL of CUDA < 12.8
   does not have, while CMakeLists.txt builds it from CUDA 12.0. Below 12.8 it is left out, as
   upstream does when no target architecture fits it (VLLM_ENABLE_COOPERATIVE_TOPK unset: the op
   is not registered). None of Gladia's engines runs a DSA indexer model.
2. FA3 (vllm-flash-attn's Hopper kernels, `_vllm_fa3_C`): nvcc 12.6.3 segfaults on
   hopper/instantiations/flash_fwd_hdim128_bf16_paged_split_softcap_sm90.cu (deterministic, at the
   vllm-flash-attn commit v0.33.0 pins); upstream only builds it with CUDA 12.9 / 13 now. Below 12.8
   setup.py no longer asks for it (it already leaves it out below 12.3): vLLM's backend selection
   (`is_fa_version_supported(3)`) and solaria-4's then run FA2 on Hopper, from its compute_80 PTX.
3. DeepGEMM (the FP8 block GEMMs of DeepSeek models) static_asserts CUDA_VERSION and
   CUDART_VERSION >= 12.9 at the commit v0.33.0 vendors, while cmake/external_projects/deepgemm.cmake
   still enables its sm90 build from 12.3. Below 12.9 it gets no architecture, so the cmake file
   creates its empty `_deep_gemm_C` target, as it does on an unsupported GPU (vLLM's has_deep_gemm()
   is then false).
"""

from __future__ import annotations

import sys
from pathlib import Path

COOPERATIVE_GATE = """     DEFINED CMAKE_CUDA_COMPILER_VERSION AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)

    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      cuda_archs_loose_intersection(COOPERATIVE_TOPK_ARCHS"""


def version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def cooperative_topk(src: Path, nvcc: tuple[int, int]) -> str:
    if nvcc >= (12, 8):
        return "cooperative_topk: CUDA >= 12.8, unchanged"
    cmake = src / "CMakeLists.txt"
    text = cmake.read_text()
    patched = COOPERATIVE_GATE.replace("EQUAL 12.0)", "EQUAL 12.8)")
    if patched in text:
        return "cooperative_topk: already left out below CUDA 12.8"
    if COOPERATIVE_GATE not in text:
        raise SystemExit("cooperative_topk: CMakeLists.txt's gate changed upstream: look at it again")
    cmake.write_text(text.replace(COOPERATIVE_GATE, patched, 1))
    return "cooperative_topk: left out below CUDA 12.8 (CMakeLists.txt gate 12.0 -> 12.8)"


FA3_GATE = """    if USE_PRECOMPILED_EXTENSIONS or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.3")
    ):"""


def fa3(src: Path, nvcc: tuple[int, int]) -> str:
    if nvcc >= (12, 8):
        return "fa3: CUDA >= 12.8, unchanged"
    setup = src / "setup.py"
    text = setup.read_text()
    patched = FA3_GATE.replace('Version("12.3")', 'Version("12.8")')
    if patched in text:
        return "fa3: already left out below CUDA 12.8"
    if FA3_GATE not in text:
        raise SystemExit("fa3: setup.py's _vllm_fa3_C gate changed upstream: look at it again")
    setup.write_text(text.replace(FA3_GATE, patched, 1))
    return "fa3: left out below CUDA 12.8 (setup.py gate 12.3 -> 12.8; Hopper runs FA2)"


DEEPGEMM_GATE = """if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
  list(APPEND DEEPGEMM_SUPPORT_ARCHS "9.0a")"""


def deepgemm(src: Path, nvcc: tuple[int, int]) -> str:
    if nvcc >= (12, 9):
        return "deepgemm: CUDA >= 12.9, unchanged"
    cmake = src / "cmake" / "external_projects" / "deepgemm.cmake"
    text = cmake.read_text()
    patched = DEEPGEMM_GATE.replace("EQUAL 12.3)", "EQUAL 12.9)")
    if patched in text:
        return "deepgemm: already left out below CUDA 12.9"
    if DEEPGEMM_GATE not in text:
        raise SystemExit("deepgemm: deepgemm.cmake's sm90 gate changed upstream: look at it again")
    cmake.write_text(text.replace(DEEPGEMM_GATE, patched, 1))
    return "deepgemm: left out below CUDA 12.9 (deepgemm.cmake sm90 gate 12.3 -> 12.9)"


def main() -> int:
    src, nvcc = Path(sys.argv[1]), version(sys.argv[2])
    for change in (cooperative_topk, fa3, deepgemm):
        print(f"patch: {change(src, nvcc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
