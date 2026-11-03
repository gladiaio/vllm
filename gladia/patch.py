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


def main() -> int:
    src, nvcc = Path(sys.argv[1]), version(sys.argv[2])
    for change in (cooperative_topk,):
        print(f"patch: {change(src, nvcc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
