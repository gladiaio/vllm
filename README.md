# gladiaio/vllm

Gladia's CUDA 12 build of [vLLM](https://github.com/vllm-project/vllm). From 0.33.0 upstream publishes
CUDA 13 wheels only (vllm-project/vllm#62575), and the torch that 0.33.0 pins (2.14.0) has no cu128 or
cu129 build. Gladia's `-cu128` inference images still need vLLM: they serve hosts whose NVIDIA driver is
older than 580 (525 or later). This fork builds each upstream release from source for CUDA 12. Those
images then run a fixed upstream version, and pip-audit / OSV see that version without any exception.

- `main` mirrors `vllm-project/vllm` `main`, synced every day.
- `gladia` (default branch) holds the build: `gladia/build.sh` (the build itself, run in an
  `nvidia/cuda:<toolkit>-cudnn-devel-ubuntu24.04` container), `gladia/plan.sh` (which release, torch and
  CUDA), `gladia/cuda12.env` (settings: torch's CUDA 12 builds by preference, the toolkit of each, the
  compute capabilities), `gladia/check_wheel.py` (the wheel's checks) and `.github/workflows/release.yml`.
- Every day at 00:11 UTC the workflow takes PyPI's latest vLLM release and finds the torch it pins
  (`requirements/cuda.txt`). It picks the first CUDA 12 build of that torch on PyTorch's index, by
  preference cu129, cu128, cu126 (0.33.0: torch 2.14.0, cu126 only), and compiles the upstream tag with the
  same CUDA toolkit. The arch list is limited to Gladia's GPUs: 7.5 8.0 8.6 8.9 9.0, plus 10.0 and 12.0
  when the toolkit is 12.8 or later. The workflow then checks the wheel (`check_wheel.py`: version,
  metadata, no CUDA 13 requirement, the SASS of every extension; then install and import). It fails if
  Trivy finds a fixable HIGH or CRITICAL vulnerability, or if pip-audit reports the upstream version. Last,
  it publishes the GitHub release `v<version>-gladia-cu128.<REV>` with the wheel, `SHA256SUMS` and
  `build.json` (the version, torch, torchvision, torchaudio and CUDA variant it was built against). The
  package version is `<version>+<cuda variant>.gladia<REV>`, for example `0.33.0+cu126.gladia1`. It is a
  local version of the upstream release, so `vllm>=0.33.0` and the advisory scanners see 0.33.0.

**Names.** The tag says the flavor it serves (`-gladia-cu128`: gladia-enterprise's `-cu128` images,
NVIDIA driver >= 525). The package version says the CUDA it is compiled for (`+cu126`): torch 2.14.0, the
exact torch that vLLM 0.33 pins, has no cu128 or cu129 build, so vLLM is compiled against torch's cu126 build
with the CUDA 12.6 toolkit, and both share one CUDA runtime. A cu126 wheel runs on any CUDA 12 driver
(>= 525), so it covers the whole `-cu128` flavor. When a later torch ships cu128 or cu129 again, `plan.sh`
picks that build, and the version follows (`+cu128` / `+cu129`), with Blackwell in the arch list.

**Platform.** The wheel is built in `nvidia/cuda:<toolkit>-cudnn-devel-ubuntu24.04` (GCC 13 is required)
and tagged `manylinux_2_39` (glibc 2.39, Ubuntu 24.04), the glibc of the images it goes into (every
gladia-enterprise CUDA base is ubuntu24.04). It is `cp38-abi3`, as upstream's wheel is.

**What differs from upstream's source** (`gladia/patch.py`, only below the toolkit each needs; each change
fails the build if upstream changes the text it edits):

1. `cooperative_topk` (the DSA indexer's top-k: DeepSeek-V3.2 / V4, Qwen4-Exp) is left out below CUDA 12.8.
   Its `cuda::ptx` calls need the CCCL of CUDA 12.8.
2. FA3 (vllm-flash-attn's Hopper kernels) is left out below CUDA 12.8. nvcc 12.6.3 segfaults on one of
   its instantiations. Hopper runs FA2 instead, as vLLM falls back to it.
3. DeepGEMM is left out below CUDA 12.9. It static-asserts CUDA 12.9.

Install it on the torch it was built against (pin the sha256 from `SHA256SUMS`):

    uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0
    uv pip install "vllm @ https://github.com/gladiaio/vllm/releases/download/v0.33.0-gladia-cu128.1/vllm-0.33.0%2Bcu126.gladia1-cp38-abi3-manylinux_2_39_x86_64.whl#sha256=485bb50872a12453ab4e45206ff9c9f8ee522a66faeb6f5c057e620deb60b35a"

Rebuild: bump `REV` in the workflow when `gladia/` changes, then run it
(`gh workflow run release.yml -R gladiaio/vllm`, or `-f version=0.33.1`, or `-f force=true` to replace
an existing release). gladia-enterprise picks up a new release through a pull request that bumps its pins
(`.github/scripts/vllm_cuda12_bump.py`, `vllm-cuda12-bump.yml`), which the GPU sweep then smokes.
How it fits together, and when to drop this fork: gladia-enterprise's `docs/internal/vllm-cuda12-wheels.md`.

The 0.32.0 backport (`v0.32.0-gladia.cve1`, branch `v0.32.0-gladia-cve`) is superseded by this build.
