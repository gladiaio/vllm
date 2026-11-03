#!/usr/bin/env bash
# Builds one CUDA 12 wheel of an upstream vLLM release from source, inside an
# nvidia/cuda:<toolkit>-cudnn-devel-ubuntu24.04 container (GCC 13: DeepGEMM and DeepSelect need C++20 <format>) (see plan.sh for the toolkit).
#
#   build.sh <vllm source dir> <out dir>
#
# Settings (environment, all required unless noted):
#   VLLM_VERSION           upstream version, e.g. 0.33.0 (the source dir is its tag)
#   CUDA_VARIANT           the torch CUDA build to compile against, e.g. cu126
#   REV                    Gladia's build revision (bumped when this recipe changes)
#   TORCH_CUDA_ARCH_LIST   the compute capabilities to compile, e.g. "7.5 8.0 8.6 8.9 9.0"
#   MAX_JOBS               parallel compile jobs (optional, default: nproc)
#   NVCC_THREADS           nvcc threads per job (optional, default 2)
#   CCACHE_DIR             compiler cache (optional; used when ccache is installed)
#
# The wheel's version is <VLLM_VERSION>+<CUDA_VARIANT>.gladia<REV>: a local version of the upstream
# release, so `vllm>=<VLLM_VERSION>` and advisory scanners see the upstream version.
set -euo pipefail

SRC="$(cd "${1:?vllm source dir}" && pwd)"
OUT="${2:?out dir}"
: "${VLLM_VERSION:?}" "${CUDA_VARIANT:?}" "${REV:?}" "${TORCH_CUDA_ARCH_LIST:?}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-stable}"
UV_VERSION="${UV_VERSION:-0.9.5}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VARIANT}"
LOCAL_VERSION="${VLLM_VERSION}+${CUDA_VARIANT}.gladia${REV}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

log() { printf '[build %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  log "system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # libdw-dev (DeepGEMM's JIT: elfutils/libdwfl.h), libibverbs-dev: upstream's build image has
  # elfutils-devel and rdma-core-devel (docker/Dockerfile)
  apt-get install -y -qq --no-install-recommends git curl ca-certificates ccache libdw-dev libibverbs-dev >/dev/null
fi

if ! command -v uv >/dev/null; then
  log "uv ${UV_VERSION}"
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
fi
if ! command -v cargo >/dev/null; then
  log "rust ${RUST_TOOLCHAIN} (setuptools-rust)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y -q --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

# torch: the exact version the release pins (requirements/cuda.txt), its CUDA 12 build
TORCH_VERSION="$(sed -n 's/^torch==\([0-9.]*\).*/\1/p' "$SRC/requirements/cuda.txt")"
[ -n "$TORCH_VERSION" ] || { echo "no torch== pin in requirements/cuda.txt" >&2; exit 1; }
log "vLLM ${VLLM_VERSION} -> ${LOCAL_VERSION}, torch ${TORCH_VERSION}+${CUDA_VARIANT}, archs ${TORCH_CUDA_ARCH_LIST}"

VENV=/tmp/vllm-build-venv
uv venv -q --python "$PYTHON_VERSION" "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
uv pip install -q "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
# the build requirements, minus torch (installed above from the CUDA 12 index: PyPI's is CUDA 13)
grep -v -E '^\s*torch\b' "$SRC/requirements/build/cuda.txt" > /tmp/vllm-build-req.txt
uv pip install -q -r /tmp/vllm-build-req.txt -r "$SRC/requirements/build/rust.txt"
python -c "import torch; v = torch.version.cuda; print('torch', torch.__version__, 'cuda', v); assert v.startswith('12.'), v"

export VLLM_TARGET_DEVICE=cuda
export VLLM_VERSION_OVERRIDE="$LOCAL_VERSION"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
unset VLLM_USE_PRECOMPILED
if command -v ccache >/dev/null; then
  export CCACHE_DIR="${CCACHE_DIR:-/tmp/ccache}" CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-10G}"
  export CCACHE_NOHASHDIR=1 CCACHE_BASEDIR="$SRC"
  ccache -z >/dev/null
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
python "$HERE/patch.py" "$SRC" "$NVCC_VERSION"

log "compile (MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS})"
cd "$SRC"
rm -rf build  # a stale CMake cache keeps the previous compiler (ccache keeps the objects)
git config --global --add safe.directory "$SRC" 2>/dev/null || true
# abi3 (cp38), as upstream's release wheel (docker/Dockerfile: bdist_wheel --py-limited-api=cp38)
uv build --wheel --no-build-isolation --config-setting=--build-option=--py-limited-api=cp38 \
  --out-dir "$OUT/raw" .
if command -v ccache >/dev/null; then ccache -s || true; fi

# Tag the platform by the glibc this was built on (ubuntu 24.04: 2.39), as manylinux, so resolvers
# that require a manylinux tag accept it. Its CUDA libraries come from torch's own nvidia-* wheels.
glibc="$(ldd --version 2>&1 | sed -n '1s/.* \([0-9]*\)\.\([0-9]*\)$/\1_\2/p')"  # sed reads it all: no SIGPIPE
for whl in "$OUT"/raw/*.whl; do
  (cd /tmp && uvx -q --from wheel==0.45.1 wheel tags --remove \
    --platform-tag "manylinux_${glibc}_x86_64" "$whl" >/dev/null)
done
mv "$OUT"/raw/*manylinux*.whl "$OUT"/
rm -rf "$OUT/raw"
(cd "$OUT" && sha256sum ./*.whl)
log "done"
