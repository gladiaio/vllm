#!/usr/bin/env bash
# Plans one CUDA 12 build: prints KEY=VALUE lines (GitHub step outputs).
#
#   plan.sh <version|""> <rev> <force:true|false>
#
# version: an upstream vLLM release (default: PyPI's latest, the version pip-audit and OSV know);
# its sources are the upstream tag v<version>. The release tag is v<version>-gladia-<FLAVOR>.<rev>;
# build=false when it is already released (unless force).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/cuda12.env"

version="${1#v}"
rev="${2:?rev}"
force="${3:-false}"
repo="${GITHUB_REPOSITORY:-gladiaio/vllm}"
if [ -z "$version" ]; then
  version="$(curl -fsSL https://pypi.org/pypi/vllm/json | python3 -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])')"
fi
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$ ]] || { echo "not a release version: $version" >&2; exit 1; }

# the torch the release pins, from its own requirements
reqs="$(curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/v${version}/requirements/cuda.txt")"
torch="$(sed -n 's/^torch==\([0-9.]*\).*/\1/p' <<<"$reqs")"
[ -n "$torch" ] || { echo "v${version}: no torch== pin in requirements/cuda.txt" >&2; exit 1; }
torchvision="$(sed -n 's/^torchvision==\([0-9.]*\).*/\1/p' <<<"$reqs")"
torchaudio="$(sed -n 's/^torchaudio==\([0-9.]*\).*/\1/p' <<<"$reqs")"

variant=""
for cu in $TORCH_CUDA_VARIANTS; do
  index="$(curl -fsSL "https://download.pytorch.org/whl/${cu}/torch/")"
  if grep -q "torch-${torch}%2B${cu}-cp312-cp312-manylinux_2_28_x86_64" <<<"$index"; then
    variant="$cu"; break
  fi
done
[ -n "$variant" ] || { echo "torch ${torch} has no CUDA 12 build (${TORCH_CUDA_VARIANTS}) on PyTorch's index" >&2; exit 1; }
toolkit_var="TOOLKIT_${variant}"; archs_var="ARCHS_${variant}"

tag="v${version}-gladia-${FLAVOR}.${rev}"
build=true
if gh release view "$tag" -R "$repo" >/dev/null 2>&1 && [ "$force" != "true" ]; then
  build=false
  echo "$tag is released: nothing to do" >&2
fi
echo "version=${version}"
echo "tag=${tag}"
echo "build=${build}"
echo "torch=${torch}"
echo "torchvision=${torchvision}"
echo "torchaudio=${torchaudio}"
echo "cuda_variant=${variant}"
echo "toolkit=${!toolkit_var}"
echo "archs=${!archs_var}"
echo "local_version=${version}+${variant}.gladia${rev}"
