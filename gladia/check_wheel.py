"""Checks a CUDA 12 vLLM wheel built by build.sh before it is released.

    python check_wheel.py <wheel> --version 0.33.0 --cuda-variant cu126 --rev 1 \
        --archs "7.5 8.0 8.6 8.9 9.0" [--cuobjdump /usr/local/cuda/bin/cuobjdump]

Fails (exit 1, one line per problem) unless:
  - the wheel's file name and METADATA say <version>+<cuda variant>.gladia<rev>;
  - METADATA requires torch at exactly the version upstream pins (requirements/cuda.txt), and no
    CUDA 13 package is left (no `[cu13]` extra, no `*-cu13` distribution): setup.py rewrites them
    for a CUDA 12 torch, a wheel still asking for them would pull CUDA 13 into a CUDA 12 venv;
  - every compiled CUDA extension (`*.so` with device code) carries SASS for exactly the requested
    compute capabilities (a `+PTX` / `a` suffix aside), none for an architecture outside the list:
    the wheel runs on the fleet's GPUs and nothing else was compiled for nothing;
  - `vllm/_version.py` holds the same version.
Standard library only: it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import email.parser
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

CU13 = re.compile(r"\[cu13\]|-cu13\b|cu13[=<>~ ;]", re.IGNORECASE)
SM = re.compile(r"\bsm_(\d+)[af]?\b")


def metadata(whl: zipfile.ZipFile) -> email.message.Message:
    name = next(n for n in whl.namelist() if n.endswith(".dist-info/METADATA"))
    return email.parser.Parser().parsestr(whl.read(name).decode())


def sm_of(arch: str) -> int:
    """'8.6' -> 86, '9.0a' -> 90, '8.0+PTX' -> 80."""
    major, minor = arch.split("+")[0].rstrip("af").split(".")
    return int(major) * 10 + int(minor)


def device_archs(so: Path, cuobjdump: str) -> set[int]:
    run = subprocess.run([cuobjdump, "--list-elf", str(so)], capture_output=True, text=True, check=False)
    if run.returncode != 0:
        return set()  # host code only: nothing to check
    return {int(m) for m in SM.findall(run.stdout)}


def check(args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    local = f"{args.version}+{args.cuda_variant}.gladia{args.rev}"
    wheel = Path(args.wheel)
    if not wheel.name.startswith(f"vllm-{local}-"):
        problems.append(f"file name {wheel.name} is not vllm-{local}-*")
    with zipfile.ZipFile(wheel) as whl:
        meta = metadata(whl)
        if meta["Version"] != local:
            problems.append(f"METADATA Version {meta['Version']} != {local}")
        requires = meta.get_all("Requires-Dist") or []
        torch = [r for r in requires if re.match(r"torch\s*[=<>!~]", r)]
        if torch != [f"torch=={args.torch}"]:
            problems.append(f"torch requirement {torch} != ['torch=={args.torch}']")
        problems += [f"CUDA 13 requirement left: {r}" for r in requires if CU13.search(r)]
        version_py = whl.read("vllm/_version.py").decode()
        if f"'{local}'" not in version_py and f'"{local}"' not in version_py:
            problems.append(f"vllm/_version.py does not hold {local}")
        wanted = {sm_of(a) for a in args.archs.split()}
        with tempfile.TemporaryDirectory() as tmp:
            sos = [n for n in whl.namelist() if n.endswith(".so")]
            seen: set[int] = set()
            for name in sos:
                path = Path(whl.extract(name, tmp))
                found = device_archs(path, args.cuobjdump)
                if found - wanted:
                    problems.append(f"{name}: SASS for {sorted(found - wanted)} outside {sorted(wanted)}")
                seen |= found
            if not seen:
                problems.append("no CUDA device code in any extension (cuobjdump found none)")
            elif wanted - seen:
                problems.append(f"no extension has SASS for {sorted(wanted - seen)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wheel")
    parser.add_argument("--version", required=True)
    parser.add_argument("--cuda-variant", required=True)
    parser.add_argument("--rev", required=True)
    parser.add_argument("--torch", required=True, help="the torch version upstream pins")
    parser.add_argument("--archs", required=True, help='TORCH_CUDA_ARCH_LIST, e.g. "7.5 8.0 8.6 8.9 9.0"')
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    problems = check(parser.parse_args())
    for p in problems:
        print(f"FAIL {p}")
    if not problems:
        print("wheel OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
