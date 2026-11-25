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
4. Qwen3-ASR / Qwen3-Omni audio tower LoRA (`enable_tower_connector_lora`): the encoder runs
   `conv_out`, a LoRA-wrapped linear, on the chunk-padded rows, while the tower's LoRA mapping
   holds one row per valid position. A batch of clips whose lengths are not whole chunks crashes
   at `lora_shrink_op.py` (`assert token_lora_mapping.size(0) == M`), or applies another row's
   adapter when the counts happen to fit (vllm-project/vllm#61608). The valid rows are gathered
   before `conv_out` and the positional embedding is taken at their positions: the same values
   without a LoRA, the right ones with it (upstream's fix, open as vllm-project/vllm#63379, does
   the same on main). Applied for every toolkit; skipped once upstream has the fix.
5. General plugins (vllm/plugins/__init__.py, load_plugins_by_group): a plugin that fails to load
   is logged as one WARNING line, its exception's type and message, never an ERROR with its
   traceback (one log entry per traceback line in a JSON log pipeline: ~46 ERROR entries per start
   for NeMo's nemo_speechlm in gladia-enterprise, gladiaio/gladia-enterprise#6033); a plugin that
   VLLM_PLUGINS leaves out is named in one WARNING line. Loading is unchanged: the same plugins
   load, a failed one is still skipped. Applied for every toolkit.
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


QWEN3_AUDIO_FILE = Path("vllm") / "model_executor" / "models" / "qwen3_omni_moe_thinker.py"
QWEN3_AUDIO_PADDED = """        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        # Add positional embedding
        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        with gpu_sync_allowed():
            hidden_states = padded_embed[padded_mask_after_cnn]
            # Use tolist() for efficient batch conversion from tensor to Python.
            aftercnn_lens_list = aftercnn_lens.tolist()
"""
QWEN3_AUDIO_VALID = """        b, c, f, t = padded_embed.size()
        padded_embed = padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # gladia: conv_out (LoRA-wrapped with enable_tower_connector_lora) projects the valid
        # rows only, as the tower's LoRA mapping counts them (vllm-project/vllm#61608)
        positions = torch.arange(t, device=padded_embed.device).expand(b, t)
        with gpu_sync_allowed():
            hidden_states = self.conv_out(padded_embed[padded_mask_after_cnn])
            valid_positions = positions[padded_mask_after_cnn]
            # Use tolist() for efficient batch conversion from tensor to Python.
            aftercnn_lens_list = aftercnn_lens.tolist()
        hidden_states = hidden_states + self.positional_embedding.positional_embedding[
            valid_positions
        ].to(hidden_states.dtype)
"""
# upstream's own fix (vllm-project/vllm#63379), once merged
QWEN3_AUDIO_UPSTREAM = "# Tower LoRA mappings contain only valid rows, not chunk padding."


def qwen3_audio_lora(src: Path, nvcc: tuple[int, int]) -> str:
    del nvcc  # a correctness fix: every toolkit
    path = src / QWEN3_AUDIO_FILE
    text = path.read_text()
    if QWEN3_AUDIO_UPSTREAM in text:
        return "qwen3_audio_lora: upstream has the fix, unchanged"
    if QWEN3_AUDIO_VALID in text:
        return "qwen3_audio_lora: already applied"
    if QWEN3_AUDIO_PADDED not in text:
        raise SystemExit(
            "qwen3_audio_lora: the Qwen3 audio encoder's conv_out block changed upstream: "
            "look at it again"
        )
    path.write_text(text.replace(QWEN3_AUDIO_PADDED, QWEN3_AUDIO_VALID, 1))
    return "qwen3_audio_lora: conv_out projects the valid rows only (tower LoRA mapping)"


PLUGINS_FILE = Path("vllm") / "plugins" / "__init__.py"
PLUGINS_UPSTREAM = """    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("Loading plugin %s", plugin.name)

            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)
"""
PLUGINS_ONE_LINE = """    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("Loading plugin %s", plugin.name)

            # gladia: one WARNING line, never an ERROR traceback (one log entry per line)
            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception as e:
                logger.warning(
                    "Failed to load plugin %s (%s): %s: %s",
                    plugin.name,
                    plugin.value,
                    type(e).__name__,
                    e,
                )
        else:
            logger.warning(
                "Plugin %s (%s) skipped: not in VLLM_PLUGINS", plugin.name, plugin.value
            )
"""


def plugin_log_lines(src: Path, nvcc: tuple[int, int]) -> str:
    del nvcc  # a logging change: every toolkit
    path = src / PLUGINS_FILE
    text = path.read_text()
    if PLUGINS_ONE_LINE in text:
        return "plugin_log_lines: already applied"
    if PLUGINS_UPSTREAM not in text:
        raise SystemExit(
            "plugin_log_lines: load_plugins_by_group changed upstream: look at it again"
        )
    path.write_text(text.replace(PLUGINS_UPSTREAM, PLUGINS_ONE_LINE, 1))
    return "plugin_log_lines: a failed or skipped plugin is one WARNING line"


def main() -> int:
    src, nvcc = Path(sys.argv[1]), version(sys.argv[2])
    for change in (cooperative_topk, fa3, deepgemm, qwen3_audio_lora, plugin_log_lines):
        print(f"patch: {change(src, nvcc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
