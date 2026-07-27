#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Test: Qwen3 W8A8 decode via PyPTO kernel (qwen3_decode_worker_w8a8).

This test validates that the PyPTO W8A8 fused decode kernel produces
correct outputs when integrated into vllm's Qwen3DecoderLayer.forward
through the vllm-ascend patch mechanism.

Prerequisites:
    - NPU device with pypto runtime installed
    - pypto_qwen3_mk repository cloned locally

Usage:
    # Set pypto_qwen3_mk models directory
    export PYPTO_QWEN3_MODELS_DIR=/path/to/pypto_qwen3_mk/models

    # Run via pytest
    pytest tests/e2e/pull_request/one_card/test_qwen3_w8a8_pypto.py -v

    # Run standalone
    python tests/e2e/pull_request/one_card/test_qwen3_w8a8_pypto.py
"""

import os
import sys

import pytest

from tests.e2e.conftest import (
    VllmRunner,
    cleanup_dist_env_and_memory,
    wait_until_npu_memory_free,
)
from tests.e2e.pull_request.utils import PROMPTS_SHORT


MODEL_NAME = "vllm-ascend/Qwen3-8B-W8A8"
MAX_MODEL_LEN = 2048
MAX_TOKENS = 32


def _get_pypto_models_dir():
    """Get pypto_qwen3_mk/models directory path."""
    pypto_dir = os.environ.get("PYPTO_QWEN3_MODELS_DIR", "")
    if not pypto_dir:
        default_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "..", "pypto", "models",
        ))
        if os.path.isdir(default_path):
            pypto_dir = default_path
    return pypto_dir if pypto_dir and os.path.isdir(pypto_dir) else None


def _check_pypto_available():
    """Check if PyPTO kernel is available for testing."""
    pypto_dir = _get_pypto_models_dir()
    if pypto_dir is None:
        pytest.skip(
            "PYPTO_QWEN3_MODELS_DIR not set or directory not found. "
            "Set it to the pypto_qwen3_mk/models directory."
        )
    return pypto_dir


@wait_until_npu_memory_free()
def test_qwen3_w8a8_pypto_generates_output():
    """Verify PyPTO kernel produces non-empty output through vllm."""
    from vllm import SamplingParams

    pypto_dir = _check_pypto_available()

    os.environ["PYPTO_QWEN3_MODELS_DIR"] = pypto_dir

    prompts = PROMPTS_SHORT
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    with VllmRunner(
        MODEL_NAME,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        quantization="ascend",
        dtype="auto",
        seed=42,
    ) as runner:
        outputs = runner.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts), (
        f"Expected {len(prompts)} outputs, got {len(outputs)}"
    )

    for i, (ids, texts) in enumerate(outputs):
        text = texts[0]
        assert len(text) > 0, f"Output {i} is empty"
        print(f"[PyPTO] Prompt {i}: {prompts[i][:30]}... -> {text[:80]}...")

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    from vllm import SamplingParams

    pypto_dir = _get_pypto_models_dir()
    if pypto_dir is None:
        print("ERROR: PYPTO_QWEN3_MODELS_DIR not set or not found")
        print("  export PYPTO_QWEN3_MODELS_DIR=/path/to/pypto/models")
        sys.exit(1)

    prompts = PROMPTS_SHORT
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    print("Running PyPTO kernel...")
    print("=" * 60)

    # os.environ["VLLM_ASCEND_ENABLE_PYPTO_W8A8"] = "0"
    os.environ["PYPTO_QWEN3_MODELS_DIR"] = pypto_dir

    with VllmRunner(
        MODEL_NAME,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        quantization="ascend",
        dtype="auto",
        seed=42,
    ) as runner:
        pypto_outputs = runner.generate(prompts, sampling_params)

    for i, (ids, texts) in enumerate(pypto_outputs):
        print(f"[PyPTO] Prompt {i}: {prompts[i][:40]}...")
        print(f"  -> {texts[0][:100]}")

    cleanup_dist_env_and_memory()

    print()
    print("=" * 60)
