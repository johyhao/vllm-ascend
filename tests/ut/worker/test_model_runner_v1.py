import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestContiguousKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = False
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        runner.shared_kv_cache_layers = {}
        return runner

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_can_use_contiguous_kv_cache_homogeneous(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["model.layers.0.attn"]),
                KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["model.layers.1.attn"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"], kv_cache_spec=kv_cache_spec
                )
            ],
        )

        result = runner._can_use_contiguous_kv_cache(kv_cache_config)
        self.assertTrue(result)

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_can_use_contiguous_kv_cache_disabled_by_sparse(self):
        runner = self._build_runner()
        runner.use_sparse = True
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=kv_cache_spec)],
        )

        result = runner._can_use_contiguous_kv_cache(kv_cache_config)
        self.assertFalse(result)

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_allocate_contiguous_kv_cache_tensors(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        page_size = kv_cache_spec.page_size_bytes
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(size=page_size * 2, shared_by=["model.layers.0.attn"]),
                KVCacheTensor(size=page_size * 2, shared_by=["model.layers.1.attn"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"], kv_cache_spec=kv_cache_spec
                )
            ],
        )

        kv_cache_raw_tensors, global_tensor = runner._allocate_contiguous_kv_cache_tensors(kv_cache_config)

        total_expected_size = page_size * 2 * 2
        self.assertEqual(global_tensor.numel(), total_expected_size)
        self.assertTrue(global_tensor.is_contiguous())

        self.assertIn("model.layers.0.attn", kv_cache_raw_tensors)
        self.assertIn("model.layers.1.attn", kv_cache_raw_tensors)

        k0, v0 = kv_cache_raw_tensors["model.layers.0.attn"]
        k1, v1 = kv_cache_raw_tensors["model.layers.1.attn"]

        self.assertEqual(k0.numel(), page_size)
        self.assertEqual(v0.numel(), page_size)
        self.assertEqual(k1.numel(), page_size)
        self.assertEqual(v1.numel(), page_size)

        self.assertEqual(k0.data_ptr(), global_tensor.data_ptr())
        self.assertEqual(v0.data_ptr(), global_tensor.data_ptr() + page_size)
        self.assertEqual(k1.data_ptr(), global_tensor.data_ptr() + page_size * 2)
        self.assertEqual(v1.data_ptr(), global_tensor.data_ptr() + page_size * 3)

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_get_all_layers_kv_cache(self):
        runner = self._build_runner()
        self.assertIsNone(runner.get_all_layers_kv_cache())

        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        page_size = kv_cache_spec.page_size_bytes
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=page_size * 2, shared_by=["attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=kv_cache_spec)],
        )

        runner._allocate_contiguous_kv_cache_tensors(kv_cache_config)
        global_tensor = runner.get_all_layers_kv_cache()
        self.assertIsNotNone(global_tensor)
        self.assertTrue(global_tensor.is_contiguous())

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_can_use_contiguous_kv_cache_with_kv_transfer_config(self):
        runner = self._build_runner()
        runner.vllm_config.kv_transfer_config = MagicMock()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=kv_cache_spec)],
        )

        result = runner._can_use_contiguous_kv_cache(kv_cache_config)
        self.assertTrue(result)

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_can_use_contiguous_kv_cache_with_shared_layers(self):
        runner = self._build_runner()
        runner.shared_kv_cache_layers = {"model.layers.1.attn": "model.layers.0.attn"}
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["model.layers.0.attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["model.layers.0.attn", "model.layers.1.attn"], kv_cache_spec=kv_cache_spec)],
        )

        result = runner._can_use_contiguous_kv_cache(kv_cache_config)
        self.assertTrue(result)

    @patch("vllm_ascend.envs.VLLM_ASCEND_CONTIGUOUS_KV_CACHE", True)
    def test_allocate_contiguous_kv_cache_with_alignment(self):
        runner = self._build_runner()
        runner.vllm_config.kv_transfer_config = MagicMock()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        page_size = kv_cache_spec.page_size_bytes
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(size=page_size * 2, shared_by=["model.layers.0.attn"]),
                KVCacheTensor(size=page_size * 2, shared_by=["model.layers.1.attn"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"], kv_cache_spec=kv_cache_spec
                )
            ],
        )

        kv_cache_raw_tensors, global_tensor = runner._allocate_contiguous_kv_cache_tensors(kv_cache_config)

        self.assertIn("model.layers.0.attn", kv_cache_raw_tensors)
        self.assertIn("model.layers.1.attn", kv_cache_raw_tensors)

        alignment = 2 * 1024 * 1024
        for layer_name, (k_tensor, v_tensor) in kv_cache_raw_tensors.items():
            self.assertEqual(k_tensor.numel(), page_size)
            self.assertEqual(v_tensor.numel(), page_size)
            self.assertEqual(k_tensor.data_ptr() % alignment, 0,
                           f"K tensor for {layer_name} not 2MB aligned")
            self.assertEqual(v_tensor.data_ptr() % alignment, 0,
                           f"V tensor for {layer_name} not 2MB aligned")


if __name__ == "__main__":
    unittest.main()