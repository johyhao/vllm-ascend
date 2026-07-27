import os
import sys
from typing import Dict

import math
import torch
import numpy as np

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

logger = init_logger(__name__)

_pypto_loaded = False
_AttentionConfig = None
_AttentionTileConfig = None
_Qwen3DecodeConfig = None
_qwen3_decode_worker_w8a8 = None


def _load_pypto_kernel():
    global _pypto_loaded
    global _AttentionConfig, _AttentionTileConfig, _Qwen3DecodeConfig
    global _qwen3_decode_worker_w8a8

    if _pypto_loaded:
        return _qwen3_decode_worker_w8a8 is not None

    pypto_models_dir = os.environ.get("PYPTO_QWEN3_MODELS_DIR", "")
    if not pypto_models_dir:
        pypto_models_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "pypto", "models",
        ))

    pypto_qwen3_32b_dir = os.path.join(pypto_models_dir, "qwen3_32b")
    pypto_parent_dir = os.path.dirname(pypto_models_dir)

    for path in [pypto_parent_dir, pypto_qwen3_32b_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from models.qwen3_32b.w8a8_dynamic.qwen3_w8a8_dynamic_kernel import (
            AttentionConfig as _AC,
            AttentionTileConfig as _ATC,
            Qwen3DecodeConfig as _QDC,
            TensorConfig as _TC,
            qwen3_decode_worker_w8a8 as _worker,
        )
        _AttentionConfig = _AC
        _AttentionTileConfig = _ATC
        _Qwen3DecodeConfig = _QDC
        _TensorConfig = _TC
        _qwen3_decode_worker_w8a8 = _worker
        logger.info("PyPTO W8A8 kernel loaded successfully from %s", pypto_models_dir)
    except ImportError as e:
        logger.warning("Failed to load PyPTO W8A8 kernel: %s", e)

    _pypto_loaded = True
    return _qwen3_decode_worker_w8a8 is not None


def _is_pypto_enabled():
    if os.environ.get("VLLM_ASCEND_ENABLE_PYPTO_W8A8", "0") != "1":
        return False
    return _load_pypto_kernel()


def golden_per_channel_quantize(x):
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1.0 / scale_quant).reshape(-1).to(torch.float32)  # float32 for npu_quant_matmul
    return y_int8, scale_dequant


def generate_tensors_from_config(self_attn, device) -> Dict[str, torch.Tensor]:
    b, s1 = 16, 1
    bs = b * s1
    hidden_size = self_attn.hidden_size
    intermediate_size = 6400
    n1, n2, d = 16, 2, self_attn.head_dim

    hidden_states = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=device)

    input_layernorm_weight = torch.rand(1, hidden_size, dtype=torch.bfloat16, device=device)
    output_layernorm_weight = torch.rand(1, hidden_size, dtype=torch.bfloat16, device=device)

    total_head_size = n1 * d + 2 * n2 * d
    # W8A8 Dynamic: quantize QKV weight per-channel, per-layer
    qkv_weight_scale_list = []
    qkv_proj_weight_list = []
    qkv_proj_weight_bf16_layer = torch.rand(hidden_size, total_head_size, dtype=torch.bfloat16, device=device)
    qkv_proj_weight_layer, qkv_weight_scale_layer = golden_per_channel_quantize(qkv_proj_weight_bf16_layer)
    qkv_proj_weight_list.append(qkv_proj_weight_layer)
    qkv_weight_scale_list.append(qkv_weight_scale_layer)
    qkv_proj_weight = torch.cat(qkv_proj_weight_list, dim=0)
    qkv_weight_scale = torch.stack(qkv_weight_scale_list, dim=0).to(torch.bfloat16)
    q_size = n1 * d

    # W8A8 Dynamic: quantize O proj weight per-channel, per-layer
    o_proj_weight_scale_list = []
    o_proj_weight_list = []
    o_proj_weight_bf16_layer = torch.rand(q_size, hidden_size, dtype=torch.bfloat16, device=device)
    o_proj_weight_layer, o_proj_weight_scale_layer = golden_per_channel_quantize(o_proj_weight_bf16_layer)
    o_proj_weight_list.append(o_proj_weight_layer)
    o_proj_weight_scale_list.append(o_proj_weight_scale_layer)
    o_proj_weight = torch.cat(o_proj_weight_list, dim=0)
    o_proj_weight_scale = torch.stack(o_proj_weight_scale_list, dim=0).to(torch.bfloat16)

    q_norm_weight = torch.rand(1, 1, d, dtype=torch.bfloat16, device=device)
    q_norm_bias = torch.rand(1, 1, d, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.rand(1, 1, d, dtype=torch.bfloat16, device=device)
    k_norm_bias = torch.rand(1, 1, d, dtype=torch.bfloat16, device=device)

    # W8A8 Dynamic: quantize Gate-Up weight per-channel, per-layer
    w13_scale_list = []
    w13_list = []
    w13_bf16_layer = torch.rand(hidden_size, intermediate_size * 2, dtype=torch.bfloat16, device=device)
    w13_layer, w13_scale_layer = golden_per_channel_quantize(w13_bf16_layer)
    w13_list.append(w13_layer)
    w13_scale_list.append(w13_scale_layer)
    w13 = torch.cat(w13_list, dim=0)
    w13_scale = torch.stack(w13_scale_list, dim=0).to(torch.bfloat16)
    w2 = torch.rand(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)

    return {
        "hidden_states": hidden_states,
        "residual": residual,
        "qkv_proj_weight": qkv_proj_weight,
        "qkv_weight_scale": qkv_weight_scale,
        "o_proj_weight": o_proj_weight,
        "o_proj_weight_scale": o_proj_weight_scale,
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "input_layernorm_weight": input_layernorm_weight,
        "output_layernorm_weight": output_layernorm_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "hidden_size": hidden_size,
        "total_head_size": total_head_size,
        "intermediate_size": intermediate_size,
    }


def _extract_layer_weights(hidden_size, self_attn, mlp, input_layernorm, post_attention_layernorm, device):
    if getattr(_EXTRA_CTX, "in_profile_run", False):
        return generate_tensors_from_config(self_attn, device)

    head_dim = self_attn.head_dim
    num_heads = self_attn.num_heads
    num_kv_heads = self_attn.num_kv_heads
    q_size = self_attn.q_size
    kv_size = self_attn.kv_size
    total_head_size = q_size + 2 * kv_size
    intermediate_size = mlp.gate_up_proj.output_size_per_partition // 2

    qkv_weights = []
    qkv_scales = []
    o_proj_weights = []
    o_proj_scales = []
    w13_weights = []
    w13_scales = []
    w2_weights = []
    w2_scales = []
    input_ln_weights = []
    output_ln_weights = []
    q_norm_weights = []
    k_norm_weights = []

    qkv_w = self_attn.qkv_proj.weight.data
    qkv_weights.append(qkv_w)

    qkv_scale = getattr(self_attn.qkv_proj, "weight_scale", None)
    if qkv_scale is not None:
        qkv_scales.append(qkv_scale.data.flatten())
    else:
        qkv_scales.append(torch.ones(total_head_size, dtype=torch.bfloat16, device=qkv_w.device))

    o_w = self_attn.o_proj.weight.data
    o_proj_weights.append(o_w)
    o_scale = getattr(self_attn.o_proj, "weight_scale", None)
    if o_scale is not None:
        o_proj_scales.append(o_scale.data.flatten())
    else:
        o_proj_scales.append(torch.ones(hidden_size, dtype=torch.bfloat16, device=o_w.device))

    gate_up_w = mlp.gate_up_proj.weight.data
    w13_weights.append(gate_up_w)
    gate_up_scale = getattr(mlp.gate_up_proj, "weight_scale", None)
    if gate_up_scale is not None:
        w13_scales.append(gate_up_scale.data.flatten())
    else:
        w13_scales.append(torch.ones(intermediate_size * 2, dtype=torch.bfloat16, device=gate_up_w.device))

    down_w = mlp.down_proj.weight.data
    w2_weights.append(down_w)
    down_scale = getattr(mlp.down_proj, "weight_scale", None)
    if down_scale is not None:
        w2_scales.append(down_scale.data.flatten())
    else:
        w2_scales.append(torch.ones(hidden_size, dtype=torch.bfloat16, device=down_w.device))

    input_ln_weights.append(input_layernorm.weight.data)
    output_ln_weights.append(post_attention_layernorm.weight.data)

    q_norm_w = self_attn.q_norm.weight.data
    k_norm_w = self_attn.k_norm.weight.data
    q_norm_weights.append(q_norm_w.unsqueeze(0).unsqueeze(0))
    k_norm_weights.append(k_norm_w.unsqueeze(0).unsqueeze(0))

    qkv_proj_weight = torch.cat(qkv_weights, dim=0)
    qkv_weight_scale = torch.stack(qkv_scales, dim=0).to(torch.bfloat16)
    o_proj_weight = torch.cat(o_proj_weights, dim=0)
    o_proj_weight_scale = torch.stack(o_proj_scales, dim=0).to(torch.bfloat16)
    w13 = torch.cat(w13_weights, dim=0).T.contiguous()
    w13_scale = torch.stack(w13_scales, dim=0).to(torch.bfloat16)
    w2 = torch.cat(w2_weights, dim=0).T.contiguous()
    w2_scale = torch.stack(w2_scales, dim=0).to(torch.bfloat16)
    input_layernorm_weight = torch.stack(input_ln_weights, dim=0)
    output_layernorm_weight = torch.stack(output_ln_weights, dim=0)
    q_norm_weight = torch.cat(q_norm_weights, dim=0)
    k_norm_weight = torch.cat(k_norm_weights, dim=0)

    return {
        "qkv_proj_weight": qkv_proj_weight,
        "qkv_weight_scale": qkv_weight_scale,
        "o_proj_weight": o_proj_weight,
        "o_proj_weight_scale": o_proj_weight_scale,
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "input_layernorm_weight": input_layernorm_weight,
        "output_layernorm_weight": output_layernorm_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "q_size": q_size,
        "kv_size": kv_size,
        "total_head_size": total_head_size,
        "intermediate_size": intermediate_size,
    }


def _extract_kv_cache_and_metadata(self_attn, hidden_states, in_profile_run):
    if in_profile_run:
        return get_kv_cache_config(self_attn, hidden_states)

    key_cache = None
    value_cache = None
    block_tables = None
    slot_mapping = None
    actual_seq_lens = None

    # Attention 实例
    kv_cache = self_attn.attn.kv_cache
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]

    forward_context = get_forward_context()
    attn_metadata_dict = forward_context.attn_metadata

    layer_name = self_attn.attn.layer_name if hasattr(self_attn.attn, "layer_name") else None

    if layer_name is not None and attn_metadata_dict is not None and layer_name in attn_metadata_dict:
        meta = attn_metadata_dict[layer_name]
        block_tables = getattr(meta, "block_tables", None)
        slot_mapping = getattr(meta, "slot_mapping", None)
        actual_seq_lens = getattr(meta, "seq_lens", None)

    return key_cache, value_cache, block_tables, slot_mapping, actual_seq_lens


def _compute_cos_sin(self_attn, hidden_states, device):
    bs = hidden_states.shape[0]
    if getattr(_EXTRA_CTX, "in_profile_run", False):
        bs = 16
    head_dim = self_attn.head_dim
    half_rotary_dim = (head_dim // 2) // 2

    cos = torch.zeros(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=device)
    sin = torch.zeros(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=device)

    return cos, sin


def _build_tile_config(world_size, self_attn):
    s2 = self_attn.hidden_size
    n1 = self_attn.num_heads // world_size
    n2 = self_attn.num_kv_heads // world_size
    cube_tile = 128
    vector_tile = 128
    s2_tile = min(512, s2)
    g = n1 // n2 if n2 > 0 else 1
    tile_cfg = _AttentionTileConfig(
        g_tile=g,
        s2_tile=s2_tile,
        c1_tile_shape=[[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        v1_tile_shape=[vector_tile, s2_tile],
        c2_tile_shape=[[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        v2_tile_shape=[vector_tile, vector_tile],
    )
    return tile_cfg


def gen_block_table(actual_seq_len):
    block_num_per_batch = []
    block_num = 0

    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq.item() / 128))
        block_num += math.ceil(actual_seq.item() / 128)

    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    block_table = torch.full([16, 4], -1, dtype=torch.int32, device=actual_seq_len.device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def get_kv_cache_config(self_attn, hidden_states):
    device = hidden_states.device

    num_heads = self_attn.num_kv_heads
    head_dim = self_attn.head_dim

    kv_cache_shape = [64, 128, num_heads, head_dim]
    key_cache = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device=device)

    actual_seq_values = [512] * 16
    actual_seq_lens = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    slot_mapping = torch.randperm(8192, dtype=torch.int32, device=device)[:16]

    block_tables = gen_block_table(actual_seq_lens)
    return key_cache, value_cache, block_tables, slot_mapping, actual_seq_lens


_original_forward = Qwen3DecoderLayer.forward


@torch.compiler.disable
def _qwen3_decoder_layer_forward_pypto(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_pypto_enabled():
        return _original_forward(self, positions, hidden_states, residual)

    in_profile_run = False
    if getattr(_EXTRA_CTX, "in_profile_run", False):
        in_profile_run = True

    self_attn = self.self_attn
    world_size = get_tensor_model_parallel_world_size()
    tile_cfg = _build_tile_config(world_size, self.self_attn)

    head_dim = self_attn.head_dim
    softmax_scale = head_dim ** -0.5

    key_cache, value_cache, block_tables, slot_mapping, actual_seq_lens = \
        _extract_kv_cache_and_metadata(self_attn, hidden_states, in_profile_run)

    device = hidden_states.device
    cos, sin = _compute_cos_sin(self.self_attn, hidden_states, device)

    group_name = get_tp_group().device_group._get_backend(device).get_hccl_comm_name(get_tensor_model_parallel_rank())

    params = _extract_layer_weights(self.hidden_size, self.self_attn, self.mlp, self.input_layernorm,
                                    self.post_attention_layernorm, device)

    if residual is None:
        residual = torch.rand(hidden_states.shape, dtype=torch.bfloat16, device=device)

    out_torch, residual_out, _ = _qwen3_decode_worker_w8a8(
        layer_num=1,
        hidden_states=params["hidden_states"] if in_profile_run else hidden_states,
        residual=params["hidden_states"] if in_profile_run else residual,
        input_layernorm_weight=params["input_layernorm_weight"],
        output_layernorm_weight=params["output_layernorm_weight"],
        qkv_proj_weight=params["qkv_proj_weight"],
        qkv_weight_scale=params["qkv_weight_scale"],
        o_proj_weight=params["o_proj_weight"],
        o_proj_weight_scale=params["o_proj_weight_scale"],
        q_norm_weight=params["q_norm_weight"],
        k_norm_weight=params["k_norm_weight"],
        w13=params["w13"],
        w13_scale=params["w13_scale"],
        w2=params["w2"],
        cos=cos,
        sin=sin,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        actual_seq_lens=actual_seq_lens,
        slot_mapping=slot_mapping,
        eps=1e-5,
        enable_residual=True,
        num_decode_tokens=0,
        softmax_scale=softmax_scale,
        tile_cfg=tile_cfg,
        group_name=group_name,
        world_size=world_size,
    )
    return out_torch, residual_out


Qwen3DecoderLayer.forward = _qwen3_decoder_layer_forward_pypto
logger.info("PyPTO W8A8 patch applied to Qwen3DecoderLayer.forward "
            "(set VLLM_ASCEND_ENABLE_PYPTO_W8A8=1 to activate)")
