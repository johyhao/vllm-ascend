import os
import sys
from typing import Dict, List

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

logger = init_logger(__name__)

_pypto_loaded = False
_AttentionConfig = None
_AttentionTileConfig = None
_Qwen3DecodeConfig = None
_qwen3_decode_worker_w8a8 = None


def _load_pypto_kernel():
    global _pypto_loaded, _AttentionConfig, _AttentionTileConfig
    global _Qwen3DecodeConfig, _qwen3_decode_worker_w8a8

    if _pypto_loaded:
        return _qwen3_decode_worker_w8a8 is not None

    pypto_models_dir = os.environ.get("PYPTO_QWEN3_MODELS_DIR", "")
    if not pypto_models_dir:
        pypto_models_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..",
                "pypto", "models",
            )
        )

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
            qwen3_decode_worker_w8a8 as _worker,
        )
        _AttentionConfig = _AC
        _AttentionTileConfig = _ATC
        _Qwen3DecodeConfig = _QDC
        _qwen3_decode_worker_w8a8 = _worker
        logger.info("PyPTO W8A8 kernel loaded from %s", pypto_models_dir)
    except ImportError as e:
        logger.warning("Failed to load PyPTO W8A8 kernel: %s", e)

    _pypto_loaded = True
    return _qwen3_decode_worker_w8a8 is not None


def _is_pypto_enabled():
    if os.environ.get("VLLM_ASCEND_ENABLE_PYPTO_W8A8", "0") != "1":
        return False
    return _load_pypto_kernel()


def _quantize_per_channel(x: torch.Tensor):
    x_fp32 = x.to(torch.float32)
    max_val = x_fp32.abs().max(dim=0, keepdim=True)[0]
    scale_q = 127.0 / (max_val + 1e-12)
    y_fp32 = x_fp32 * scale_q
    y_int8 = torch.round(y_fp32).to(torch.int8)
    scale_deq = (1.0 / scale_q).reshape(-1).to(torch.float32)
    return y_int8, scale_deq


def _stack_all_layer_weights(
    model: Qwen2Model,
    layer_list,
    num_layers: int,
    hidden_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Stack weights from all decoder layers into kernel-expected format.

    Kernel weight shapes (stacked across layers):
      qkv_proj_weight:  [layer_num * hidden_size, total_head_size]  (INT8)
      o_proj_weight:    [layer_num * q_size, hidden_size]           (INT8)
      w13:              [layer_num * hidden_size, intermediate_size*2] (INT8)
      w2:               [layer_num * intermediate_size, hidden_size] (INT8)
      input_layernorm_weight:      [layer_num, hidden_size]         (BF16)
      output_layernorm_weight:     [layer_num, hidden_size]         (BF16)
      q_norm_weight:               [layer_num, 1, head_dim]         (BF16)
      k_norm_weight:               [layer_num, 1, head_dim]         (BF16)
      *_scale:                     [layer_num, out_features]         (BF16)
    """
    qkv_w_list: List[torch.Tensor] = []
    qkv_s_list: List[torch.Tensor] = []
    o_w_list: List[torch.Tensor] = []
    o_s_list: List[torch.Tensor] = []
    w13_list: List[torch.Tensor] = []
    w13_s_list: List[torch.Tensor] = []
    w2_list: List[torch.Tensor] = []
    w2_s_list: List[torch.Tensor] = []
    input_ln_list: List[torch.Tensor] = []
    output_ln_list: List[torch.Tensor] = []
    q_norm_list: List[torch.Tensor] = []
    k_norm_list: List[torch.Tensor] = []
    has_qk_norm = False

    for layer in layer_list:
        self_attn = layer.self_attn
        mlp = layer.mlp

        qkv_w = self_attn.qkv_proj.weight.data
        qkv_w_list.append(qkv_w)
        qkv_scale = getattr(self_attn.qkv_proj, "weight_scale", None)
        if qkv_scale is not None:
            qkv_s_list.append(qkv_scale.data.flatten().to(torch.bfloat16))
        else:
            qkv_int8, qkv_deq = _quantize_per_channel(qkv_w)
            qkv_w_list[-1] = qkv_int8
            qkv_s_list.append(qkv_deq.to(torch.bfloat16))

        o_w = self_attn.o_proj.weight.data
        o_w_list.append(o_w)
        o_scale = getattr(self_attn.o_proj, "weight_scale", None)
        if o_scale is not None:
            o_s_list.append(o_scale.data.flatten().to(torch.bfloat16))
        else:
            o_int8, o_deq = _quantize_per_channel(o_w)
            o_w_list[-1] = o_int8
            o_s_list.append(o_deq.to(torch.bfloat16))

        gate_up_w = mlp.gate_up_proj.weight.data
        w13_list.append(gate_up_w)
        gate_up_scale = getattr(mlp.gate_up_proj, "weight_scale", None)
        if gate_up_scale is not None:
            w13_s_list.append(gate_up_scale.data.flatten().to(torch.bfloat16))
        else:
            w13_int8, w13_deq = _quantize_per_channel(gate_up_w)
            w13_list[-1] = w13_int8
            w13_s_list.append(w13_deq.to(torch.bfloat16))

        down_w = mlp.down_proj.weight.data
        w2_list.append(down_w)
        down_scale = getattr(mlp.down_proj, "weight_scale", None)
        if down_scale is not None:
            w2_s_list.append(down_scale.data.flatten().to(torch.bfloat16))
        else:
            w2_int8, w2_deq = _quantize_per_channel(down_w)
            w2_list[-1] = w2_int8
            w2_s_list.append(w2_deq.to(torch.bfloat16))

        input_ln_list.append(layer.input_layernorm.weight.data)
        output_ln_list.append(layer.post_attention_layernorm.weight.data)

        has_qk_norm = hasattr(self_attn, "q_norm") and self_attn.q_norm is not None
        if has_qk_norm:
            q_norm_list.append(
                self_attn.q_norm.weight.data.unsqueeze(0).unsqueeze(0)
            )
            k_norm_list.append(
                self_attn.k_norm.weight.data.unsqueeze(0).unsqueeze(0)
            )

    qkv_proj_weight = torch.cat(qkv_w_list, dim=0)
    qkv_weight_scale = torch.stack(qkv_s_list, dim=0)
    o_proj_weight = torch.cat(o_w_list, dim=0)
    o_proj_weight_scale = torch.stack(o_s_list, dim=0)
    w13 = torch.cat(w13_list, dim=0)
    w13_scale = torch.stack(w13_s_list, dim=0)
    w2 = torch.cat(w2_list, dim=0)
    w2_scale = torch.stack(w2_s_list, dim=0)
    input_layernorm_weight = torch.stack(input_ln_list, dim=0)
    output_layernorm_weight = torch.stack(output_ln_list, dim=0)

    if has_qk_norm:
        q_norm_weight = torch.cat(q_norm_list, dim=0)
        k_norm_weight = torch.cat(k_norm_list, dim=0)
    else:
        head_dim = self_attn.head_dim
        q_norm_weight = torch.ones(
            num_layers, 1, head_dim, dtype=torch.bfloat16, device=device
        )
        k_norm_weight = torch.ones(
            num_layers, 1, head_dim, dtype=torch.bfloat16, device=device
        )

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
    }


def _concat_kv_caches(kv_cache_list, num_layers):
    """
    Concatenate per-layer KV caches into kernel-expected format.

    vllm per-layer kv_cache: list/tuple of (key_cache, value_cache)
      each shape: [num_blocks, block_size, num_kv_heads, head_dim]

    Kernel expects:
      key_cache / value_cache shape:
        [layer_num * num_blocks, block_size, n2, d]
    """
    key_caches = []
    value_caches = []
    for i in range(num_layers):
        kv = kv_cache_list[i]
        key_caches.append(kv[0])
        value_caches.append(kv[1])
    key_cache = torch.cat(key_caches, dim=0)
    value_cache = torch.cat(value_caches, dim=0)
    return key_cache, value_cache


def _extract_attn_metadata(self_attn_layer):
    """
    Extract block_tables, slot_mapping, actual_seq_lens from ForwardContext.

    These are set by the model runner via set_forward_context() before
    model.forward() is called. Stored per-layer in forward_context.
    """
    forward_context = get_forward_context()
    attn_metadata_dict = forward_context.attn_metadata

    layer_name = getattr(self_attn_layer.attn, "layer_name", None)
    block_tables = None
    slot_mapping = None
    actual_seq_lens = None

    if layer_name and attn_metadata_dict and layer_name in attn_metadata_dict:
        meta = attn_metadata_dict[layer_name]
        block_tables = getattr(meta, "block_tables", None)
        slot_mapping = getattr(meta, "slot_mapping", None)
        actual_seq_lens = getattr(meta, "seq_lens", None)

    return block_tables, slot_mapping, actual_seq_lens


def _compute_cos_sin_from_cache(model, positions, hidden_states, device):
    """
    Compute cos/sin from rotary_emb's precomputed cos_sin_cache.

    vllm's rotary_emb stores cos_sin_cache of shape:
      [max_position_embeddings, rotary_dim]
    where rotary_dim = head_dim, and the cache contains
    [cos(half), sin(half)] concatenated along dim=-1,
    so each of cos/sin has size head_dim//2.

    Kernel expects:
      cos, sin shape: [bs, 1, half_rotary_dim]
      where half_rotary_dim = (head_dim // 2) // 2 = head_dim // 4

    This is because the kernel's rope_data function splits each head's
    rotary portion into two halves (q1, q2) and applies:
      o1 = q1*cos - q2*sin
      o2 = q2*cos + q1*sin
    then concatenates [o1, o2] along the head dimension.
    """
    bs = hidden_states.shape[0]
    self_attn = model.layers[0].self_attn
    head_dim = self_attn.head_dim
    half_rotary_dim = (head_dim // 2) // 2

    cos_sin_cache = self_attn.rotary_emb.cos_sin_cache
    pos_flat = positions.flatten().long()
    cos_sin = cos_sin_cache.index_select(0, pos_flat)
    cos_full, sin_full = cos_sin.chunk(2, dim=-1)

    cos = cos_full[:bs, :half_rotary_dim].reshape(
        bs, 1, half_rotary_dim
    ).to(torch.bfloat16)
    sin = sin_full[:bs, :half_rotary_dim].reshape(
        bs, 1, half_rotary_dim
    ).to(torch.bfloat16)

    return cos, sin


def _build_decode_config(
    world_size, num_layers, hidden_size, intermediate_size,
    num_heads, num_kv_heads, head_dim, softmax_scale,
    block_table_batch, kv_num_blocks, block_size, max_num_blocks_per_query,
):
    """
    Build Qwen3DecodeConfig with AttentionConfig and AttentionTileConfig.

    AttentionConfig fields map to model parameters:
      b     = batch size (number of requests)
      s1    = query tokens per request (1 for decode)
      s2    = max KV sequence length
      n1    = num query heads per rank
      n2    = num KV heads per rank
      q_d   = head dimension
      kv_d  = KV head dimension (same as q_d for Qwen2/3)
    """
    n1 = num_heads // world_size
    n2 = num_kv_heads // world_size
    s2 = 512

    attn_cfg = _AttentionConfig(
        b=block_table_batch,
        s1=1,
        s2=s2,
        n1=n1,
        n2=n2,
        softmax_scale=softmax_scale,
        kv_layout="PA_BSND",
        q_d=head_dim,
        kv_d=head_dim,
        block_table_batch=block_table_batch,
        kv_num_blocks=kv_num_blocks,
        hidden_size=hidden_size,
    )
    attn_cfg.max_num_blocks_per_query = max_num_blocks_per_query

    cube_tile = 128
    vector_tile = 128
    s2_tile = min(512, s2)
    g = n1 // n2 if n2 > 0 else 1

    tile_cfg = _AttentionTileConfig(
        g_tile=g,
        s2_tile=s2_tile,
        c1_tile_shape=[
            [cube_tile, cube_tile],
            [cube_tile, cube_tile],
            [cube_tile, cube_tile],
        ],
        v1_tile_shape=[vector_tile, s2_tile],
        c2_tile_shape=[
            [cube_tile, cube_tile],
            [cube_tile, cube_tile],
            [cube_tile, cube_tile],
        ],
        v2_tile_shape=[vector_tile, vector_tile],
    )

    decode_cfg = _Qwen3DecodeConfig(
        attn_cfg=attn_cfg,
        tile_cfg=tile_cfg,
        intermediate_size=intermediate_size,
        layer_num=num_layers,
        eps=1e-5,
    )
    return decode_cfg


_original_forward = Qwen2Model.forward


@torch.compiler.disable
def _qwen2_model_forward_pypto(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
):
    """
    Replacement for Qwen2Model.forward that fuses the entire layer loop
    into a single qwen3_decode_worker_w8a8 kernel call.

    Original flow (per-layer loop):
      for layer in self.layers:
          hidden_states, residual = layer(positions, hidden_states, residual)

    Replaced with (single fused call):
      out, residual_out, _ = qwen3_decode_worker_w8a8(
          hidden_states, residual,
          [all layer weights stacked],
          [all KV caches concatenated],
          block_tables, actual_seq_lens, slot_mapping,
          cos, sin,
          qwen3_decode_cfg, group_name, world_size,
      )

    Parameter mapping from layer(positions, hidden_states, residual):
    ┌──────────────────────┬──────────────────────────────────────────────┐
    │ layer() parameter    │ qwen3_decode_worker_w8a8 parameter          │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ hidden_states        │ hidden_states (direct pass-through)         │
    │ residual             │ residual (direct; zeros if None)            │
    │ positions            │ cos, sin (precomputed from rotary_emb cache │
    │                      │   indexed by positions)                     │
    │ layer.input_layernorm│ input_layernorm_weight [layer_num, hidden]  │
    │ layer.post_attn_ln   │ output_layernorm_weight [layer_num, hidden] │
    │ layer.self_attn.     │ qkv_proj_weight [L*hidden, total_head] INT8│
    │   qkv_proj           │ qkv_weight_scale [L, total_head] BF16      │
    │ layer.self_attn.     │ o_proj_weight [L*q_size, hidden] INT8      │
    │   o_proj             │ o_proj_weight_scale [L, hidden] BF16       │
    │ layer.self_attn.     │ q_norm_weight [L, 1, head_dim] BF16        │
    │   q_norm             │                                            │
    │ layer.self_attn.     │ k_norm_weight [L, 1, head_dim] BF16        │
    │   k_norm             │                                            │
    │ layer.mlp.           │ w13 [L*hidden, intermediate*2] INT8        │
    │   gate_up_proj       │ w13_scale [L, intermediate*2] BF16         │
    │ layer.mlp.           │ w2 [L*intermediate, hidden] INT8           │
    │   down_proj          │ w2_scale [L, hidden] BF16                  │
    │ attn.kv_cache        │ key_cache [L*blocks, blk_sz, n2, d]        │
    │                      │ value_cache [L*blocks, blk_sz, n2, d]      │
    │ attn_metadata.       │ block_tables (direct from ForwardContext)   │
    │   block_tables       │                                            │
    │ attn_metadata.       │ actual_seq_lens (direct from ForwardContext)│
    │   seq_lens           │                                            │
    │ attn_metadata.       │ slot_mapping (direct from ForwardContext)   │
    │   slot_mapping       │                                            │
    └──────────────────────┴──────────────────────────────────────────────┘
    """
    if not _is_pypto_enabled():
        return _original_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )

    from vllm.distributed import get_pp_group

    if not get_pp_group().is_first_rank:
        return _original_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )

    if self.start_layer != 0 or self.end_layer != len(self.layers):
        return _original_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )

    in_profile_run = getattr(_EXTRA_CTX, "in_profile_run", False)

    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)

    residual = None
    device = hidden_states.device
    world_size = get_tensor_model_parallel_world_size()
    num_layers = self.end_layer - self.start_layer
    hidden_size = self.config.hidden_size
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    head_dim = getattr(
        self.config, "head_dim", hidden_size // num_heads
    )
    intermediate_size = self.config.intermediate_size
    eps = getattr(self.config, "rms_norm_eps", 1e-6)
    softmax_scale = head_dim ** -0.5

    if residual is None:
        residual = torch.zeros_like(hidden_states)

    layer_list = list(self.layers[self.start_layer:self.end_layer])

    weights = _stack_all_layer_weights(
        self, layer_list, num_layers, hidden_size, device
    )

    kv_cache_list = [
        layer_list[i].self_attn.attn.kv_cache for i in range(num_layers)
    ]
    key_cache, value_cache = _concat_kv_caches(kv_cache_list, num_layers)

    first_layer_attn = layer_list[0].self_attn
    block_tables, slot_mapping, actual_seq_lens = _extract_attn_metadata(
        first_layer_attn
    )

    cos, sin = _compute_cos_sin_from_cache(
        self, positions, hidden_states, device
    )

    rank = get_tensor_model_parallel_rank()
    group_name = get_tp_group().device_group._get_backend(
        device
    ).get_hccl_comm_name(rank)

    bs = hidden_states.shape[0]
    block_table_batch = (
        block_tables.shape[0] if block_tables is not None else bs
    )
    kv_num_blocks = (
        key_cache.shape[0] // num_layers if key_cache is not None else 1
    )
    block_size = (
        key_cache.shape[1] if key_cache is not None else 128
    )
    max_blocks_per_query = (
        block_tables.shape[1] if block_tables is not None else 1
    )

    decode_cfg = _build_decode_config(
        world_size=world_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        block_table_batch=block_table_batch,
        kv_num_blocks=kv_num_blocks,
        block_size=block_size,
        max_num_blocks_per_query=max_blocks_per_query,
    )

    out_torch, residual_out, _ = _qwen3_decode_worker_w8a8(
        hidden_states=hidden_states,
        residual=residual,
        input_layernorm_weight=weights["input_layernorm_weight"],
        output_layernorm_weight=weights["output_layernorm_weight"],
        qkv_proj_weight=weights["qkv_proj_weight"],
        qkv_weight_scale=weights["qkv_weight_scale"],
        o_proj_weight=weights["o_proj_weight"],
        o_proj_weight_scale=weights["o_proj_weight_scale"],
        q_norm_weight=weights["q_norm_weight"],
        k_norm_weight=weights["k_norm_weight"],
        w13=weights["w13"],
        w13_scale=weights["w13_scale"],
        w2=weights["w2"],
        w2_scale=weights["w2_scale"],
        cos=cos,
        sin=sin,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        actual_seq_lens=actual_seq_lens,
        slot_mapping=slot_mapping,
        eps=eps,
        enable_residual=True,
        num_decode_tokens=bs,
        qwen3_decode_cfg=decode_cfg,
        group_name=group_name,
        world_size=world_size,
    )

    hidden_states = out_torch
    residual = residual_out

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


Qwen2Model.forward = _qwen2_model_forward_pypto
logger.info(
    "PyPTO W8A8 patch applied to Qwen2Model.forward "
    "(set VLLM_ASCEND_ENABLE_PYPTO_W8A8=1 to activate)"
)
