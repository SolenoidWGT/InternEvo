import pytest
import torch
from torch import nn

import internlm  # noqa # pylint: disable=W0611
import internlm.model.modules.embedding as InternEMBED
from internlm.accelerator import get_accelerator
from internlm.model.modules.embedding import (
    ApplyRotaryEmb,
    ApplyRotaryEmbQKV_,
    RotaryEmbedding,
    _torch_apply_rotary_func,
    apply_torch_npu_rotary_mul,
    rotary_emb_in_rotate_half_style,
)
from internlm.model.ops.fusion_ops_import_helper import try_import_fused_rotary
from internlm.utils.common import get_current_device

internlm_accelerator = get_accelerator()


MICRO_BSZ_LIST = [1, 2]
DTYPE_LIST = [torch.bfloat16, torch.float16]
NUM_KV_HEAD_LIST = [8, 32]


def npu_rope_fwd(B, KV_N, dtype, H=128, N=32, S=4096, rope_base=10000):
    device = get_current_device()
    INDEXES = torch.tensor(list(range(S)), dtype=torch.long, device=device)
    qkv = torch.randn((B, S, 3, N, H), dtype=dtype, device=device)
    q = torch.randn((B, S, N, H), dtype=dtype, device=device)
    k = torch.randn((B, S, KV_N, H), dtype=dtype, device=device)

    q = nn.init.normal_(q, mean=0.0, std=1.0)
    k = nn.init.normal_(k, mean=0.0, std=1.0)

    # Test normal torch.
    InternEMBED.apply_rotary_emb = ApplyRotaryEmb.apply
    InternEMBED.apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
    InternEMBED.apply_rotary_func = _torch_apply_rotary_func

    embed = RotaryEmbedding(dim=H, base=rope_base, scale_base=0, device=device)
    q1 = embed._single_forward(q.clone(), indexes=INDEXES)
    k1 = embed._single_forward(k.clone(), indexes=INDEXES)
    qkv1 = embed(qkv.clone(), indexes=INDEXES)

    # Test rotate_half torch.
    InternEMBED.apply_rotary_emb = rotary_emb_in_rotate_half_style
    InternEMBED.apply_rotary_emb_qkv_ = rotary_emb_in_rotate_half_style
    InternEMBED.apply_rotary_func = apply_torch_npu_rotary_mul
    embed = RotaryEmbedding(dim=H, base=rope_base, scale_base=0, device=device)
    q2 = embed._single_forward(q.clone(), indexes=INDEXES)
    k2 = embed._single_forward(k.clone(), indexes=INDEXES)
    qkv2 = embed(qkv.clone(), indexes=INDEXES)

    # Test rotate_half torch_npu fused.
    _, _, apply_npu_rotary_mul = try_import_fused_rotary()
    assert apply_npu_rotary_mul is not None
    InternEMBED.apply_rotary_func = apply_npu_rotary_mul
    embed = RotaryEmbedding(dim=H, base=rope_base, scale_base=0, device=device)
    q3 = embed._single_forward(q.clone(), indexes=INDEXES)
    k3 = embed._single_forward(k.clone(), indexes=INDEXES)
    qkv3 = embed(qkv.clone(), indexes=INDEXES)

    assert torch.allclose(q1, q2, rtol=1e-4, atol=1e-5)
    assert torch.allclose(q2, q3, rtol=1e-4, atol=1e-5)
    assert torch.allclose(k1, k2, rtol=1e-4, atol=1e-5)
    assert torch.allclose(k2, k3, rtol=1e-4, atol=1e-5)
    assert torch.allclose(qkv1, qkv2, rtol=1e-4, atol=1e-5)
    assert torch.allclose(qkv2, qkv3, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
def test_NPU_fa(micro_bsz, test_dtype, num_kv_head):
    npu_rope_fwd(B=micro_bsz, KV_N=num_kv_head, dtype=test_dtype)
