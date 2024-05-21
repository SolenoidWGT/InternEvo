"""
TODO: add NPU CI
"""

import math
import random

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.model.modules.multi_head_attention import (
    AscendFlashAttention,
    CrossAttention,
)
from internlm.utils.common import set_random_seed

HEAD_NUM = 32
HIDDEN_SZIE = 4096
SEQ_LEN = 2048
MICRO_BSZ = 1
HEAD_DIM = HIDDEN_SZIE // HEAD_NUM
VOCAB_SIZE = 32000
NUM_KV_HEAD_LIST = [8, 32]
MICRO_BSZ_LIST = [1, 2]
DTYPE_LIST = [torch.bfloat16, torch.float16]

internlm_accelerator = get_accelerator()


def npu_transform(B, S, N_KV, dtype):
    set_random_seed(1024)
    softmax_scale = 1 / math.sqrt(HEAD_DIM)
    cross_attn = CrossAttention(causal=True, softmax_scale=softmax_scale, attention_dropout=0.0).to(dtype)
    npu_flash_attn = AscendFlashAttention(causal=True, softmax_scale=softmax_scale, attention_dropout=0.0).to(dtype)

    x = torch.LongTensor([[i + 1 for i in range(S)] for _ in range(B)]).npu()  # no-padiing
    cu_seqlens = [0] + sorted(random.sample(list(range(x.numel())), 4))
    if cu_seqlens[-1] != x.numel():
        cu_seqlens.append(x.numel())
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device="npu")
    x = rearrange(x, "b s -> (b s)").unsqueeze(0)

    KV_DIM = HEAD_DIM * N_KV
    Q_PER_KV = HEAD_NUM // N_KV
    wqkv = torch.rand((HIDDEN_SZIE + 2 * KV_DIM, HIDDEN_SZIE), dtype=dtype, device="npu")
    wembed = torch.rand((VOCAB_SIZE, HIDDEN_SZIE), dtype=dtype, device="npu")

    # It is very important to set appropriate initialization values for parameters so
    # that the values fall within an appropriate precision range to prevent overflow or underflow.
    with torch.no_grad():
        wqkv.data = nn.init.normal_(wqkv.data)
        wembed = nn.init.normal_(wembed.data, std=0.02)

    embed_x = F.embedding(x, wembed).to(dtype)
    qkv = F.linear(embed_x, wqkv)  # pylint: disable=E1102
    qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=Q_PER_KV + 2, d=HEAD_DIM)
    q, k, v = (qkv[..., :Q_PER_KV, :], qkv[..., -2, :], qkv[..., -1, :])
    q = rearrange(q, "b t h gs d -> b t (h gs) d")
    kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
    q_fa, kv_fa = q.clone(), kv.clone()

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    q, kv = unpack_qkv_before_attn(q, cu_seqlens), unpack_qkv_before_attn(kv, cu_seqlens)
    a = cross_attn(q, kv)  # pylint: disable=E1102
    c = npu_flash_attn(q=q, kv=kv)  # pylint: disable=E1102
    a, c = rearrange(a, "b s h d -> b s (h d)"), rearrange(c, "b s h d -> b s (h d)")
    a, c = pack_output_after_attn(a, cu_seqlens, packed_len=B * S), pack_output_after_attn(
        c, cu_seqlens, packed_len=B * S
    )

    q, kv = q_fa.squeeze(0), kv_fa.squeeze(0)
    b = npu_flash_attn(  # pylint: disable=E1102
        q=q,
        kv=kv,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )
    b = rearrange(b, "s h d -> s (h d)").unsqueeze(0)

    assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    if dtype == torch.bfloat16:
        # torch_npu's equal not support bfloat16 by now.
        assert torch.allclose(a.to(torch.float32), b.to(torch.float32), atol=5e-2, rtol=1e-4)
        assert torch.allclose(a.to(torch.float32), c.to(torch.float32), atol=5e-2, rtol=1e-4)
    else:
        assert torch.allclose(a, b, atol=5e-2, rtol=1e-4), f"a: {a}, b: {b}"
        assert torch.allclose(a, c, atol=5e-2, rtol=1e-4), f"a: {a}, c: {c}"
    print("done!", flush=True)


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
def test_NPU_fa(micro_bsz, test_dtype, num_kv_head):
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        npu_transform(micro_bsz, SEQ_LEN, num_kv_head, test_dtype)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_npu_ops.py"])
