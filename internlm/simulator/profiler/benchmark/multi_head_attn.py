#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

from internlm.simulator.profiler.profiler import Timer
from internlm.utils.common import get_current_device

BENCH_TYPE = "flash_attn"


def run_fa_lat_test(micro_bsz, seqlen, hidden_size, q_head, kv_head, dtype, warmups=2, trials=5):
    # 1, S, N, H
    def run():
        device = get_current_device()
        cu_seqlens = torch.tensor([i * seqlen for i in range(micro_bsz + 1)], dtype=torch.int32, device=device)

        tfwd, tbwd = Timer(True), Timer(True)
        q = torch.rand(
            [micro_bsz * seqlen, q_head, hidden_size // q_head],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        kv = torch.rand(
            [micro_bsz * seqlen, 2, kv_head, hidden_size // q_head],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        torch.cuda.synchronize()
        tfwd.start()
        context = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=seqlen,
            max_seqlen_q=seqlen,
            causal=True,
        )
        t_fwd = tfwd.end()

        grad = torch.randn_like(context) / 32  # avoid grad is too large.
        torch.cuda.synchronize()

        tbwd.start()
        context.backward(grad, retain_graph=True)
        t_bwd = tbwd.end()
        return t_fwd, t_bwd

    for i in range(warmups):
        run()

    t_fwds, t_bwds = 0, 0
    for i in range(trials):
        t_fwd, t_bwd = run()
        t_fwds += t_fwd
        t_bwds += t_bwd

    return t_fwds / trials, t_bwds / trials
