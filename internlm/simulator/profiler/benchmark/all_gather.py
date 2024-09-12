import torch
import torch.distributed as dist

from internlm.model.registry import benchmark_initializer
from internlm.simulator.common import *
from internlm.utils.common import get_current_device

from .base_benchmark import UnitBench

BENCH_TYPE = "all_gather"


# @benchmark_initializer.register_module(module_name=BENCH_TYPE)
class UnitBenchAllGather(UnitBench):
    test_loop = {
        "global_size": GLOBAL_ELEM_SIZES_LIST,
        "async_op": [False],  # it is not work!! False,
        "dtype": [torch.bfloat16],
    }

    def __init__(self, async_op, dtype, group, global_size=None, unit_size=None) -> None:
        assert global_size is None or unit_size is None

        world_size = dist.get_world_size(group)
        self.unit_size = unit_size if unit_size else global_size // world_size  # elements_per_gpu
        self.world_size = world_size
        self.dtype = dtype
        self.async_op = async_op
        self.group = group

        device = get_current_device()

        if self.group is not None:
            self.output = torch.ones(self.world_size, self.unit_size, dtype=self.dtype, device=device)
            self.input = torch.ones(self.unit_size, dtype=self.dtype, device=device)
            self.output_buffer_size = self.output.element_size() * self.output.numel()

    def run(self):
        if self.group is None:
            return

        handler = dist._all_gather_base(self.output, self.input, async_op=self.async_op, group=self.group)
        if self.async_op:
            handler.wait()

    def bw_complexity(self):
        return self.output_buffer_size

    def algo_complexity(self):
        return self.output_buffer_size