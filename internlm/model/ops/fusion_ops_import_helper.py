from typing import Callable, Tuple, Union

import torch
from torch import Tensor, nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

internlm_accelerator = get_accelerator()

fused_rotary_is_warning = False


# RMSNorm
def try_import_RMSNorm():
    """
    Try import MixFusedRMSNorm from apex, if failed, return our RMSNorm

    """
    try:
        device_backend = internlm_accelerator.get_accelerator_backend()
        if device_backend == AcceleratorType.DIPU:
            from deeplink_ext.internlm_ops.rms_norm import (
                DeepLinkRMSNormWithNormalizedShape as RMSNorm,
            )

            if gpc.is_rank_for_log():
                logger.warning("Use DeepLinkRMSNormWithNormalizedShape, Please note this!")

            return RMSNorm
        else:
            from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm

            if gpc.is_rank_for_log():
                logger.warning("Use apex MixedFusedRMSNorm, Please note this!")

            return RMSNorm
    except (ModuleNotFoundError, ImportError):
        if gpc.is_rank_for_log():
            logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
        from internlm.model.ops.norm import RMSNormTorch as RMSNorm

        return RMSNorm


# RotaryEmb
def try_import_fused_rotary() -> Tuple[Union[None, Callable], Union[None, Callable], Union[None, Callable]]:
    """try_import_fused_rotary

    Returns:
        Tuple[Union[None, Callable], Union[None, Callable], Union[None, Callable]]:
            Returns if there is a mixing operator available, otherwise returns None.
    """
    global fused_rotary_is_warning
    try:
        device_backend = internlm_accelerator.get_accelerator_backend()
        if device_backend is AcceleratorType.GPU:
            import rotary_emb

            if gpc.is_rank_for_log() and not fused_rotary_is_warning:
                logger.warning("Use flash_attn rotary_emb, Please note this!")
                fused_rotary_is_warning = True

            return None, None, rotary_emb.apply_rotary
        elif device_backend is AcceleratorType.DIPU:
            from deeplink_ext.internlm_ops.rotary.deeplink import (
                DeeplinkApplyRotaryEmb,
                DeeplinkApplyRotaryEmbQKV_,
            )

            if gpc.is_rank_for_log() and not fused_rotary_is_warning:
                logger.warning("Use DeeplinkApplyRotaryEmb, Please note this!")
                fused_rotary_is_warning = True

            return DeeplinkApplyRotaryEmb.apply, DeeplinkApplyRotaryEmbQKV_.apply, None
        elif device_backend is AcceleratorType.NPU:
            import torch_npu

            from internlm.model.modules.embedding import rotary_emb_in_rotate_half_style

            def apply_npu_rotary_mul(x: Tensor, cos: Tensor, sin: Tensor, out_dtype: torch.dtype):
                """
                Implement RotaryEmbedding rotation position encoding. Support FakeTensor mode.
                Ref: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/
                        apiref/fmkadptapi/ptaoplist_000451.html
                Args:
                    x (Tensor): q or k, shape is [B, S, N, D].
                    cos (Tensor): cos, shape is [1, S, 1, D].
                    sin (Tensor): sin, shape is [1, S, 1, D].
                    out_dtype (torch.dtype): output dtype.
                """
                return torch_npu.npu_rotary_mul(x, cos, sin).to(out_dtype)

            if gpc.is_rank_for_log() and not fused_rotary_is_warning:
                logger.warning("Use npu_rotary_mul, Please note this!")
                fused_rotary_is_warning = True

            return rotary_emb_in_rotate_half_style, rotary_emb_in_rotate_half_style, apply_npu_rotary_mul

    except (ModuleNotFoundError, ImportError):
        pass

    if gpc.is_rank_for_log() and not fused_rotary_is_warning:
        logger.warning(
            "The torch implementation for apply_rotary is slower" "than flash atten rotary_emb. Please note this!"
        )
        fused_rotary_is_warning = True

    return None, None, None


# ParallelGPT2Embeddings
def try_import_ParallelGPT2Embeddings(embed_split_hidden):
    try:
        device_backend = internlm_accelerator.get_accelerator_backend()

        if not embed_split_hidden:
            if device_backend is AcceleratorType.GPU:
                from flash_attn.modules.embedding import ParallelGPT2Embeddings

                return ParallelGPT2Embeddings

    except (ModuleNotFoundError, ImportError):
        pass

    return None


# CrossEntropyLoss
def internlm_init_CrossEntropyLoss(
    parallel_output: bool, reduction="none", label_smoothing=0, inplace_backward=True, process_group=None, **kwargs
):
    """
    Try import FlashCrossEntropyLoss from flash_attn, if failed, return our CrossEntropyLoss

    """
    if parallel_output:
        try:
            if internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU:
                from flash_attn.losses.cross_entropy import (
                    CrossEntropyLoss as FlashCrossEntropyLoss,
                )

                if process_group is None:
                    gpc.get_group(ParallelMode.TENSOR)

                if gpc.is_rank_for_log():
                    logger.warning("Use flash_attn FlashCrossEntropyLoss, Please note this!")

                return FlashCrossEntropyLoss(
                    reduction=reduction,
                    inplace_backward=inplace_backward,
                    process_group=process_group,
                    label_smoothing=label_smoothing,
                    **kwargs,
                )
        except (ModuleNotFoundError, ImportError):
            pass

    if gpc.is_rank_for_log():
        logger.warning(
            "Use nn.CrossEntropyLoss rather than CrossEntropyLoss."
            "parallel_output must be set false. Please note this!"
        )

    if "process_group" in kwargs:
        kwargs.pop("process_group")
    if "inplace_backward" in kwargs:
        kwargs.pop("inplace_backward")

    return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing, **kwargs)


# Adamw
def try_import_FusedAdamW():
    """
    Try import FusedAdamW from torch_npu/torch

    """
    adam_extra_kwargs = {}
    backend = internlm_accelerator.get_accelerator_backend()
    try:
        if backend is AcceleratorType.GPU:
            adam_extra_kwargs["fused"] = True

            if gpc.is_rank_for_log():
                logger.warning(
                    "Use fused AdamaW to avoid nan grad norm when "
                    "model size is larger and use_fp32_norm=True, Please note this!"
                )
            return adam_extra_kwargs, torch.optim.AdamW
        elif backend is AcceleratorType.NPU:

            if gpc.is_rank_for_log():
                logger.warning(
                    "Use normal AdamaW, NPU fused_adamw currently has"
                    "accuracy issues and is not supported yet. Please note this!"
                )
            # return adam_extra_kwargs, torch_npu.optim.NpuFusedAdamW
    except (ModuleNotFoundError, ImportError):
        pass

    if gpc.is_rank_for_log():
        logger.warning("Use torch.optim.AdamW rather than FusedAdamW. Please note this!")
    return adam_extra_kwargs, torch.optim.AdamW


# scatter_sum
def try_import_scatter_sum():
    """
    Try import scatter_sum from cuda, if failed, return None

    """
    try:
        if internlm_accelerator.get_accelerator_backend() in [AcceleratorType.GPU, AcceleratorType.DIPU]:
            from torch_scatter import scatter as cuda_scatter

            if gpc.is_rank_for_log():
                logger.warning("Use cuda_scatter. Please note this!")

            return cuda_scatter

    except (ModuleNotFoundError, ImportError):
        pass

    if gpc.is_rank_for_log():
        logger.warning("Use vanilla_scatter rather than cuda_scatter. Please note this!")

    return None


# FlashAttn
def try_import_linear_bias_wgrad():
    """
    Try import linear_bias_wgrad from flash_attn, if failed, return None

    """
    try:
        if internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU:
            import fused_dense_lib as fused_dense_cuda

            if gpc.is_rank_for_log():
                logger.warning("Use flash_attn linear_bias_wgrad. Please note this!")

            return fused_dense_cuda.linear_bias_wgrad

    except (ModuleNotFoundError, ImportError):
        pass

    if gpc.is_rank_for_log():
        logger.warning("Use linear_bias_wgrad_torch. Please note this!")

    return None
