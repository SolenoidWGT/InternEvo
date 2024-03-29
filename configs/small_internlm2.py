# Copyright (c) InternLM. All rights reserved.
JOB_NAME = "100b_internlm2_demo"

# !!!!!!!! Important configuration !!!!!!!!
model_type = "INTERNLM2_PUBLIC"
SAVE_CKPT_FOLDER = f"./{JOB_NAME}_checkpoints"
LOAD_MODEL_PATH = "/mnt/petrelfs/wangguoteng.p/InternEvo/small_model_align"
SAVED_DATA_PATH = "/mnt/inspurfs/zhangshuo/datasets/1125_attr_filtered_100m_metric_v13_dumped/"
CHECKPOINT_EVERY = 250
LEARNING_RATE = 1e-2
MIN_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.028
OPTIMIZER_WARMUP_STEP = 0

VOCAB_SIZE = 92544
HIDDEN_SIZE = 768
NUM_ATTENTION_HEAD = 12
MLP_RATIO = 8 / 3
NUM_LAYER = 12
SEQ_LEN = 4096
MICRO_NUM = 4
MICRO_BSZ = 16
TOTAL_STEP = 90000
PACK_SAMPLE_INTO_ONE = True
SEQUENCE_PARALLEL_MODE = "mtp"  # Megatron-Sequence parallelism implemented by Tri Dao.
TP = 1  # # Megatron-Tensor parallelism
PP = 1  # 1F1B Pipeline parallel
ZERO = 8
USE_FLASH_ATTN = True
RE_COMUPTED = False
DTYPE = "torch.bfloat16"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

model = dict(
    checkpoint=False,
    num_chunks=1,  # 每张卡需要forward多少次。
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=False,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,  # 清华的embedding放缩技巧，如果为1的话，不放缩
    parallel_output=False,  # 最后的输出是否需要gather起来，如果不gather的话，每个tensor parallel获取的就是自己对应的结果
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    no_bias=True,
    deepnorm=False,
    dtype=DTYPE,
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    embedding_init_std=0.02,
    attn_wqkv_init_std=0.02,
    attn_other_init_std=0.02,
    ffn_uplayer_init_std=0.02,
    ffn_other_init_std=0.02,
    out_head_init_std=0.02,
)
"""
zero1 parallel (dict):
    1. size: int
        * if size <= 0, the size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        * if size == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        * if size > 1 and size <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
    2. fsdp: bool, enable/disable torch's fully sharded data parallel, defaults to False.
tensor parallel (dict):
    1. size: int, the size of tensor parallel.
    2. mode: str, the tensor parallel mode, should be in ['mtp', 'msp', 'fsp', 'isp'],
        defaults to 'mtp', means the pure megatron tensor parallel without sequence parallel.
        msp: megatron tensor parallel with sequence parallel, sequence parallel size = tensor parallel size.
        fsp: tensor parallel by flash-attn with sequence parallel, sequence parallel size = tensor parallel size.
        isp: customed intern sequence parallel without tensor parallel, can be used with weight parallel.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler,
        defaults to False.
weight parallel (dict):
    1. size: int, the size of weight parallel.
    2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
    3. memory_pool: bool, enable/disable memory pool, defaults to False.
"""
parallel = dict(
    zero1=dict(size=ZERO),
    tensor=dict(size=TP, mode=SEQUENCE_PARALLEL_MODE),
    pipeline=dict(size=PP, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=True),
)
hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,  # overlap grad all_reduce/reduce_scatter
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

data = dict(
    seq_len=SEQ_LEN,
    micro_num=MICRO_NUM,
    micro_bsz=MICRO_BSZ,
    valid_micro_num=4,
    valid_every=0,
    pack_sample_into_one=PACK_SAMPLE_INTO_ONE,
    total_steps=TOTAL_STEP,
    skip_batches="",
    rampup_batch_size="",
    min_length=50,
    train_folder=None,
    valid_folder=None,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
)
loss = dict(label_smoothing=0.0)
adam = dict(
    lr=LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=OPTIMIZER_WARMUP_STEP,  # optimizer_warmup_step
    warmup_ratio=WARMUP_RATIO,
    eta_min=MIN_LEARNING_RATE,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)
cudnn_deterministic = False
cudnn_benchmark = False
monitor = dict(
    alert=dict(
        enable_feishu_alert=False,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)
grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**14,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)
ckpt = dict(
    enable_save_ckpt=True,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=False,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=CHECKPOINT_EVERY,  # snapshot ckpt save frequency.
    load_ckpt_info=dict(path=LOAD_MODEL_PATH, content=("model",), ckpt_type="internevo"),
)
