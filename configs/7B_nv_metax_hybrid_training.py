JOB_NAME = "7b_internlm2_train"
model_type = "INTERNLM2_PUBLIC"
DO_ALERT = False

clusters = [
    {
        "name": "nv",
        "peak_tflops": 320,
        "capacity": 80 * 1024**3,
        "intra_bw": 150,
        "inter_bw": 100,
        "gpu_per_node": 8,
        "node_num": 1,
    },
    {
        "name": "mx",
        "peak_tflops": 240,
        "capacity": 64 * 1024**3,
        "intra_bw": 150,
        "inter_bw": 100,
        "gpu_per_node": 8,
        "node_num": 1,
    },
]

GLOBAL_BSZ = 4 * 1024**2
VOCAB_SIZE = 92544
SEQ_LEN = 2048
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 3.5
NUM_LAYER = 32
MICRO_SIZE = 1
MICRO_NUM = -1

data = dict(
    seq_len=SEQ_LEN,
    micro_num=MICRO_NUM,
    micro_bsz=MICRO_SIZE,
    valid_micro_num=4,
    valid_every=0,
    pack_sample_into_one=False,
    total_steps=20,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    train_folder=None,
    valid_folder=None,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
    global_bsz=GLOBAL_BSZ,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
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

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

use_fp32_norm = False
model = dict(
    checkpoint=False,
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    use_flash_attn=True,
    qk_interleaved=False,
)

parallel = dict(
    zero1=dict(size=8),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=True),
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)

ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=None,  # Path to save training ckpt.
    auto_resume=False,
    checkpoint_every=50,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(50 / 2),  # snapshot ckpt save frequency.
)
