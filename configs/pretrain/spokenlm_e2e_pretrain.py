project = "speechlm"
runner = 'SpokenLmPretrainRunner'

# increment to add and override args.
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_6wh', # 
        # 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    ],
    train_file_list=[
        '["sub_{}".format(i) for i in range(1333)]', # 6w 小时数据
        # '["train_sub{}".format(i) for i in range(103)]',
    ],
    valid_data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/aishell1/dev/wav_ark_text/"
    ],
    valid_file_list=[
        '["dev"]',
    ],
    chunk_size=20,
    bucket_schedule='200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_val='5,100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_key='length',
    max_batch_size=20000,
    batch_strategy='FixedBucketBatching',
    use_old_bucket=True,
    prefetch_worker_num=4,
    next_retry=20,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    train_item_transform=[
        dict(type="PickleParser"),
        dict(type='WavParser'),
        dict(type="WavResample", sample_rate=16000, key="waveform"),
        dict(type="WavCrop", key="waveform", crop_sample_length=480000),
        dict(type="WavConvert", in_key="waveform"),
        dict(type="CalculateFrameLength"),
    ],
    valid_item_transform=[
        dict(type="PickleParser"),
        dict(type='WavParser'),
        dict(type="WavResample", sample_rate=16000, key="waveform"),
        dict(type="WavCrop", key="waveform", crop_sample_length=480000),
        dict(type="WavConvert", in_key="waveform"),
        dict(type="CalculateFrameLength"),
    ],
    batch_transform=[
        dict(type="WaveformCollate", wav_key="waveform", frame_chunk_size=1),
    ],
    inference_batch_transform=[
        dict(type="WaveformCollate", wav_key="waveform"),
    ],
)
solution = dict(
    model_type='SpokenLmE2EPretrainModel',

    # speech encoder relative
    wav2vec_type='PretrainedData2Vec',
    use_fused_kernel=False,
    activation_checkpoint=False,
    data2vec_finetuning=True,
    extractor_mode="layer_norm",
    normalize=True,
    encoder_layers=24,
    encoder_embed_dim=1024,
    encoder_ffn_embed_dim=4096,
    encoder_attention_heads=16,
    activation_fn="gelu",
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    layer_norm_first=True,
    encoder_layerdrop=0.0,
    pad_to_multiple=True,
    required_seq_len_multiple=4,
    conv_feature_layers="[(512, 10, 5)] + [(512, 3, 2)] * 5 + [(512, 2, 2)] * 2",
    feature_grad_mult=0.0,
    fix_w2v_model=True,
    # mask
    mask_length=5,
    mask_prob=0.5,
    mask_selection="static",
    mask_other=0.0,
    mask_minlen_type="random",
    no_mask_overlap=False,
    mask_min_space=1,
    mask_dropout=0,
    # mask channel
    mask_channel_length=10,
    mask_channel_prob=0.0,
    mask_channel_selection="static",
    mask_channel_other=0.0,
    mask_channel_minlen_type="random",
    mask_channel_before=False,
    no_mask_channel_overlap=False,
    mask_channel_min_space=1,
    require_same_masks=True,
    # dropout
    dropout_input=0.1,
    conv_pos=95,
    conv_pos_groups=16,
    pos_conv_depth=5,
    conv_bias=True,
    data2vec_feature_conv_type="default",
    final_proj_num_layer=1,
    final_dropout=0,
    final_dim=256,
    weighted_data2vec=False,
    apply_mask=True,
    freeze_finetune_updates=10000000,
    freeze_encoder_layers=-1,
    average_top_k_layers=0,
    loss_beta=0,
    loss_scale=-1,
    ema_transformer_only=True,

    # gumbel
    init_temp=0.1,
    temp_decay_factor=0.999995,

    # lm
    vocab_size=8192, # 和codebook一样大
    lm_backbone_type='TransformerBackbone',
    causal_transformer=True,
    backbone_topology='[[-1,0,1]]*24',  # 这两行是搞出来上三角mask。18是层数。
    chunk_mask=False,
    backbone_pos_embd=True,  # 这里用的是绝对位置编码
    backbone_after_norm=True,
    backbone_layer_num=24,
    backbone_memory_size=1024,
    backbone_hidden_size=4096,
    self_attn_heads=16,
    self_attn_dropout=0.1,
    positional_dropout_rate=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=True,  # 这个是pre-norm吗
    self_attn_activation_fn='gelu',
    backbone_act_glu=True,

    # Loss
    label_smooth_factor=0,
    diversity_loss_factor=1.0,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    resume_pretrain_chkpt="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/"
                          "donglinhao/dolphin/models/supervised_60k/data2vec_pretrain_large_bs12000_exp1_25hz_2/"
                          "data2vec_pretrain/large_lr3e-4_poly600k_mask0.65_ld0.00_bsz63m/checkpoints/step_600000.pth",
    max_epochs=40,
    max_iters=300000,
    lr_scheduler=dict(
        by_epoch=False,
        # policy="LayerwiseTriStage",
        policy="TriStage",
        lr=1e-3,
        phase_ratio="[0.1,0.4,0.5]",
        init_lr_scale=0.01,
        final_lr_scale=0.05,
        max_train_steps=300000,
        # lr_prefix_multiplier_list=[('w2v_model.', 0.02)]
    ),
    checkpoint_config=dict(
        interval=1600,  # steps
        max_keep_ckpts=5,
    ),
    save_root='./spokenlm_e2e',
    save_dir='spokenlm_e2e_pretrain',
    save_name='spokenlm_e2e_pretrain',
    freeze_feature_extractor=False,
    compare_metric="UF",
    validation_log=True,
    amp_level='O0',
    # gradient accumulation
    grad_accum_step=1,
    )
valid = dict(
    output_probs=False,
    not_show_results=False,
    interval=1600,
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='AdamW',
    # lr=[('w2v_model.', 2e-5), 1e-3],
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-8,
    weight_decay=0.01,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=100.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    max_grad_clip=0.0,
    bmuf_config=0,
 )
log_level = 'INFO'
log_config = dict(
    interval=100,  # log at every 100 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])