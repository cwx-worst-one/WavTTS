project = "speechlm"
runner = 'AuLlmPretrainRunner'

# increment to add and override args.
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav_testset_basic_7_21/',
    train_file_list='["train_sub{}".format(i) for i in range(102)]',
    valid_file_list='["cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta_ag350k_lzl",
    chunk_size=20,
    bucket_schedule='318, 398, 468, 522, 592, 646, 704, 781, 838, 898, 947, 998, 1098, 1209, 1314, 1398, 1498, 1558, 2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='length',
    max_batch_size=15000,
    # max_batch_size=1,
    batch_strategy='FixedBucketBatching',
    use_old_bucket=True,
    prefetch_worker_num=4,
    next_retry=20,
    shuffle=True,
    global_shuffle=True,
    drop_last=False,
    batch_means_tokens=1,
    use_eos=1,
    fbank_dim=80,
    fbank_channel=1,
    fbank=None,
    train_item_transform=[
        dict(type="PickleParser"),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type='LengthFilter', key='length', min_len=20, max_len=2000),
        dict(type="WavResample", sample_rate=16000, key="waveform"),
        dict(type='BPE', use_eos=True),
        dict(type='LabelLengthFilter', in_out_ratio=1, key='char'),
    ],
    valid_item_transform=[
        dict(type="PickleParser"),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type="WavResample", sample_rate=16000, key="waveform"),
        dict(type='BPE', use_eos=True),
        dict(type='LabelLengthFilter', in_out_ratio=1, key='char'),
    ],
    batch_transform=[
        dict(type='WaveformCollate', wav_key='waveform'),
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='label'),
        dict(type='CharCollate'),
    ],
    inference_batch_transform=[
        dict(type='WaveformCollate', wav_key='waveform'),
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='label'),
        dict(type='CharCollate'),
    ],
)
solution = dict(
    model_type='AuLlmPretrainModel',

    # text content
    use_continuous_rep=True,
    vocab_size=9759,

    # speech encoder relative
    fix_w2v_model=False,
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
    freeze_finetune_updates=0,
    # freeze_finetune_updates=4000,
    freeze_encoder_layers=-1,
    average_top_k_layers=0,
    loss_beta=0,
    loss_scale=-1,
    ema_transformer_only=True,

    # criterion type
    # criterion_type='SpokenLmLoss',

    # lm (for discrete)
    # vocab_size=8192, # 和codebook一样大
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

    # lm (here used)
    self_attention_type='dot_product',
    num_heads=8,
    hidden_size=640,
    filter_size=2560,
    ffn_layer='dense_relu_dense',
    layer_preprocess_sequence='n',
    layer_postprocess_sequence='da',
    norm_type='layer',
    norm_epsilon=0.000001,
    proximity_bias=True,
    num_decoder_layers=2,
    decoder_self_attention_type='dot_product',
    layer_prepostprocess_dropout=0.1,
    relu_dropout=0.0,
    extra_decode_length=20,

    # LLM
    use_llm=True,
    prompt="请识别语音中的文本内容。",
    am_hidden_size=1024,
    llm_hidden_size=4096,
    llm_dir="./huggingface_models/bloomz-7b1-mt",
    unfreeze_llm=False,
    unfreezed_llm_layers="[0,29]",

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
    max_iters=40000,
    lr_scheduler=dict(
        by_epoch=False,
        # policy="LayerwiseTriStage",
        policy="TriStage",
        lr=1e-3,
        phase_ratio="[0.1,0.4,0.5]",
        init_lr_scale=0.01,
        final_lr_scale=0.05,
        max_train_steps=40000,
        # lr_prefix_multiplier_list=[('w2v_model.', 0.03)],
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
    interval=100000000,
    # interval=1600,
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='AdamW',
    # lr=[('w2v_model.', 3e-5), 1e-3],
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
