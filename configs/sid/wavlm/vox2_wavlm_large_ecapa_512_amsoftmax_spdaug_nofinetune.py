project = "sid_training"
runner = 'BaseSidRunner'

# data
data = dict(
    train_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/wav_pkg/voxceleb2_train',
    ],
    train_file_list=[
        '["train{}".format(i) for i in range(16)]',
    ],
    meta_file=[
        'meta',
    ],
    valid_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/wav_pkg/voxceleb2_train',
    ],
    valid_file_list=[
        '["valid{}".format(i) for i in range(16)]',
    ],
    eval_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/wav_pkg/voxceleb1_test',
    ],
    eval_file_list=[
        '["eval0"]',
    ],
    trials=[
        'trials2',
    ],

    fbank_dim=1024,
    # Will consume more memory
    prefetch_worker_num=5,
    global_shuffle=1,
    prefetch_block_num=1,

    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', in_key='wav', min_len=0.01, max_len=10000),
        dict(
            type='SpeedPerturbation',
            speed_rate_list=[0.9, 1.1],
            p=2.0/3,
            cls_relabel=True),
        dict(type='PreSelAug', aug_probs={'no_aug':1, 'music':1, 'speech':1, 'noise':1, 'reverb':1}),
        dict(type='KaldiAddNoise', noise_prefix='music', noise_type='music', noise_snr=[10,7,5], ground_mode='background'),
        dict(type='KaldiAddNoise', noise_prefix='speech', noise_type='speech', noise_snr=[19,17,15,13,11], noise_times=[3,4,5,6,7],  ground_mode='background'),
        dict(type='KaldiAddNoise', noise_snr=[5, 10], ground_mode='foreground', noise_interval=1),
        dict(type='KaldiAddRir'),
        dict(
            type='SelfSplicing',
            max_len=400,
            random_clip=True,
            fake_vad=True),
        dict(type='CalculateFrameLength'),
        dict(type='DictTrans', in_key='spk', out_key='spk', relabel=True)
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', in_key='wav', min_len=0.01, max_len=10000),
        dict(
            type='SelfSplicing',
            max_len=400,
            random_clip=True,
            fake_vad=True),
        dict(type='CalculateFrameLength'),
        dict(type='DictTrans', in_key='spk', out_key='spk')
    ],
    eval_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', in_key='wav', min_len=0.01, max_len=10000),
        dict(type='CalculateFrameLength'),
        dict(type='EerFilter')
    ],

    train_batch_transform=[
        dict(
            type='SidCollate',
            src_key='waveform',
            sample_rate=16000,
            random_clip=True,
            min_len=200,
            max_len=400,
            input_type='wav')
    ],
    valid_batch_transform=[
        dict(
            type='SidCollate',
            src_key='waveform',
            sample_rate=16000,
            min_len=300,
            max_len=300,
            random_clip=True,
            input_type='wav')
    ],
    eval_batch_transform=[
        dict(
            type='EerCollate',
            src_key='waveform',
            input_type='wav',
            vad_key='no_vad_required')
    ],

    train_device_transform=[
        dict(type='VADFilter', key='feature', vad_key='vad', input_type='wav')
    ],
    valid_device_transform=[
        dict(type='VADFilter', key='feature', vad_key='vad', input_type='wav')
    ],
    eval_device_transform=[
        dict(type='VADFilter', key='feature', vad_key='vad', input_type='wav')
    ],

    # used for training
    batch_size_per_class=1,
    batch_class_num=64,
    # used for validation
    max_batch_size=128,
    batch_means_tokens=False,
    drop_last=False,
)

# solution
solution = dict(
    type='wavlm_sid_solution',

    transformer_model="RelTransformerLayer",
    attention_model="PosMultiHeadAttention",
    feature_grad_mult=0,
    freeze_upstream=True,
    fused_transformer=False,

    normalize=True,
    apply_mask=True,
    conv_feature_layers="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    extractor_mode="layer_norm",
    conv_bias=False,
    quantize_input=False,
    encoder_embed_dim=1024,
    mask_prob=0.8,
    mask_selection="static",
    mask_other=0.0,
    mask_length=10,
    no_mask_overlap=False,
    mask_min_space=1,
    mask_channel_prob=0.0,
    mask_channel_selection="static",
    mask_channel_other=0.0,
    mask_channel_length=10,
    no_mask_channel_overlap=False,
    mask_channel_min_space=1,
    dropout_input=0.0,
    dropout=0.0,
    conv_pos=128,
    conv_pos_groups=16,
    encoder_layers=24,
    layer_norm_first=True,
    encoder_layerdrop=0.00,
    encoder_ffn_embed_dim=4096,
    encoder_attention_heads=16,
    attention_dropout=0.0,
    activation_dropout=0.0,
    pretrained_activation_fn="gelu",
    use_wavlm_encoder=True,
    relative_position_embedding=True,
    num_buckets=320,
    max_distance=800,
    gru_rel_pos=True,

    acoustic_backbone_type='ECAPABackbone',
    channels='[512, 512, 512, 512, 1536]',
    block_type="full",
    kernel_sizes='[5, 3, 3, 3, 1]',
    dilations='[1, 2, 3, 4, 1]',
    activation_fn='relu',
    res2net_scale=8,
    se_channels=128,
    normalization_fn='batch_norm',
    normalization_after=True,
    batchnorm_momentum=0.1,
    batchnorm_eps=1e-5,
    padding_mode='reflect',

    # Pooling layer
    pooling_layer_type='AttentivePooling',
    attentive_bottleneck_dim=128,
    norm_after_pooling=True,
    use_global_context=True,

    # Segment-level network
    segment_network_type="MLPSoftmax",
    # For utt-level network, the topology is [n_nodes].
    segment_topology='[192]',
    segment_use_conv=True,
    segment_last_layer_use_norm=False,
    segment_last_layer_use_nonlinear=False,

    # Loss function
    softmax_type='additive_margin_softmax',
    softmax_renorm=30,
    softmax_margin=0.2,
    softmax_lambda_base=0,
    softmax_lambda_min=0,
    softmax_lambda_gamma=1,
    softmax_lambda_power=1,
    softmax_margin_by_epoch=True,
    softmax_margin_steps=[3+i for i in range(10)],

    # Criterion
    criterion_type="Xentropy",
    label_smooth_factor=0.0,

    # EER evaluation
    used_epoch_eval=True,
    embedding_dim=192,
    embedding_layer='utt_output'
)

# runtime settings
work_dir = './vox_cn'

# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-3,
)

# Optimizer Hook Configuration
optimizer_config = dict(
    max_grad_clip=0.0,
    grad_clip=0,
    bmuf_config=0
)

# training and testing settings
train = dict(
    iters_per_epoch=10000,
    max_epochs=24,
    max_iters=100000000000,
    final_lr=1e-9,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=30000,
        warmup_ratio=1e-5,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[13 + i for i in range(10)]
        ),

    checkpoint_config=dict(
        interval=10000000000, # steps
        max_keep_ckpts=10,
    ),

    save_root='wavlm',
    save_dir='vox2',
    save_name='wavlm_large_ecapa_512_nofinetune',
    amp_level='O1',

    resume_pretrain_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/wavlm/WavLM-Large.pt',
    resume='latest.pth',
    save_after_epoch=True,
)

valid = dict(
)

# training and testing settings
log_level = 'INFO'
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
