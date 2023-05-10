# 4 GPUs: trial id: 2889262
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/overlap_detection/v1_8k/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/overlap_detection/v2_8k/',
    ],
    train_file_list=[
        '["train_sub{}".format(i) for i in range(64)]',
        '["train_sub{}".format(i) for i in range(80)]',
    ],
    valid_file_list=[
        '["cv"]',
        '["cv"]',
    ],
    meta_file="meta",
    fbank_dim=60,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    max_batch_scale=3,
    prefetch_worker_num=4,
    next_retry=1000,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='CMVN', key='fbank'),
        dict(
            type='DynamicTimeMask',
            time_mask_size=20,
            time_mask_block=60,
            replace_with_zero=False,
            inplace=True,
            time_mask_num_dither=1
        ),
        dict(
            type='FreqMask',
            freq_mask_size=27,
            freq_mask_num=1,
            replace_with_zero=False,
            inplace=True
        )
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='overlap'),
        dict(type='FbankCollate', fbank_dim=60),
        dict(type='TimeRange2FrameTarget', downsample=320),
        dict(type='BinaryTargetCollect', key='overlap', out_key='target'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='overlap'),
        dict(type='FbankCollate', fbank_dim=60),
        dict(type='BinaryTargetCollect', key='overlap', out_key='target'),
    ])
project = 'overlap_detection'
runner = 'OverlapDetectionRunner'
solution = dict(
    type='base_overlap_detection_solution',
    model_type='BaseOverlapDetectionModel',
    # front_end
    front_end_type='VGGPosFrontEnd',
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='OfflineTransformerBackbone',
    backbone_topology='[[-1,-1,1]]*16',
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    self_attn_heads=8,
    self_attn_dropout=0.2,
    dropout=0.2,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=1,
    self_attn_activation_fn='relu',
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # frame-level MTL
    frame_mtl=True,
    frame_mtl_scale=1.0,
    frame_soft_mask=False,
    # pool
    jointer_hidden_size=640,
    backbone_pool_type='MaxPoolLayer',
    # criterion
    tgt_size=2,
    criterion_type='Xentropy',
    label_smooth_factor=0.1,
    # avoid nan/inf
    backbone_clamp_inf=True,
    mtl_clamp=True,
    head_clamp=True,
)
work_dir = './demo'
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    max_iters=400000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.9,
    ),
    checkpoint_config=dict(interval=10000, max_keep_ckpts=100),
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/',
    save_root='./demo',
    save_dir='overlap_detection',
    save_name='overlap_detection_baseline',
    amp_level='O1')
valid = dict(interval=10000)
inference = dict(
    resume='latest.pth',
    test_sets='cv',
    prob_thres='[0.3,0.8,0.05]',
    min_frame='[1,2,3,4,5]',
    sen_infer=1)
optimizer = dict(
    type='FusedAdam',
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=50.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.75,
        warmup_steps=5000,
        average_sync=True,
        warmup_sync=True,
        use_nesterov=True))
log_level = 'INFO'
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
