data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/smb_gogokid/',
    train_file_list='["train_sub{}".format(i) for i in range(128)]',
    valid_file_list='["cv"]',
    meta_file="../meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=7680,
    max_batch_scale=10,
    prefetch_worker_num=4,
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
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60,
             replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             replace_with_zero=False, inplace=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', need_adapt_target=True),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
    next_retry=1000,
    )
project = 'tel_rnnt'
runner = 'UniversalRNNTRunner'
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='RnntUniversalModel',
    front_end_type='VGGPosFrontEnd',
    input_concat_size=8,
    downsampling_size=4,
    acoustic_backbone_type='OfflineTransformerBackbone',
    backbone_topology='[[-1,0,1]]*20',
    backbone_layer_gap=100,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    self_attn_heads=8,
    self_attn_dropout=0.1,
    dropout=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=1,
    self_attn_activation_fn='relu',
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    predictor_type='LSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=2048,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    adaptive_head_size=968,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    criterion_type='RnntUniversalCE',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    xavier_init=False,
    # solve nan
    backbone_clamp_inf=True,
    mtl_clamp=True,
    head_clamp=True,
    # universal config
    causal_transformer=True,
    distillation_scheduler=0.00001,
    symmetric=0, #1 for use symmetric distillation
    peak=0, #1 for add spike loss and weight
    frame_shift=0,
    inference_mode=2, # 2 for streaming when inference
    )
work_dir = './demo'
train = dict(
    check_loss=True,
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
        policy='step',
        gamma=0.92,
        step=[150000 + i * 5000 for i in range(80)],
        key='cer'
        ),
    checkpoint_config=dict(interval=10000, max_keep_ckpts=40),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/sunjingyu/saved_models/dolphin',
    save_root='./demo',
    save_dir='universal_transformer',
    save_name='universal_transformer_base',
    amp_level='O1',# set O0 when infernce in universal transformer
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,100000',
    test_sets='../test/bytebot_test_set_20210318_18867',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    fst_path="",
    fst_weight=0.7,
    blk_scale=1.0,
    rnnt_temperature=1.0,
    len_penalty_scale=0.01,
    #remote_stat_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/",
    use_batch_beam=True,
    )
nnlm = dict(lm_type='')
optimizer = dict(
    type='FusedAdam',
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.75,
        warmup_steps=20000,
        warmup_sync=True,
        average_sync=True,
        use_nesterov=True,
    )
)
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')
    ]
)
