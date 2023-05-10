data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/caijing/caijing_3rd/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dongchedi_3rd/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/bb_online_2nd',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dianshang_1st',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dianshang_2nd',
    ],
    train_file_list=[
        '["train_sub{}".format(i) for i in range(64)]',  # caijing_3rd
        '["train_sub{}".format(i) for i in range(64)]',  # dongchedi_3rd
        '["train_sub{}".format(i) for i in range(64)]', # bb_online_2nd
        '["train_sub{}".format(i) for i in range(64)]', # dianshang_1st
        '["train_sub{}".format(i) for i in range(32)]', # dianshang_2nd
    ],
    valid_file_list=[
        '["cv"]',  # caijing_3rd
        '["cv"]',  # dongchedi_3rd
        '["cv"]',  # bb_online_2nd
        '["cv"]',  # dianshang_1st
        '["cv"]',  # dianshang_2nd
    ],
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/test',
    meta_file="../../tele/smb_gogokid/meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    prefetch_worker_num=5,
    prefetch_torch_thread=4,
    next_retry=1024,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,2250,2500,2750,3000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    max_batch_scale=5,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', max_len=30.),
        dict(type='AppendSilence', dur=0.2,
             sil_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/xiansuotong_padding_silence',
             skip_enough=True, is_eos_sil_frame=True,
             sil_snr_max=60, sil_snr_min=40,
             sil_prefix='train_sub', sil_shards=16),
        dict(type='SpeedKaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60,
             replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=False, inplace=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type="AppendFrames", frame=20),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
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
    ])
project = 'tel_rnnt'
runner = 'UniversalRNNTRunner'
solution = dict(
    model_type='RnntUniversalModel',
    # front_end
    front_end_type='Conv2dPooling4',
    input_concat_size=4,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='ConformerBackbone',
    conformer_dual_mode=True,
    conformer_normalize_before=True,
    conformer_attention_heads=8,
    conformer_mask_topology='[[128,128]]*12',
    stream_conformer_mask_topology='[[128,0]]*12',
    conformer_linear_units=2048,
    conformer_num_blocks=12,
    conformer_dropout_rate=0.1,
    conformer_positional_dropout_rate=0.1,
    conformer_attention_dropout_rate=0.1,
    conformer_positionwise_layer_type='linear',
    conformer_activation_fn='gelu',
    conformer_positionwise_conv_kernel_size=1,
    conformer_macaron_style=1,
    conformer_pos_enc_layer_type='abs_pos',
    conformer_selfattention_layer_type='selfattn',
    conformer_layer_order='conv_before_mhsa',
    conformer_use_cnn_module=1,
    conformer_cnn_module='ConvolutionModule',
    conformer_cnn_module_kernel='15,15',
    conformer_cnn_norm_type='layer_norm',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_memory_size=512,
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.1,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=968,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    # criterion
    criterion_type='RnntUniversalCE',
    cer_update_freq=500,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    #mtl_type='ctc',
    #mtl_head='HybridCEHead',
    backbone_clamp_inf=True,
    mtl_clamp=True,
    head_clamp=True,
    #fast_emit 1.001 到 1.01可调
    fast_emit=1.004,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    resume_progress=True,
    resume_lr_scheduler=True,
    max_epochs=90,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.7,
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    save_root='./demo',
    save_dir='dual_mode_ct',
    save_name='v1',
    amp_level='O1',
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    test_sets='bytebot_test_set_20210318_18867',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    use_batch_beam=True,
    )
nnlm = dict(
    lm_type="",
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=50,
        norm_type=2,
        ),
    max_grad_clip=50.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=5000,
        warmup_sync=True,
        average_sync=True,
        use_nesterov=True,
    )
)
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=500,
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
