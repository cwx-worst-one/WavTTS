data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dongchedi_2nd/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/caijing/caijing_1st/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/caijing/caijing_2nd/',
    ],
    train_file_list=[
        '["train_sub{}".format(i) for i in range(64)]',
        '["train_sub{}".format(i) for i in range(20)]',
        '["train_sub{}".format(i) for i in range(24)]',
    ],
    valid_file_list=[
        '["cv"]',
        '["cv"]',
        '["cv"]',
    ],
    meta_file="../meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,10000',
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
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='AppendFrames', frame=12, value=0),
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
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='AppendFrames', frame=12, value=0),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80, reserve_frames=1),
        dict(type='CharCollate', need_adapt_target=True),
        dict(type='PreCharCollate')
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80)
    ])
project = 'tel_rnnt'
runner = 'RNNTRunner'
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    acoustic_backbone_type='LSTMPBackbone',
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=False,
    backbone_residual=True,
    dropout=0.1,
    head_type='RnntSimpleHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    predictor_type='LSTMReLUPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    adaptive_head_size=968,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    criterion_type='RnntAdaptiveCE',
    # fast emit
    fast_emit=1.004,
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    mtl_type='ctc',
    mtl_head='HybridCEHead',
    backbone_mask=1)
work_dir = './demo'
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    max_iters=800000,
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
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/sunjiaming.s/f_e',
    save_root='./demo',
    save_dir='f_e',
    save_name='tel_rnnt_lstm_fastemit_4kh',
    amp_level='O1')
valid = dict(interval=10000)
inference = dict(
    resume='latest.pth',
    test_sets='../test/bytebot_test_set_20210318_18867',
    beam_size=10,
    lm_path='',
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    use_batch_beam=True)
nnlm = dict(lm_type='')
optimizer = dict(
    type='AdamW',
    lr=5e-5,
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
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=200,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
