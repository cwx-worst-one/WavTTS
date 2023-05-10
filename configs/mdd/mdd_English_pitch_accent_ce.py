project = "mdd_stress_prediction"
runner = 'BaseMddRunner'

# increment to add and override args.
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xiaohai/data/dolphin/stress/English_TTS_update/',
    # add ce sy+ctc label
    train_file_list = '["f0_mcc_sp_mel_duration_phrase_accent_pitch_accent-train-{}".format(i) for i in range(10)]',
    valid_file_list = '["f0_mcc_sp_mel_duration_phrase_accent_pitch_accent-valid-{}".format(i) for i in range(1)]',
    eval_file_list  = '["f0_mcc_sp_mel_duration_phrase_accent_pitch_accent-eval_preserve-{}".format(i) for i in range(1)]',
    eval_file_list1 = '["f0_mcc_sp_mel_duration_phrase_accent_pitch_accent-eval-{}".format(i) for i in range(4)]',

    chunk_size=20,
    fetch_block_size=200,
    bucket_schedule='100,200,300,400,500,600,700,800,900,1000,1200,1350,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='src',
    batch_means_tokens=0,
    max_batch_size=49,
    # 是否保证每次实验获取的数据完全一致，用于支持模型可复现；稍微影响速度
    deterministic=False,
    # 是否进行全局shuffle
    global_shuffle=True,
    shuffle=True,
    drop_last=False,

    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='CMNutt', key='f0', dim=-1),
        dict(type='CMNutt', key='mcc', dim=0),
        dict(type='MergeStressFeats', in_key=['sp', 'f0', 'mcc'], out_key='src'),
        dict(type='AlignStressFeature', key='src', ce_key='pitch_accent'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='CMNutt', key='f0', dim=-1),
        dict(type='CMNutt', key='mcc', dim=0),
        dict(type='MergeStressFeats', in_key=['sp', 'f0', 'mcc'], out_key='src'),
        dict(type='AlignStressFeature', key='src', ce_key='pitch_accent'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='StressFeatCollate', feat_dim=260),
        dict(type='CeLabelCollate', key='pitch_accent')
    ],

    inference_item_transform=[
        dict(type='PickleParser'),
        dict(type='CMNutt', key='f0', dim=-1),
        dict(type='CMNutt', key='mcc', dim=0),
        dict(type='MergeStressFeats', in_key=['sp', 'f0', 'mcc'], out_key='src'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='StressFeatCollate', feat_dim=260),
    ],
)
solution = dict(
    type='base_mdd_solution',
    mdd_model_type='StressClassificationNet',
    encoder_type='FrameCMLEncoder',
    input_dim=260,
    out_dim=7,
    hidden_dim=64,

    # cnn config
    num_cnn=5,
    kernel_size=7,
    stride=1,

    # selfattention config
    multiheadaAttention_layer=False,
    num_heads=4,

    # lstm config
    num_lstm_hidden=1,
    bidirectional=True,

    # criterion
    criterion_type='Xentropy',
    cer_update_freq=200,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=False,
    # pretrain or not
    pretrain_encoder=True,

    # only use speech part for loss calculation
    speech_loss=False,

    # evaluation
    evaluation='frame',
    eval_seg='oveall'
)
# runtime settings
work_dir = './dfsmn_english_23kh_pretrain'
# training and testing settings
train = dict(
    #resume='latest.pth',
    #resume_optimizer=True,
    #resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/shenchen/dolphin/MDD/test/dfsmnln_online_pretrain/dfsmnln_online_pretrain/checkpoints/step_1000.pth',
    max_epochs=90,
    max_iters=160000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup=None,
        warmup_iters=0,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.8,
        step=[3 + i for i in range(50)]
        ),
    checkpoint_config=dict(
        interval=1000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xiaohai/arnold_dolphin/MDD/stress_prediction',
    save_root='./stress_prediction',
    save_dir='English_TTS',
    save_name='pitch_accent_7_classes_v2',
    amp_level='O1',
    )
valid = dict(
    interval=1000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    # eval_remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xiaohai/arnold_dolphin/MDD/stress_prediction/pitch_accent_7_classes_eval',
    eval_save_root='./stress_eval',
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=2e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-3,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=50,
        norm_type=2,
        ),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.9,
        warmup_steps=5000,
        use_nesterov=True,
    )
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=100,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
