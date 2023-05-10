project = "tdnn_rnnt"
runner = 'RNNTRunner'
data = dict(
    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/general_18000h',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_54000h',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_6wh',
    ],
    train_file_list=[
        '["sub_{}".format(i) for i in range(443)]',
        '["sub_{}".format(i) for i in range(1149)]',
        '["sub_{}".format(i) for i in range(1332)]',
    ],
    valid_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/'
    ],
    valid_file_list=[
        '["input/cv", "video/cv", "other/cv", "smb_gogokid/cv", "feiyu/cv"]'
    ],
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/Datatang_child_2000h',
    meta_file="meta",
    fbank_dim=80,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=24000,
    max_batch_scale=5,
    shuffle=True,
    drop_last=False,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='DynamicTimeMask',
             time_mask_size=20,
             time_mask_block=60,
             replace_with_zero=False,
             inplace=True,
             time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             inplace=True, replace_with_zero=False),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    acoustic_backbone_type='TDNNLSTMPBackbone',
    tdnn_backbone_topology='[[2,2,0]]*2 + [[2,2,1]]*3',
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=False,
    backbone_residual=True,
    dropout=0.1,
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    layer_norm_after=True,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=512,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.0,
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
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=True,
    # NOTE: if do finetune, should close xavier_init!
    xavier_init=False
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    max_iters=1200000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.7,
        step=[200000 + i * 10000 for i in range(140)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=20,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    #remote_save_root='',
    save_root='./demo',
    save_dir='tdnn_lstm_rnnt',
    save_name='tdnn_lstm_rnnt_240ms',
    amp_level='O1'
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    # --data.data_root hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/ag20k_new
    # test_sets='aishell_2',
    # --data.data_root hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/
    test_sets='test/ev_teacher_homework_comment_20200609_730',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    blank_thresh=1.0,
    len_penalty_scale=0.01,
    # remote_stat_dir="hdfs://haruna/xxxx",
    )
nnlm = dict(
    lm_type="",
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=20,
        norm_type=2,
        ),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=5000,
        use_nesterov=True,
        average_sync=True,
    )
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=200,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
