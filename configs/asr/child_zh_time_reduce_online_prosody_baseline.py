# pylint: disable-all

project = "tel_rnnt"
runner = 'RNNTRunner'
# increment to add and override args.
data = dict(
    data_root=[
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/child_zh_asr/Datatang_child_2000h_new',
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/child_zh_asr/Huiting_child_1000h_new',
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/child_zh_asr/adults_15kh'],
    # only use 3000 hours of speech data in adults_15kh dataset, that is,
    # train_sub index ranges from 0 to 64.
    train_file_list = ['["train_sub{}".format(i) for i in range(200)]',
                       '["train_sub{}".format(i) for i in range(50)]',
                       '["train_sub{}".format(i) for i in range(64)]'],
    valid_data_root=[
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/child_zh_asr/Datatang_child_2000h_new',
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/child_zh_asr/Huiting_child_1000h_new'],
    valid_file_list=['["cv_dev"]', '["cv_dev"]'],
    tgt_dict_dir='total_bpe.dict',
    bpe_code='total.code',
    cmvn_file='fbank_80dim_cmvn.kaldi',
    meta_file="meta",
    fbank_dim=80,
    fetch_block_size=200,
    bucket_schedule=
    '50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=15360,
    shuffle=True,
    drop_last=False,

    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='ModifyProsody', sample_rate=16000, tempo_factor=0.8, \
                speed_factor=1.25, adjust_prob=1.0),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='PunctuationFilter', key='label'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='PunctuationFilter', key='label'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80, reserve_frames=1),
        dict(type='CharCollate', need_adapt_target=True),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80)
        ],
)
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
    # front_end
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='LSTMPBackbone',
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    backbone_layer_num=5,
    backbone_bilstm=False,
    backbone_residual=True,
    dropout=0.1,
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=2048,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
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
    cer_update_freq=50,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    #export onnx
    onnx_stack_frame=320,
)
# runtime settings
work_dir = './exp'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    max_iters=800000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='exp',
        warmup_iters=1000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.97,
        step=[40000 + i * 5000 for i in range(160)]
        ),
    checkpoint_config=dict(
        interval=20000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root= 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/saved_models',
    save_root='./exp',
    save_dir='child_zh_asr',
    save_name='online_prosody_baseline_exp',
    amp_level='O1',
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    test_sets='datatang_huiting_testsets',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    # remote_stat_dir= "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/inference_results/child_zh_asr",
    )
nnlm = dict(
    lm_type="",
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax',\
#                  'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']

optimizer = dict(
    type='FusedAdam',
    lr=2e-5,
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
    max_grad_clip=10.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.9,
        warmup_steps=5000,
        use_nesterov=True
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
