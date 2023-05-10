project = "multich_dfsmn_rnnt"
runner = 'MCRunner'
#_base_ = [
#    '../_base_/datasets/ag20kh_lzl.py',
#]
# increment to add and override args.
data = dict(
    # data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/1wh_data_arnold_trial/',
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    # add ce sy+ctc label
    train_file_list = '["train_sub{}".format(i) for i in range(1024)]',
    valid_file_list = '["cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='length',
    batch_means_tokens=1,
    max_batch_size=10240,
    shuffle=True,
    drop_last=False,
    #chunk_size=20,
    #use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    prefetch_worker_num=1,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='MultiChannelModuleSimu', simulator_config_path='./configs/se/simulator_config_jointtrain.py'),
        # dict(type='GetRirData'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        # dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='MultiChannelModuleSimu', simulator_config_path='./configs/se/simulator_config_jointtrain.py'),
        # dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        # dict(type='FbankCollate', fbank_dim=80),
        dict(type='MultiChannelWaveformCollate'),
        # dict(type='RirDataCollate'),
        dict(type='CharCollate', need_adapt_target=True),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        # dict(type='FbankCollate', fbank_dim=80),
    ],
    # do after batch_transform and before model.forward.
    # It's in prefetch thread of main preprocess.
    # the input and output are batch_data.
    train_device_transform=[
        # dict(type='GPURir', mic_num=1, simu_ratio=0.5, speech_rir_ratio=1.0, noise_ratio=1.0, noise_rir_ratio=1.0),
        dict(type='SpeedKaldiFbank', out_key='src', dither=1.0, device='cuda', out_numpy=False),
        dict(type='CMVN', key='src'),
        dict(type='TimeMask', key='src', time_mask_size=20, time_mask_num=10,
             replace_with_zero=False),
        dict(type='FreqMask', key='src', freq_mask_size=27, freq_mask_num=1,
             replace_with_zero=False),
    ],
)
solution = dict(
    type='base_mc_rnnt_solution',
    rnnt_type='BaseMCRnntModel',
    # multi-channel front_end
    mc_front_end_type='MulChFrontend',
    mic_num=6,
    fft_len=512,
    n_mel=80,
    frame_len=400,
    frame_shift=160,
    wav_key='mc_waveform',
    fbank_key='src',
    # front_end
    front_end_type='VGGFrontEnd',
    clamp_residual=0,
    input_concat_size=8,
    downsampling_size=4,
    fsmn_left_kernel_size=40,
    fsmn_right_kernel_size=1,
    fsmn_dilation=1,
    # backbone
    acoustic_backbone_type='DFSMNBackboneLN',
    backbone_topology='[[20,20,2]]*30',
    backbone_layer_gap=100,
    dfsmn_bn_flag=1,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    self_attn_heads=8,
    self_attn_dropout=0.0,
    dropout=0.0,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=1,
    self_attn_activation_fn='relu',
    # head
    head_type='RnntBaseHead',
    head_hidden_size=768,
    head_lowrank_size=512,
    layer_norm_after=True,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=512,
    predictor_lstm_hidden_size=2048,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.0,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=768,
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
    # inference
    cer_blank_scale=0.2,
    # NOTE: if do finetune, should close xavier_init!
    xavier_init=False

)
# runtime settings
work_dir = './dfsmn_ag20k_opt'
# training and testing settings
train = dict(
    # resume='latest.pth',
    # resume_optimizer=False,
    # resume_progress=False,
    # dolphin will try to resume from these path in order:
    #   1. ${train.out_dir}/${train.resume}
    #   2. ${train.resume_hdfs_chkpt}
    #   3. ${train.remote_save_root}/${train.out_dir}/${train.resume}
    # note: ${train.out_dir} is ${train.save_root}/${train.save_dir}/${train.save_name}/checkpoints,
    # if it's not set in config file.
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/input_10k_pretrain/dfsmnln_offline_pretrain/dfsmnln_online_pretrain/checkpoints/step_180000.pth',
    max_epochs=90,
    max_iters=2000000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=2000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.8,
        step=[4 + i for i in range(100)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    save_root='./dfsmnln_offline',
    save_dir='dfsmnln_offline',
    save_name='dfsmnln_offline',
    amp_level='O0',
    )
valid = dict(
    interval=10000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='lark_chinese|lark_mixed|aishell_2|ceo_external|smart_dog|meeting',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    rnnt_temperature=1.0,
    use_batch_beam=True,
    use_fsmn_beam=False
    #remote_stat_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/",
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
    eps=1e-8,
    weight_decay=0.05,
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
    interval=1,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
