project = "chinglish_rnnt"
runner = 'RNNTMBRRunner'
_base_ = [
    '../_base_/datasets/haitian_chinglish.py',
]

# increment to add and override args.
data = dict(
    chunk_size=200,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_scale=5,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='SpaceAdd'),
        dict(type='BPE', use_eos=False, skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>']),
        dict(type='CMVN', key='fbank'),
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60, replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=False, inplace=True),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='SpaceAdd'),
        dict(type='BPE', use_eos=False, skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>']),
        dict(type='CMVN', key='fbank'),
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
    ])
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    fsmn_left_kernel_size=20,
    fsmn_right_kernel_size=1,
    fsmn_dilation=2,
    acoustic_backbone_type='LSTMPBackbone',
    backbone_topology='[[-1,-1,1]]*30',
    backbone_layer_gap=100,
    dfsmn_bn_flag=1,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=0,
    backbone_residual=1,
    backbone_mask=1,
    self_attn_heads=8,
    self_attn_dropout=0.1,
    dropout=0.3,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=1,
    self_attn_activation_fn='relu',
    head_type='RnntSimpleHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    predictor_type='LSTMReLUPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.3,
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    adaptive_head_size=1056,
    adaptive_tail_size=150,
    adaptive_tail_groups=40,
    # criterion
    criterion_type='RnntAdaptiveMBR',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    mtl_type='ctc',
    mtl_head='HybridCEHead',
    # nbest generation
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.05,
    language='en',
    rnnt_temperature=1.0,
    mbr_beam_size=10,
    use_batch_beam=True,
    one_step_expand=True,
    # mbr criterion
    use_wer=False,
    rnnt_regular_factor=0.0,)
work_dir = './demo'
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=2,
    max_iters=100000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.8,
        step=[100000 + i * 5000 for i in range(140)]),
    checkpoint_config=dict(interval=1000, max_keep_ckpts=-1),
    curriculum_end_epoch=0,
    curriculum_length=520,
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/',
    #resume_hdfs_chkpt='',
    save_root='./demo',
    save_dir='tel_dy_hotsoon_80kh_rnnt',
    save_name='rnnt_transformer_base',
    amp_level='O1',)
valid = dict(interval=1000)
inference = dict(
    bucket_schedule='100000',
    test_sets='80dim_subshard.ez_chinglish_20200113_1554',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.0,
    language='en',
    nbest_out=False,
    filter_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>'],
    #remote_stat_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/",
    )

nnlm = dict(lm_type='')
optimizer = dict(
    type='FusedAdam',
    lr=1e-6,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=1000,
        average_params=True,
        use_nesterov=True))
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=10,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
