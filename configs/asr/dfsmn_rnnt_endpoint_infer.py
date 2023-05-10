project = "dfsmn_rnnt"
runner = 'RNNTRunner'
#_base_ = [
#    '../_base_/datasets/ag20kh_lzl.py',
#]
# increment to add and override args.
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/aiot_12_9_data',
    # add ce sy+ctc label
    train_file_list = '["sub_{}".format(i) for i in range(2048)]',
    valid_file_list = '["fbank_80dim_cv"]',
    meta_data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/endpoints_data',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/endpoints_data',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta",
    fbank_dim=80,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    shuffle=True,
    drop_last=False, 
    chunk_size=20,
    max_batch_scale=15,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='LengthFilter',key='fbank', max_len=2000),
        dict(type='BadNumeralsFilter', key1='uttid', key2='label',
             filter_set='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/filter_list/general_1w8_uttid.txt'),
        dict(type='CodeSwitchTagAdd', key='label', add_token='^',
             special_pattern_list=['<[a-z]+>', '<[A-Z]+>']),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10,
             replace_with_zero=False),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             replace_with_zero=False),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='CodeSwitchTagAdd', key='label', add_token='^',
             special_pattern_list=['<[a-z]+>', '<[A-Z]+>']),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
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
)
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
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
    backbone_topology='[[40,1,1]]*10+[[20,0,2]]*20',
    backbone_layer_gap=100,
    dfsmn_bn_flag=1,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    self_attn_heads=8,
    self_attn_dropout=0.1,
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
    adaptive_tail_small=True,
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
    # resume_optimizer=True,
    # dolphin will try to resume from these path in order:
    #   1. ${train.out_dir}/${train.resume}
    #   2. ${train.resume_hdfs_chkpt}
    #   3. ${train.remote_save_root}/${train.out_dir}/${train.resume}
    # note: ${train.out_dir} is ${train.save_root}/${train.save_dir}/${train.save_name}/checkpoints,
    # if it's not set in config file.
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/pretrain_models/pretrain_16w/bytespeech_endpoints_model.pth',
    max_epochs=90,
    max_iters=99999999999,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=2000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[3 + i for i in range(140)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/dbg_for_pretrain',
    save_root='./endpoints',
    save_dir='endpoints',
    save_name='endpoints',
    amp_level='O1',
    )
valid = dict(
    interval=10000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    #test_sets='lark_chinese|lark_mixed|aishell_2|ceo_external|smart_dog|meeting', 
    # test_sets='aishell_2|aishell_2019a|aishell_2019c|ceo_external|h_test_kuaisu|h_test_mobile|h_test_putong|lark_chinese|lark_meeting_remove_zeros_dither0|lark_mixed|learning|meeting|primewords_10h|smart_dog|st_cmds_10h|thchs|toutiao_school',
    test_sets='h_real',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    rnnt_temperature=0.8,
    use_batch_beam=True,
    use_aiot_beam=True,
    temp_before_softmax=False,
    blk_scale_add=False,
    eos_scale=1.0,
    endpoints_indexs=None,
    endpoints_scales=None,
    keep_non_proun_tokens=['，', '。', '！', '？', ',', '.', '!', '?'],
    # NOTE one_step_expand only use for use_aiot_beam=False
    filter_list=['^']
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
    interval=100,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
