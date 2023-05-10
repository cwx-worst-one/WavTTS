# please use 8 GPU and train more than 400k steps
# training trial: loss/nll_loss/cer on cv: 7.7627/7.3694/7.60 (400k)
# https://cloud.bytedance.net/arnold/job/9213/task/875527/trial/2204866
# inference trial: CER 20.05
# https://cloud.bytedance.net/arnold/job/9213/task/875527/trial/2217128
project = "tel_rnnt"
runner = 'RNNTRunner'
_base_ = [
    '../_base_/datasets/tel_dy_hostoon.py',
]
# increment to add and override args.
data = dict(
    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    max_batch_size=20480,
    max_batch_scale=10,
    prefetch_worker_num=5,
    next_retry=1024,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='LengthFilter',max_len=2000),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60,
             replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             replace_with_zero=False, inplace=True)
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
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
    front_end_type='VGGPosFrontEnd',
    input_concat_size=8,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='OfflineTransformerBackbone',
    backbone_topology='[[-1,-1,1]]*30',
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
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    mtl_type='ctc',
    mtl_head='HybridCEHead',
    backbone_pool_type='Conv1dTimeReduce',
    # avoid nan/inf
    backbone_clamp_inf=True,
    mtl_clamp=True,
    head_clamp=True,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    check_loss=True,
    resume='latest.pth', # local checkpoint file to resume.
    resume_optimizer=True,
    # dolphin will try to resume from these path in order:
    #   1. ${train.out_dir}/${train.resume}
    #   2. ${train.resume_hdfs_chkpt}
    #   3. ${train.remote_save_root}/${train.out_dir}/${train.resume}
    # note: ${train.out_dir} is ${train.save_root}/${train.save_dir}/${train.save_name}/checkpoints,
    # if it's not set in config file.
    # resume_hdfs_chkpt='hdfs://haruna/xxx/pretrained/models/step100.pth',
    max_epochs=9,
    max_iters=600000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.5,
        key='loss',
        step=[40000 + i * 5000 for i in range(140)]),
    checkpoint_config=dict(
        interval=20000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/xxx'
    save_root='./demo',
    save_dir='tel_dy_hotsoon_80kh_rnnt',
    save_name='rnnt_transformer_base_drop0.15',
    amp_level='O1',
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,100000',
    test_sets='tele_general_test', # 'call_analysis_test|smb_itm_test'
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
