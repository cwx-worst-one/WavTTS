project = "dfsmn_rnnt_pretrain"
runner = 'RNNTCePretrainRunner'
_base_ = [
    '../_base_/datasets/aiot_english_23kh.py',
]
# increment to add and override args.
data = dict(
    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=False,
    tgt_vocab_size=41,
    max_batch_size=40960,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        # dict(type='BPE', use_eos=False),
        dict(type='AlignFeature', key='fbank', ce_key='ce_label'),
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
        # dict(type='BPE', use_eos=False),
        dict(type='AlignFeature', key='fbank', ce_key='ce_label'),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CeLabelCollate', frame_chunk_size=32)
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    type='rnnt_ce_pretrain_solution',
    rnnt_type='BaseRnntModelEncoder',
    # front_end
    front_end_type='VGGFrontEnd',
    clamp_residual=1,
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
    head_type='RnntEncoderBaseHead',
    head_hidden_size=768,
    head_lowrank_size=512,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=768,
    concate_U=1,
    jointer_simple_fusion=0,
    # criterion
    criterion_type='Xentropy',
    cer_update_freq=200,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=False,
    # pretrain or not
    pretrain_encoder=True
)
# runtime settings
work_dir = './dfsmn_english_23kh_pretrain'
# training and testing settings
train = dict(
    #resume='latest.pth',
    #resume_optimizer=True,
    max_epochs=90,
    max_iters=1600000,
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
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/english_23kh_pretrain',
    save_root='./dfsmnln_online_pretrain',
    save_dir='dfsmnln_online_pretrain',
    save_name='dfsmnln_online_pretrain',
    amp_level='O1',
    )
valid = dict(
    interval=10000, # iter
    )

# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=1e-4,
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
