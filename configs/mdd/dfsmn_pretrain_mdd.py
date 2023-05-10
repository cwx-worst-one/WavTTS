project = "dfsmn_rnnt_pretrain"
runner = 'RNNTCePretrainRunner'
# increment to add and override args.
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guyiwei/dolphin/data/train_eng_ey_350w_ext70w_ext30w_5432_39dim/',
    # add ce sy+ctc label
    train_file_list = '["sub_{}".format(i) for i in range(40)]',
    valid_file_list = '["cv"]',
    meta_file="meta",
    fbank_dim=39,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    shuffle=True,
    drop_last=False,


    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=False,
    tgt_vocab_size=5432,

    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='AlignFeature', key='fbank', ce_key='ce_label'),
        dict(type='CMVN', key='fbank'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='AlignFeature', key='fbank', ce_key='ce_label'),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=39),
        dict(type='CeLabelCollate')
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=39),
    ],
)
solution = dict(
    type='rnnt_ce_pretrain_solution',
    rnnt_type='BaseRnntModelEncoder',
    # front_end
    front_end_type='FSMNFrontEnd',
    clamp_residual=1,
    input_concat_size=1,
    downsampling_size=1,
    fsmn_left_kernel_size=40,
    fsmn_right_kernel_size=5,
    fsmn_dilation=1,
    # backbone
    acoustic_backbone_type='DFSMNBackboneLN',
    backbone_topology='[[40,4,1],[40,3,1],[40,2,1],[40,1,1]]+[[20,0,2]]*20',
    backbone_layer_gap=100,
    dfsmn_bn_flag=1,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    dropout=0.1,
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
    max_epochs=90,
    max_iters=500000,
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
    save_root='./dfsmnln_am',
    save_dir='dfsmnln_am',
    save_name='ey_350w_ext40w_ext30w_5432',
    amp_level='O1',
    )
valid = dict(
    interval=1000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='test_english_children_ey_4381',
    #remote_stat_dir="",
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=2e-4,
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
