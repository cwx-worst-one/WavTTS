project = "kws_ce_train"
runner = 'BaseKwsRunner'
# increment to add and override args.
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/kws/for_h/v34/dolphin/rnnt_with_ce_label',
    train_file_list = '["train_sub{}".format(i) for i in range(128)]',
    valid_file_list = '["cv"]',
    meta_file="meta_asr_syl1892_to_syl483_reorder_dali",
    fbank_dim=80,
    chunk_size=20,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000,5000,7000,10000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=40960,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    tgt_vocab_size=483,
    prefetch_worker_num=3,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='UttidKeywordsFilter', key='uttid', keywords=['sp1.1-', 'sp0.9-','kws']),
        dict(type='AlignFeature', key='fbank', ce_key='ce_label'),
        dict(type='CMVN', key='fbank'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LengthFilter', key='fbank', max_len=2000),
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
    type='kws_ce_solution',
    ce_type='KwsCeAdaptModel',
    # encoder
    encoder_type='RNNTConvDFSMNEncoder',
    input_concat_size=8,
    downsampling_size=3,
    clamp_residual=0,
    fsmn_left_kernel_size=7,
    fsmn_right_kernel_size=1,
    fsmn_dilation=1,
    backbone_topology='[[7,1,1]]*3+[[7,0,1]]*7',
    backbone_memory_size=128,
    backbone_hidden_size=480,
    backbone_weight_scale=1.0,
    dropout=0.0,
    encoder_out_layer_norm=1,
    rnnt_hidden_size=256,
    selected_dfsmn_layer=-1,
    # head
    head_type='DNNHead',
    head_hidden_layer_num=0,
    head_input_size=256,
    head_hidden_size=256,
    head_layer_norm=0,
    head_use_selected_encoder_out=False,
    # criterion
    criterion_type='Xentropy',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    xavier_init=False,
    # pretrain model
    freeze_encoder=False,
    mtl_type=None,
    # slim param for binary quantizer
    slim_encoder_skip_scope='input_trans_fc,pred_fc',
    slim_encoder_binary_act_quant_func='Sign',
    slim_encoder_binary_param_quant_func='Sign',
    slim_encoder_binary_act_scale_updator=None,
    slim_encoder_binary_param_scale_updator='AbsMeanScale',
)
# runtime settings
work_dir = './kws_ce_train'
# training and testing settings
train = dict(
    max_epochs=15,
    max_iters=200000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup=None,
        warmup_iters=0,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[4 + i for i in range(50)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    save_root='./kws_ce_train',
    save_dir='kws_ce_train',
    save_name='kws_ce_train',
    amp_level='O0',
    )
valid = dict(
    interval=10000, # iter
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
    interval=200,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
