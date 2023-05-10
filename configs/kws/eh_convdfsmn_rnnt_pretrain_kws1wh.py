project = "kws_rnnt_train"
runner = 'KwsRnntRunner'
# increment to add and override args.
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/kws/for_h/v34/dolphin/rnnt_with_ce_label',
    train_file_list = '["train_sub{}".format(i) for i in range(128)]',
    valid_file_list = '["cv"]',
    meta_file="meta_syl483_reorder_dali",
    fbank_dim=80,
    chunk_size=20,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000,5000,7000,10000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    tgt_vocab_size=483,
    prefetch_worker_num=3,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='UttidKeywordsFilter', key='uttid', keywords=['sp1.1-', 'sp0.9-','kws']),
        dict(type='LabelMap', key='s2s_label'),
        dict(type='CMVN', key='fbank'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelMap', key='s2s_label'),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', need_adapt_target=False),
        dict(type='PreCharCollate', bos_id=0)
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    type='kws_rnnt_solution',
    rnnt_type='KwsRnntModel',
    # encoder
    encoder_type='ConvDFSMNEncoder',
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
    # predictor
    predictor_type='DNNPredictor',
    predictor_embedding_dim=256,
    predictor_layer_num=2,
    predictor_hidden_dim=512,
    predictor_label_dropout_factor=0.0,
    predictor_out_layer_norm=1,
    # jointer
    jointer_type='ShallowJointer',
    rnnt_hidden_size=256,
    rnnt_softmax_hidden_size=256,
    # mtl head
    mtl_type='rnnt_ctc',
    ctc_weight=1.0,
    ctc_head_type='DNNHead',
    ctc_use_selected_encoder_out=False,
    ctc_head_hidden_layer_num=0,
    ctc_head_input_size=256,
    ctc_head_hidden_size=256,
    ctc_head_layer_norm=0,
    ctc_tgt_vocab_size=483,
    selected_dfsmn_layer=-1,
    # criterion
    rnnt_weight=1.0,
    criterion_type='KwsRnnt',
    label_smooth_weight=0.1,
    rnnt_exclude_data_tag=None,
    cer_update_freq=100,
    # pretrain model
    pretrain_encoder_model_path=None,
    pretrain_predictor_model_path=None,
    pretrain_jointer_model_path=None,
    xavier_init=False,
    # slim param
    slim_encoder_quant_type='binary',
    slim_encoder_skip_scope='input_trans_fc,pred_fc',
    slim_encoder_binary_act_quant_func='Sign',
    slim_encoder_binary_param_quant_func='Sign',
    slim_encoder_binary_act_scale_updator=None,
    slim_encoder_binary_param_scale_updator='AbsMeanScale',
    slim_skip_jointer=True,
)
# runtime settings
work_dir = './kws_rnnt_train'
# training and testing settings
train = dict(
    max_epochs=15,
    max_iters=300000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup=None,
        warmup_iters=0,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[3 + i for i in range(50)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    save_root='./kws_rnnt_train',
    save_dir='rnnt_dfsmn',
    save_name='rnnt_dfsmn',
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
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
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
    interval=100,  # log at every 100 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
