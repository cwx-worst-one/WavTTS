project = "sid_training"
runner = 'BaseSidRunner'

# data
data = dict(
    train_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/fbank80_pkg/voxceleb2_train_clean',
    ],
    train_file_list=[
        '["train{}".format(i) for i in range(8)]',
    ],
    meta_file=[
        'meta',
    ],

    valid_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/fbank80_pkg/voxceleb2_train_clean',
    ],
    valid_file_list=[
        '["valid{}".format(i) for i in range(8)]',
    ],

    eval_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/fbank80_pkg/voxceleb1_test',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/data/new/fbank80_pkg/voxmovies_test',
    ],
    eval_file_list=[
        '["eval0"]',
        '["eval0"]',
    ],
    trials=[
        'trials2',
        'trials_voxmovies_e1',
    ],

    fbank_dim=80,
    prefetch_worker_num=2,
    global_shuffle=1,
    prefetch_block_num=1,

    train_item_transform=[
        dict(type='PickleParser'),
        dict(type="CutFeature", max_len=400),
        dict(type='DictTrans', in_key='spk', out_key='spk'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='DictTrans', in_key='spk', out_key='spk'),
        dict(type='SplitFeature', splited_len=300, in_key='src', out_key='src', max_pieces=5),
    ],
    eval_item_transform=[
        dict(type='PickleParser'),
        dict(type='EerFilter'),
    ],

    train_batch_transform=[
        dict(type='SidCollate', min_len=200, max_len=400)
    ],
    valid_batch_transform=[
        dict(type='SidCollate', min_len=300, max_len=300)
    ],
    eval_batch_transform=[
        dict(type='EerCollate', min_len=100, max_len=10000, random_clip=False)
    ],

    # used for training
    batch_size_per_class=1,
    batch_class_num=16,
    # used for validation
    max_batch_size=128,
    batch_means_tokens=False,
    drop_last=False,
)

# solution
solution = dict(
    type='base_sid_solution',

    acoustic_backbone_type='ECAPABackbone',
    channels='[1024, 1024, 1024, 1024, 3072]',
    kernel_sizes='[5, 3, 3, 3, 1]',
    dilations='[1, 2, 3, 4, 1]',
    activation_fn='relu',
    res2net_scale=8,
    se_channels=128,
    normalization_fn='batch_norm',
    normalization_after=True,
    batchnorm_momentum=0.1,
    batchnorm_eps=1e-5,
    padding_mode='reflect',

    # Pooling layer
    pooling_layer_type='AttentivePooling',
    attentive_bottleneck_dim=128,
    norm_after_pooling=True,
    use_global_context=True,

    # Segment-level network
    segment_network_type="MLPSoftmax",
    # For utt-level network, the topology is [n_nodes].
    segment_topology='[192]',
    segment_use_conv=True,
    segment_last_layer_use_norm=False,
    segment_last_layer_use_nonlinear=False,

    # Loss function
    softmax_type='additive_margin_softmax',
    softmax_renorm=30,
    softmax_margin=0.2,
    softmax_lambda_base=0,
    softmax_lambda_min=0,
    softmax_lambda_gamma=1,
    softmax_lambda_power=1,
    softmax_margin_by_epoch=True,
    softmax_margin_steps=[3+i for i in range(10)],

    # Criterion
    criterion_type="Xentropy",
    label_smooth_factor=0.0,

    # EER evaluation
    used_epoch_eval=True,
    embedding_dim=192,
    embedding_layer='utt_output'
)

# runtime settings
work_dir = './vox_cn'

# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-3,
)

# Optimizer Hook Configuration
optimizer_config = dict(
    max_grad_clip=0.0,
    grad_clip=0,
    bmuf_config=0
)

# training and testing settings
train = dict(
    iters_per_epoch=30000,
    max_epochs=24,
    max_iters=100000000000,
    final_lr=1e-9,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=30000*3,
        warmup_ratio=1e-5,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[13 + i for i in range(10)]
        ),

    checkpoint_config=dict(
        interval=10000000000, # steps
        max_keep_ckpts=10,
    ),

    save_root='./vox_cn',
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin/data_fbank64/',
    save_dir='vox_cn_lark_record',
    save_name='resnet101_stride_thin_256_fn30_am0.20',
    amp_level='O1',

    # Load a pretrained model from the following path.
    # If the checkpoint is saved in HDFS, the path is $resume_hdfs_chkpt.
    # While the path is ${save_root}/${save_dir}/${save_name}/checkpoint/${resume}
    # if it is in local.
    # resume='latest.pth',
    # resume_optimizer=True,
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin/data_fbank64/vox_cn_lark_record/resnet101_stride_thin_256_fn30_am0.20/checkpoints/latest.pth',
)

valid = dict(
)

# training and testing settings
log_level = 'INFO'
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
