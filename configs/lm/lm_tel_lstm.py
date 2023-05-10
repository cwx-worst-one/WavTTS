project = "tel_lm"
runner = 'BaseLMRunner'
_base_ = [
    '../_base_/datasets/lm_tel.py',
]
# increment to add and override args.
data = dict(
    chunk_size=30,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_bpe=True),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_bpe=True),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='LMCharCollate'),
        dict(type='LMPreCharCollate'),
    ],
)
solution = dict(
    type='base_nnlm_solution',
    nnlm_type='BaseNNLMModel',
    # lstm
    nnlm_model_type='LSTMLM',
    embedding_size=1024,
    lstm_cell_size=4096,
    lstm_num_layers=2,
    pred_fc_hidden_size=1024,
    # criterion
    criterion_type='Xentropy',
    label_smooth_factor=0.1,
    reorder_dict_by_freq=False,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',  # local checkpoint file to resume.
    resume_optimizer=True,
    max_epochs=100,
    max_iters=800000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=1,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.4,
        key='loss',
    ),
    checkpoint_config=dict(
        interval=12850,  # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/xxx'
    save_root='./demo',
    save_dir='lm_tel',
    save_name='lm_lstm_l2e640c2048_tel_o0Adam_gpu8',
    amp_level='O0',
)
valid = dict(
    interval=12850,  # iter
)
optimizer = dict(
    type='Adam',
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
)
optimizer_config = dict(
    grad_clip=dict(
        max_norm=20,
        norm_type=2,
    ),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.9,
        warmup_steps=5000,
        use_nesterov=True,
        average_params=True,
    ),
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
    ],
)
