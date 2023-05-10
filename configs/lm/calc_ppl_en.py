project = "tel_lm"
runner = 'BaseLMRunner'
# increment to add and override args.
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/nnlm_data/label_chinglish_haitian_replaceStarToken/',
    train_file_list = '["train_subshard{}".format(i) for i in range(29)]',
    valid_file_list = '["subshard.cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    meta_file="meta",
    fetch_block_size=200,
    bucket_schedule='5,10,20,30,40,50,60,70,80,90,100,200',
    bucket_schedule_val='100000',
    bucket_schedule_key='char',
    batch_means_tokens=1,
    max_batch_size=15360,
    shuffle=True,
    drop_last=False,
    chunk_size=1000,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='SpaceAddForZhLabel'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>']),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='SpaceAddForZhLabel'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>']),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='LMCharCollate'),
        dict(type='LMPreCharCollate'),
    ],
    inference_batch_transform=[
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
    embedding_size=512,
    lstm_cell_size=1024,
    lstm_num_layers=2,
    pred_fc_hidden_size=512,

    # criterion
    criterion_type='Xentropy',
    label_smooth_factor=0.1,
    reorder_dict_by_freq=True,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth', # local checkpoint file to resume.
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
        interval=12850, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/xxx'
    save_root='./nnlm_ez',
    save_dir='jinei_training',
    save_name='lm_lstm_512emb_1024lstmcell_512prec_326M',
    amp_level='O0',
    )
valid = dict(
    interval=12850, # iter
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
    )
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=1000,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
