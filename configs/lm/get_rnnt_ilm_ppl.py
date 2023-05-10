project = "rnnt_ilm_ppl"
runner = 'BaseLMRunner'
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liboyu.622/traindata/text/alltext_4_normallm/lark_mt_800M',
    ],
    train_file_list=[
        '["text.{}".format(i) for i in range(1, 2)]',   # hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liboyu.622/traindata/text/alltext_4_normallm/lark_mt_800M
    ],
    valid_data_root = [
         'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav_testset_basic_7_21/',
    ],
    valid_file_list = [
         '["merged_cv"]', # multidomain
    ],
#    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liboyu.622/wukong_vs/data/testset/text',
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele',
    meta_file="meta_man_sc_yue",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    prefetch_worker_num=4,
    next_retry=60000,
    bucket_schedule='5,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000',
    bucket_schedule_val='5,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000',
# bucket_schedule_key用char
    bucket_schedule_key='char',
    batch_means_tokens=1,
    native_parallel_file_num=16,
    max_batch_size=4096,
    max_batch_scale=13,
    shuffle=True,
    global_shuffle=0,
    drop_last=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    use_old_bucket=1,
    train_item_transform=[
    ],

    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='SpaceAddForZhLabel'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_bpe=False),
    ],
    batch_transform=[
    ],

# 计算ppl时需要配置上CharCollate和PreCharCollate
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
)
solution = dict(
    type='rnnt_ilm_solution',
    model_type='RnntIlmModel',
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=512,
    predictor_lstm_hidden_size=2048,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.0,
    # jointer
    head_hidden_size=768,
    jointer_type='ShallowJointer',
    jointer_hidden_size=768,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=1936,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=500,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=True,

    xavier_init=False,
)
# runtime settings
work_dir = './dfsmn_ag20k_opt'
# training and testing settings
train = dict(
    # resume='latest.pth',
    resume_optimizer=True,
    resume_progress=True,
    # dolphin will try to resume from these path in order:
    #   1. ${train.out_dir}/${train.resume}
    #   2. ${train.resume_hdfs_chkpt}
    #   3. ${train.remote_save_root}/${train.out_dir}/${train.resume}
    # note: ${train.out_dir} is ${train.save_root}/${train.save_dir}/${train.save_name}/checkpoints,
    # if it's not set in config file.
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/input_10k_pretrain/dfsmnln_online_pretrain/dfsmnln_online_pretrain/checkpoints/step_180000.pth',
    max_iters=2000000,
    max_epochs=200,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.98,
        step=[250000 + i * 5000 for i in range(400)],
        ),
    checkpoint_config=dict(
        interval=20000, # steps
        max_keep_ckpts=20,
    ),
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/',
    save_root='./demo',
    save_dir='202304_lm_pretrain',
    save_name='lm_pretrain',
    amp_level='O1',
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    test_sets='wukong_js_202303_pack/text.1',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    rnnt_temperature=1.0,
    use_batch_beam=True,
    use_fsmn_beam=False,
    output_ilm_ppl=True,
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
        max_norm=20,
        norm_type=2,
        ),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=5000,
        warmup_sync=True,
        average_sync=True,
        use_nesterov=True,
    )
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=500,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
