project = "bestrq-extraction"
runner = 'AcousticRepresentationDumpingRunner'

# increment to add and override args.
data = dict(
    data_root=[
        #'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_6wh', # 
        # 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
        '/mnt/bn/by-nas/Speech/Codes/dolphin-usm_develop-gspt-infer/model'
    ],
    train_file_list=[
        '["sub_{}".format(i) for i in range(1333)]', # 这里没用
    ],
    valid_file_list=[
        #'["sub_{}".format(i) for i in range(22)]', # 大概1000小时
        '["lark_5hmm_test"]'
    ],
    meta_data_root="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/",
    meta_file="meta_ag350k_lzl",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    chunk_size=20,
    bucket_schedule='200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_val='5,100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_key='length',
    fbank_dim=80,
    fbank_channel=1,
    max_batch_size=20000,
    batch_means_tokens=1,
    use_eos=False,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type='LengthFilter', key='length', min_len=20, max_len=2000),
        dict(type='WavResample', sample_rate=16000, key='wav'),
        # dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),

    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type='LengthFilter', key='length', min_len=20, max_len=2000),
        dict(type='WavResample', sample_rate=16000, key='wav'),
    ],
    batch_transform=[
        dict(type='WaveformCollate', wav_key='waveform'),
        dict(type='ListCollate', key='uttid'),
        
    ],
    inference_batch_transform=[
        dict(type='WaveformCollate', wav_key='waveform'),
        dict(type='ListCollate', key='uttid'),
    ],
)
solution = dict(
    model_type='AcousticTokenizationModel',
    ## wrap an encoder
    # front_end
    acoustic_tokenizer_type='OriginalHubertTokenizer',
    acoustic_tokenizer_hubert_model_path='/mnt/bn/by-nas/Speech/opensource_pretrained/chinese_speech_pretrain/chinese-hubert-base',
    acoustic_tokenizer_kmeans_model_path='/mnt/bn/by-nas/Speech/opensource_pretrained/chinese_speech_pretrain/hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl',
    acoustic_tokenizer_kmlayer=9,
    # Loss
    label_smooth_factor=0,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=60,
    max_iters=600000,
    lr_scheduler=dict(
        policy='Transformer',
        by_epoch=False,
        warmup_steps=25000,
        d_model=1024,
        warmup=None,
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    save_root='./bestrq_conformerxl_nonstream_input6w',
    save_dir='bestrq',
    save_name='bestrq',
    amp_level='O1',
    # gradient accumulation
    grad_accum_step=4,
    # 'AVG' 'SUM'
    grad_accum_mode='AVG'
    )
valid = dict(
    output_probs=False,
    not_show_results=False,
    interval=1600,
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='AdamW',
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=1.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    grad_clip_mode='accum_before',
    max_grad_clip=5.0,
 )

inference = dict(
    extraction_mode="extract_hubert_codes",
    override_directory=True,
    saving_interval=103,
    n_shard_per_proc=13,
    local_saving_dir="./hubert",
    #remote_saving_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/baiye/experiments/bestrq_conformerxl_nonstream_input6w_expsetting/maskprob0.12_span40_codebooksize4096_codebookdim16_nsoftmax8_warmup50000_accum2_gcmaxnorm20_accumafter/bestrq/dumped-extract_codes-video_live_6wh-video_live_6wh-sub0_sub21",
)
log_level = 'INFO'
log_config = dict(
    interval=10,  # log at every iteration
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
