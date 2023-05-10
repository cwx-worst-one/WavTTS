project = "se_music_aec_dfsmn_r4x4"
runner = 'BaseSeRunner'

# increment to add and override args. 
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/gongyuzhou/',
    # add ce sy+ctc label
    train_file_list = ['dolphin_music_aec_offline'],
    valid_file_list = ['dolphin_music_aec_offline'],
    eval_file_list  = ['dolphin_music_aec_offline'],

    # shuffle的颗粒度
    chunk_size=20,
    bucket_schedule='1000000',
    bucket_schedule_val='1000000',
    bucket_schedule_key='wav_shape_in_bytes',
    # 常设为0，保持固定的句子数组batch，为0的条件：batch_size >= max_batch_size
    batch_means_tokens=0,
    # max_batch_size 需要设置为batch_size-1
    max_batch_size=63,
    max_batch_scale=0,
    # 是否保证每次实验获取的数据完全一致，用于支持模型可复现；稍微影响速度
    deterministic=True,
    # 是否进行全局shuffle
    global_shuffle=True,
    shuffle=True,
    drop_last=False,

    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', in_key="mic_wav"),
        dict(type='WavParser', in_key="ref_wav"),
        dict(type='WavParser', in_key="aec_wav"),
        dict(type='WavParser', in_key="speech_wav"),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', in_key="mic_wav"),
        dict(type='WavParser', in_key="ref_wav"),
        dict(type='WavParser', in_key="aec_wav"),
        dict(type='WavParser', in_key="speech_wav"),
    ],
    batch_transform=[
        dict(type='ListCollate', key='mic_waveform'),
        dict(type='ListCollate', key='aec_waveform'),
        dict(type='ListCollate', key='ref_waveform'),
        dict(type='ListCollate', key='speech_waveform'),
        dict(type='ListCollate', key='path'),
        dict(type='WaveformCollateNoSplit', wav_key='mic_waveform', out_key='mic_waveform'),
        dict(type='WaveformCollateNoSplit', wav_key='aec_waveform', out_key='aec_waveform'),
        dict(type='WaveformCollateNoSplit', wav_key='ref_waveform', out_key='ref_waveform'),
        dict(type='WaveformCollateNoSplit', wav_key='speech_waveform', out_key='speech_waveform'),
    ],

    inference_item_transform=[
    ],
    inference_batch_transform=[
    ],
)
solution = dict(
    type='base_se_solution',
    se_model_type='BaseAECSolution',
    feat_extractor_type='AecSubbandFeatExtractor',
    feat_extractor=dict(
        frame_length=128,
        oversample_ratio=2,
        sampling_rate=16000,
        delay_version='high'
    ),
    aec_net_type='TcnLowDelaySub',
    aec_net=dict(
        n_feats=258,
        n_b=16,
        C=172,
        B=96,
        H=128,
        P=5
    ),
    # criterion
    criterion_type='MixSisnrMelMseAsyn',
    criterion=dict(
        alpha=2,
        lamda=0,
        fs=16000,
        n_fft=256,
        n_mels=80,
        fmin=0
    ),
    # evaluation
    evaluation='frame',
    eval_seg='oveall'
)
# runtime settings
work_dir = './se_test'
# training and testing settings
train = dict(
    #resume='latest.pth',
    #resume_optimizer=True,
    #resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/shenchen/dolphin/MDD/test/dfsmnln_online_pretrain/dfsmnln_online_pretrain/checkpoints/step_1000.pth',
    max_epochs=100,
    max_iters=100000,
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
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/gongyuzhou/',
    save_root='./',
    save_dir='dolphin_test',
    save_name='test',
    amp_level='O1',
    )
valid = dict(
    interval=100000000, # iter
    )
inference = dict(
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='Adam',
    lr=1e-4,
    betas=(0.5, 0.9),
    eps=1e-8,
    weight_decay=0,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=10,
        norm_type=2,
        ),
    max_grad_clip=0.0,
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=10,  # log at every interval iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
