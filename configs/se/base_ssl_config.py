project = "base_ssl"
runner = 'BaseSeModuleRunner'
seed = 8888
# increment to add and override args. 
data = dict(
    # train_data_root='/opt/tiger/workspace/dataset/TrainSet',
    # valid_data_root='/opt/tiger/workspace/dataset/TestSet',
    # eval_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/robin_record',
    train_data_root='/mnt/bd/speech-se',
    valid_data_root='/mnt/bd/speech-se',
    eval_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_test_data/ssl',
    # add ce sy+ctc label
    train_file_list = '["clean_aishell_1ch_16k"]',
    valid_file_list = '["clean_aishell_1ch_16k"]',
    eval_file_list  = '["nonovlp_data"]',

    # shuffle的颗粒度
    chunk_size=20,
    # schedule大于data中最大的长度即可
    bucket_schedule='1000000',
    bucket_schedule_val='1000000',
    bucket_schedule_key='length',
    # bucket strategy
    bucket_strategy='BucketBatching',
    batch_means_tokens=0, # 常设为0，保持固定的句子数组batch，为0的条件：batch_size >= max_batch_size
    # max_batch_size 需要设置为batch_size-1
    max_batch_size=99,
    max_batch_scale=0,
    # 是否保证每次实验获取的数据完全一致，用于支持模型可复现；稍微影响速度
    deterministic=False,
    # sampler config
    global_shuffle=True, # 是否进行全局shuffle
    shuffle=True,
    drop_last=True,

    train_item_transform=[
        dict(type='NumpyParser', dtype='np.int16', out_key='waveform'),
        dict(type='LoadSimulatorData', simulate_type='simulate_ssl', simulator_config_path='./configs/se/simulator_config_ssl_2mic.py', cached_name='train'),
    ],
    valid_item_transform=[
        dict(type='NumpyParser', dtype='np.int16', out_key='waveform'),
        dict(type='LoadSimulatorData', simulate_type='simulate_ssl', simulator_config_path='./configs/se/simulator_config_ssl_2mic.py', cached_name='valid'),
    ],
    batch_transform=[
        dict(type='SimulatorWaveformCollate', simulate_type='simulate_ssl', simulator_config_path='./configs/se/simulator_config_ssl_2mic.py'),
        dict(type='WaveformCollateNoSplit', wav_key='direction', out_key='direction'),
        dict(type='FlagCollate', wav_key='source_num', out_key='source_num'),
        dict(type='FlagCollate', wav_key='noise_flag', out_key='noise_flag'),
    ],

    device_transform = [dict(type='SimulatorModule', simulate_type='simulate_ssl', simulator_config_path='./configs/se/simulator_config_ssl_2mic.py')],

    inference_item_transform=[
        dict(type='PickleParser'),
        dict(type='SimpleWavParser', in_key='mc_waveform', out_key='mc_waveform', out_dtype='np.float32'),

    ],
    inference_batch_transform=[
        # for output zone prob
        # dict(type='ListCollate', key='wavname', out_key='wavname'),
        # dict(type='MultiChannelWaveformCollateNoSplit', wav_key='mc_waveform', out_key='mc_waveform'),
        # for output location accuracy
        dict(type='FlagCollate', wav_key='length', out_key='length'),
        dict(type='FlagCollate', wav_key='direction', out_key='direction'),
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='mc_waveform', out_key='mc_waveform'),
        dict(type='WaveformCollateNoSplit', wav_key='vad', out_key='vad')
    ],
)

solution = dict(
    type='base_se_solution',
    se_model_type='BaseSSLSolution',
    feat_extractor_type='SSLSubbandFeatExtractor_linear',
    feat_extractor=dict(
        frame_length=128,
        oversample_ratio=2,
        sampling_rate=16000,
        delay_version='high',
        mic_num=2,
        zone_num=36,
        block_frame=8,
        uniform_type="rad",
        max_source_num=2,
        disable_freq=dict(
            enable=True,
            freq_bin_range=(0,1),  # [min, max)
            freq_range=None,
        ),
        gauss_smooth=dict(
            enable=True,
            gauss_const=2.0,
        )
    ),
    ssl_net_type='nnlocation',
    ssl_net=dict(
        mic_num=2,
        zone_num=36,
        inter_channel1=10,
        inter_channel2=20,
        inter_channel3=None,  # "None" means the same value as zone_num
    ),
    # criterion
    criterion_type='SSL',
    criterion=dict(
        criterion_subtype="mse"
    ),
    # evaluation
    evaluation='frame',
    eval_seg='oveall',
    # inference
    inference = dict(
        save_dir = './inference/ssl',
        remote_save_dir = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_weekly_test/ssl/save_metrics/',
        type = 'save_location_accuracy',   # 'save_location_accuracy': 保存定位精度； 'save_probability': 保存音区输出概率
        # configs used in "save_location_accuracy" mode
        array_type = 'linear',
        zone_num = 36,
        angle_config = dict(
            angle_min=0,
            angle_max=180
        ),  # specify angle_min and angle_max, so that zone and angle are aligned by np.linspace(angle_min, angle_max, zone_num, endpoint=False)
        eval_angle_bins = [0, 15.1, 20.1, 30.1, 50.1, 370],  # optional
        accuracy_threshold = 15,    # optional
        metric = "SslInferMetricSaveAcc",
        log_metric=False,
        final_call="process"
    )
)
# runtime settings
work_dir = './se_test'
# training and testing settings
train = dict(
    # resume='latest.pth',
    # resume_optimizer=True,
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/shenchen/dolphin/MDD/test/dfsmnln_online_pretrain/dfsmnln_online_pretrain/checkpoints/step_1000.pth',
    save_after_epoch=True,  # save checkpoint after each epoch
    drop_when_epoch_end=True,
    max_epochs=100,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=True,
        policy='StepLR',  # use torch lr_scheduler
        gamma=0.9,
        step_size=1
        ),
    checkpoint_config=dict(
        interval=1000000, # steps
        max_keep_ckpts=-1,
    ),
    best_metric_config=dict(
        best_metric_name='backward_loss',  # None or not provided, won't trigger best metric comparison
        best_metric_type='min',
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_weekly_test/ssl/',
    save_root='./',
    save_dir='dolphin_ssl',
    save_name='ssl_2mic',
    amp_level='O0',
    deterministic=False,
    max_cufft_plan_cache=512,
    metric='SslTrainMetric'
    )
valid = dict(
    interval=100000000, # iter
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='Adam',
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=100000000,
        norm_type=2,
        ),
    max_grad_clip=10.0,
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
