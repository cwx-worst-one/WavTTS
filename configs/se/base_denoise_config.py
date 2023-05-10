project = "base_denoise"
runner = 'BaseSeModuleRunner'
seed = 8888
# increment to add and override args. 
data = dict(
    train_data_root='/mnt/bd/lgz-data/denoise_data/',
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y',
    eval_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y',
    # add ce sy+ctc label
    train_file_list = '["clean/DNSSpeech_aishell3_vocal_1ch_16k"]',
    valid_file_list = '["valid_lark"]',
    eval_file_list = '["test_-10dB", "test_-5dB", "test_0dB", "test_5dB"]',

    # shuffle的颗粒度
    chunk_size=20,
    prefetch_worker_num=6,
    
    # bucket strategy
    batch_strategy='RandomSimpleBatching',
    bucket_size=20,
    bucket_schedule='',
    max_batch_size=64,
    # 是否保证每次实验获取的数据完全一致，用于支持模型可复现；稍微影响速度
    deterministic=False,
    # sampler config
    global_shuffle=True, # 是否进行全局shuffle
    shuffle=True,
    drop_last=True,

    train_item_transform=[
        dict(type='NumpyParser', dtype='np.int16', out_key='waveform'),
        dict(type='LoadSimulatorData', simulate_type='simulate_denoise', simulator_config_path='./configs/se/simulator_config_denoise_3mic.py', cached_name='train'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LoadSimulatorData', simulate_type='simulate_denoise_valid', simulator_config_path='./configs/se/simulator_config_denoise_3mic_valid.py', cached_name='valid'),
    ],
    train_batch_transform=[
        dict(type='WaveformCollateNoSplit', wav_key='noisy_noreverb', out_key='noisy_noreverb'),
        dict(type='WaveformCollateNoSplit', wav_key='early_reverb_signal', out_key='early_reverb_signal'),
        dict(type='WaveformCollateNoSplit', wav_key='speech', out_key='speech'),
        dict(type='FlagCollate', wav_key='sign_clip', out_key='sign_clip'),
        dict(type='SimulatorWaveformCollate', simulate_type='simulate_denoise', simulator_config_path='./configs/se/simulator_config_denoise_3mic.py'),
    ],
    valid_batch_transform=[
        dict(type='WaveformCollateNoSplit', wav_key='noisy_noreverb', out_key='noisy_noreverb'),
        dict(type='FlagCollate', wav_key='sign_clip', out_key='sign_clip'),
        dict(type='FlagCollate', wav_key='agc', out_key='agc'),
        dict(type='SimulatorWaveformCollate', simulate_type='simulate_denoise_valid', simulator_config_path='./configs/se/simulator_config_denoise_3mic_valid.py'),
    ],
    train_device_transform = [dict(type='SimulatorModule', simulate_type='simulate_denoise', simulator_config_path='./configs/se/simulator_config_denoise_3mic.py')],
    valid_device_transform = [dict(type='SimulatorModule', simulate_type='simulate_denoise_valid', simulator_config_path='./configs/se/simulator_config_denoise_3mic_valid.py')],

    inference_item_transform=[
        dict(type='PickleParser'),
    ],
    inference_batch_transform=[
        # general transform
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='clean_waveform', out_key='clean_waveform'),
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='noisy_waveform', out_key='noisy_waveform'),
        dict(type='ListCollate', key='uttid'),
    ],
)
solution = dict(
    type='base_se_solution',
    se_model_type='BaseDenoiseSolution',
    feat_extractor_type='DenoiseSubbandFeatExtractor',
    feat_extractor=dict(
        frame_length=160,
        oversample_ratio=2,
        sampling_rate=16000,
        delay_version='high',
        eq_ratio=0.0,
    ),
    denoise_net_type='DenoiseMcmodelFullsubnet3CIrmCplv2',
    denoise_net=dict(
        num_freqs=161,
        fb_layers=4,
        sb_layers=2,
        fb_num_neighbors=0,
        fb_model_hidden_size=160,
        sb_model_hidden_size=120,
        sb_stride=20,
        sb_kernel=40,
        faam=False,
    ),
    # criterion
    criterion_type='DenoisePcsloss',
    criterion=dict(
        compress_coeff = 0.5,
    ),
    # inference
    inference = dict(
        save_dir ='./inference/denoise',
        remote_save_dir=0,
        # remote_save_dir = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_weekly_test/denoise/save_metrics/',
        metric = 'DenoiseInferMetric',
        log_metric=True,
        eval_multi_cer=True, # 是否分别对每个eval file进行评估
        save_enh_wav=False, # 是否保存enh wav 音频文件
        ),
)
# runtime settings
work_dir = './denoise_test'
# training and testing settings
train = dict(
    # resume='latest.pth',
    # resume_optimizer=True,
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dolphin_weekly_test/denoise/dolphin_denoise/denoise_3mic/checkpoints/latest.pth',
    save_after_epoch=True,  # save checkpoint after each epoch
    eval_after_epoch=True, # do inference after each epoch
    max_epochs=200,
    iters_per_epoch=125, # 每个轮次多少iters
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=True,
        policy='StepLR',  # use torch lr_scheduler
        gamma=0.98,
        step_size=10
        ),
    checkpoint_config=dict(
        interval=1000000, # steps
        max_keep_ckpts=-1,
    ),
    best_metric_config = dict(),
    remote_save_root=0,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dolphin_weekly_test/denoise/',
    save_root='./',
    save_dir='dolphin_denoise',
    save_name='denoise_3mic',
    amp_level='O0',
    deterministic=False,
    metric='DenoiseTrainMetric',
    )
valid = dict(
    # data_epoch_interval=100,
    interval=10000000000, # iter
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='Adam',
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=5,
        norm_type=2,
        ),
    max_grad_clip=0,
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