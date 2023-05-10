project = "base_denoise"
runner = 'BaseSeModuleRunner'
seed = 777
# increment to add and override args. 
data = dict(
    train_data_root=[
        '/mnt/bd/lgz-data-aec/dataset/'
    ],
    train_file_list=['["DNSSpeech_1ch_16k"]'],
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y',
    eval_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/se/aec/data/',
    # add ce sy+ctc label
    valid_file_list = '["valid_lark"]',
    eval_file_list = '["eval_-20dB", "eval_-10dB","eval_0dB", "eval_clean", "eval_echo"]',

    # shuffle的颗粒度
    chunk_size=20,
    prefetch_worker_num=6,
    
    # bucket strategy
    batch_strategy='RandomSimpleBatching',
    bucket_size=20,
    bucket_schedule='',
    max_batch_size=32,
    # 是否保证每次实验获取的数据完全一致，用于支持模型可复现；稍微影响速度
    deterministic=False,
    # sampler config
    global_shuffle=True, # 是否进行全局shuffle
    shuffle=True,
    drop_last=True,

    train_item_transform=[
        dict(type='NumpyParser', dtype='np.int16', out_key='waveform'),
        dict(type='LoadSimulatorData', simulate_type='simulate_aec', simulator_config_path='./configs/se/simulator_config_aec.py', cached_name='train'),
    ],
    valid_item_transform=[
    ],
    train_batch_transform=[
        dict(type='WaveformCollateNoSplit', wav_key='speech_mask', out_key='speech_mask'),
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='echo', out_key='echo'),
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='ref', out_key='ref'),
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='time_delay', out_key='time_delay'),
        dict(type='FlagCollate', wav_key='speech_flag', out_key='speech_flag'),
        dict(type='FlagCollate', wav_key='noise_flag', out_key='noise_flag'),
        dict(type='FlagCollate', wav_key='echo_flag', out_key='echo_flag'),
        dict(type='SimulatorWaveformCollate', simulate_type='simulate_aec', simulator_config_path='./configs/se/simulator_config_aec.py'),
    ],
    valid_batch_transform=[
    ],
    train_device_transform = [dict(type='SimulatorModule', simulate_type='simulate_aec', simulator_config_path='./configs/se/simulator_config_aec.py')],

    inference_item_transform=[
        dict(type='PickleParser'),
    ],
    inference_batch_transform=[
        # general transform
        dict(type='WaveformCollateNoSplit', wav_key='speech', out_key='speech'),
        dict(type='WaveformCollateNoSplit', wav_key='mic', out_key='mic'),
        dict(type='WaveformCollateNoSplit', wav_key='ref_tde', out_key='ref_tde'),
        dict(type='WaveformCollateNoSplit', wav_key='aec', out_key='aec'),
        dict(type='ListCollate', key='uttid'),
    ],
    inference_device_transform=[
        dict(type='SimulatorModule', simulate_type='simulate_aec_infer', simulator_config_path='./configs/se/simulator_config_aec.py')
    ],
)
solution = dict(
    type='base_se_solution',
    se_model_type='BaseAECSolution',
    aec_net_type='AECSmallV42Align',
    aec_net=dict(),
    # pretrained_model_path='/opt/tiger/se/cnnpooltfgrurealmask_1c_v10_fremse7_dnsdata_noclearnrir2_188.pth',
    feat_extractor_type='AecSubbandFeatExtractor',
    feat_extractor=dict(
        frame_length=160,
        oversample_ratio=2,
        sampling_rate=16000,
        delay_version='high',
    ),
    # criterion
    criterion_type='EchoAwareLoss',
    criterion=dict(
        compress_coeff=0.3,
    ),
    # inference
    inference = dict(
        save_dir ='./inference/aec',
        remote_save_dir=0,
        # remote_save_dir = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_weekly_test/denoise/save_metrics/',
        metric = 'AECInferMetric',
        log_metric=True,
        save_enh_wav=False, # 是否保存enh wav 音频文件
        eval_multi_cer=True, # 是否分别对每个eval file进行评估
        ),
)
# runtime settings
work_dir = './aec_training'
# training and testing settings
train = dict(
    save_after_epoch=True,  # save checkpoint after each epoch
    eval_after_epoch=True, # do inference after each epoch
    max_epochs=200,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=True,
        policy='LambdaLR',  # use torch lr_scheduler
        lr_lambda='lambda epoch: 0.98**epoch if (0.98**epoch) > 5e-7 else 5e-7',
        last_epoch=-1
        ),
    checkpoint_config=dict(
        interval=1000000, # steps
        max_keep_ckpts=-1,
    ),
    best_metric_config = dict(),
    remote_save_root=0,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dolphin_weekly_test/aec/',
    save_root='./',
    save_dir='dolphin_aec',
    save_name='aec',
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
    lr=0.0005,
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