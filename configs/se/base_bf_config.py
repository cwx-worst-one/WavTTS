project = "base_bf"
runner = 'BaseSeModuleRunner'
seed = 8888
# increment to add and override args. 
data = dict(
    # train_data_root='/opt/tiger/workspace/dataset/TrainSet',
    # valid_data_root='/opt/tiger/workspace/dataset/TestSet',
    # eval_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/eh_record',
    train_data_root='/mnt/bd/speech-se',
    valid_data_root='/mnt/bd/speech-se',
    eval_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_test_data/bf',
    # add ce sy+ctc label
    train_file_list = '["clean_aishell_1ch_16k"]',
    valid_file_list = '["clean_aishell_1ch_16k"]',
    eval_file_list  = '["ovlp_data"]',

    # shuffle的颗粒度
    chunk_size=20,
    # schedule大于data中最大的长度即可
    bucket_schedule='10000000000',
    bucket_schedule_val='10000000000',
    bucket_schedule_key='length',
    # bucket strategy
    bucket_strategy='BucketBatching',
    batch_means_tokens=0, # 常设为0，保持固定的句子数组batch，为0的条件：batch_size >= max_batch_size
    # max_batch_size 需要设置为batch_size-1
    max_batch_size=239,
    max_batch_scale=0,
    # 是否保证每次实验获取的数据完全一致，用于支持模型可复现；稍微影响速度
    deterministic=False,
    # sampler config
    global_shuffle=True, # 是否进行全局shuffle
    shuffle=True,
    drop_last=False,

    train_item_transform=[
        dict(type='NumpyParser', dtype='np.int16', out_key='waveform'),
        dict(type='LoadSimulatorData', simulate_type='simulate_bf', simulator_config_path='./configs/se/simulator_config_bf_2mic.py', cached_name='train'),
    ],
    valid_item_transform=[
        dict(type='NumpyParser', dtype='np.int16', out_key='waveform'),
        dict(type='LoadSimulatorData', simulate_type='simulate_bf', simulator_config_path='./configs/se/simulator_config_bf_2mic.py', cached_name='valid'),
    ],
    batch_transform=[
        dict(type='SimulatorWaveformCollate', simulate_type='simulate_bf', simulator_config_path='./configs/se/simulator_config_bf_2mic.py'),
        # dict(type='MultiChannelWaveformCollateNoSplit', wav_key='mc_waveform', out_key='mc_waveform'),
        # dict(type='WaveformCollateNoSplit', wav_key='target_sou_data', out_key='target_sou_data'),
        dict(type='WaveformCollateNoSplit', wav_key='direction_all', out_key='direction_all'),
        dict(type='FlagCollate', wav_key='clean_direction_use', out_key='clean_direction_use'),
        dict(type='FlagCollate', wav_key='source_num', out_key='source_num'),
        dict(type='FlagCollate', wav_key='noise_flag', out_key='noise_flag'),
    ],

    device_transform = [dict(type='SimulatorModule', simulate_type='simulate_bf', simulator_config_path='./configs/se/simulator_config_bf_2mic.py')],

    inference_item_transform=[
        dict(type='PickleParser'),
        dict(type='SimpleWavParser', in_key='mc_waveform', out_key='mc_waveform', out_dtype='np.float32'),
    ],
    inference_batch_transform=[
        # general transform
        dict(type='MultiChannelWaveformCollateNoSplit', wav_key='mc_waveform', out_key='mc_waveform'),
        dict(type='FlagCollate', wav_key='target_direction', out_key='target_direction'),
        # transform for save_wav
        # dict(type='ListCollate', key='wavname', out_key='wavname'),
        # transform for spatial_response_and_sisnr
        dict(type='FlagCollate', wav_key='net_direction', out_key='net_direction'),
        dict(type='WaveformCollateNoSplit', wav_key='target_data', out_key='target_data'),
        dict(type='WaveformCollateNoSplit', wav_key='ovlp_vad', out_key='ovlp_vad'),
        dict(type='WaveformCollateNoSplit', wav_key='nonovlp_vad', out_key='nonovlp_vad'),
    ],
)
solution = dict(
    type='base_se_solution',
    se_model_type='BaseBFSolution',
    feat_extractor_type='BFSubbandFeatExtractor',
    feat_extractor=dict(
        frame_length=128,
        oversample_ratio=2,
        sampling_rate=16000,
        delay_version='high',
        mic_num=2,
        mic_space = 0.027,  # mic distance in linear array, radius in circular array
        array_type = 'linear',
        mask_type = "mask_clean_fixbeam",
        net_in = "noisy_steer_mic",
    ),
    bf_net_type='spatialbeam',
    bf_net=dict(
        mic_num = 2,
        out_channel1 = 2,
        out_channel2 = 2,
    ),
    # criterion
    criterion_type='BFMseRelMseSisnr',
    criterion=dict(
        mask_coeff = 40.0,
        rel_mask_coeff = 25.0,
        sisnr_coeff = 0.0,
        mask_under_penalty = 1.1,
    ),
    # evaluation
    evaluation='frame',
    eval_seg='oveall',
    # inference
    inference = dict(
        save_dir ='./inference/bf',
        remote_save_dir = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_weekly_test/bf/save_metrics/',
        type='spatial_response_and_sisnr',  # save_wav: 保存增强后的wav；spatial_response_and_sisnr: 空间响应以及sisnr
        # configs used in 'save_wav' mode
        # fix_gain_on=True,
        # target_level=32768,
        # configs used in 'spatial_response_and_sisnr' mode
        # net_direction_config=dict(
        #     start = 20,
        #     end = 90,
        #     steps = 8,
        # ),
        src_direction_step=10,
        metric = 'BfInferMetricSpatialResponseAndSisnr',
        log_metric=False,
        final_call='process'
        ),
)
# runtime settings
work_dir = './se_test'
# training and testing settings
train = dict(
    # resume='latest.pth',
    #resume_optimizer=True,
    #resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/shenchen/dolphin/MDD/test/dfsmnln_online_pretrain/dfsmnln_online_pretrain/checkpoints/step_1000.pth',
    save_after_epoch=True,  # save checkpoint after each epoch
    drop_when_epoch_end=True,
    max_epochs=10000,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=True,
        policy='StepLR',  # use torch lr_scheduler
        gamma=0.5,
        step_size=5
        ),
    checkpoint_config=dict(
        interval=1000000, # steps
        max_keep_ckpts=-1,
    ),
    best_metric_config=dict(
        # best_metric_name='backward_loss',  # None or not provided, won't trigger best metric comparison
        # best_metric_type='min',
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_weekly_test/bf/',
    save_root='./',
    save_dir='dolphin_bf',
    save_name='bf_2mic',
    amp_level='O0',
    deterministic=False,
    max_cufft_plan_cache=512,
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
