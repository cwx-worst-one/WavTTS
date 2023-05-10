project = "ts_vad_train"
runner = 'BaseVadTsRunner'

# increment to add and override args.
data = dict(
    train_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/ts_vad/AliMeeting_data/pkg/wav/Train_Ali_far_reorder',
    ],
    train_file_list=[
        '["train{}".format(i) for i in range(8)]',
    ],
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/ts_vad/AliMeeting_data/pkg/wav/Eval_Ali_far_reorder',
    valid_file_list='["train0"]',

    fbank_dim=80,
    batch_size_per_class=1,
    batch_class_num=16,
    shuffle=True,
    drop_last=False,
    valid_chunk_size=2,

    prefetch_worker_num=5,

    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WaveFormat', key='src', swap_dim=True),
        dict(type='SelectWavChannel', wav_key='src', target_channel=0),
        dict(
            type='WavSplicing',
            wav_key='src',
            target_len=16,
            mode='time',
            sample_rate=16000,
            label_key='label',
            random_clip=True),
        dict(
            type='AlignSpeakerLabel', speaker_key='speakers', label_key='label', target_speakers=4),
        # Online augmentation
        dict(type='PreSelAug', aug_probs={'no_aug': 1, 'music': 1, 'speech': 1, 'noise': 1, 'reverb': 1}),
        dict(type='KaldiAddNoise', key='src', noise_prefix='music', noise_type='music', noise_snr=[10,7,5], ground_mode='background'),
        dict(type='KaldiAddNoise', key='src', noise_prefix='speech', noise_type='speech', noise_snr=[19,17,15,13,11], noise_times=[3,4,5,6,7],  ground_mode='background'),
        dict(type='KaldiAddNoise', key='src', noise_snr=[5, 10], ground_mode='foreground', noise_interval=1),
        dict(type='KaldiAddRir', key='src', rir_key='waveform'),
    ],

    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WaveFormat', key='src', swap_dim=True),
        dict(type='SelectWavChannel', wav_key='src', target_channel=0),
        dict(
            type='WavSplicing',
            wav_key='src',
            target_len=16,
            mode='time',
            sample_rate=16000,
            label_key='label',
            random_clip=False),
        dict(
            type='AlignSpeakerLabel', speaker_key='speakers', label_key='label', target_speakers=4),
    ],

    batch_transform=[
        dict(type='StackCollate', key='src'),
        dict(type='StackCollate', key='label'),
        dict(type='ListCollate', key='speakers'),
    ],

    train_device_transform=[
        dict(
            type='SpeedKaldiFbank',
            in_key='src',
            out_key='feature',
            dither=1.0,
            device='cuda',
            out_numpy=False,
            fbank_dim=80),
        dict(
            type='DynamicCmvn',
            key='feature',
            cmvn_type='sliding_dynamic_C',
            center=True),
    ],

    valid_device_transform=[
        dict(
            type='SpeedKaldiFbank',
            in_key='src',
            out_key='feature',
            dither=1.0,
            device='cuda',
            out_numpy=False,
            fbank_dim=80),
        dict(
            type='DynamicCmvn',
            key='feature',
            cmvn_type='sliding_dynamic_C',
            center=True),
    ],
)

solution = dict(
    type='vad_ce_solution',
    ce_type='VadTsCeModel',
    # encoder
    encoder_type='TsVadEncoder',
    freeze_frontend=True,
    normalize_embedding=True,
    use_same_utt_embedding=True,

    # Choose the embedding we use
    embedding_list=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/ts_vad/AliMeeting_data/pkg/embedding/resnet34/Train_Ali_0/emb_train',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/ts_vad/AliMeeting_data/pkg/embedding/resnet34/Eval_Ali_0/emb_train',
    ],

    # resnet front-end
    backbone_topology='[["conv", 32, [3, 3], [1, 1]],' \
        '["basic_block", 3, 32, [3, 3], [1, 1]],' \
        '["basic_block", 4, 64, [3, 3], [2, 2]],' \
        '["basic_block", 6, 128, [3, 3], [2, 2]],' \
        '["basic_block", 3, 256, [3, 3], [2, 2]]]',
    resnet_groups=1,
    resnet_base_width=64,
    resnet_zero_init_residual=True,
    activation_fn='relu',
    normalization_fn='batch_norm',
    normalization_after=False,
    batchnorm_momentum=0.01,
    batchnorm_eps=0.001,

    # local pooling
    pooling_orders=[1, 2],
    pooling_frames=2,

    # linear proj
    segment_topology='[128]',
    segment_last_layer_use_norm=True,
    segment_last_layer_use_nonlinear=False,

    # backend encoder
    tsvad_encoder_type='transformer',
    encoder_num_layers=2,
    encoder_attn_heads=4,
    encoder_ffn_embed_dim=1024,
    encoder_dropout=0.1,
    encoder_self_attn_activation_fn='relu',

    # backend decoder
    lstm_hidden_size=1024,
    lstm_num_layers=2,
    lstm_dropout=0.1,
    lstm_bidirectional=True,

    target_speaker_num=4,

    # criterion
    criterion_type='Bientropy',
    xavier_init=False,
    detection_threshold=0.5,
    label_pooling=8,
)

# runtime settings
work_dir = './ts_vad_train'

# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    # type='FusedAdam',
    type='AdamW',
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
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
    max_epochs=100,
    max_iters=8000000000,

    # Set a larger value for data simulation.
    iters_per_epoch=200,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=200*3,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='fixed',
        # gamma=0.5,
        # step=[33 + 5 * i for i in range(10)]
    ),
    checkpoint_config=dict(
        max_keep_ckpts=100,
    ),

    save_root='./alimeeting',
    save_dir='ts_vad_train',
    save_name='resnet34_transformer_freeze',
    amp_level='O1',

    # The pretrained resnet
    resume_pretrain_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin/data_fbank80/m2met/resnet34_128_amsoftmax_m0.20_spdaug/checkpoints/latest.pth',
    resume='latest.pth',
    save_after_epoch=True,
)

valid = dict(
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
    ]
)