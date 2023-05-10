project = "aed_classification"
runner = 'AedClassificationRunner'

# increment to add and override args.
data = dict(
    #train_data_root='hdfs://haruna/home/byte_arnold_lq_speech_asr/user/hekexin/feature/aed_general/tfrecord/aed_14_v0_10h/train_torch_meta/',
    #valid_data_root='hdfs://haruna/home/byte_arnold_lq_speech_asr/user/hekexin/feature/aed_general/tfrecord/aed_14_v0/eval_torch/',
    #train_file_list='["train-sub{}".format(i) for i in range(4)]',
    #valid_file_list='["eval-sub{}".format(i) for i in range(4)]',
    train_file_pattern='hdfs://haruna/home/byte_arnold_lq_speech_asr/user/hekexin/feature/aed_general/tfrecord/aed_14_v0/train_10h/train-sub*',
    valid_file_pattern='hdfs://haruna/home/byte_arnold_lq_speech_asr/user/hekexin/feature/aed_general/tfrecord/aed_14_v0/eval_10h/eval-sub*',
    filter_len=50,
    min_len=100,
    max_len=100,
    eval_len=100,
    num_classes=14,
    fbank_dim=64,
    fbank_channel=1,
    chunk_size=20,
    bucket_schedule='',
    # bucket_schedule_val='2000',
    # bucket_schedule_key='wav_mel_spectrigram',
    batch_strategy='SimpleBatching',
    max_batch_size=256,
    drop_last=True,
    batch_means_tokens=1,
    use_eos=True,
    global_shuffle=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LengthFilter', key='wav_mel_spectrogram', max_len=-1),
        dict(type='AppendFrames', value=0, key='wav_mel_spectrogram'),
        dict(type='CutFeature', random_cut=True, key='wav_mel_spectrogram'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LengthFilter', key='wav_mel_spectrogram', max_len=-1),
        dict(type='AppendFrames', value=0, key='wav_mel_spectrogram'),
        dict(type='CutFeature', random_cut=False, key='wav_mel_spectrogram'),
    ],
    train_batch_transform=[
        dict(type='ListCollate', key='video_id'),
        dict(type='OnehotLabelCollate', key='labels'),
        dict(type='LengthRandomClipCollate', random_clip=True, same_start=True, key='wav_mel_spectrogram', out_key='features'),
    ],
    valid_batch_transform=[
        dict(type='ListCollate', key='video_id'),
        dict(type='OnehotLabelCollate', key='labels'),
        dict(type='LengthRandomClipCollate', random_clip=False, same_start=True, key='wav_mel_spectrogram', out_key='features'),
    ],
)
solution = dict(
    model_type='AedClassificationModel',
    classifier='Vggish10',
    use_batch_norm=True,
    batch_norm_decay=0.997,
    batch_norm_eps=1e-5,
    feature_dim=512,
    dropout_rate=0.5,
    multi_labels=False,
    criterion_type='XentropyWithoutMask',
    label_smoothing=0.0,
    weight_decay=0.0001,
    topk=2,
    use_detection=False,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    max_epochs=200,
    max_iters=0,
    drop_when_epoch_end=True,
    lr_scheduler=dict(
        policy='Fixed',
        by_epoch=True,
    ),
    checkpoint_config=dict(
        interval=1000, # steps
        max_keep_ckpts=-1,
    ),
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhangjun.jarry/saved_models/dolphin_test',
    save_root='./aed',
    save_dir='classification',
    save_name='14',
    amp_level='O0',

    # gradient accumulation
    grad_accum_step=1,
)
valid = dict(
    interval=1000,
    best_metric_name='top1',
    best_metric_type='topk',
    topk=5,
)
inference = dict()
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    # grad_clip=dict(
    #     max_norm=1.0,
    #     norm_type=2,
    # ),
    grad_clip=None,
    # 'accum_after' & 'accum_before'
    # grad_clip_mode='accum_after',
    max_grad_clip=0.0,
)
log_level = 'INFO'
log_config = dict(
    interval=100,  # log at every 50 iterations
)
