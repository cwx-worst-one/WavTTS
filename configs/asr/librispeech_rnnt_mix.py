# writed by sunjingyu
# transformer transducer baseline 4*V100 40w steps
# trial:https://cloud.bytedance.net/arnold/job/11108/task/800534/trial/2144930
# CER : test celan/other 3.67/8.27
project = "librispeech_rnnt"
runner = 'RNNTRunner'
# increment to add and override args.
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/librispeech_wav/',
    train_file_list='["train_sub{}".format(i) for i in range(16)]',
    valid_file_list='["dev_clean", "dev_other"]',
    meta_file="meta",
    fbank_dim=80,
    io_cache_size=2048, # cache_size used by FalconReader
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=12800,
    shuffle=True,
    drop_last=False,
    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='ProtoParser'),
        dict(type='DecordRaw', key2type={'frames':'int16', 'transcript':'bytes', 'uttid':'bytes'}),
        dict(type='LabelParser', in_key='transcript'),
        dict(type='WavConvert', in_key='frames'),
        dict(type='WavParser'),
        dict(type='SpeedPerturbation',p=0.666,speed_rate_list=[0.95,1.05]),
        dict(type='AddNoise',
            noise_shards=1,
            p=0.4,
            noise_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/librispeech_noise',
            min_snr=7.5, max_snr=20.0
            ),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False, bpe_dropout=0.1),
        dict(type='CMVN', key='fbank'),
        dict(type='DynamicTimeMask',
             time_mask_size=20,
             time_mask_block=60,
             replace_with_zero=False,
             inplace=True,
             time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             inplace=True, replace_with_zero=False),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='ProtoParser'),
        dict(type='DecordRaw', key2type={'frames':'int16', 'transcript':'bytes', 'uttid':'bytes'}),
        dict(type='LabelParser', in_key='transcript'),
        dict(type='WavConvert', in_key='frames'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
    weight=1, # kv weight
    parquet={
        'path_list':['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/datasets/parquet/librispeech_wav/train_sub000/data{}.parquet'.format(i) for i in range(16)],
        'weight':1,
    }
)
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
    # front_end
    front_end_type='VGGPosFrontEnd',
    input_concat_size=8,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='OfflineTransformerBackbone',
    backbone_topology='[[-1,-1,1]]*20',
    backbone_layer_gap=100,
    dfsmn_bn_flag=1,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    self_attn_heads=8,
    self_attn_dropout=0.1,
    dropout=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=1,
    self_attn_activation_fn='relu',
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=3900,
    adaptive_tail_size=148,
    adaptive_tail_groups=1,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    mtl_type='ctc',
    mtl_head='HybridCEHead',
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    max_iters=400000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.95,
        step=[100000 + i * 5000 for i in range(140)]
        ),
    checkpoint_config=dict(
        interval=5000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin',
    save_root='./demo',
    save_dir='librispeech_rnnt',
    save_name='rnnt_transformer_base',
    amp_level='O1',
    )
valid = dict(
    interval=5000, # iter
    )
inference = dict(
    test_sets='test_clean|test_other',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    language='en',
    # remote_stat_dir="hdfs://haruna/xxxx",
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
    weight_decay=1e-5,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=20,
        norm_type=2,
        ),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.75,
        warmup_steps=20000,
        warmup_sync=True,
        use_nesterov=True,
        average_sync=True,
    )
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
    ])
