project = "vs_tt"
runner = 'RNNTCASERunner'
# increment to add and override args.
data = dict(
    data_root = [
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/YT_en-US_speech/YT_en-US_speech_12kh_uppercase/dolphin/',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/TT_en-US_speech_18.5kh_new/dolphin/',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_en_us_labelled_5.5kh_new_20220204/tf_record/train/',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_en_gb_labelled_5kh_20220204/tf_record/train/',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_cla_feedback_en_us_7kh_20220204/tf_record/train/',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_en_au_labelled_5kh_20220204/tf_record/train/',
        '/mnt/bd/speech-asr-vi-k8s/user/tangyu.yt/train_data/en_common_voice',
        ],
    train_file_list = [
        '["train_sub{}".format(i) for i in list(range(53)) + list(range(54, 88))]',
        '["train_sub{}".format(i) for i in range(124)]',
        '["tfrecord-sub{}".format(i) for i in range(184)]',
        '["tfrecord-sub{}".format(i) for i in range(328)]',
        '["tfrecord-sub{}".format(i) for i in range(191)]',
        '["tfrecord-sub{}".format(i) for i in range(266)]',
        '["train_sub{}".format(i) for i in range(1, 24)]',
        ],
    valid_data_root = [
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_en_us_labelled_5.5kh_new_20220204/tf_record/cv',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_en_gb_labelled_5kh_20220204/tf_record/cv',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_cla_feedback_en_us_7kh_20220204/tf_record/cv/',
        '/mnt/bd/speech-asr-vi-k8s/data/training_platform/lingvo/input/tt_en_au_labelled_5kh_20220204/tf_record/cv/',
        ],
    valid_file_list = [
        '["tfrecord-sub%d"]',
        '["tfrecord-sub%d"]',
        '["tfrecord-sub%d"]',
        '["tfrecord-sub%d"]',
        ],
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    shuffle=True,
    drop_last=False,
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
        dict(type='WavDurationFilter',key='waveform',min_duration=0.01,max_duration=30),
        dict(type='SpeedPerturbation',p=0.666,speed_rate_list=[0.95,1.05]),
        dict(type='BadNumeralsFilter'),
        dict(type='PunctuationFilter'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        #dict(type='SpecialTokenFilter', filter_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<b>']),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
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
        dict(type='BPE', use_eos=False, keep_empty_label=True),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', need_adapt_target=True),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    model_type='RnntCaseModel',
    # front_end
    front_end_type='Conv2dPooling4',
    input_concat_size=1,
    downsampling_size=4,
    front_end_conv0_ch=128,
    front_end_conv1_ch=128,
    # backbone
    acoustic_backbone_type='ConformerBackbone',
    # conformer_dual_mode=True,
    conformer_normalize_before=True,
    conformer_attention_heads=8,
    conformer_mask_topology='[[128,1]]*2 + [[128,0]]*10',
    conformer_linear_units=2048,
    conformer_num_blocks=12,
    conformer_dropout_rate=0.1,
    conformer_positional_dropout_rate=0.1,
    conformer_attention_dropout_rate=0.1,
    conformer_positionwise_layer_type='linear',
    conformer_activation_fn='gelu',
    conformer_positionwise_conv_kernel_size=1,
    conformer_macaron_style=1,
    conformer_pos_enc_layer_type='fix_rel_pos',
    conformer_selfattention_layer_type='rel_selfattn',
    conformer_layer_order='mhsa_before_conv',
    conformer_use_cnn_module=1,
    conformer_cnn_module='ConvolutionModule',
    conformer_cnn_module_kernel='15,0',
    conformer_cnn_norm_type='layer_norm',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_memory_size=512,
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # 2-pass encoder
    twopass_acoustic_args = dict(
        # backbone
        acoustic_backbone_type='ConformerBackbone',
        conformer_normalize_before=True,
        conformer_attention_heads=8,
        conformer_mask_topology='[[128,13]]*2',
        conformer_linear_units=2048,
        conformer_num_blocks=2,
        conformer_dropout_rate=0.1,
        conformer_positional_dropout_rate=0.1,
        conformer_attention_dropout_rate=0.1,
        conformer_positionwise_layer_type='linear',
        conformer_activation_fn='gelu',
        conformer_positionwise_conv_kernel_size=1,
        conformer_macaron_style=1,
        conformer_pos_enc_layer_type='fix_rel_pos',
        conformer_selfattention_layer_type='rel_selfattn',
        conformer_layer_order='mhsa_before_conv',
        conformer_use_cnn_module=1,
        conformer_cnn_module='ConvolutionModule',
        conformer_cnn_module_kernel='15,0',
        conformer_cnn_norm_type='layer_norm',
        conformer_layernorm_interval=0,
        conformer_weight_scale=1.0,
        conformer_half_pooling=0,
        backbone_memory_size=512,
        backbone_pool_type='MaxPoolTimeReduce',
    ),
    # layer_norm_after=True,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.1,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=1056,
    adaptive_tail_size=150,
    adaptive_tail_groups=40,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=500,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    backbone_clamp_inf=True,
    mtl_clamp=True,
    head_clamp=True,
    dropout=0.1,
    #fast_emit 1.001 到 1.01可调
    fast_emit=1.004,
    causal_loss_weight=0.5,
)

# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=40,
    max_iters=2000000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.8,
        step=[4 + i for i in range(100)]
        ),
    checkpoint_config=dict(
        interval=20000, # steps
        max_keep_ckpts=100,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    #remote_save_root='',
    save_root='./demo',
    save_dir='conformer_case',
    save_name='tmp',
    amp_level='O1',
    )
valid = dict(
    interval=20000, # iter
    )
inference = dict(
    test_sets='test_clean|test_other',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    language='en-US',
    use_batch_beam=True,
    causal_mode=True,
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
        max_norm=50,
        norm_type=2,
        ),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        #block_momentum跟卡数有关系，0.97/32, 0.94/16,0.9/8
        block_momentum=0.94,
        warmup_steps=5000,
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
