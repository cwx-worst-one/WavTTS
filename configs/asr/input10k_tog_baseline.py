project = "rnnt"
runner = 'RNNTRunner'
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    # add ce sy+ctc label
    train_file_list = '["train_sub{}".format(i) for i in range(1024)]',
    valid_file_list = '["cv"]',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav_testset_basic_7_21',
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    next_retry=6000,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,2250,2500,2750,3000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000',
    use_old_bucket=True,
    bucket_schedule_key='length',
    batch_means_tokens=1,
    max_batch_size=40960,
    max_batch_scale=13,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', max_len=30., skip_prob=0.2),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='Char2Phone', mask_rate=0.15),
        dict(type='LabelLengthFilter', in_out_ratio=4, strict_mode=False),
        dict(type='CMVN', key='fbank'),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10,
             replace_with_zero=False, inplace=True),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             replace_with_zero=False, inplace=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80, strict_mode=False),
        dict(type='PhoneCollate'),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80, strict_mode=False)
    ],
    device_transform=[
        dict(type='PhonePredict')
    ]
)
solution = dict(
    type='base_rnnt_solution',
    model_type='RnntTogModel',
    # init
    xavier_init=False,
    # front_end
    front_end_type='Conv2dPooling4',
    front_end_conv0_ch=512,
    front_end_conv1_ch=512,
    # backbone
    acoustic_backbone_type='MaskedConformerBackbone',
    conformer_normalize_before=1,
    conformer_attention_heads=8,
    conformer_mask_topology='[[160, 160]]*12',
    conformer_linear_units=2048,
    conformer_num_blocks=12,
    conformer_dropout_rate=0.1,
    conformer_positional_dropout_rate=0.1,
    conformer_attention_dropout_rate=0.0,
    conformer_positionwise_layer_type='linear',
    conformer_activation_fn='gelu',
    conformer_positionwise_conv_kernel_size=1,
    conformer_macaron_style=1,
    conformer_pos_enc_layer_type='fix_rel_pos',
    conformer_selfattention_layer_type='rel_selfattn',
    conformer_use_cnn_module=1,
    conformer_cnn_module='ConvolutionModule',
    conformer_cnn_module_kernel='15,15',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_memory_size=512,
    # tog
    tog_front_end_args=dict(
        token_frontend_type='TextEmbeddingTransformer',
        input_token_size=300,
        backbone_memory_size=512,
        backbone_hidden_size=2048,
        self_attn_heads=8,
        self_attn_dropout=0.1,
        dropout=0.1,
        self_attn_activation_dropout=0,
        self_attn_layer_norm_before=1,
        self_attn_activation_fn='relu',
    ),
    # head
    head_type='RnntBaseHead',
    head_hidden_size=768,
    head_lowrank_size=512,
    layer_norm_after=True,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=512,
    predictor_lstm_hidden_size=2048,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.0,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=768,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=968,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=500,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=True,
    # inference
    cer_blank_scale=0.2,
    # use pretrain encoder, 1. can be a local path 2. support hdfs path
    pretrain_encoder_path=None,
    freeze_pretrain_acoustic_model=False,
    # use ilmt
    ilmt_weight=0.4,
)
# runtime settings
work_dir = './dfsmn_ag20k_opt'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    resume_progress=True,
    resume_lr_scheduler=True,
    max_epochs=90,
    max_iters=2000000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=10000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.9,
        step=[10 + i for i in range(100)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    save_root='./demo',
    save_dir='202207_text_adaptation',
    save_name='v1',
    amp_level='O1')
valid = dict(
    interval=10000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets="aishell_2/aishell_2|aishell_2019a/aishell_2019a|aishell_2019c/aishell_2019c|lark_new_energe_61_split/lark_new_energe_61_split|dongchedi_hot_words_20220715_7645",
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    rnnt_temperature=1.0,
    use_batch_beam=True,
    )
nnlm = dict(
    lm_type="",
    )
# optimizer
optimizer = dict(
    type='FusedAdam',
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.05,
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
        block_momentum=0.9,
        warmup_steps=5000,
        use_nesterov=True,
    )
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=500,  # log at every 500 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
        ])
