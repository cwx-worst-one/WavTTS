project = "san_cif"
runner = 'BaseCifRunner'

# increment to add and override args.
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/cif_data_convert/convert_data/meta_path_ag160k_wav/',
    train_file_list = '["train_sub{}".format(i) for i in range(1024)]',
    valid_file_list = '["cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta_cif_add_pie",
    fbank_dim=80,
    fbank_channel=1,
    chunk_size=20,
    bucket_schedule='318, 398, 468, 522, 592, 646, 704, 781, 838, 898, 947, 998, 1098, 1209, 1314, 1398, 1498, 1558, 2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    max_batch_size=81920,
    batch_means_tokens=1,
    use_eos=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='ConcateEnLetters'),
        dict(type='BPE', use_eos=True),
        dict(type='CMVN', key='fbank'),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10, replace_with_zero=True),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='ConcateEnLetters'),
        dict(type='BPE', use_eos=True),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate', rnnt_format=False),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='CharCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    model_type='BaseCifModel',
    vocab_size=9759,
    scp_feat_total_dim=80,
    scp_feat_channel_num=1,

    # init
    xavier_init=False,
    # front_end
    front_end_type='Conv2dPooling',
    front_end_conv0_ch=128,
    front_end_conv1_ch=128,
    # backbone
    acoustic_backbone_type='ConformerBackbone',
    conformer_mask_topology='[[160, 160]]*15',
    backbone_memory_size=512,
    conformer_linear_units=2048,
    conformer_num_blocks=15,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    conformer_dropout_rate=0.1,
    conformer_positional_dropout_rate=0.1,
    conformer_pos_enc_layer_type='rel_pos',
    conformer_normalize_before=1,
    conformer_attention_heads=8,
    conformer_attention_dropout_rate=0.0,
    conformer_positionwise_layer_type='linear',
    conformer_positionwise_conv_kernel_size=1,
    conformer_activation_fn='gelu',
    conformer_macaron_style=1,
    conformer_selfattention_layer_type='rel_selfattn',
    conformer_use_cnn_module=1,
    conformer_cnn_module='ConvolutionModule',
    conformer_cnn_module_kernel='15,15',
    conformer_layernorm_interval=0,
    backbone_layer_gap=100,

    # estimator
    conv_norm_type='layer_correct',

    # # front end settings
    # use_cnn_frontend=True,
    # conv_norm_type='layer_correct',
    # down_sample_conv_num_layers=1,
    # down_sample_conv_num_filters=64,
    # additional_module='new_mu33_2',
    # additional_module_num_layers=1,
    # additional_module_num_filters=64,
    # num_frame_stacking=3,
    # num_frame_striding=3,

    # self-attention encoder/decoder setting
    self_attention_type='dot_product',
    num_heads=8,
    hidden_size=512,
    filter_size=2048,
    ffn_layer='dense_relu_dense',
    layer_preprocess_sequence='n',
    layer_postprocess_sequence='da',
    norm_type='layer',
    norm_epsilon=0.000001,
    pos=None,
    proximity_bias=True,

    # attention_key_channels:
    # attention_value_channels:
    # parameter_attention_key_channels:
    # parameter_attention_value_channels:
    max_relative_position=0,

    num_encoder_layers=15,
    encoder_self_attention_type='dot_product',
    sa_pooling_layers=[5, 10],

    num_decoder_layers=2,
    decoder_self_attention_type='dot_product',

    # use teacher_forcing / pss
    use_teacher_forcing=True,
    use_pss=False,
    pss_rate=0.0,

    # chunk hopping
    use_chunk_hopping=False,
    chunk_size=128,
    hop_size=64,
    ch_type='middle',
    ch_fixed_future_size=32,
    load_trained_offline_model=True,
    ch_cif_dir='./exp/ag20k_80fbank_char_end2end_stl_asr/cif_base_exp1/models',

    # regularizer
    layer_prepostprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.0,

    # extra
    loss_multiplier=2.0,
    shared_embedding_and_softmax_weights=False,
    multiply_embedding_mode='sqrt_depth',
    symbol_modality_num_shards=1,

    # cif part
    produce_weights_type='conv',
    conv_cif_num_layers=1,
    conv_cif_width_string='3',
    conv_cif_num_filters=512,
    conv_cif_dropout=0.0,
    dense_cif_units=512,

    cif_weight_threshold=0.99999,
    use_scaling_strategy=True,
    use_tail_handling=True,
    add_eos_to_target=False,

    # all loss setting
    calculated_loss='ce_loss, quantity_loss, ctc_loss_on_encoder',

    # ce loss
    label_smoothing=0.1,
    ls_type='uniform',

    # confidence penalty
    using_confidence_penalty=False,
    cp_lambda=0.0,

    # ctc loss on the encoder
    ctc_loss_on_encoder=True,
    ctc_loss_lambda=0.5,

    # quantity loss
    quantity_loss_lambda=1.0,

    # Fusion methods
    # use sf to represent shallow fusion
    use_shallow_fusion=False,
    sf_lm_dir='./exp/lm_dir/.../models',
    sf_lm_score_lambda=0,

    sf_lm_num_layers=3,
    sf_lm_filter_size=2048,
    sf_lm_hidden_size=512,
    sf_lm_num_heads=4,
    sf_lm_proximity_bias=True,
    sf_lm_attention_type='dot_product',
    sf_lm_attention_dropout=0.0,

    # use lm rescore
    use_lm_rescore=False,
    rescore_beam_size=10,
    rescore_lm_lambda=0.2,

    joint_ctc_cif_decoding=False,
    joint_decoding_lambda=0.08,
    joint_only_ctc=False,
    joint_ctc_temperature=1.0,
    joint_cif_temperature=1.0,

    eos_id=2,
    # criterion
    criterion_type='CifLoss',
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    max_epochs=40,
    max_iters=100000000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=2000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.8,
        step=[14 + i for i in range(100)]
        ),
    checkpoint_config=dict(
        interval=4000, # steps
        max_keep_ckpts=-1,
    ),
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhangjun.jarry/saved_models/dolphin_test',
    save_root='./cif',
    save_dir='san_cif',
    save_name='san_cif',
    amp_level='O1',
    # evaluation setting
    text_decode_func='ag20k_cif_text_decode',
    beam_size=1,
    top_beams=1,

    # gradient calculation
    used_loss_names='ce_loss, quantity_loss, ctc_loss_on_encoder',
    summary_freq=999999,
    logging_freq=100,

    # training process control
    early_stop_metric='wer',
    metric_not_grow_limits=20000,
    save_checkpoint_freq=4000,
    total_train_step=90000,
    just_eval=False,

    # extra
    write_results=True,
    eval_on_dev=True,
    use_daisy_chain_getter=False,
    max_saved_checkpoints=20,

    # gradient accumulation
    grad_accum_step=2,
    # 'AVG' 'SUM'
    grad_accum_mode='SUM'

    )
valid = dict(
    average_ckpts=True, # iter
    num_averaging_ckpts=10,
    text_decode_func='ag20k_cif_text_decode',
    beam_size=10,
    top_beams=1,
    output_probs=False,
    not_show_results=False,
    interval=4000,
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='eval_aishell_2|eval_aishell_2019a|eval_aishell_2019c|eval_thchs|eval_ceo_external_refine_20201223|eval_smart_dog_refine_20201223|eval_learning|eval_toutiao_school|eval_meeting_refine_20201223|eval_lark_meeting_remove_zeros_refine_20201223|eval_lark_chinese_refine_20201223|eval_lark_mixed_refine_20201223|eval_h_test_kuaisu|eval_h_test_putong|eval_h_test_mobile',
    beam_size=10,
    nbest=1,
    cif_temperature=1.0
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='Adam',
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.0,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=1.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    grad_clip_mode='accum_before',
    max_grad_clip=0.0,
 )
log_level = 'INFO'
log_config = dict(
    interval=100,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
