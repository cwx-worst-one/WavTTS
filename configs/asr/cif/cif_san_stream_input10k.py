project = "san_cif"
runner = 'BaseCifRunner'

# increment to add and override args.
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav_testset_basic_7_21/',
    train_file_list = '["train_sub{}".format(i) for i in range(1024)]',
    valid_file_list = '["cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta_ag350k_lzl",
    fbank_dim=80,
    fbank_channel=1,
    chunk_size=20,
    bucket_schedule='318, 398, 468, 522, 592, 646, 704, 781, 838, 898, 947, 998, 1098, 1209, 1314, 1398, 1498, 1558, 2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    max_batch_size=18000,
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
        #dict(type='TimeMask', time_mask_size=20, time_mask_num=10, replace_with_zero=True),
        #dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        # in_out_ratio is used for fair comparison with other models in evaluation, 8 is actually used in cif
        dict(type='LabelLengthFilter', in_out_ratio=4),
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
        dict(type='CharCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    model_type='BaseCifModel',
    vocab_size=9759,
    use_fused_kernel=True,

    # front end settings
    use_cnn_frontend=True,
    conv_norm_type='layer_correct',
    down_sample_conv_num_layers=1,
    down_sample_conv_num_filters=64,
    additional_module='new_mu33_2',
    additional_module_num_layers=1,
    additional_module_num_filters=64,

    # self-attention encoder/decoder setting
    self_attention_type='dot_product',
    num_heads=8,
    hidden_size=640,
    filter_size=2560,
    ffn_layer='dense_relu_dense',
    layer_preprocess_sequence='n',
    layer_postprocess_sequence='da',
    norm_type='layer',
    norm_epsilon=0.000001,
    proximity_bias=True,

    front_end_type='CifOriginalFrontend',
    acoustic_backbone_type='CifSelfAttentionEncoder',

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
    use_chunk_hopping=True,
    chunk_size=80,
    hop_size=40,
    ch_type='fixed_future',
    ch_fixed_future_size=40,
    last_chunk_pad=True,
    num_history_chunk=8,

    # regularizer
    layer_prepostprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.0,

    # extra
    loss_multiplier=2.0,
    shared_embedding_and_softmax_weights=False,
    multiply_embedding_mode='sqrt_depth',

    # cif part
    produce_weights_type='conv',
    conv_cif_num_layers=1,
    conv_cif_width_string='3',
    conv_cif_num_filters=640,
    conv_cif_dropout=0.0,
    dense_cif_units=640,

    cif_weight_threshold=0.99999,
    use_scaling_strategy=True,
    use_tail_handling=True,

    # all loss setting
    calculated_loss='ce_loss, quantity_loss, ctc_loss_on_encoder',

    # ce loss
    label_smoothing=0.1,
    ls_type='uniform',

    # ctc loss on the encoder
    ctc_loss_on_encoder=True,
    ctc_loss_lambda=0.5,

    # quantity loss
    quantity_loss_lambda=1.0,

    # criterion
    criterion_type='CifLoss',
    eos_id=2,
    save_until_pad=True,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    max_epochs=50,
    max_iters=32000,
    lr_scheduler=dict(
        policy='Plateau',
        by_epoch=False,
        start_decay_step=0,
        end_decay_step=1,
        each_decay_step=1,
        peak_lr=0.0001,
        init_lr=0,
        end_lr=0.0001,
        warmup='linear',
        warmup_iters=1,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        ),
    checkpoint_config=dict(
        interval=1600, # steps
        max_keep_ckpts=-1,
    ),
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhangjun.jarry/saved_models/dolphin_test',
    save_root='./cif',
    save_dir='san_cif',
    save_name='san_cif',
    amp_level='O1',
    # evaluation setting
    beam_size=1,
    top_beams=1,

    # gradient accumulation
    grad_accum_step=1,
    # 'AVG' 'SUM'
    grad_accum_mode='SUM'
    )
valid = dict(
    average_ckpts=True, # iter
    num_averaging_ckpts=10,
    beam_size=10,
    top_beams=1,
    output_probs=False,
    not_show_results=False,
    interval=1600,
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
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-8,
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
    interval=100,  # log at every 100 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
