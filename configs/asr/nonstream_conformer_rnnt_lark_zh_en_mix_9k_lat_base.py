project = "dfsmn_rnnt"
runner = 'RNNTRunner'
#_base_ = [
#    '../_base_/datasets/ag20kh_lzl.py',
#]
# increment to add and override args.
data = dict(
    data_root = [
        'hdfs://haruna/home/byte_ailab_speech_algo/data/zy/train_data/pure_zh_3000h_20220610',
        'hdfs://haruna/home/byte_ailab_speech_algo/data/zy/train_data/zh_en_2000h_20220610',
        'hdfs://haruna/home/byte_ailab_speech_algo/data/zy/train_data/zh_en_1000h_20220411',
        'hdfs://haruna/home/byte_arnold_lq_speech_asr/data/lingvo_trainset/asr_video_dou' + \
                'yin_english_5000h/train',
    ],
    train_file_list = [
        '["sub_{}".format(i) for i in range(160)]',
        '["sub_{}".format(i) for i in range(315)]',
        '["sub_{}".format(i) for i in range(217)]',
        '["tfrecord-sub{}".format(i) for i in range(80)]',
    ],

    valid_data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav_testset_basic_7_21/',
    #valid_file_list = '["lark_5hmm_test_right/lark_5hmm_test_right", "lark9_10_test/lark9_10_test", "lark_e2e_7_8_english/lark_e2e_7_8_english", "lark_en_zy_2022q1_3230/lark_en_zy_2022q1_3230"]',
    valid_file_list = '["lark9_10_test/lark9_10_test"]',
    meta_data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/53w_meta_new_new',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta",
    fbank_dim=80,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    shuffle=True,
    global_shuffle=True,
    use_old_bucket=1,
    drop_last=False,
    chunk_size=20,
    max_batch_scale=15,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='ProtoParser'),
        dict(
            type='DecordRaw',
            key2type=dict(frames='int16', transcript='bytes', uttid='bytes')),
        dict(type='LabelParser', in_key='transcript'),
        dict(type='WavConvert', in_key='frames'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='PunctuationFilter', key='label', invalid_tokens_pattern=r"[^ 0-9a-zA-Z\u4e00-\u9fa5\']"),
        #dict(type='CodeSwitchTagAdd', key='label', add_token='^',
        #     special_pattern_list=['<[a-z]+>', '<[A-Z]+>']),
        #dict(type='BadCaseRemoveOrFilter', key='label', do_remove_punc=True, do_filter_special_list=False,
        #     external_remove_key=['哈哈哈','嗯','啊','恩']),
        # dict(type='MaskLabelByLang', keep_lang='zh', mask='<EN>', key='label', out_key='zh_label'),
        # dict(type='BPE', key='zh_label', out_key='zh_char', use_eos=False, skip_bpe=True, skip_list=['<pad>', '<unk>', '<EN>', '<CN>']),
        # dict(type='MaskLabelByLang', keep_lang='en', mask='<CN>', key='label', out_key='en_label'),
        # dict(type='BPE', key='en_label', out_key='en_char', use_eos=False, skip_bpe=True, skip_list=['<pad>', '<unk>', '<EN>', '<CN>']),
        # dict(type='BPE', use_eos=False, skip_bpe=True, skip_list=['<pad>', '<unk>', '<EN>', '<CN>']),
        dict(type='MaskLabelByLang', keep_lang='zh', mask='<unk>', key='label', out_key='zh_label'),
        dict(type='BPE', key='zh_label', out_key='zh_char', use_eos=False, skip_list=['<pad>', '<unk>']),
        dict(type='MaskLabelByLang', keep_lang='en', mask='<unk>', key='label', out_key='en_label'),
        dict(type='BPE', key='en_label', out_key='en_char', use_eos=False, skip_list=['<pad>', '<unk>']),
        dict(type='BPE', key='label', out_key='char', use_eos=False, skip_list=['<pad>', '<unk>']),
        dict(type='CMVN', key='fbank'),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10,
             replace_with_zero=False),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             replace_with_zero=False),
    ],

    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        # dict(type='CalculateFrameLength'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        #dict(type='CodeSwitchTagAdd', key='label', add_token='^',
        #     special_pattern_list=['<[a-z]+>', '<[A-Z]+>']),
        #dict(type='BPE', use_eos=False, split_code_switch=True, skip_list=['<pad>']),
        dict(type='MaskLabelByLang', keep_lang='zh', mask='<unk>', key='label', out_key='zh_label'),
        dict(type='BPE', key='zh_label', out_key='zh_char', use_eos=False, skip_list=['<pad>', '<unk>']),
        dict(type='MaskLabelByLang', keep_lang='en', mask='<unk>', key='label', out_key='en_label'),
        dict(type='BPE', key='en_label', out_key='en_char', use_eos=False, skip_list=['<pad>', '<unk>']),
        dict(type='BPE', use_eos=False, skip_list=['<pad>', '<unk>']),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='RefLabelCollate', out_key='label'),
        dict(type='CharCollate', key='zh_char'),
        dict(type='CharCollate', key='en_char'),
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
    type='base_rnnt_model',
    rnnt_type='LangAwareRnntModel',
    model_type='LangAwareRnntModel',
    # front_end
    front_end_type='Conv2dPooling4',
    # backbone
    acoustic_backbone_type='ConformerBackbone',
    language_aware_training=True,
    conformer_normalize_before=1,
    #conformer_normalize_before=0,
    conformer_attention_heads=8,
    conformer_mask_topology='[[160, 160]]*6',
    conformer_num_blocks=6,
    conformer_linear_units=2048,
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
    conformer_cnn_norm_type='layer_norm',
    conformer_cnn_module_kernel='15,15',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_topology='[[_1,_1,1]]*10',
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
    lang_aware_encoder_config = dict(
        acoustic_backbone_type='ConformerBackbone',
        language_aware_training=True,
        conformer_normalize_before=1,
        conformer_attention_heads=8,
        conformer_mask_topology='[[160, 160]]*6',
        conformer_num_blocks=6,
        conformer_linear_units=2048,
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
        conformer_cnn_norm_type='layer_norm',
        conformer_cnn_module_kernel='15,15',
        conformer_layernorm_interval=0,
        conformer_weight_scale=1.0,
        conformer_half_pooling=0,
        backbone_topology='[[_1,_1,1]]*10',
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
        # # head
        # head_type='RnntBaseHead',
        # head_hidden_size=768,
        # head_lowrank_size=512,
        # layer_norm_after=True,
        # mtl_head='HybridCEHead',
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
    mtl_type='ctc',
    mtl_head='HybridCEHead',
    lat_weight=0.5,
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=True,
    # inference
    cer_blank_scale=0.2,
    # NOTE: if do finetune, should close xavier_init!
    xavier_init=False
)
# runtime settings
work_dir = './dfsmn_ag20k_opt'
# training and testing settings
train = dict(
    # resume='latest.pth',
    # resume_optimizer=True,
    # dolphin will try to resume from these path in order:
    #   1. ${train.out_dir}/${train.resume}
    #   2. ${train.resume_hdfs_chkpt}
    #   3. ${train.remote_save_root}/${train.out_dir}/${train.resume}
    # note: ${train.out_dir} is ${train.save_root}/${train.save_dir}/${train.save_name}/checkpoints,
    # if it's not set in config file.
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/shenchen/dolphin/aiot_53w_finetune_from40w_32card_lrdecayhalf_online_model_tts_data_finetune_1e-7/dfsmnln_online/dfsmnln_online/checkpoints/step_350000.pth',
    resume_optimizer=False,
    resume_progress=False,
    max_epochs=50,
    max_iters=99999999999,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=2000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        key='cer',
        gamma=0.8,
        ),
    checkpoint_config=dict(
        interval=5000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/dbg_for_pretrain',
    save_root='./dfsmnln_online_finetune',
    save_dir='dfsmnln_online_finetune',
    save_name='dfsmnln_online_finetune',
    amp_level='O1',
    )
valid = dict(
    interval=5000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='lark_chinese|lark_mixed|aishell_2|ceo_external|smart_dog|meeting',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    rnnt_temperature=1.0,
    use_batch_beam=True,
    use_fsmn_beam=False,
    filter_list=['^']
    #remote_stat_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/",
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
    interval=100,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
