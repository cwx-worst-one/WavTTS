project = "san_cif"
runner = 'CifMostRunner'

# increment to add and override args.
data = dict(
    #data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/librispeech/wav/',
    add_text_only_data_loader=True,
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/librispeech-train-clean-100/',
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/librispeech/wav/',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/librispeech/wav/',
    text_only_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/data/text_only_data/librispeech_norm_lm/dolphin_data/',
    train_file_list = '["train_sub{}".format(i) for i in range(36)]',
    valid_file_list = '["dev_other"]',
    text_only_train_file_list='["train_sub{}".format(i) for i in range(36)]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_data_root="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/data_package/librispeech/dolphin_data/meta_files",
    meta_file="meta_cif_bpe",
    fbank_dim=80,
    fbank_channel=1,
    chunk_size=20,
    bucket_schedule='690,952,1116,1212,1274,1319,1355,1385,1411,1435,1456,1476,1496,1514,1533,1551,1569,1588,1617,2000',
    bucket_schedule_val='100000',
    text_only_bucket_schedule='3,6,12,24,36,48,60,72,84,96,112,128,144,160,184,208,256,320,480,640,1000',
    bucket_schedule_key='length',
    max_batch_size=5000,
    text_only_max_batch_size=40000,
    text_only_bucket_schedule_key='char_length',
    shuffle=True,
    global_shuffle=True,
    batch_means_tokens=1,
    use_eos=True,
    train_item_transform=[
        dict(type='PickleParser'),
        #dict(type='WavConvert', in_key="wav"),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type='LengthFilter', key='length', min_len=20, max_len=2000),
        dict(type='WavResample', sample_rate=16000, key='wav'),
        #dict(type='ConvertToChar', use_eos=True),
        dict(type='BPE', use_eos=True, skip_bpe=True, keep_bpe_words=True),
        dict(type='LabelLengthFilter', in_out_ratio=1, key='char'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='CalculateFrameLength'),
        dict(type='WavResample', sample_rate=16000, key='wav'),
        #dict(type='ConvertToChar', use_eos=True, convert_bpe_to_text=True),
        dict(type='BPE', use_eos=True, skip_bpe=True, keep_bpe_words=True),
        dict(type='LabelLengthFilter', in_out_ratio=1, key='char'),
    ],
    text_only_train_item_transform=[
        dict(type='PickleParser'),
        dict(type='BPE', use_eos=True, skip_bpe=True, keep_bpe_words=True),
        dict(type='LabelLength', key='char'),
    ],
    batch_transform=[
        dict(type='WaveformCollate', wav_key='waveform'),
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='label_bpe'),
        dict(type='CharCollate'),
    ],
    inference_batch_transform=[
        dict(type='WaveformCollate', wav_key='waveform'),
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='label_bpe'),
        dict(type='CharCollate'),
    ],
    text_only_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='label_bpe'),
        dict(type='CharCollate'),
    ],
)
solution = dict(
    model_type='Data2vecCifModel',
    wav2vec_type='PretrainedData2Vec',
    vocab_size=3727,
    use_fused_kernel=False,

    activation_checkpoint=False,
    data2vec_finetuning=True,

    data2vec_speech_only_encoder=True,
    fix_speech_only_encoder=True,
    shared_encoder_type='ConformerV2',
    shared_encoder_conformerv2_args=dict(
        num_encoder_layers=3,
        encoder_self_attention_type='dot_product',
        sa_pooling_layers=[],
	self_attention_type='dot_product',
	num_heads=8,
	hidden_size=512,
	filter_size=2048,
	ffn_layer='dense_relu_dense',
	layer_preprocess_sequence='n',
	layer_postprocess_sequence='da',
	norm_type='layer',
	norm_epsilon=0.000001,
	proximity_bias=False,
	conformer_conv_width=31,
	conformer_conv_norm_type='none',
        layer_prepostprocess_dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.0,
    ),

    mm_loss_type='mse',
    text_only_encoder_type='RelTransformer',
    text_only_reltransformer_args=dict(
        front_end_type='VGGFrontEndNoFsmn',
        position_encoding_type='RelPositionalEncoding',
        backbone_layer_num=2,
        backbone_memory_size=768,
        self_attn_heads=12,
        backbone_hidden_size=3072,
        dropout=0.1,
        self_attn_dropout=0.1,
        self_attn_activation_dropout=0.0,
        self_attn_activation_fn='relu',
        self_attn_layer_norm_before=1,
    ),

    extractor_mode="layer_norm",
    normalize=True,
    #encoder
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn="gelu",
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    layer_norm_first=False,
    encoder_layerdrop=0.0,
    pad_to_multiple=True,
    required_seq_len_multiple=2,
    # convolutional feature extraction layers
    conv_feature_layers="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2",
    feature_grad_mult=0.0,
    # connection parameters between w2v and cif
    close_reset_parameters=True,
    use_input_padding=True,

    # self-attention decoder/decoder setting
    self_attention_type='dot_product',
    num_heads=8,
    hidden_size=512,
    filter_size=2048,
    ffn_layer='dense_relu_dense',
    layer_preprocess_sequence='n',
    layer_postprocess_sequence='da',
    norm_type='layer',
    norm_epsilon=0.000001,
    proximity_bias=True,

    num_decoder_layers=2,
    decoder_self_attention_type='dot_product',
    # use teacher_forcing / pss
    use_teacher_forcing=True,
    use_pss=False,
    pss_rate=0.0,

    # regularizer
    layer_prepostprocess_dropout=0.1,
    #attention_dropout=0.1,
    relu_dropout=0.0,

    # extra
    loss_multiplier=2.0,
    shared_embedding_and_softmax_weights=False,
    multiply_embedding_mode='sqrt_depth',

    # mask
    mask_length=10,
    mask_prob=0.5,
    mask_selection="static",
    mask_other=0.0,
    mask_minlen_type="random",
    no_mask_overlap=False,
    mask_min_space=1,
    mask_dropout=0,
    # mask channel
    mask_channel_length=10,
    mask_channel_prob=0.0,
    mask_channel_selection="static",
    mask_channel_other=0.0,
    mask_channel_minlen_type="random",
    mask_channel_before=False,
    no_mask_channel_overlap=False,
    mask_channel_min_space=1,
    require_same_masks=True,
    # dropout
    dropout_input=0.1,
    conv_pos=95,
    conv_pos_groups=16,
    pos_conv_depth=5,
    conv_bias=False,
    data2vec_feature_conv_type="default",
    final_proj_num_layer=1,
    final_dropout=0,
    final_dim=256,
    weighted_data2vec=False,
    apply_mask=True,
    freeze_finetune_updates=300000,
    freeze_encoder_layers=-1,
    
    # cif part
    produce_weights_type='conv',
    conv_cif_num_layers=1,
    conv_cif_width_string='3',
    conv_cif_num_filters=512,
    conv_cif_dropout=0.0,
    dense_cif_units=512,
    conv_norm_type='layer_correct',

    cif_weight_threshold=0.99999,
    use_scaling_strategy=True,
    use_tail_handling=True,

    # all loss setting
    calculated_loss='ce_loss,quantity_loss,text_only_ce_loss,modality_match_loss',

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
    val_criterion_type='CifLoss',
    eos_id=2,

    #for da2vec fine-tune
    average_top_k_layers=0,
    loss_beta=0,
    loss_scale=-1,
    ema_transformer_only=True,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    # resume='step_10000.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
 
    max_epochs=40,
    max_iters=40000,
    lr_scheduler=dict(
        by_epoch=False,
        policy="TriStage",
        lr=5e-4,
        phase_ratio="[0.1,0.4,0.5]",
        init_lr_scale=0.01,
        final_lr_scale=0.05,
        max_train_steps=40000,
    ),
    checkpoint_config=dict(
        interval=1600, # steps
        max_keep_ckpts=5,
    ),
    save_root='./cif',
    save_dir='data2vec_cif',
    save_name='data2vec_cif',
    #resume_pretrain_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liyong/dolphin/data2vec/lyz_baseline/pretrain/checkpoint_last.pth',
    #resume_pretrain_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/models/librilight/cif_data2vec_pretrain_large_top20_lr3e-4_O1_bs12k_layernormFirst/data2vec_pretrain/large_lr3e-4_poly600k_mask0.65_ld0.00_bsz63m/checkpoints/step_350000.pth',
    freeze_feature_extractor=False,
    # try to figure out what this metric mean
    compare_metric="UF",
    validation_log=True,
    amp_level='O0',
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
    bucket_schedule='144, 167, 185, 198, 211, 222, 233, 243, 256, 268, 281, 294, 313, 332, 358, 390, 428, 479, 563, 2400, 100000',
    test_sets='test',
    beam_size=10,
    nbest=1,
    cif_temperature=1.0
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.98),
    eps=1e-8,
    weight_decay=0.01,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=100.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    max_grad_clip=0.0,
    bmuf_config=0,
 )
log_level = 'INFO'
log_config = dict(
    interval=100,  # log at every 100 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
