project = 'conformerxl_baseline'
runner = 'RNNTRunner'
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav_testset_basic_7_21/',
    train_file_list = '["train_sub{}".format(i) for i in range(103)]',
    valid_file_list = '["cv"]',
    tgt_vocab_size=9759,
    tgt_dict_dir='total_bpe.dict', # 这个词表里面有blk吗？？？？？？？？？？
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta_ag350k_lzl",
    fbank_dim=80,
    global_shuffle=True,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,1900,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000,4000,5000,10000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=30000,
    max_batch_scale=10,
    prefetch_worker_num=4,
    native_parallel_file_num=8,
    next_retry=1024,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', max_len=20.02),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
        dict(
            type='DynamicTimeMask',
            time_mask_size=20,
            time_mask_block=60,
            replace_with_zero=False,
            inplace=True,
            time_mask_num_dither=1),
        dict(
            type='FreqMask',
            freq_mask_size=27,
            freq_mask_num=1,
            replace_with_zero=False,
            inplace=True)
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate')
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80)
    ]
)
solution = dict(
    model_type='BaseRnntModel',
    # front_end
    ## wrap an encoder # 608M
    # front_end
    front_end_type='Conv2dPooling',
    front_end_conv0_ch=256,
    front_end_conv1_ch=256,
    front_end_padding=1,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='MaskedConformerBackbone',
    conformer_normalize_before=True,
    conformer_attention_heads=8,
    conformer_mask_topology=None,
    conformer_linear_units=4096,
    conformer_num_blocks=24,
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
    conformer_cnn_module_kernel='5',
    conformer_cnn_norm_type='layer_norm',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_memory_size=1024,
    dropout=0.1,
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
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
    adaptive_head_size=760,
    adaptive_tail_size=200,
    adaptive_tail_groups=45,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=500,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    backbone_clamp_inf=True,
    head_clamp=True,
    mtl_clamp=True,
    backbone_pool_type='Conv1dTimeReduce')
work_dir = './demo'
train = dict(
    max_epochs=50,
    max_iters=75200,
    check_loss=True,
    resume='latest.pth',
    resume_optimizer=True,
    lr_scheduler=dict(
        policy='Transformer',
        by_epoch=False,
        warmup_steps=8000,
        d_model=1024,
        warmup=None,
    ),
    checkpoint_config=dict(
        interval=1600, # steps
        max_keep_ckpts=-1,
    ),
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/',
    save_root='./rnnt_conformer_nonstream_input1k_expsetting',
    save_dir='conformer_transducer',
    save_name='conformer_transducer',
    amp_level='O1',
    # gradient accumulation
    grad_accum_step=2,
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
    test_sets='ceo_external_refine/ceo_external_refine|aishell_2019c/aishell_2019c',
    beam_size=8,
    lm_path='',
    lm_weight=1.0,
    blk_scale=0.7,
    use_batch_beam=True,
    len_penalty_scale=0.01)
optimizer = dict(
    type='AdamW',
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01,
)
optimizer_config = dict(
    grad_clip=dict(
        max_norm=1.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    grad_clip_mode='accum_before',
    max_grad_clip=5.0,
 )
log_level = 'INFO'
log_config = dict(
    interval=1,
    reset_flag=True,
    hooks=[dict(type='TextLoggerHook')])
