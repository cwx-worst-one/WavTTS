project = "speechlm_las_baseline"
runner = 'BaseLASRunner'

# increment to add and override args.
data = dict(
    data_root=[
        # 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_6wh',  
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    ],
    train_file_list=[
        # '["sub_{}".format(i) for i in range(1333)]', # 6w 小时数据
        '["train_sub{}".format(i) for i in range(1024)]',
    ],
    valid_data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/aishell1/dev/wav_ark_text/"
    ],
    valid_file_list=[
        '["dev"]',
    ],
    meta_data_root="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/",
    meta_file="meta_ag350k_lzl",
    tgt_vocab_size=9759, # original: 9759
    extra_special_symbols=None,
    tgt_dict_dir='total_bpe.dict', # 这个词表里面有blk吗？？？？？？？？？？
    reorder_dict_by_freq=0,   # 这个 reorder_dict_by_freq 是干啥的？
    cmvn_file="fbank_80dim_cmvn.kaldi",
    chunk_size=20,
    bucket_schedule='200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_val='5,100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_key='fbank',
    fbank_dim=80,
    fbank_channel=1,
    max_batch_size=20000,
    global_shuffle=True,
    batch_means_tokens=1,
    next_retry=1024,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=True, # eos 在solution里面加
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        dict(type='CMVN', key='fbank'),
        # dict(type='TimeMask', time_mask_size=20, time_mask_num=10, replace_with_zero=True),
        # dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=True),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=True, skip_list=['<pad>']), # eos 在solution里面加
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        dict(type='CMVN', key='fbank'),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=True, skip_list=['<pad>']), # eos 在solution里面加 
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', pad_eos=True, las_format=True),
        dict(type='PreCharCollate', rnnt_format=False, las_format=True),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
    ],
)
solution = dict(
    type='base_las_solution',
    las_type='BaseLASModel',
    ## wrap an encoder
    using_discrete_token=True,
    # acoustic_tokenizer backbone
    acoustic_tokenizer_type="KmeansTokenizerWithEncoder",
    kmeans_model_path='/mnt/bn/by-nas/representation_dumping/bestrq_conformerxl_nonstream_input6w_expsetting/maskprob0.12_span40_codebooksize4096_codebookdim16_nsoftmax8_warmup50000_accum2_gcmaxnorm20_accumafter/bestrq/dumped-extract_encoder_backbone_out-video_live_6wh-sub0_sub21-kmeans1024.bin',
    acoustic_tokenizer_pretrained_model='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/baiye/experiments/bestrq_conformerxl_nonstream_input6w_expsetting/maskprob0.12_span40_codebooksize4096_codebookdim16_nsoftmax8_warmup50000_accum2_gcmaxnorm20_accumafter/bestrq/checkpoints/step_400000.pth',
    acoustic_tokenizer_front_end_type='Conv2dPooling',
    acoustic_tokenizer_fbank_dim=80,
    acoustic_tokenizer_front_end_conv0_ch=256,
    acoustic_tokenizer_front_end_conv1_ch=256,
    acoustic_tokenizer_front_end_padding=1,
    acoustic_tokenizer_downsampling_size=4,
    acoustic_tokenizer_acoustic_backbone_type='MaskedConformerBackbone',
    acoustic_tokenizer_conformer_normalize_before=True,
    acoustic_tokenizer_conformer_attention_heads=8,
    acoustic_tokenizer_conformer_mask_topology=None,
    acoustic_tokenizer_conformer_linear_units=4096,
    acoustic_tokenizer_conformer_num_blocks=24,
    acoustic_tokenizer_conformer_dropout_rate=0.1,
    acoustic_tokenizer_conformer_positional_dropout_rate=0.1,
    acoustic_tokenizer_conformer_attention_dropout_rate=0.1,
    acoustic_tokenizer_conformer_positionwise_layer_type='linear',
    acoustic_tokenizer_conformer_activation_fn='gelu',
    acoustic_tokenizer_conformer_positionwise_conv_kernel_size=1,
    acoustic_tokenizer_conformer_macaron_style=1,
    acoustic_tokenizer_conformer_pos_enc_layer_type='fix_rel_pos',
    acoustic_tokenizer_conformer_selfattention_layer_type='rel_selfattn',
    acoustic_tokenizer_conformer_layer_order='mhsa_before_conv',
    acoustic_tokenizer_conformer_use_cnn_module=1,
    acoustic_tokenizer_conformer_cnn_module='ConvolutionModule',
    acoustic_tokenizer_conformer_cnn_module_kernel='5',
    acoustic_tokenizer_conformer_cnn_norm_type='layer_norm',
    acoustic_tokenizer_conformer_layernorm_interval=0,
    acoustic_tokenizer_conformer_weight_scale=1.0,
    acoustic_tokenizer_conformer_half_pooling=0,
    acoustic_tokenizer_backbone_memory_size=1024,
    acoustic_tokenizer_dropout=0.1,
    dropout=0.1, # transformer.py
    # tokenizer
    acoustic_vocab_size=1024,
    acoustic_token_embedding_size=1792,
    using_enc_proj_in_dim=False,
    # transformer lm
    # decoder
    las_decoder_type='TransformerLASDecoder',
    max_target_positions=600,
    embedding_size=1792,
    decoder_embed_dim=1792,
    decode_residue=True,
    apply_embed_scale=True,
    decoder_pos_embd_type='abs',
    decoder_layer_num=26,
    decoder_ffn_embed_dim=7168,
    decoder_attention_heads=14,
    attention_dropout=0,
    activation_dropout=0.1,
    decoder_dropout=0.1,
    decoder_normalize_before=True,
    activation_fn='gelu',
    decoder_act_glu=True,
    # output
    tgt_vocab_size=9759,
    # criterion
    criterion_type='LasCE',
    label_smooth_factor=0.0, # 和 SpokenLLM目前的训练一致
    # frontend_fix=True,
    # backbone_fix=True,
    acoustic_token_fix=True
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=60,
    max_iters=600000,
    lr_scheduler=dict(
        policy='Transformer',
        by_epoch=False,
        warmup_steps=25000,
        d_model=1792,
        warmup=None,
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=20,
    ),
    save_root='./speechlm',
    save_dir='speechlm',
    save_name='speechlm',
    amp_level='O1',
    # gradient accumulation
    grad_accum_step=1,
    # 'AVG' 'SUM'
    grad_accum_mode='AVG'
    )
valid = dict(
    output_probs=False,
    not_show_results=False,
    interval=1600,
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='AdamW',
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=20.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    grad_clip_mode='accum_after',
    max_grad_clip=5.0,
 )
log_level = 'INFO'
log_config = dict(
    interval=1,  # log at every 100 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])