project = "speechlm"
runner = 'SpokenLmPretrainRunner'

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
    tgt_vocab_size=9761, # original: 9759
    extra_special_symbols=["<speech>", "</speech>"],
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
    use_eos=False, # eos 在solution里面加
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
        dict(type='BPE', use_eos=False, skip_list=['<pad>']), # eos 在solution里面加
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        dict(type='CMVN', key='fbank'),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']), # eos 在solution里面加 
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
    ],
)
solution = dict(
    model_type='SpokenLmPretrainModel',
    ## wrap an encoder
    # front_end
    front_end_type='StackingFrameFrontEnd', # 这里要不要降采样?
    stacking_frontend_kernel_size=4,
    stacking_frontend_stride=4,
    stacking_frontend_dilation=1,
    stacking_frontend_padding=0,
    # acoustic_tokenizer
    acoustic_tokenizer_type='RandomProjectionTokenizer',
    acoustic_tokenizer_input_dim=320,
    acoustic_tokenizer_codebook_size=8192,
    acoustic_tokenizer_codebook_dim=16,
    prototype_initialization_method='gaussian',
    projection_initialization_method='xavier',
    # transformer lm
    tgt_vocab_size=9761,
    acoustic_vocab_size=8192, # 和codebook一样大. kmeans 用的4096
    embedding_size=1792, # code embedding的维度
    # backbone
    acoustic_backbone_type='TransformerBackbone',
    causal_transformer=True,
    backbone_topology='[[-1,0,1]]*26', # 这两行是搞出来上三角mask。18是层数。
    chunk_mask=False,
    backbone_pos_embd=True, # 这里用的是绝对位置编码
    backbone_after_norm=True,
    backbone_layer_num=26,
    backbone_memory_size=1792,
    backbone_hidden_size=7168,
    self_attn_heads=14,
    self_attn_dropout=0.0,
    dropout=0.1,
    positional_dropout_rate=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=True, # 这个是pre-norm吗
    self_attn_activation_fn='gelu',
    backbone_act_glu=True,
    squeeze_mem=1,
    # token combination
    token_combination_mode="simple_asr",
    # Loss
    label_smooth_factor=0,
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