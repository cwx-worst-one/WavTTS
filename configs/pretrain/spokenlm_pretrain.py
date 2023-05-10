project = "speechlm"
runner = 'SpokenLmPretrainRunner'

# increment to add and override args.
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_6wh', # 
        # 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    ],
    train_file_list=[
        '["sub_{}".format(i) for i in range(1333)]', # 6w 小时数据
        # '["train_sub{}".format(i) for i in range(103)]',
    ],
    valid_data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanzhiyun/data/dolphin/aishell1/dev/wav_ark_text/"
    ],
    valid_file_list=[
        '["dev"]',
    ],
    meta_data_root="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/",
    meta_file="meta_ag350k_lzl",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    chunk_size=20,
    bucket_schedule='200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_val='5,100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_key='fbank',
    fbank_dim=80,
    fbank_channel=1,
    max_batch_size=20000,
    batch_means_tokens=1,
    use_eos=False,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        dict(type='CMVN', key='fbank'),
        # dict(type='TimeMask', time_mask_size=20, time_mask_num=10, replace_with_zero=True),
        # dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    model_type='SpokenLmPretrainModel',
    ## wrap an encoder
    # front_end
    front_end_type='StackingFrameFrontEnd', # 这里要不要降采样?
    stacking_frontend_kernel_size=1,
    stacking_frontend_stride=1,
    stacking_frontend_dilation=1,
    stacking_frontend_padding=0,
    # acoustic_tokenizer
    acoustic_tokenizer_type='RandomProjectionTokenizer',
    tokenizer_input_dim=80,
    codebook_size=8192,
    codebook_dim=16,
    prototype_initialization_method='gaussian',
    projection_initialization_method='xavier',
    # lm
    vocab_size=8192, # 和codebook一样大
    embedding_size=768, # code embedding的维度
    # backbone
    acoustic_backbone_type='TransformerBackbone',
    causal_transformer=True,
    backbone_topology='[[-1,0,1]]*18', # 这两行是搞出来上三角mask。18是层数。
    chunk_mask=False,
    backbone_pos_embd=True, # 这里用的是绝对位置编码
    backbone_after_norm=True,
    backbone_layer_num=18,
    backbone_memory_size=768,
    backbone_hidden_size=2048,
    self_attn_heads=4,
    self_attn_dropout=0.0,
    dropout=0.1,
    positional_dropout_rate=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=True, # 这个是pre-norm吗
    self_attn_activation_fn='gelu',
    backbone_act_glu=True,
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
        d_model=1024,
        warmup=None,
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
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
        max_norm=1.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    grad_clip_mode='accum_before',
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
