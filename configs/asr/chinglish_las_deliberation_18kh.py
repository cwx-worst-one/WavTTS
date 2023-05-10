data = dict(
    data_root = ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/native_yt_12kh_16k_wav', \
                 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/lls_filtered_by_WerMdd', \
                 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/ez_sample10times', \
                 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/chinglish_haitian_2kh_16k_wav', \
                 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/wemeet/', \
                 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/ac_data/'],
    train_file_list = ['["train_sub{}".format(i) for i in range(245)]', \
            '["train_sub{}".format(i) for i in range(13)]', \
            '["train_sub{}".format(i) for i in range(3)]', \
            '["train_sub{}".format(i) for i in range(56)]', \
            '["train_sub{}".format(i) for i in range(32)]', \
            '["train_sub{}".format(i) for i in range(19)]'],
    valid_file_list=['["cv"]', '["wemeet_cv"]', '["wemeet_cv"]', '["cv"]', '["cv"]', '["wemeet_cv"]'],
    meta_file="meta",
    fbank_dim=80,
    chunk_size=30,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,10000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    max_batch_scale=5,
    shuffle=True,
    drop_last=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', max_len=20.),
        dict(type='SpeedPerturbation', p=0.666, speed_rate_list=[0.95, 1.05]),
        dict(
            type='AddNoise',
            noise_prefix='noise_sub',
            noise_dir=
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/noiseX92_rand_cut5_20',
            noise_shards=1,
            p=0.4,
            min_snr=5,
            max_snr=20.0),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=True, skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>']),
        dict(type='CMVN', key='fbank'),
        dict(type='DynamicTimeMask',
             time_mask_size=20,
             time_mask_block=60,
             replace_with_zero=False,
             inplace=True,
             time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
             inplace=True, replace_with_zero=False),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=True, skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>']),
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
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
project='chinglish_lasdeliberation'
runner='RNNTDeliberationRunner'
solution = dict(
    model_type='RnntDeliberationModel',
    # acoustic_front_end_module
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    # acoustic_backbone_module, output size should be backbone_memory_size
    acoustic_backbone_type='LSTMPBackbone',
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=0,
    backbone_residual=1,
    backbone_mask=1,
    dropout=0.1,
    # acoustic_head_module, only used for RNN-T jointer
    head_type='RnntSimpleHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # predictor_module
    predictor_type='LSTMReLUPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
    # jointer_module
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    # criterion_module
    adaptive_head_size=1056,
    adaptive_tail_size=150,
    adaptive_tail_groups=40,
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    reorder_dict_by_freq=True,
    mtl_head='HybridCEHead',
    twopass_acoustic_args = dict(
        backbone_type='OfflineTransformerBackbone',
        backbone_topology='[[-1,-1,1]]*5',
        backbone_memory_size=512,
        backbone_hidden_size=2048,
        dropout=0.1,
        attn_dropout=0.1,
        self_attn_heads=8,
        self_attn_dropout=0.1,
        self_attn_activation_dropout=0,
        self_attn_layer_norm_before=1,
        self_attn_activation_fn='relu'),
    twopass_hyp_args = dict(
        hyp_encoder_type='TextEncoder',
        backbone_type='LSTMPBackbone', # TransformerEncoder 
        embedding_size=512,
        backbone_memory_size=512,
        backbone_hidden_size=1024,
        backbone_layer_num=2,
        backbone_bilstm=True,
        backbone_mask=True,
        hyp_residual=True),
    twopass_decoder_args = dict(
        las_decoder_type='LSTMDelibDecoder',
        embedding_size=512,
        decoder_lstm_layer_num=2,
        decoder_input_size=1024, # acoustic_args.backbone_memory_size + hyp_args.backbone_memory_size
        decoder_lstm_hidden_size=2048,
        decoder_dropout=0.15,
        decoder_pred_fc_size=2048,
        decoder_pred_fc_proj_size=768,
        # head
        las_atten_type='MHAttention',
        atten_hidden_size=2048,
        multi_head_num=8,
        multi_att_weight_drop=0.1,
        # criterion
        las_criterion_type='LasCE',
        label_smooth_factor=0.1,
        # schedule sampling
        schedule_sample_begin=30000,
        schedule_sample_increase_steps=15000,
        schedule_ratio=0.3),
    beam_schedule = dict(use_batch_beam=True,
                       rnnt_beam_size=10,
                       nbest_input_size=4,
                       recombine_sum=True,
                       blk_scale=1.0,
                       len_penalty_scale=0.0)
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    check_loss=True,
    resume='latest.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    max_epochs=20,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        cooldown=0,
        gamma=0.6,
        key='loss',
        step=[100000 + i * 5000 for i in range(140)],
        reset_loss_step=30000
        ),
    checkpoint_config=dict(
        interval=5000, # steps
        max_keep_ckpts=-1,
    ),
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xuwh15/dolphin/0312_Rerun_libri_las_rescore_stream/saved_models',
    save_root='./demo',
    save_dir='librispeech_las_delib',
    save_name='librispeech_lstm_rnnt',
    amp_level='O1',
    )
valid = dict(
    interval=5000, # iter
    )
inference = dict(
    test_sets='test_clean|test_other',
    twopass_infer_cfg = dict(
        rnnt_beam_size=10,
        nbest=10,
        enable_twopass_rescore=False,
        las_beam_size=10,# las delib beam_size
        nbest_input_size=4
        ),
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.0,
    language='en',
    use_batch_beam=False,
    one_step_expand=True,
    recombine_sum=True,
    filter_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '**', '<space>'],
    nbest=10, # rnnt nbest out
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
        max_norm=20,
        norm_type=2,
        ),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=5000,
        average_params=True,
        use_nesterov=True,
    )
 )

criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=200,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
