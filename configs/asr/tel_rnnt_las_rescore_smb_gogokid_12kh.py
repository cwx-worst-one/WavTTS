data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/smb_gogokid/',
    train_file_list='["train_sub{}".format(i) for i in range(128)]',
    valid_file_list='["cv"]',
    meta_file="../meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    max_batch_scale=5,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=True),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='AppendFrames', frame=12),
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60,
                  replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=False, inplace=True),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=True),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='AppendFrames', frame=12),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', reverse=True),
        dict(type='PreCharCollate', rnnt_format=False, reverse=True),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ])
project = 'tel_rnnt'
runner = 'RNNTLASRescoreRunner'
solution = dict(
    model_type='RnntLasRescoreModel',
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    acoustic_backbone_type='LSTMPBackbone',
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=False,
    backbone_residual=True,
    backbone_mask=1,
    dropout=0.1,
    head_type='RnntSimpleHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    predictor_type='LSTMReLUPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    adaptive_head_size=968,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    reorder_dict_by_freq=True,
    twopass_acoustic_args = dict(
        backbone_type='OfflineTransformerBackbone',
        backbone_topology='[[-1,-1,1]]*5',
        backbone_memory_size=512,
        backbone_hidden_size=2048,
        self_attn_heads=8,
        self_attn_dropout=0.1,
        dropout=0.1,
        self_attn_activation_dropout=0,
        self_attn_layer_norm_before=1,
        self_attn_activation_fn='relu'),
    # decoder
    twopass_decoder_args = dict(
        las_decoder_type='SimpleLSTMLASDecoder', # SimpleLSTMLASDecoderOnnx, # export onnx
        embedding_size=256,
        decoder_input_size=512,
        decoder_lstm_hidden_size=512,
        decoder_lstm_layer_num=2,
        decoder_dropout=0.15,
        decoder_pred_fc_size=256,
        decoder_pred_fc_proj_size=256,
        las_forward_decoder=True,
        las_backward_decoder=True,
        fw_decoder_weight=0.5,
        ## head
        las_atten_type='MHAttention',
        atten_hidden_size=256,
        multi_head_num=4,
        multi_att_weight_drop=0.1,
        # ce criterion
        las_criterion_type='LasCE',
        label_smooth_factor=0.1,
       
        # schedule sampling
        schedule_sample_begin=100000,
        schedule_sample_increase_steps=50000,
        schedule_ratio=0.3),
    # export onnx
    # onnx_stack_frame=320,
    )
work_dir = './demo'
train = dict(
    # resume='latest.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    max_epochs=9,
    max_iters=800000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.7),
    checkpoint_config=dict(interval=10000, max_keep_ckpts=1000),
    curriculum_end_epoch=0,
    curriculum_length=520,
    resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/miaohaoran/saved_models/dolphin/smb_gogokid_202101/rnnt_lstmp/checkpoints/best.pth',
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xxx/',
    save_root='./demo',
    save_dir='smb_gogokid_202101',
    save_name='rnnt_las_rescore',
    amp_level='O1')
valid = dict(interval=10000)
inference = dict(
    resume='latest.pth',
    test_sets='cv',
    twopass_infer_cfg = dict(
        rnnt_beam_size=10,
        nbest=10,
        enable_twopass_rescore=True,
        #enable grid search for scale in range [start:end:stride]
        grid_search_params='[0.0, 2.0, 0.1]',
        # disable grid search by specified scale
        #las_forward_score_scale=50.0,
        #las_backward_score_scale=50.0,
        # las fst scale
        las_fst_score_scale=0.02
        ),
    fst_weight=0.3,
    lm_path='',
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    use_batch_beam=False,
    )
nnlm = dict(lm_type='')
optimizer = dict(
    type='FusedAdam',
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=5000,
        average_params=True,
        use_nesterov=True))
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=200,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
