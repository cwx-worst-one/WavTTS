project = "context-aware_rnnt"
runner = 'RNNTContextAwareRunner'
# increment to add and override args.
data = dict(
    data_root = [
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/chinese/child_datasets/ef_training_set.ctx',
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/chinese/child_datasets/er_training_set.ctx',
            ],
    train_file_list = [
                       '["part_{}".format(i) for i in range(1, 298)]',
                       '["part_{}".format(i) for i in range(1, 166)]',
                       ],
    valid_data_root = [
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/chinese/child_datasets/ef_training_set.ctx',
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/chinese/child_datasets/er_training_set.ctx',
                       ],
    valid_file_list=['["cv_6k"]', '["cv_4k"]'],
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/chinese/all_testsets/',
    tgt_dict_dir='total_bpe.dict',
    bpe_code='total.code',
    cmvn_file='fbank_80dim_cmvn.kaldi',
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/resources/zh_meta_data',
    meta_file="meta",
    fbank_dim=80,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    max_batch_scale=3,
    shuffle=True,
    drop_last=False,
    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    golbal_shuffle=1,
    next_retry=1200,
    prefetch_block_num=50,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='PunctuationFilter', key='label'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False, key='context_text'),
        # dict(type='BertTokenizer', key='context_text', out_key='context_text', max_chars_num=512),  # for BERT
        dict(type='BPE', use_eos=False, key='label'),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
        dict(
            type='DynamicTimeMask',
            time_mask_size=20,
            time_mask_block=60,
            replace_with_zero=False,
            inplace=True,
            time_mask_num_dither=1,
        ),
        dict(
            type='FreqMask',
            freq_mask_size=27,
            freq_mask_num=1,
            inplace=True,
            replace_with_zero=False,
        ),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='PunctuationFilter', key='label'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False, key='context_text'),
        # dict(type='BertTokenizer', key='context_text', out_key='context_text', max_chars_num=512),  # for BERT
        dict(type='BPE', use_eos=False, key='label'),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', key='context_text'),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='CharCollate', key='context_text'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    type='base_rnnt_solution',
    rnnt_type='ContextAwareRnntModel',
    # front_end
    front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='LSTMPBackbone',
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=False,
    backbone_residual=True,
    dropout=0.1,
    # head
    head_type='RnntSimpleHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # context encoder
    context_encoder_module='ContextEncoder',
    context_aware_method='CAP', # context-aware predictor
    context_encoder_type='LSTM',
    context_embd_size=512,
    context_lstm_hidden_size=1024,
    context_lstm_layer_num=2,
    context_bidirectional=False,
    context_attn_heads=4,
    context_attn_qdim=1024,
    context_attn_kvdim=1024,
    context_out_dim=640,
    context_dropout=0.1,  # dropout prob inside of the context encoder and attention layer
    context_global_dropout=0.2,  # dropout prob of discarding the sentence-level context
    context_encoder_ckpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/saved_models/lm/lm_lstm_512emb_1024lstmcell_512prec_newdata_prehandle_final/best.pth',
    context_encoder_pname_prefix='nnlm_model.',  # parameter name prefix in the pretrained context encoder
    context_encoder_fixed=False,
    context_encoder_fixed_steps=0,
    context_query_with_prev_embd=False,
    # context_encoder_type='BERT',
    # context_encoder_ckpt='hdfs://path/to/bert_ckpt/pytorch_model.bin',
    # context_encoder_config='hdfs://path/to/bert/config.json',
    # bert_vocab_dict='hdfs://path/to/bert//vocab.txt',
    # bert_char2index=True,
    # context_attn_kvdim=768,

    # predictor
    predictor_type='ContextAwareLSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.15,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=968,
    adaptive_tail_size=200,
    adaptive_tail_groups=60,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    soft_label=False,
    left_factor=0.2,
    right_factor=0.6,
    mtl_head='HybridCEHead',
    backbone_mask=1,
    encoder_fix_steps=0,
)
# runtime settings
work_dir = './exp'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.75,
    ),
    checkpoint_config=dict(
        interval=10000,  # steps
        max_keep_ckpts=200,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    remote_save_root='',
    save_root='./exp',
    save_dir='child_zh_asr',
    save_name='',
    amp_level='O1',
)
valid = dict(
    interval=10000,  # iter
)
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='ef_testset_with_context_0712|er_dataset_v1_2898_with_context',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=0.6,
    blank_thresh=1.0,
    len_penalty_scale=0.01,
    remote_stat_dir="",
)
nnlm = dict(
    lm_type="",
)
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax',\
#                  'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']

optimizer = dict(
    type='FusedAdam',
    lr=3e-6,
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
        use_nesterov=True,
        average_sync=True,
    ),
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
    ],
)
