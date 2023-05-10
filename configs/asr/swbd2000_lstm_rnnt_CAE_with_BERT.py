project = "switchboard-2000"
runner = 'RNNTContextAwareRunner'

# increment to add and override args.
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/english/swbd_300h/train_nodup_espnet_sp1.0.ctx',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/english/fisher_2000h/training_nodev',
        ],
    train_file_list=[
        '["part_{}".format(i) for i in range(39)]',
        '["shard_{}".format(i) for i in range(187)]',
        ],
    valid_data_root = [
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/english/swbd_300h/train_dev_espnet',
        ],
    valid_file_list=['["cv_4k.ctx"]'],
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/english/all_testsets/',
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/resources/en_meta_data',
    meta_file="meta_swbd_fisher_vocab2k_fs8k",
    tgt_dict_dir='char_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    fbank_dim=80,
    max_batch_size=10240,
    shuffle=True,
    drop_last=False,
    chunk_size=200,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_scale=5,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    golbal_shuffle=1,
    next_retry=1000,
    prefetch_block_num=50,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='SpeedPerturbation', p=0.666, speed_rate_list=[0.9, 1.1]),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(
            type='DialogHistToContext',
            key='dialogue_history',
            out_key='context_text',
            turns=3,
            max_context_len=256,
            mask_prob=0.0,
            mask_token='<pad>',
            perturb_prob=0.1,  # refers to WER
            edit_ops_probs='0.6,0.2,0.2',  # prob of Sub, Ins, Del
            ),
        dict(type='BertTokenizer', key='context_text', out_key='context_text', max_chars_num=512),
        dict(type='BPE', use_eos=False, key='label', skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>', '<pad>', '[laughter]', '[noise]', '[vocalized-noise]', '<s>', '</s>']),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='CMVN', key='fbank'),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(
            type='DialogHistToContext',
            key='dialogue_history',
            out_key='context_text',
            turns=3,
            max_context_len=256,
            mask_prob=0.0,
            mask_token='<pad>',
            perturb_prob=0.1,  # refers to WER
            edit_ops_probs='0.6,0.2,0.2',  # prob of Sub, Ins, Del
            ),
        dict(type='BertTokenizer', key='context_text', out_key='context_text', max_chars_num=512),
        dict(type='BPE', use_eos=False, key='label', skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>', '<pad>', '[laughter]', '[noise]', '[vocalized-noise]', '<s>', '</s>']),
        dict(type='AppendFrames', frame=12),
        dict(type='AppendFeatureDim', total_dim=80),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        # the ContextMakePairsCollate must be placed at the first position to make data-pairs
        dict(type='ContextMakePairsCollate', key='uttid', context_loss_scale=0.8),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', key='context_text', set_tgt_length=False),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='CharCollate', key='context_text', set_tgt_length=False),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ])
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
    context_aware_method='CAE',
    context_encoder_type='BERT',
    context_encoder_fixed=True,
    context_pred_concat_query=0,
    context_embd_size=512,
    context_bidirectional=False,
    context_attn_heads=4,
    context_attn_qdim=512,
    context_attn_kvdim=768,
    context_out_dim=512,
    context_dropout=0.2,  # dropout prob inside of the context encoder and attention layer
    context_global_dropout=0.0,  # dropout prob of discarding the sentence-level context
    context_joint_training=True,
    context_loss_scale=0.8,  # this has higher priority than that in batch_transform
    context_encoder_ckpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/saved_models/dolphin_tools/context_aware_rnnt/bert_init_model/english/pytorch_model.bin',
    context_encoder_config='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/saved_models/dolphin_tools/context_aware_rnnt/bert_init_model/english/config.json',
    bert_vocab_dict='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/saved_models/dolphin_tools/context_aware_rnnt/bert_init_model/english/vocab.txt',
    # predictor
    predictor_type='LSTMReLUPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=1,
    predictor_dropout_factor=0.15,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=640,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=2008,
    adaptive_tail_size=72,
    adaptive_tail_groups=1,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    soft_label=False,
    left_factor=0.2,
    right_factor=0.6,
    # CTC
    mtl_type='ctc',
    mtl_head='HybridCEHead',
    backbone_mask=1,
    encoder_fix_steps=0
)
work_dir = './exp'
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=20,
    max_iters=2000000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='exp',
        warmup_iters=10000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.85,
        step=[80000 + i * 10000 for i in range(40)]
        ),
    checkpoint_config=dict(
        interval=5000, # steps
        max_keep_ckpts=50,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root= 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/saved_models',
    save_root='./exp',
    save_dir='swbd',
    save_name='swbd_lstm_rnnt_cae',
    amp_level='O1',
    )
valid = dict(
    interval=5000, # iter
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    language='en',
    nbest_out=False,
    filter_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>', '<pad>', '[laughter]', '[noise]', '[vocalized-noise]'],
    # remote_stat_dir= "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/inference_results/child_en_asr",
    )

nnlm = dict(lm_type='')
optimizer = dict(
    type='FusedAdam',
    lr=5e-6,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=200,
        block_lr=0.5,
        block_momentum=0.9,
        warmup_steps=3000,
        average_params=True,
        use_nesterov=True))
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
