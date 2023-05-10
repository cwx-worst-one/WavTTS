project = 'tel_rnnt'
runner = 'RNNTRunner'

_base_ = [
    '../_base_/datasets/tel_dy_hostoon.py',
]

data = dict(
    chunk_size=20,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule=
    '50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    shuffle=True,
    drop_last=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
        dict(
            type='DynamicTimeMask',
            time_mask_size=20,
            time_mask_block=60,
            replace_with_zero=False,
            inplace=True,
            time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1)
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='AppendFrames', frame=12),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', need_adapt_target=True),
        dict(type='PreCharCollate')
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80)
    ],
    max_batch_scale=5)
solution = dict(
    type='base_rnnt_solution',
    stage='student_rnnt_ctc',
    rnnt_type='RnntHmmFreeModel',
    front_end_type='TimeReduceLSTMP',
    teacher_front_end_type='TimeReduceLSTMP',
    input_concat_size=4,
    downsampling_size=4,
    acoustic_backbone_type='LSTMPBackbone',
    teacher_acoustic_backbone_type='LSTMPBackbone',
    backbone_memory_size=512,
    backbone_hidden_size=1024,
    backbone_layer_num=5,
    backbone_bilstm=False,
    backbone_residual=True,
    dropout=0.1,
    head_type='RnntSimpleHead',
    teacher_head_type='RnntSimpleHead',
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
    criterion_type='RnntPretrainAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    # hmm_free
    soft_label=False,
    left_factor=0.2,
    right_factor=0.6,
    mtl_head='HybridCEHead',
    backbone_mask=1,
    encoder_fix_steps=0)
work_dir = './demo'
train = dict(
    resume='latest.pth',
    resume_optimizer=0,
    max_epochs=9,
    max_iters=800000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=5000,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.5,
        step=[40000 + i * 5000 for i in range(140)]),
    checkpoint_config=dict(interval=10000, max_keep_ckpts=5),
    curriculum_end_epoch=0,
    curriculum_length=520,
    save_root='./demo',
    save_dir='tel_dy_hotsoon_80kh_rnnt_pretrain',
    save_name='lstm_rnnt_ctc_append12_lr5e-5_fix0_new_O1',
    amp_level='O1',
    resume_progress=0)
valid = dict(interval=10000)
inference = dict(
    test_sets='tele_general_test',
    beam_size=10,
    lm_path='',
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01)
nnlm = dict(lm_type='')
optimizer = dict(
    type='FusedAdam',
    lr=5e-05,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=5.0,
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
