project = "conformer_rnnt"
runner = 'RNNTRunner'
_base_ = [
    '../_base_/datasets/general10.py',
]

# increment to add and override args.
data = dict(
    chunk_size=20,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='BPE'),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=False),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim'),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='BPE'),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='CMVN', key='fbank'),
        dict(type='AppendFeatureDim'),
    ],
    batch_transform=[
        dict(type='ListCollate'),
        dict(type='FbankCollate', frame_chunk_size=12),
        dict(type='CharCollate'),
        dict(type='PreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', frame_chunk_size=12),
    ],
)

solution = dict(
    type='base_rnnt_solution',
    rnnt_type='BaseRnntModel',
    # init
    xavier_init=False,
    # front_end
    front_end_type='Conv2dPooling',
    front_end_conv0_ch=128,
    front_end_conv1_ch=128,
    # backbone
    acoustic_backbone_type='ConformerBackbone',
    conformer_mask_topology='[[40, 40]]*12',
    backbone_memory_size=512,
    conformer_linear_units=2048,
    conformer_num_blocks=12,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    conformer_dropout_rate=0.1,
    conformer_positional_dropout_rate=0.1,
    conformer_pos_enc_layer_type='rel_pos',
    conformer_normalize_before=1,
    conformer_attention_heads=8,
    conformer_attention_dropout_rate=0.0,
    conformer_positionwise_layer_type='linear',
    conformer_positionwise_conv_kernel_size=1,
    conformer_activation_fn='gelu',
    conformer_macaron_style=1,
    conformer_selfattention_layer_type='rel_selfattn',
    conformer_use_cnn_module=1,
    conformer_cnn_module='ConvolutionModule',
    conformer_cnn_module_kernel='15,15',
    conformer_layernorm_interval=0,
    backbone_layer_gap=100,
    # head
    head_type='RnntBaseHead',
    head_hidden_size=768,
    head_lowrank_size=512,
    # predictor
    predictor_type='TransformerXLPredictor',
    predictor_n_layer=8,
    predictor_d_embed=512,
    predictor_d_model=512,
    predictor_n_head=8,
    predictor_d_head=64,
    predictor_d_inner=2048,
    predictor_dropout=0.1,
    predictor_dropatt=0.0,
    predictor_pre_lnorm=1,
    predictor_ext_len=0,
    predictor_mem_len=10,
    predictor_activation_fn='relu',
    predictor_relu_inplace=True,
    init='normal',
    init_range=0.1,
    init_std=0.02,
    predictor_dropout_factor=0.0,
    # jointer
    jointer_type='ShallowJointer',
    jointer_hidden_size=768,
    concate_U=1,
    jointer_simple_fusion=0,
    # Adaptive Softmax
    adaptive_head_size=1500,
    adaptive_tail_size=2867,
    adaptive_tail_groups=4,
    # criterion
    criterion_type='RnntAdaptiveCE',
    cer_update_freq=200,
    label_smooth_factor=0.1,
    reorder_dict_by_freq=True,
    cer_blank_scale=1.0,
)

# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='checkpoint_best.pt',
    resume_optimizer=False,
    resume_progress=False,
    max_epochs=15,
    max_iters=195000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=2000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[13000 * 4 + i * 13000 for i in range(20)]
        ),
    checkpoint_config=dict(
        interval=13000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/saved_models/dolphin',
    save_root='./demo',
    save_dir='general10k_rnnt',
    save_name='conformer_rnnt_base',
    amp_level='O1',
    )
valid = dict(
    interval=13000, # iter
    )

inference = dict(
    bucket_schedule='100000',
    test_sets='aishell_2|ceo_external|lark_chinese|lark_meeting_remove_zeros_dither0|lark_mixed|smart_dog|smb_itm_18569_fb60|tele_general_test_20191025_20000_fb60',
    beam_size=-1,
    lm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    language='zh',
    utt_key='utt_id',
    # remote_stat_dir="hdfs://haruna/xxxx",
    )
nnlm = dict(
    lm_type="",
    )

# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=5e-2,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=50,
        norm_type=2,
        ),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.85,
        warmup_steps=5000,
        use_nesterov=True,
        average_sync=True,
    )
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
    ])
