project = "chinglish_las"
runner = 'BaseLASRunner'
_base_ = [
    '../_base_/datasets/haitian_chinglish.py',
]

# increment to add and override args.
data = dict(
    chunk_size=20,
    use_recombine=1,
    use_code_switch=1,
    use_eos=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='SpaceAdd'),
        dict(type='BPE', use_eos=True, skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>']),
        dict(type='CMVN', key='fbank'),
        dict(type='TimeMask', time_mask_size=20, time_mask_num=10),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='SpaceAdd'),
        dict(type='BPE', use_eos=True, skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>']),
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
solution = dict(
    type='base_las_solution',
    las_type='BaseLASModel',
    # front_end
    front_end_type='VGGPosFrontEnd',
    input_concat_size=8,
    downsampling_size=4,
    fsmn_left_kernel_size=20,
    fsmn_right_kernel_size=1,
    fsmn_dilation=2,
    # backbone
    acoustic_backbone_type='TransformerBackbone',
    backbone_topology='[[-1,-1,1]]*10',
    backbone_layer_gap=100,
    dfsmn_bn_flag=1,
    backbone_weight_scale=1.0,
    backbone_memory_size=512,
    backbone_hidden_size=2048,
    self_attn_heads=8,
    self_attn_dropout=0.1,
    dropout=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=1,
    self_attn_activation_fn='relu',
    # attention
    # decoder
    las_decoder_type='LSTMLASDecoder',
    embedding_size=512,
    decoder_input_size=512,
    decoder_lstm_hidden_size=2048,
    decoder_lstm_layer_num=2,
    decoder_dropout=0.15,
    decoder_pred_fc_size=2048,
    decoder_pred_fc_proj_size=768,
    ## head
    las_atten_type='MHAttention',
    atten_hidden_size=2048,
    multi_head_num=8,
    multi_att_weight_drop=0.1,
    # criterion
    criterion_type='LasCE',
    cer_update_freq=50,
    label_smooth_factor=0.1,
    # schedule sampling
    schedule_sample_begin=150000,
    schedule_sample_increase_steps=50000,
    schedule_ratio=0.3,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=50,
    max_iters=800000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=10000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.8,
        step=[100000 + i * 20000 for i in range(140)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhangjun.jarry/saved_models/dolphin_test',
    save_root='./',
    save_dir='las_chinglish',
    save_name='las_test',
    amp_level='O1',
    )
valid = dict(
    interval=5000, # iter
    )
inference = dict(
    bucket_schedule='100000',
    test_sets='80dim_subshard.ez_chinglish_20200113_1554',
    beam_size=10,
    lm_path="",
    lm_weight=1.0,
    language='en',
    nbest_out=False,
    filter_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>', '<space>'],
    #remote_stat_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/",
    )
nnlm = dict(
    lm_type="",
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam',
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=20,
        norm_type=2,
        ),
    max_grad_clip=0.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.9,
        warmup_steps=5000,
        use_nesterov=True,
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
