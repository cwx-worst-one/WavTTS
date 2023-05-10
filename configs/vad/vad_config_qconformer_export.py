project = "vad_ce_train"
runner = 'BaseVadRunner'
# increment to add and override args.
data = dict(
    data_root = [
    '/mnt/bd/speech-vad-0/yuanzeyu/datasets/Datatang_child_2000h/feat_pack/',
    ],
    train_file_list = [
        '["train_sub{}".format(i) for i in range(0, 200)]',
    ],
    valid_data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guxinyi/workspace/20210704monophoneVAD/cv_set/',
    valid_file_list = '["EH_data_2nd_1ch_400h", "EH_data_3rd_2ch_3000h", "EH_data_4th_2ch_900h", "EH_data_whisper_100h"]',
    meta_file="meta_online",
    fbank_dim=80,
    chunk_size=20,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000,4000,5000,6000,7000,8000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=81920,
    shuffle=True,
    drop_last=False,
    tgt_vocab_size=2,
    prefetch_worker_num=8,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='SilenceRatioFilter'),
        dict(type='AlignFeature', key='fbank', ce_key='bi_vad_label'),
        dict(type='LengthFilter', key='length', max_len=3000),
        dict(type='CMVN', key='fbank'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='SilenceRatioFilter'),
        dict(type='AlignFeature', key='fbank', ce_key='bi_vad_label'),
        dict(type='LengthFilter', key='length', max_len=3000),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CeLabelCollate', frame_chunk_size=32, key='bi_vad_label')
      #  dict(type='CeLabelCollate', frame_chunk_size=32)
      #  dict(type='CeLabelCollate', frame_chunk_size=32)
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CeLabelCollate', frame_chunk_size=32, key='bi_vad_label')
        # dict(type='ListCollate', key='uttid'),
        # dict(type='RefLabelCollate'),
        # dict(type='FbankCollate', fbank_dim=80),
    ],
)

solution = dict(
    type='vad_ce_solution',
    ce_type='VadCeModel',
    
    # backbone
    encoder_type='ConformerEncoder',

    acoustic_backbone_type='MaskedConformerBackbone',
    conformer_normalize_before=True,
    conformer_attention_heads=8,
    conformer_mask_topology='[[64,0]]*6',
    conformer_linear_units=128,
    conformer_num_blocks=6,
    conformer_dropout_rate=0.1,
    conformer_positional_dropout_rate=0.1,
    conformer_attention_dropout_rate=0.1,
    conformer_positionwise_layer_type='linear',
    conformer_activation_fn='gelu',
    conformer_positionwise_conv_kernel_size=1,
    conformer_macaron_style=1,
    conformer_pos_enc_layer_type='fix_rel_pos',
    conformer_selfattention_layer_type='rel_selfattn',
    conformer_layer_order='mhsa_before_conv',
    conformer_use_cnn_module=1,
    conformer_cnn_module='ConvolutionModule',
    conformer_cnn_module_kernel='15,0',
    conformer_cnn_norm_type='layer_norm',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_memory_size=64,
    backbone_hidden_size=64,
    tgt_vocab_size=2,
    input_concat_size = 8,
    conformer_linearPredict_dropout=0.8,
    downsampling_size = 1,
    # criterion
    criterion_type='Xentropy',
    cer_update_freq=200,
    label_smooth_factor=0.0,
    xavier_init=False,
    

    # slim
    slim_config = {
        'Quantizer': {
            'Linear': {},
            'default': {
                'per_channel' : True,
                'act_preprocess': {
                    'ScaleUpdater': {
                        'initializer': 'moving_average_abs_max',
                        'moving_const': 0.1,
                        'update_type': 'statistic',
                        'update_step': 10000
                    }
                },
                'param_preprocess': {
                    'ScaleUpdater': {
                        'initializer': 'aciq',
                        'moving_const': 0.1,
                        'update_type': 'statistic',
                        'update_step': 10000
                    }
                }
            }
        }
    }
)
# runtime settings
work_dir = './vad_ce_train'
# training and testing settings
train = dict(
    max_epochs=50,
    max_iters=8000000000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup=None,
        warmup_iters=0,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.5,
        step=[4 + i for i in range(50)]
        ),
    checkpoint_config=dict(
        interval=10000, # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    save_root='./vad_ce_train',
    save_dir='vad_ce_train',
    save_name='vad_ce_train',
    amp_level='O0',
    resume_pretrain_chkpt='hdfs://haruna/usr/lab/speech/panther/pantherslim/luxiaojin/vad_conformer_qat_fulldata/1118_quant_full_dataC3_1e-5_/vad_ce_train/vad_config_conformer_fulldataC3/checkpoints/step_25000.pth'
    )
valid = dict(
    interval=10000, # iter
    )

# optimizer
optimizer = dict(
    type='FusedAdam',
    lr=1e-4,
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
        warmup_steps=50,
        average_sync=True,
        use_nesterov=True))

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

