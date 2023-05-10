project = 'tel_rnnt'
runner = 'RNNTRunner'
data = dict(
    data_root= [
        ] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/Datatang_child_2000h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/Huiting_child_1000h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/voice_input_first_chn_eng_4500h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/King_ASR_60h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/ez_3400h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/libri-light_1_3600h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/libri-light_2_4600h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/zhiyuan_20h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/shenchen/data/zhiyuan/wav_data'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/livechat_internal_asr_annotation_lark'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/livechat_internal_asr_annotation_lark_new'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_54000h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/video_live_6wh'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/dou' + \
        'yin_1wh'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/dou' + \
        'yin_ch_en_1300h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_data/daliremoval/eh_finetune.202203/h_data_3rd_2ch_3000h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_data/daliremoval/eh_finetune.202203/h_data_4th_2ch_900h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_data/daliremoval/eh_finetune.202203/h_data_5th_2ch_2000h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_data/daliremoval/eh_finetune.202203/h_data_1ch_1st_2000h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_data/daliremoval/eh_finetune.202203/h_data_ch_en_1st_500h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_data/daliremoval/eh_finetune.202203/h_data_ch_en_2nd_200h'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/smb_gogokid'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/feiyu_9th'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/feiyu_10th'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/feiyu_11th'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/ivr'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/ivr_2nd'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/ivr_3rd'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/xingfuli'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dongchedi_3rd'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/caijing/caijing_3rd'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dianshang_1st'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele/dianshang_2nd'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/chinese/dou' + \
        'yin_vs/sami_share_10kh'] + \
        ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/wemeet'], 
    train_file_list = [
        ] + \
        ['["sub_{}".format(i) for i in range(20)]'] + \
        ['["sub_{}".format(i) for i in range(5)]'] + \
        ['["sub_{}".format(i) for i in range(23)]'] + \
        ['["sub_{}".format(i) for i in range(1)]'] + \
        ['["sub_{}".format(i) for i in range(10)]'] + \
        ['["sub_{}".format(i) for i in range(17)]'] + \
        ['["sub_{}".format(i) for i in range(23)]'] + \
        ['["sub_{}".format(i) for i in range(1)]'] + \
        ['["train_sub{}".format(i) for i in range(1)]'] + \
        ['["sub_{}".format(i) for i in range(1)]'] + \
        ['["sub_{}".format(i) for i in range(1)]'] + \
        ['["sub_{}".format(i) for i in range(115)]'] + \
        ['["sub_{}".format(i) for i in range(133)]'] + \
        ['["sub_{}".format(i) for i in range(19)]'] + \
        ['["sub_{}".format(i) for i in range(10)]'] + \
        ['["sub_{}".format(i) for i in range(40)]'] + \
        ['["sub_{}".format(i) for i in range(13)]'] + \
        ['["sub_{}".format(i) for i in range(26)]'] + \
        ['["sub_{}".format(i) for i in range(51)]'] + \
        ['["sub_{}".format(i) for i in range(6)]'] + \
        ['["sub_{}".format(i) for i in range(3)]'] + \
        ['["train_sub{}".format(i) for i in range(13)]'] + \
        ['["train_sub{}".format(i) for i in range(26)]'] + \
        ['["train_sub{}".format(i) for i in range(6)]'] + \
        ['["train_sub{}".format(i) for i in range(1)]'] + \
        ['["train_sub{}".format(i) for i in range(13)]'] + \
        ['["train_sub{}".format(i) for i in range(3)]'] + \
        ['["train_sub{}".format(i) for i in range(26)]'] + \
        ['["train_sub{}".format(i) for i in range(3)]'] + \
        ['["train_sub{}".format(i) for i in range(6)]'] + \
        ['["train_sub{}".format(i) for i in range(6)]'] + \
        ['["train_sub{}".format(i) for i in range(6)]'] + \
        ['["train_sub{}".format(i) for i in range(3)]'] + \
        ['["sub_{}".format(i) for i in range(39)]'] + \
        ['["train_sub{}".format(i) for i in range(6)]'], 
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/data/multidomain/cv/',
    valid_file_list = '["merged_cv"]',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/data/multidomain/test',
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/tele',
    meta_file="meta_v2",
    fbank_dim=80,
    global_shuffle=True,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,1900,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,3000,4000,5000,10000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    max_batch_scale=10,
    prefetch_worker_num=4,
    native_parallel_file_num=8,
    next_retry=1024,
    shuffle=True,
    drop_last=False,
    use_recombine=True,
    use_code_switch=False,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser', max_len=20.02),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
        dict(
            type='DynamicTimeMask',
            time_mask_size=20,
            time_mask_block=60,
            replace_with_zero=False,
            inplace=True,
            time_mask_num_dither=1),
        dict(
            type='FreqMask',
            freq_mask_size=27,
            freq_mask_num=1,
            replace_with_zero=False,
            inplace=True)
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='LabelLengthFilter', in_out_ratio=8),
        dict(type='BPE', use_eos=False, skip_list=['<pad>']),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate')
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80)
    ])
solution = dict(
    model_type='BaseRnntModel',
    # front_end
    front_end_type='Conv2dPooling4',
    front_end_conv0_ch=256,
    front_end_conv1_ch=256,
    input_concat_size=4,
    downsampling_size=4,
    # backbone
    acoustic_backbone_type='MaskedConformerBackbone',
    conformer_normalize_before=True,
    conformer_attention_heads=8,
    conformer_mask_topology='[[160,160]]*16',
    conformer_linear_units=2048,
    conformer_num_blocks=16,
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
    conformer_cnn_module_kernel='15',
    conformer_cnn_norm_type='layer_norm',
    conformer_layernorm_interval=0,
    conformer_weight_scale=1.0,
    conformer_half_pooling=0,
    backbone_memory_size=512,
    dropout=0.1,
    moe_args=dict(
        activation='gelu',
        moe_expert=8,
        moe_topk=2,
        moe_loss_scale=1e-5,
        moe_layers='[i for i in range(8, 16)]',
        use_lego=True,
    ),
    # head
    head_type='RnntBaseHead',
    head_hidden_size=640,
    head_lowrank_size=512,
    # predictor
    predictor_type='LSTMPredictor',
    predictor_emb_size=640,
    predictor_lstm_hidden_size=1024,
    predictor_lstm_layer_num=2,
    predictor_dropout_factor=0.1,
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
    cer_update_freq=500,
    label_smooth_factor=0.0,
    reorder_dict_by_freq=True,
    backbone_clamp_inf=True,
    head_clamp=True,
    mtl_clamp=True,
    backbone_pool_type='Conv1dTimeReduce')
work_dir = './demo'
train = dict(
    resume='latest.pth',
    resume_optimizer=True,
    max_epochs=9,
    check_loss=True,
    max_iters=1000000,
    lr_scheduler=dict(
        by_epoch=True,
        warmup='linear',
        warmup_iters=10000,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='step',
        gamma=0.85,
        step=[10 + i for i in range(100)]
    ),
    checkpoint_config=dict(interval=20000, max_keep_ckpts=20, max_local_keep_ckpts=10),
    #remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/',
    save_root='./demo',
    save_dir='conformer_nonstreaming',
    save_name='moe',
    amp_level='O1')
valid = dict(interval=20000)
inference = dict(
    test_sets='merged_public|lark_5hmm_test_right|bytebot_tele_analysis_20210412_31224|tomato_v1_v2_v3_merged|merged_eh',
    beam_size=8,
    lm_path='',
    lm_weight=1.0,
    blk_scale=0.7,
    use_batch_beam=True,
    len_penalty_scale=0.01)
optimizer = dict(
    type='FusedAdam',
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.05)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=5.0,
    bmuf_config=dict(
        block=50,
        block_lr=1.0,
        block_momentum=0.9,
        warmup_steps=5000,
        warmup_sync=True,
        average_sync=True,
        use_nesterov=True)
    )
criterion = None
super_criterion = None
log_level = 'INFO'
log_config = dict(
    interval=500,
    reset_flag=True,
    hooks=[dict(type='TextLoggerHook')])
