project = "conformer_cif"
runner = 'BaseCifRunner'

# increment to add and override args.
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
        ['["train_sub{}".format(i) for i in range(6)]'], 
    valid_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/data/multidomain/cv/',
    valid_file_list = '["merged_cv_v1.01"]',
    test_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/donglinhao/dolphin/data/multidomain/test/',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    meta_file="meta_ag350k_lzl",
    fbank_dim=80,
    fbank_channel=1,
    chunk_size=20,
    bucket_schedule='181, 226, 264, 300, 334, 370, 407, 448, 497, 549, 608, 680, 760, 854, 966, 1124, 1334, 1526, 2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    max_batch_size=51200,
    batch_means_tokens=1,
    use_eos=True,
    global_shuffle=True,
    shuffle=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='PunctuationFilter', key='label'),
        dict(type='ConcateEnLetters'),
        dict(type='BPE', use_eos=True, skip_list=['</s>']),
        dict(type='CMVN', key='fbank'),
        #dict(type='AppendFeatureDim', total_dim=80),
        dict(type='LengthFilter', key='fbank', min_len=20, max_len=2000),
        #dict(type='TimeMask', time_mask_size=20, time_mask_num=10, replace_with_zero=True),
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60, 
            replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1, replace_with_zero=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='PunctuationFilter', key='label'),
        dict(type='ConcateEnLetters'),
        dict(type='BPE', use_eos=True, skip_list=['</s>']),
        dict(type='CMVN', key='fbank'),
        #dict(type='AppendFeatureDim', total_dim=80),
        # in_out_ratio is used for fair comparison with other models in evaluation, 8 is actually used in cif
        dict(type='LabelLengthFilter', in_out_ratio=4),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate', rnnt_format=False),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='CharCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)
solution = dict(
    model_type='BaseCifModel',
    vocab_size=9759,
    use_fused_kernel=True,

    # front end settings
    use_cnn_frontend=True,
    conv_norm_type='layer_correct',
    down_sample_conv_num_layers=1,
    down_sample_conv_num_filters=64,
    additional_module='new_mu33_2',
    additional_module_num_layers=1,
    additional_module_num_filters=64,

    # self-attention encoder/decoder setting
    self_attention_type='dot_product',
    num_heads=8,
    hidden_size=512,
    filter_size=2048,
    ffn_layer='dense_relu_dense',
    layer_preprocess_sequence='n',
    layer_postprocess_sequence='da',
    norm_type='layer',
    norm_epsilon=0.000001,
    proximity_bias=True,

    front_end_type='CifOriginalFrontend',
    acoustic_backbone_type='CifConformerV2Encoder',
    conformer_conv_width=31,
    conformer_conv_norm_type='none',

    num_encoder_layers=20,
    encoder_self_attention_type='dot_product',
    sa_pooling_layers=[7, 14],

    num_decoder_layers=2,
    decoder_self_attention_type='dot_product',

    # use teacher_forcing / pss
    use_teacher_forcing=True,
    use_pss=False,
    pss_rate=0.0,

    # chunk hopping
    use_chunk_hopping=False,
    chunk_size=80,
    hop_size=40,
    ch_type='fixed_future',
    ch_fixed_future_size=40,

    # regularizer
    layer_prepostprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.0,

    # extra
    loss_multiplier=2.0,
    shared_embedding_and_softmax_weights=False,
    multiply_embedding_mode='sqrt_depth',

    # cif part
    produce_weights_type='conv',
    conv_cif_num_layers=1,
    conv_cif_width_string='3',
    conv_cif_num_filters=512,
    conv_cif_dropout=0.0,
    dense_cif_units=512,

    cif_weight_threshold=0.99999,
    use_scaling_strategy=True,
    use_tail_handling=True,

    # all loss setting
    calculated_loss='ce_loss, quantity_loss, ctc_loss_on_encoder',

    # ce loss
    label_smoothing=0.1,
    ls_type='uniform',

    # ctc loss on the encoder
    ctc_loss_on_encoder=True,
    ctc_loss_lambda=0.5,

    # quantity loss
    quantity_loss_lambda=1.0,

    # criterion
    criterion_type='CifLoss',
    eos_id=2,
    use_input_padding=True,
    save_until_pad=True,
)
# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    max_epochs=50,
    max_iters=100000,
    lr_scheduler=dict(
        policy='Plateau',
        by_epoch=False,
        start_decay_step=60000,
        end_decay_step=100000,
        each_decay_step=4000,
        peak_lr=0.001,
        init_lr=0,
        end_lr=0.0001,
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        ),
    checkpoint_config=dict(
        interval=1600, # steps
        max_keep_ckpts=20,
    ),
    save_root='./cif',
    save_dir='san_cif',
    save_name='san_cif',
    amp_level='O1',
    # evaluation setting
    beam_size=1,
    top_beams=1,

    # gradient accumulation
    grad_accum_step=3,
    # 'AVG' 'SUM'
    grad_accum_mode='SUM'
    )
valid = dict(
    average_ckpts=True, # iter
    num_averaging_ckpts=10,
    beam_size=10,
    top_beams=1,
    output_probs=False,
    not_show_results=False,
    interval=1600,
    )
inference = dict(
    bucket_schedule='5,50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    test_sets='merged_public|lark_5hmm_test_right|bytebot_tele_analysis_20210412_31224|tomato_v1_v2_v3_merged|merged_eh',
    beam_size=10,
    nbest=1,
    cif_temperature=1.0
    )
# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.04,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=1.0,
        norm_type=2,
        ),
    # 'accum_after' & 'accum_before'
    grad_clip_mode='accum_before',
    max_grad_clip=0.0,
 )
log_level = 'INFO'
log_config = dict(
    interval=100,  # log at every 100 iterations
    reset_flag=True,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])
