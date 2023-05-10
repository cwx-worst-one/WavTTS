# 1 GPU: https://cloud.bytedance.net/arnold/trial/3708018
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhousuping/telephone_emotion/dolphin_data/feature_pho_60_no_space_asr_v2v1swst',
    ],
    train_file_list=[
        '["train_pho_"]',
    ],
    valid_file_list=[
        '["valid_pho_"]',
    ],
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,5000,10000,50000,100000',
    bucket_schedule_key='length',
    max_batch_size=10000,
    use_old_bucket=True,
    batch_means_tokens=1,
    prefetch_worker_num=3,
    next_retry=1000,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        # dict(type='WavNorm', zscore=True),
        dict(type='WavResample', sample_rate=16000, key='waveform'),
        dict(type='EmotionLabelParser', key='emotion_phone'),
        dict(type='CalculateFrameLength'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        # dict(type='WavNorm', zscore=True),
        dict(type='WavResample', sample_rate=16000, key='waveform'),
        dict(type='EmotionLabelParser', key='emotion_phone'),
        dict(type='CalculateFrameLength'),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='emotion'),
        dict(type='EmotionFeatCollate', key='waveform', out_key='waveform', mask_key='src_mask'),
        dict(type='EmotionFeatCollate', key='emotion_phone', mask_key='label_mask', pad=-100),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='ListCollate', key='emotion'),
        dict(type='EmotionFeatCollate', key='waveform', out_key='waveform', mask_key='src_mask'),
        dict(type='EmotionFeatCollate', key='emotion_phone', mask_key='label_mask', pad=-100),
    ],
)
project = 'emotion_recognition'
runner = 'EmotionRecognitionRunner'
solution = dict(
    type='base_emotion_recognition_solution',
    model_type='BaseEmotionRecognitionModel',
    dim_emotion_num=6,
    dim_tgt_size=362,
    dim_neutral_label=1,
    phone_num=60,
    phone_tag=2,
    emotion_num=6,
    vocab_size=362,  # phone_num*emotion_num+phone_tag
    tgt_size=362,
    neutral_label=1,
    predict_sets='emotion_phone',
    labels_sets='emotion_phone',
    output_hidden_states=False,
    fused_transformer=False,
    pad_token_id=0,
    activation_checkpoint=False,
    encoder_type='PretrainedWav2Vec2',
    wav2vec_load_pretrained_model=False,
    wav2vec_finetuning=True,
    wav2vec_feature_conv_type='default',
    wav2vec_use_fbank=False,
    fbank_conv_embed_dim=512,
    # mode for feature extractor. default has a single group norm with
    # d groups in the first conv block, whereas layer_norm has layer
    # norms in every block (meant to use with normalize)
    # extractor_mode='layer_norm',
    extractor_mode='default',
    normalize=False,  # normalize input feature with layernorm
    # encoder
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn='gelu',
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
    final_dim=256,  # 0 means the encoder_embed_dim
    layer_norm_first=False,  # apply layernorm first in the transformer
    encoder_layerdrop=0,  # changed
    # convolutional feature extraction layers [(dim, kernel_size, stride), ...]
    conv_feature_layers="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2",
    logit_temp=0.1,  # temperature to divide logits by
    quantize_targets=True,  # use quantized targets
    quantize_input=False,  # use quantized inputs
    same_quantizer=False,  # use same quantizer for inputs and targets
    feature_grad_mult=0.0,  # multiply feature extractor var grads by this
    latent_vars=320,  # number of latent variables V in each group of the codebook
    latent_groups=2,  # number of groups G of latent variables in the codebook
    latent_dim=0,  # if set, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups
    # mask
    mask_length=10,  # mask length
    mask_prob=0.1,  # probability of replacing a token with mask changed
    mask_selection='static',  # how to choose masks
    mask_other=0.0,  # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_minlen_type='random',  # how to choose minlen masks
    no_mask_overlap=False,  # whether to allow masks to overlap
    mask_min_space=1,  # min space between spans (if no overlap is enabled)
    # mask channel
    mask_channel_length=64,  # repeat the mask indices multiple times
    mask_channel_prob=0.1,  # probability of replacing a token with mask
    mask_channel_selection='static',  # how to choose masks
    mask_channel_other=0.0,  # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_channel_minlen_type='random',  # how to choose minlen masks
    no_mask_channel_overlap=False,  # whether to allow masks to overlap
    mask_channel_min_space=1,  # min space between spans (if no overlap is enabled)
    # dropout
    dropout_input=0.1,  # dropout to apply to the input (after feat extr)
    dropout_features=0.0,  # dropout to apply to the features (after feat extr)
    final_dropout=0.1,  # dropout after transformer and before final projection
    num_negatives=100,  # number of negative examples
    negatives_from_everywhere=False,  # sample negatives from everywhere, not just masked states
    cross_sample_negatives=0,  # num of cross sampled negatives
    codebook_negatives=0,  # num of codebook sampled negatives
    conv_pos=128,  # number of filters for convolutional positional embeddings
    conv_pos_groups=16,  # number of groups for convolutional positional embedding
    latent_temp='(2,0.5,0.999995)',  # temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)
    target_glu=False,  # adds projection + glu to targets
    conv_bias=False,  # include bias in conv encoder
    # head
    head_type='Wav2Vec2SeqOutHead',
    # criterion
    criterion_type='CTC',
    # loss
    loss_weights='[0.1, 10]',  # weights for additional loss terms (not first one)
    infonce=True,  # if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)
    apply_mask=True,
    freeze_finetune_updates=0,  # dont finetune wav2vec for this many updates
    freeze_encoder_layers=-1,  # freeze bottom # of encoder layers
    # ctc
    ctc_use_lm=0,
    zero_infinity=1,
    sentence_avg=1,
    lid_mask_ratio=1.0,
    lid_loss_scale=1.0,
    ctc_loss_reduction='mean',
)
work_dir = './emotion'
train = dict(
    # resume='latest.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    max_epochs=30,
    max_iters=100000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=1,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.9,
    ),
    checkpoint_config=dict(interval=1000, max_keep_ckpts=2),
    resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhousuping/pretrained_models/wav2vec2_base/base_e_business_data_2.1h_3e4/checkpoints/step_400000.pth',
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhousuping/iemocap_emotion/dolphin_data/saved_models/',
    save_root='./emotion/',
    save_dir='emotion_wav2vec',
    save_name='test/',
    freeze_feature_extractor=True,
    compare_metric='WF',
    validation_log=True,
    amp_level='O1',
)
valid = dict(interval=500, save_latest_results=True)
inference = dict(
    save_results=False,
    save_original_results=False,
    save_remote=True,
    # save_results_filename='',
    test_sets='test_pho_',
    save_features=False,
    save_features_dir='../test/',
    inference_only=False,
)
optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2),
    max_grad_clip=50.0,
)
log_level = 'INFO'
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')],
)
