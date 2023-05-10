project = 'wav2vec_base_pretrain'
runner = 'Wav2vecPretrainRunner'
data = dict(
    data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/cantonese_unsup4kh/hdfs_data/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/cantonese_live_unsup6kh/hdfs_data/',
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/cantonese_xigua_unsup1kh/hdfs_data/',
    ],
    train_file_list=[
        '["shard{}".format(i) for i in range(80)]',
        '["shard{}".format(i) for i in range(128)]',
        '["shard{}".format(i) for i in range(16)]',
    ],
    valid_data_root=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/cantonese_noise1kh/hdfs_data/'
    ],
    valid_file_list=[
        '["shard.yueyu_noise"]',
    ],
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    tgt_vocab_size=0,
    use_lid=False,
    fetch_block_size=300,
    bucket_schedule='5,10000',
    bucket_schedule_val='5,10000',
    bucket_schedule_key='length', # frame length
    batch_means_tokens=1,
    # use_old_bucket=True,
    max_batch_size=18000,
    prefetch_worker_num=4,
    next_retry=1000,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavResample', sample_rate=16000, key='wav'),
        dict(type='WavCrop', key='wav', crop_sample_length=480000),
        dict(type='WavConvert', in_key='wav'),
        dict(type='CalculateFrameLength'),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavResample', sample_rate=16000, key='wav'),
        dict(type='WavCrop', key='wav', crop_sample_length=480000),
        dict(type='WavConvert', in_key='wav'),
        dict(type='CalculateFrameLength'),
    ],
    batch_transform=[
        dict(type='WaveformNarrowCollate', wav_key='waveform', max_sample_length=180000),
        dict(type='ComputeW2vMask', in_key='waveform', out_key='mask_idc', padding=True, min_masks=2),
    ],
    inference_batch_transform=[
        dict(type='WaveformNarrowCollate', wav_key='waveform', max_sample_length=180000),
    ])
solution = dict(
    type='base_wav2vec_pretrain_solution',
    model_type='BaseWav2vecPretrainModel',
    wav2vec_type='PretrainedWav2Vec2',
   
    vocab_size=0,
    tgt_size=0,

    activation_checkpoint=False,
    wav2vec_finetuning=False,
    wav2vec_feature_conv_type='default',
    wav2vec_use_fbank=False,
    fbank_conv_embed_dim=512,
    # mode for feature extractor. default has a single group norm with 
    # d groups in the first conv block, whereas layer_norm has layer 
    # norms in every block (meant to use with normalize)
    extractor_mode='default',
    normalize=False, # normalize input feature with layernorm
    # encoder
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn='gelu',
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    final_dim=256, # 0 means the encoder_embed_dim
    layer_norm_first=False, # apply layernorm first in the transformer
    encoder_layerdrop=0.0, #changed
    # convolutional feature extraction layers [(dim, kernel_size, stride), ...]
    conv_feature_layers="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2",
    logit_temp=0.1, # temperature to divide logits by
    quantize_targets=True, # use quantized targets
    quantize_input=False, # use quantized inputs
    same_quantizer=False, # use same quantizer for inputs and targets
    feature_grad_mult=0.1, # multiply feature extractor var grads by this
    latent_vars=320, # number of latent variables V in each group of the codebook
    latent_groups=2, # number of groups G of latent variables in the codebook
    latent_dim=0, # if set, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups
    # mask
    mask_length=5, # mask length
    mask_prob=0.65, # probability of replacing a token with mask changed
    mask_selection='static', # how to choose masks
    mask_other=0.0, # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_minlen_type='random', # how to choose minlen masks
    no_mask_overlap=False, # whether to allow masks to overlap
    mask_min_space=1, # min space between spans (if no overlap is enabled)
    # mask channel
    mask_channel_length=10, # repeat the mask indices multiple times
    mask_channel_prob=0.0, # probability of replacing a token with mask
    mask_channel_selection='static', # how to choose masks
    mask_channel_other=0.0, # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_channel_minlen_type='random', # how to choose minlen masks
    no_mask_channel_overlap=False, # whether to allow masks to overlap
    mask_channel_min_space=1, # min space between spans (if no overlap is enabled)
    # dropout
    dropout_input=0.1, # dropout to apply to the input (after feat extr)
    dropout_features=0.1, # dropout to apply to the features (after feat extr)
    num_negatives=100, # number of negative examples
    negatives_from_everywhere=False, # sample negatives from everywhere, not just masked states
    cross_sample_negatives=0, # num of cross sampled negatives
    codebook_negatives=0, # num of codebook sampled negatives
    conv_pos=128, # number of filters for convolutional positional embeddings
    conv_pos_groups=16, # number of groups for convolutional positional embedding
    latent_temp='(2,0.5,0.999995)', # temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)
    target_glu=False, # adds projection + glu to targets
    conv_bias=False, # include bias in conv encoder
    # loss
    loss_weights='[0.1, 10]', # weights for additional loss terms (not first one)
    infonce=True, # if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)
    apply_mask=True,
    freeze_finetune_updates=0, # dont finetune wav2vec for this many updates
    freeze_encoder_layers=-1, # freeze bottom # of encoder layers
)
work_dir = './w2v_base_pretrain'
train = dict(
    # resume='latest.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    grad_accum_step=1,
    bucket_bytes_cap=16*1024*1024,
    max_epochs=60,
    max_iters=400000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=32000,
        warmup_ratio=0.00003125, # 1/32000
        warmup_by_epoch=False,
        policy='poly',
        ignore_warmup=False,
    ),
    checkpoint_config=dict(interval=10000, max_keep_ckpts=10),
    # resume_hdfs_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhousuping/iemocap_emotion/dolphin_data/saved_models/lighteternal/wav2vec2-large-xlsr-53-greek/pytorch_model.bin',
    # remote_save_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/zhengyijie/saved_models/dolphin',
    save_root='./w2v_pretrain/',
    save_dir='wav2vec2_base',
    save_name='base_10kh_lr5e-4_poly400k_mask0.65_len5_cat11.25s_bsz1.6h_v2',
    freeze_feature_extractor=False,
    compare_metric='UF',
    validation_log=True,
    amp_level='O1')
valid = dict(interval=5000) 
optimizer = dict(
    type='FusedAdam',
    lr=5e-4,
    betas=(0.9, 0.98),
    eps=1e-06,
    weight_decay=1e-2)
optimizer_config = dict(
    grad_clip=dict(max_norm=1, norm_type=2),
    max_grad_clip=0.0,
    bmuf_config=0,
        )
log_config = dict(
    interval=100,
    reset_flag=True,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
