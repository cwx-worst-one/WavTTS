project = "data2vec_base_pretrain"
runner = "Wav2vecPretrainRunner"
data = dict(
    data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/librispeech/hdfs_data/",
    ],
    train_file_list=[
        '["shard{}".format(i) for i in range(16)]',
    ],
    valid_data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/librispeech/hdfs_data/"
    ],
    valid_file_list=[
        '["shard.dev_other"]',
    ],
    tgt_dict_dir="total_bpe.dict",
    bpe_code="total.code",
    bucket_schedule='200,300,400,500,600,800,1000,1200,1400,1600,1800,2000',
    bucket_schedule_val='5,100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_key="length",  # frame length
    batch_means_tokens=1,
    max_batch_size=24000,
    batch_strategy='FixedBucketBatching',
    use_old_bucket=True,
    prefetch_worker_num=4,
    next_retry=1000,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    train_item_transform=[
        dict(type="PickleParser"),
        dict(type="WavResample", sample_rate=16000, key="wav"),
        dict(type="WavCrop", key="wav", crop_sample_length=480000),
        dict(type="WavConvert", in_key="wav"),
        dict(type="CalculateFrameLength"),
    ],
    valid_item_transform=[
        dict(type="PickleParser"),
        dict(type="WavResample", sample_rate=16000, key="wav"),
        dict(type="WavCrop", key="wav", crop_sample_length=480000),
        dict(type="WavConvert", in_key="wav"),
        dict(type="CalculateFrameLength"),
    ],
    batch_transform=[
        dict(type="WaveformCollate", wav_key="waveform", frame_chunk_size=1),
    ],
    inference_batch_transform=[
        dict(type="WaveformCollate", wav_key="waveform"),
    ],
)
solution = dict(
    type="base_data2vec_pretrain_solution",
    model_type="BaseData2vecPretrainModel",
    data2vec_type='PretrainedData2Vec',
    activation_checkpoint=False,
    data2vec_finetuning=False,
    data2vec_feature_conv_type="default",
    # mode for feature extractor. default has a single group norm with
    # d groups in the first conv block, whereas layer_norm has layer
    # norms in every block (meant to use with normalize)
    extractor_mode="layer_norm",
    normalize=True,  # normalize input feature with layernorm
    # norm
    average_top_k_layers=8,
    layer_norm_target_layer=False,
    instance_norm_target_layer=True,
    instance_norm_targets=False,
    layer_norm_targets=False,
    batch_norm_target_layer=False,
    group_norm_target_layer=False,
    # encoder
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn="gelu",
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    layer_norm_first=False,  # apply layernorm first in the transformer
    encoder_layerdrop=0.05,  # changed
    pad_to_multiple=True,
    required_seq_len_multiple=2,
    output_layer_result=True,
    # convolutional feature extraction layers [(dim, kernel_size, stride), ...]
    conv_feature_layers="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2",
    feature_grad_mult=1.0,  # multiply feature extractor var grads by this
    # mask
    mask_length=10,  # mask length
    mask_prob=0.65,  # probability of replacing a token with mask changed
    mask_selection="static",  # how to choose masks
    mask_other=0.0,  # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_minlen_type="random",  # how to choose minlen masks
    no_mask_overlap=False,  # whether to allow masks to overlap
    mask_min_space=1,  # min space between spans (if no overlap is enabled)
    mask_dropout=0.0,
    # mask channel
    mask_channel_length=10,  # repeat the mask indices multiple times
    mask_channel_prob=0.0,  # probability of replacing a token with mask
    mask_channel_selection="static",  # how to choose masks
    mask_channel_other=0.0,  # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_channel_minlen_type="random",  # how to choose minlen masks
    mask_channel_before=False,
    no_mask_channel_overlap=False,  # whether to allow masks to overlap
    mask_channel_min_space=1,  # min space between spans (if no overlap is enabled)
    require_same_masks=True,
    # ema
    ema_decay=0.999,
    ema_end_decay=0.9999,
    ema_anneal_end_step=30000,
    ema_transformer_only=True,
    ema_layers_only=True,
    # dropout
    dropout_input=0.0,  # dropout to apply to the input (after feat extr)
    conv_pos=95,  # number of filters for convolutional positional embeddings
    conv_pos_groups=16,  # number of groups for convolutional positional embedding
    pos_conv_depth=5,
    conv_bias=False,  # include bias in conv encoder
    final_proj_num_layer=1,
    weighted_data2vec=False,
    # loss
    loss_beta=0,
    loss_scale=-1,
    min_target_var=0.1,
    min_pred_var=0.01,
)
work_dir = "./w2v_base_pretrain"
train = dict(
    resume='latest.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    grad_accum_step=1,
    bucket_bytes_cap=16 * 1024 * 1024,
    max_epochs=60,
    max_iters=400000,
    lr_scheduler=dict(
        by_epoch=False,
        policy="TriStage",
        lr=5e-4,
        phase_ratio="[0.03,0.9,0.07]",
        init_lr_scale=0.01,
        final_lr_scale=0.01,
        max_train_steps=400000,
    ),
    checkpoint_config=dict(interval=10000, max_keep_ckpts=10),
    save_root="./d2v_pretrain/",
    save_dir="data2vec2_base",
    save_name="base_lr5e-4_poly400k_mask0.65_ld0.05_bsz63m",
    freeze_feature_extractor=False,
    compare_metric="UF",
    validation_log=True,
    amp_level="O1",
)
valid = dict(interval=5000)
optimizer = dict(
    type="FusedAdam", lr=5e-4, betas=(0.9, 0.98), eps=1e-06, weight_decay=1e-2
)
optimizer_config = dict(
    grad_clip=dict(max_norm=10, norm_type=2),
    max_grad_clip=0.0,
    bmuf_config=0,
)
log_config = dict(
    interval=100,
)
