project = "data2vec_ctc_base_finetune"
runner = "Wav2vecCtcRunner"
data = dict(
    data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/librispeech/hdfs_data/",
    ],
    train_file_list=[
        '["shard.train_10h"]',
    ],
    valid_data_root=[
        "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/kv_datasets/librispeech/hdfs_data/"
    ],
    valid_file_list=[
        '["shard.dev_other"]',
    ],
    tgt_dict_dir="librispeech_phn_dict.txt",
    lexicon="librispeech_phn_lexicon.txt",
    bpe_code="librispeech_bpe4k.code",
    tgt_vocab_size=320,
    bucket_schedule='200,300,400,500,600,800,1000,1200,1400,1600,1800',
    bucket_schedule_val='5,100,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000,5000,6000,8000,10000',
    bucket_schedule_key="length",
    batch_means_tokens=1,
    use_old_bucket=True,
    max_batch_size=20000,
    prefetch_worker_num=4,
    next_retry=1000,
    global_shuffle=True,
    shuffle=True,
    drop_last=False,
    train_item_transform=[
        dict(type="PickleParser"),
        dict(type="WavConvert", in_key="wav"),
        dict(type="CalculateFrameLength"),
        dict(type='LabelLengthFilter', in_out_ratio=1, key='text'),
        dict(type="WavResample", sample_rate=16000, key="wav"),
        dict(type="W2vPhone2charLabel", preprocess=False),
    ],
    valid_item_transform=[
        dict(type="PickleParser"),
        dict(type="WavConvert", in_key="wav"),
        dict(type="CalculateFrameLength"),
        dict(type='LabelLengthFilter', in_out_ratio=1, key='text'),
        dict(type="WavResample", sample_rate=16000, key="wav"),
        dict(type="W2vPhone2charLabel", preprocess=False),
    ],
    batch_transform=[
        dict(type="WaveformCollate", wav_key="waveform"),
        dict(type="CharCollate", pad_eos=True),
        dict(type="ListCollate", key="utt_id"),
        dict(type="ListCollate", key="text"),
    ],
    inference_batch_transform=[
        dict(type="WaveformCollate", wav_key="waveform"),
        dict(type="CharCollate", pad_eos=True),
        dict(type="ListCollate", key="utt_id"),
        dict(type="ListCollate", key="text"),
    ],
)
solution = dict(
    model_type="BaseData2vecCtcModel",
    wav2vec_type='PretrainedData2Vec',
    decoder_type='Wav2vecDecoder',
    ctc_type='Wav2vecCtc',

    activation_checkpoint=False,
    # wav2vec_load_pretrained_model=False,
    wav2vec_finetuning=True,
    wav2vec_feature_conv_type="default",
    wav2vec_use_fbank=False,
    fbank_conv_embed_dim=512,
    # mode for feature extractor. default has a single group norm with
    # d groups in the first conv block, whereas layer_norm has layer
    # norms in every block (meant to use with normalize)
    extractor_mode="layer_norm",
    normalize=True,  # normalize input feature with layernorm
    # encoder
    pos_conv_depth=5,
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn="gelu",
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0,
    final_dim=256,  # 0 means the encoder_embed_dim
    layer_norm_first=False,  # apply layernorm first in the transformer
    encoder_layerdrop=0.05,  # changed
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
    mask_dropout=0,
    mask_length=10,  # mask length
    mask_prob=0.65,  # probability of replacing a token with mask changed
    mask_selection="static",  # how to choose masks
    mask_other=0.0,  # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_minlen_type="random",  # how to choose minlen masks
    no_mask_overlap=False,  # whether to allow masks to overlap
    mask_min_space=1,  # min space between spans (if no overlap is enabled)
    # mask channel
    mask_channel_length=64,  # repeat the mask indices multiple times
    mask_channel_prob=0.5,  # probability of replacing a token with mask
    mask_channel_selection="static",  # how to choose masks
    mask_channel_other=0.0,  # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
    mask_channel_minlen_type="random",  # how to choose minlen masks
    no_mask_channel_overlap=False,  # whether to allow masks to overlap
    mask_channel_min_space=1,  # min space between spans (if no overlap is enabled)
    mask_channel_before=False,
    final_proj_num_layer=1,
    weighted_data2vec=False,
    require_same_masks=True,
    pad_to_multiple=True,
    required_seq_len_multiple=2,
    # dropout
    dropout_input=0.1,  # dropout to apply to the input (after feat extr)
    dropout_features=0.0,  # dropout to apply to the features (after feat extr)
    final_dropout=0,  # dropout after transformer and before final projection
    num_negatives=100,  # number of negative examples
    negatives_from_everywhere=False,  # sample negatives from everywhere, not just masked states
    cross_sample_negatives=0,  # num of cross sampled negatives
    codebook_negatives=0,  # num of codebook sampled negatives
    conv_pos=95,  # number of filters for convolutional positional embeddings
    conv_pos_groups=16,  # number of groups for convolutional positional embedding
    latent_temp="(2,0.5,0.999995)",  # temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)
    target_glu=False,  # adds projection + glu to targets
    conv_bias=False,  # include bias in conv encoder
    data2vec_feature_conv_type="default",
    # loss
    loss_weights="[0.1, 10]",  # weights for additional loss terms (not first one)
    infonce=True,  # if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)
    apply_mask=True,
    freeze_finetune_updates=0,  # dont finetune wav2vec for this many updates
    freeze_encoder_layers=-1,  # freeze bottom # of encoder layers
    # ctc
    ctc_use_lm=False,
    zero_infinity=True,
    sentence_avg=True,
    ctc_loss_reduction="sum",
    cer_update_freq=100,
    modeling_unit_type="bpe",
    # ema
    ema_anneal_end_step=30000,
    ema_decay=0.999,
    ema_end_decay=0.9999,
    ema_layers_only=True,
    ema_transformer_only=True,
    average_top_k_layers=8,
    loss_beta=0,
    loss_scale=-1,
    data2vec_finetuning=1,
    # decode
    w2l_decoder="greedy",
    lang_formator="zh",
    nbest=1,
    beam=50,
    beam_size_token=50,
    beam_threshold=50,
    sil_weight=0,
    kenlm=None,
    lm_lexicon=None,
    word_score=-1,
    lm_weight=2,
    wfst_graph="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/lmdb_data/librispeech/wfst_graph/phone",
    wfst_decoder="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/tools/decoder/latgen.ken",
)
work_dir = "./w2v_ctc"
train = dict(
    # resume='step_10000.pth',
    resume_optimizer=False,
    resume_progress=False,
    resume_lr_scheduler=False,
    # grad_accum_step=2,
    max_epochs=40,
    max_iters=4000,
    lr_scheduler=dict(
        by_epoch=False,
        policy="TriStage",
        lr=2e-5,
        phase_ratio="[0.1,0.4,0.5]",
        init_lr_scale=0.01,
        final_lr_scale=0.05,
        max_train_steps=4000,
    ),
    checkpoint_config=dict(interval=2000, max_keep_ckpts=10),
    save_root="./d2v_ctc/",
    save_dir="data2vec2_base",
    save_name="base_10kh_lr5e-4_poly400k_mask0.65_len5_cat11.25s_bsz1.6h_v2_debug",
    resume_pretrain_chkpt='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liyong/dolphin/data2vec/lyz_baseline/pretrain/checkpoint_last.pth',
    freeze_feature_extractor=False,
    compare_metric="UF",
    validation_log=True,
    amp_level="O1",
)
valid = dict(interval=100)
inference = dict(
    test_sets="shard.dev_clean###shard.dev_other###shard.test_clean###shard.test_other",
    search_params=True,
    am_scale=0.75,
    blk_scale=0.15,
)
optimizer = dict(
    type="FusedAdam", lr=2e-5, betas=(0.9, 0.98), eps=1e-08, weight_decay=1e-2
)
optimizer_config = dict(
    grad_clip=dict(max_norm=100, norm_type=2),
    max_grad_clip=0.0,
    bmuf_config=0,
)
log_config = dict(interval=100)
