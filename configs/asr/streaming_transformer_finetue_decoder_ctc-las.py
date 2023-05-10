project = "streaming_transformer_LAS_finetune"
runner = 'BaseLASRunner'
_base_ = [
    './datasets/dy1wh.py',
]

# increment to add and override args.
data = dict(
    chunk_size=20,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        # dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=True, skip_list=['<pad>', '<blank>']),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='CMVN', key='fbank'),
        # spec Aug
        dict(type='DynamicTimeMask', time_mask_size=20, time_mask_block=60,
             replace_with_zero=False, inplace=True, time_mask_num_dither=1),
        dict(type='FreqMask', freq_mask_size=27, freq_mask_num=1,
            replace_with_zero=False, inplace=True),
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='WavParser'),
        dict(type='WavResample', sample_rate=16000),
        # dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=True, skip_list=['<pad>', '<blank>']),
        dict(type='KaldiFbank', dither=0.0),
        dict(type='CMVN', key='fbank'),
    ],
    batch_transform=[
        dict(type='ListCollate'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate', pad_eos=True, las_format=True),
        dict(type='PreCharCollate', rnnt_format=False, las_format=True),
    ],
    inference_batch_transform=[
        dict(type='ListCollate'),
        dict(type='RefLabelCollate'),
        dict(type='FbankCollate', fbank_dim=80),
    ],
)

solution = dict(
    type='base_las_solution',
    las_type='StreamingLASModel',
    # init
    xavier_init=False,
    # downsampling (implementation same with samiasr)
    front_end_type='Conv2dPooling',
    front_end_conv0_ch=32,
    front_end_conv1_ch=64,
    front_end_padding=0,
    # backbone
    acoustic_backbone_type='TransformerBackbone',
    causal_transformer=True,
    chunk_mask=True,
    streaming_chunk_size=8,
    backbone_pos_embd=True,
    backbone_after_norm=True,
    backbone_layer_num=18,
    backbone_memory_size=768,
    backbone_hidden_size=2048,
    self_attn_heads=4,
    self_attn_dropout=0.0,
    dropout=0.1,
    positional_dropout_rate=0.1,
    self_attn_activation_dropout=0,
    self_attn_layer_norm_before=True,
    self_attn_activation_fn='gelu',
    backbone_act_glu=True,
    # decoder
    las_decoder_type='TransformerLASDecoder',
    streaming_decoder=True,
    decoder_chunk_size=8,
    decoder_left_chunk_num=4,
    decoder_right_peak=1,
    max_target_positions=600,
    embedding_size=768,
    decoder_embed_dim=768,
    decode_residue=True,
    apply_embed_scale=True,
    decoder_pos_embd_type='sinusoidal',
    decoder_layer_num=6,
    decoder_ffn_embed_dim=2048,
    decoder_attention_heads=4,
    attention_dropout=0,
    activation_dropout=0.1,
    decoder_dropout=0.1,
    decoder_normalize_before=True,
    # criterion
    criterion_type='CTCLasCE',
    cer_update_freq=50,
    label_smooth_factor=0.1,
    skip_ctc_eos=True,
    # mtl
    mtl_type='ctc',
    mtl_head='CTCHead',
    mtl_dropout=0.2,
    mtl_weight=0.1,
    # freeze params
    frontend_fix=True,
    backbone_fix=True,
    mtl_fix=True,
)

# runtime settings
work_dir = './demo'
# training and testing settings
train = dict(
    # resume_pretrain_chkpt='hdfs://haruna/home/byte_speech_sv/sami_sa_share/experiments/1_transformer_enc320ms/ckpts/avg1-2_dolphin.pth',
    resume_optimizer=False,
    resume_progress=False,
    grad_accum_step=4, # if > 1, need to divide iter numbers; set to 4 will bring NaN
    max_epochs=80,
    max_iters=300000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup=None,
        warmup_iters=5000,
        warmup_ratio=1e-8,
        warmup_by_epoch=False,
        policy='inv',
        gamma=1.0,
        power=0.5,
        offset=5000,
        scale=5000**0.5,
        ),
    checkpoint_config=dict(
        interval=1000, # steps
        max_keep_ckpts=-1,
    ),
    # remote_save_root='hdfs://haruna/home/byte_speech_sv/user/mingtu/workspace/ASR/dolphin_exps/',
    save_root='./demo',
    save_dir='streaming_transformer_las_finetune_decoder',
    save_name='CTC_LAS_lr5e-4_gradacc4',
    amp_level='O0',
    )
valid = dict(
    interval=10000, # iter
    )

inference = dict(
    bucket_schedule='',
    test_sets='data',
    las_beam_size=10,
    temperature=1.4,
    merge_asr_results=True,
    ctc_max_len=False, # use ctc num of peak to decide decoding steps;
    no_repeat_ngram_size=0,
    nnlm_path="",
    lm_weight=1.0,
    blk_scale=1.0,
    len_penalty_scale=0.01,
    language='zh',
    utt_key='utt_id',
    # remote_stat_dir="hdfs://haruna/xxxx",
    )

# optimizer
# torch optimizer ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
# Apex optimizer ['FusedAdagrad', 'FusedAdam', 'FusedLAMB', 'FusedNovoGrad', 'FusedSGD']
optimizer = dict(
    type='FusedAdam', # adam or fusedadam not related to inf
    lr=5e-4, # 2e-3 will bring NaN
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
    )
# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=20,
        norm_type=2,
        ),
    max_grad_clip=5.0,
    # bmuf_config=dict( # not related to inf
    #     block=200,
    #     block_lr=0.5,
    #     block_momentum=0.9,
    #     warmup_steps=5000,
    #     use_nesterov=True,
    #     average_sync=True,
    # )
 )

log_config = dict(
    interval=100,  # log at every 50 iterations
    )
