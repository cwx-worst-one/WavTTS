project = "bert_pretrain"
runner = 'BertPretrainRunner'
data = dict(
    data_root = ['hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/nnlm_data/label_librispeech/', 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/lmdb_data/dolphin_data/nnlm_data/label_aesrc2020/'],
    train_file_list = ['["train_subshard{}".format(i) for i in range(4)]', '["train_subshard{}".format(i) for i in range(2)]'],
    valid_file_list = ['["subshard.aesrc_cv"]', '["subshard.cv"]'],
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    meta_file="meta",
    fetch_block_size=200,
    bucket_schedule='5,10,20,30,40,50,60,70,80,90,100,200',
    bucket_schedule_val='100000',
    bucket_schedule_key='char',
    batch_means_tokens=1,
    max_batch_size=15360,
    shuffle=True,
    drop_last=False,
    chunk_size=1000,
    use_lid=False,
    use_recombine=1,
    use_code_switch=1,
    use_eos=False,
    use_bpe=True,
    train_item_transform=[
        dict(type='PickleParser'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>']),
        # No TimeWarp, equal to time_warp_size = -1
    ],
    valid_item_transform=[
        dict(type='PickleParser'),
        dict(type='BPE', use_eos=False, key='text', out_key='char', skip_list=['<fil/>', '<spk/>', '<sta/>', '<non/>', '<nps/>']),
    ],
    batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='LMCharCollate'),
        dict(type='LMPreCharCollate'),
    ],
    inference_batch_transform=[
        dict(type='ListCollate', key='uttid'),
        dict(type='LMCharCollate'),
        dict(type='LMPreCharCollate'),
    ],
)
solution = dict(
    type='bert_pretrain_solution',
    model_type='BaseBertPretrainModel',
    bert_type='BertForPreTraining',
    # bert
    attention_probs_dropout_prob=0.1,
    directionality="bidi",
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=0,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-12,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    output_past=True,
    pooler_fc_size=768,
    pooler_num_attention_heads=12,
    pooler_num_fc_layers=3,
    pooler_size_per_head=128,
    pooler_type="first_token_transform",
    type_vocab_size=2,
    bert_vocab_size=21128,
    is_decoder=False,
    chunk_size_feed_forward=0,
    add_cross_attention=False,
    use_return_dict=True,
    output_attentions=False,
)
# runtime settings
work_dir = './bert_pretrain'
# training and testing settings
train = dict(
    resume='latest.pth',  # local checkpoint file to resume.
    resume_optimizer=True,
    max_epochs=100,
    max_iters=800000,
    lr_scheduler=dict(
        by_epoch=False,
        warmup='linear',
        warmup_iters=1,
        warmup_ratio=1e-3,
        warmup_by_epoch=False,
        policy='valid',
        gamma=0.4,
        key='loss',
    ),
    checkpoint_config=dict(
        interval=12850,  # steps
        max_keep_ckpts=-1,
    ),
    curriculum_end_epoch=0,
    curriculum_length=520,
    # remote_save_root='hdfs://haruna/xxx'
    save_root='./bert_pretrain',
    save_dir='bert_test',
    save_name='bert_pretrain_base_test',
    amp_level='O0',
)
valid = dict(
    interval=12850,  # iter
)
optimizer = dict(
    type='Adam',
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
)
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
        average_params=True,
    ),
)
log_config = dict(
    interval=100,  # log at every 100 iterations
    reset_flag=True,
)
