# dataset settings
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data/english_23kh/',
    # add ce sy+ctc label
    train_file_list = '["fbank_sub{}".format(i) for i in range(256)]',
    valid_file_list = '["fbank_80dim_cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=20480,
    shuffle=True,
    drop_last=False,
)