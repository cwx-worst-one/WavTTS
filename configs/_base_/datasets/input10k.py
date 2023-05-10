# dataset settings
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tianyao/data/asr/input10k/dolphin/wav/',
    # add ce sy+ctc label
    train_file_list = '["train_sub{}".format(i) for i in range(1024)]',
    valid_file_list = '["cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=15360,
    shuffle=True,
    drop_last=False,
)
