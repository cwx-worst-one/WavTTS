# dataset settings
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/dolphin_datasets/general30k_v2/',
    train_file_list = '["shard{}".format(i) for i in range(208)]',
    valid_file_list = '["cv"]',
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=18432,
    shuffle=True,
    drop_last=False,
)
