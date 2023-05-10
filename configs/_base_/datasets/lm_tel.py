# dataset settings
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/youyongbin/data/lm/bytebot/hdfs_data/',
    train_file_list='["shard{}".format(i) for i in range(128)]',
    valid_file_list='["shard_cv"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    meta_file="meta",
    chunk_size=30,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='5,10,20,30,40,50,60,70,80,90,100,200',
    bucket_schedule_val='100000',
    bucket_schedule_key='char',
    batch_means_tokens=1,
    max_batch_size=15360,
    shuffle=True,
    drop_last=False,
)
