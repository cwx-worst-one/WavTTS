# dataset settings
data = dict(
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/datasets/dolphin/librispeech_wav/',
    train_file_list = '["train_sub{}".format(i) for i in range(16)]',
    valid_file_list = '["dev_clean", "dev_other"]',
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    io_cache_size=2048, # cache_size used by FalconReader
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=12800,
    shuffle=True,
    drop_last=False,
)
