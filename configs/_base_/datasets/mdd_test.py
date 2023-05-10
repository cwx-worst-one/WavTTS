# dataset settings
data = dict(        
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guyiwei/dolphin/data/train_eng_ey_350w_ext70w_ext30w_5432_39dim/',
    train_file_list = '["sub_{}".format(i) for i in range(40)]',
    valid_file_list = '["cv"]',
    meta_file="meta",
    fbank_dim=39,
    chunk_size=20,
    use_lid=False,
    fetch_block_size=200,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_val='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,100000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=40960,
    #max_batch_scale=10,
    shuffle=True,
    drop_last=False,
)