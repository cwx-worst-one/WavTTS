# dataset settings
data = dict(        
    type = 'ag15kh_bpe',
    data_root = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/gaoxiao/data/asr/ag15kh_bpe',
    huge_hdfs=1,
    tgt_dict_dir='total_bpe.dict',
    hdfs_num=256,
    chunk_size=50,
    tgt_vocab_size=12968,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000',
    bucket_schedule_key='fbank',
    batch_means_tokens=True,
    max_batch_size=4096,
    valid_data_name='.cv',
    shuffle=True,
    drop_last=False,
    retry = 10, # retry times to wait hdfs command, each time cost 5s.
)
