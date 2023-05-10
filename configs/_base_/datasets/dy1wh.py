# dataset settings
data = dict(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/dou' + \
            'yin_1wh',
    # add ce sy+ctc label
    train_file_list = '["sub_{}".format(i) for i in range(192)]',
    valid_file_list = '["sub_192"]',
    tgt_dict_dir='total_bpe.dict',
    bpe_code="total.code",
    cmvn_file="fbank_80dim_cmvn.kaldi",
    meta_data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/resources/zh_meta_data/',
    meta_file="meta",
    fbank_dim=80,
    chunk_size=20,
    bucket_schedule='50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000,2400,2800,3200,3600,10000',
    bucket_schedule_key='fbank',
    batch_means_tokens=1,
    max_batch_size=15360,
    shuffle=True,
    drop_last=False,
    test_data_root='hdfs://haruna/home/byte_speech_sv/sami_sa_share/datasets/dou' + \
            'yin_testing_data.human_cut/',
)
