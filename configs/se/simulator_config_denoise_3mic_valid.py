"simulator config for bf with 2mic"
# pylint: disable=line-too-long
general = dict(
    wave_key="waveform",
    random_seed=233,
    noise_disturb_and_flag=False, # 是否同时加干扰和扩散噪声
    min_limit=1e-5,
    sampling_rate=16000,
    frame_size=160,
    agc_min=1000,
    agc_max=25000,
    colornoise_ratio=0.6, # color noise add radio
    mic_num=6,  # rir_mic_num
    max_speech_time=3,
    top_db=40,# vad merge的参数
)

falcon_reader = dict(
    mem_shared=True, # 多进程是否使用共享内存以减少内存占用, 若为True, 则底层会共享数据表单信息, 可缺省, 默认值为True
    chunk_size=8, # 为加速数据读取采取批读取方式, 此代表单次read操作读取的单个数据切片的长度, 可缺省, 默认值为10
    parrallel_chunk_num=5, # 一次读取操作所读取的数据切片的数量, 可缺省, 默认值为16
    io_thread_num=3, # 多线程读取采用的线程数, 可缺省, 默认值为8
    fd_cache_size=1024, # 单进程内句柄缓存上限, 可缺省, 默认值为256
)

volume_rand = dict(
    amp_low=0.841,
    amp_high=1.189,
    variance=0.2,
) #每个通道之间能量扰动的配置

audio_data = [
    dict(key="clean", path='/mnt/bd/lgz-data/denoise_data/clean/DNSSpeech_aishell3_vocal_1ch_16k'),
    dict(key="norm_noise", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/norm_noise'),
    dict(key="clip_noise", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/clip_noise'),
]
rir_data = [
    dict(key="direct", prefix='circular_6mic_245mm_test', length=2048, unserialize_type='numpy_only', path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liguangzheng/data/rir/'),
]  # 如果rir为dict序列化，length可缺省

clean = dict(
    rir_condition=dict(
        path_condition=[
            dict(prob=1.1, key='direct')
        ],
        num_source=2, # 默认值2, 加载的rir的source数, rir shape: [num_source, mic_num, rir_length]
        norm_rir=True, # 默认值True, 是否对多source rir做normlization
    ),
    snr_min=-5,
    snr_max=10,
)

colornoise = dict()
