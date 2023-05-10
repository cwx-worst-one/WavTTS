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
    colornoise_ratio=0.0, # color noise add radio
    mic_num=1,  # rir_mic_num
    max_speech_time=5,
    top_db=40,# vad merge的参数
    early_reverb_signal='use_clean', # early_reverb_signal 等于speech, 默认是全0 (use_zero)
)

falcon_reader = dict(
    mem_shared=True, # 多进程是否使用共享内存以减少内存占用, 若为True, 则底层会共享数据表单信息, 可缺省, 默认值为True
    chunk_size=8, # 为加速数据读取采取批读取方式, 此代表单次read操作读取的单个数据切片的长度, 可缺省, 默认值为10
    parrallel_chunk_num=16, # 一次读取操作所读取的数据切片的数量, 可缺省, 默认值为16
    io_thread_num=8, # 多线程读取采用的线程数, 可缺省, 默认值为8
    fd_cache_size=1024, # 单进程内句柄缓存上限, 可缺省, 默认值为256
)

volume_rand = dict(
    amp_low=0.841,
    amp_high=1.189,
    variance=0.2,
) #每个通道之间能量扰动的配置

audio_data = [
    dict(key="clean", path='/mnt/bd/lgz-dnsnoise-data/dnsnoise_data/indexed/DnsCleanData'),
    dict(key="norm_noise", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liguangzheng/data/dns_2021/noise/dns_noise'),
    dict(key="clip_noise", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/clip_noise'),
]
rir_data = [
    dict(key="direct", prefix='circular_6mic_35mm', length=8192, mic_num=6, unserialize_type='numpy_only', path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liguangzheng/data/rir/'),
]  # 如果rir为dict序列化，length可缺省, mic_num 默认为general_config中配置的mic num

clean = dict(
    src_condition=[
        dict(
            path_condition=[
                dict(prob=1.1, key='speech'),
            ],
            use_vad_merge=False,  # 可缺省，值为False
            length=None,  # 可缺省，None为使用general里的max_speech_time
            in_dtype='np.int16',  # 可缺省，值为np.int16
            out_dtype='np.float32',  # 可缺省，值为np.float32
            audio_repeat=True, # 可缺省，值为False, 是否通过重复音频覆盖最大长度
        ),   
    ],
    rir_condition=dict(), # no rir
    snr_min=-5,
    snr_max=10,
)

noise = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1.1, key='norm_noise'),
        ],
        use_vad_merge=False,
        length=None,  # 可缺省，None为使用general里的max_speech_time
        in_dtype='np.int16',  # 可缺省，值为np.int16
        out_dtype='np.float32',  # 可缺省，值为np.float32
        audio_repeat=True, # 可缺省，值为False, 是否通过重复音频覆盖最大长度
    ),
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

colornoise = dict(
    type="correlated",  # 'correlated': 使用direct rir生成；'uncorrelated': 直接生成多通道不相关
    beta_min=2.0,  # colornoise 参数, 可缺省，值为1.4
    beta_max=5.0,  # colornoise 参数, 可缺省，值为2.0
    rir_condition=dict(
        type='multi_source',  # withref: 依赖参考角度； withoutref: 不依赖参考角度; multi_source: 加载多声源的rir
        path_condition=[
            dict(prob=1.1, key='direct')
        ],
        num_source=2, # 默认值2, 加载的rir的source数, rir shape: [num_source, mic_num, rir_length]
        norm_rir=False, # 默认值True, 是否对多source rir做normlization
        select_target_source_rir=0, # 默认值None, 是否只加载特定source rir, 返回rir[select_target_source_rir], 即[mic_num, rir_len]
    ),
    colornoise_fn='powerlaw_psd_gaussian',
    fmin_min = 100,
    fmin_max = 1000,
    fs = 16000,
    snr_min=-5,
    snr_max=10,
    in_dtype='np.float32',
    out_dtype='np.float32'
)