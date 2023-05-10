"simulator config for aec training"
# pylint: disable=line-too-long
general = dict(
    wave_key="waveform",
    random_seed=233,
    noise_disturb_and_flag=False, # 是否同时加干扰和扩散噪声
    min_limit=1e-5,
    sampling_rate=16000,
    frame_size=160,
    agc_min=1000,
    agc_max=32767,
    colornoise_ratio=0.6, # color noise add radio
    mic_num=1,  # rir_mic_num
    max_speech_time=8,
    top_db=40,# vad merge的参数,
    speech_aug_ratio=0.1,
    clip_ratio=0.3,
    pitch_ratio=0.0,
    eq_ratio=0.05,
    echo_ratio=0.8,
    noise_ratio=0.8,
    speech_ratio=0.8,
    max_delay=20,
    delay_perturb_ratio=0.2,
    real_echo_ratio=0.3,
    color_noise_ratio=1.0,
    ref_clip_ratio=0.2,
    return_tde=0,
    max_target_speech_length=8,
    min_target_speech_length=8,
    ser_min=-10,
    ser_max=10,
    inference_linear_aec_flag=False, # 解码数据是否过线性AEC, 默认值False
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
    dict(key="speech", path='/mnt/bd/lgz-data-aec/dataset/DNSSpeech_1ch_16k'),
    dict(key="noise", path='/mnt/bd/lgz-data-aec/dataset/DNSNoise_1ch_16k'),
    dict(key='fake_echo', path='/mnt/bd/lgz-data-aec/dataset/librispeech_train_clean_460_16k_1ch'),
    dict(key='fake_echo_opt', path='/mnt/bd/lgz-data-aec/dataset/{}{}_500h_audio_1ch_16k'.format('dou', 'yin')),
    dict(key='ref_echo', path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/se/aec/data/ref_echo',)
]

rir_data = [
    dict(key="direct", prefix='rir_near', length=2048, unserialize_type='numpy_only',path='/mnt/bd/lgz-data-aec/dataset/'),
]  # 如果rir为dict序列化，length可缺省

clean = dict(
    src_condition=[
        dict(
            path_condition=[
                dict(prob=1.1, key='speech'),
            ],
            use_vad_merge=False,  # 可缺省，值为False
            length=-1,  # 可缺省，None为使用general里的max_speech_time, -1则使用加载的单条数据的长度
            in_dtype='np.int16',  # 可缺省，值为np.int16
            out_dtype='np.float32',  # 可缺省，值为np.float32
        ),
    ],
    rir_condition=dict(
        type='clean', # rir only
        path_condition=[
            dict(prob=0.5, key='direct')
        ],
        num_source=None, # 默认值2, 加载的rir的source数, rir shape: [num_source, mic_num, rir_length]
        norm_rir=False, # 默认值True, 是否对多source rir做normlization
    ),
)

base_noise = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1.1, key='noise'),
        ],
        use_vad_merge=False,
        length=-1,  # 可缺省，None为使用general里的max_speech_time
        in_dtype='np.int16',  # 可缺省，值为np.int16
        out_dtype='np.float32',  # 可缺省，值为np.float32
    ),
    rir_condition=dict( # norir
        path_condition=[],
    ),
    snr_min=15,
    snr_max=30,
)

colornoise = dict(
    type="norir",  # 'correlated': 使用direct rir生成；'uncorrelated': 直接生成多通道不相关
    beta_min=2.0,  # colornoise 参数, 可缺省，值为1.4
    beta_max=5.0,  # colornoise 参数, 可缺省，值为2.0
    colornoise_fn='powerlaw_psd_gaussian',
    max_iter_num=3, # 生成colornoise时使用的wnoise的条数，仅在powerlaw_psd_gaussian下生效
    fmin_min = 100,
    fmin_max = 1000,
    fs = 16000,
    snr_min=15,
    snr_max=30,
    in_dtype='np.float32',
    out_dtype='np.float32',
)

noise = [
    dict(prob=0.0, key='base_noise'),
    dict(prob=1.1, key='colornoise')
]
