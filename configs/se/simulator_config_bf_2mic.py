"simulator config for bf with 2mic"
# pylint: disable=line-too-long
general = dict(
    wave_key="waveform",
    random_seed=233,
    noise_disturb_and_flag=False, # 是否同时加干扰和扩散噪声
    array_type="linear",
    radius=0.03, #环阵生效，阵列半径
    mic_distance=0.027, #线阵生效，mic间距
    min_limit=1e-5,
    sampling_rate=16000,
    frame_size=160,
    agc_min=300,
    agc_max=5000,
    max_src_num=2,
    disturb_ratio=0.7,
    noise_ratio=0.6,
    colornoise_ratio=0.0,
    mic_num=2,  # rir_mic_num
    rir_num=18,
    top_db=40,# vad merge的参数
    use_outside_data='false', # 是否采用随机加载的干净语音（和asr的语音不同) 'false': complete decouple; 'partial': use waveform, if not enough in length, added by clean_data; 'true': use input waveform
    waveform_decouple_partial_axis=0,  # 在partial下指定的waveform通道, 可缺省，值为0
    max_speech_time=8,
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
    # dict(key="clean", path='/opt/tiger/workspace/dataset/TrainSet/clean_aishell_1ch_16k'),
    # dict(key="noise", path='/opt/tiger/workspace/dataset/TrainSet/DNSNoise_new_1ch_16k'),
    dict(key="clean", path='/mnt/bd/speech-se/clean_aishell_1ch_16k'),
    dict(key="disturb", path='/mnt/bd/speech-se/disturb/{}{}_500h_audio_1ch_16k'.format('dou', 'yin')),
    dict(key="noise", path='/mnt/bd/speech-se/noise/DNSNoise_new_1ch_16k'),
    dict(key="music", path='/mnt/bd/speech-se/music/music_douyin'),
    dict(key="colornoise", path='/mnt/bd/speech-falcon-lq/litianyu.y/data/se/colornoise_1.4_2.0_128000_50000')
]
rir_data = [
    dict(key="direct", prefix='rir_data', length=8192, unserialize_type='numpy', path='/mnt/bd/speech-se/rir_dolphin/list_split8192_pack_new'),
    dict(key="diffuse", prefix='rir_diffuse_data', length=2048, unserialize_type='numpy', path='/mnt/bd/speech-se/rir_dolphin/list_diffuse_split_pack_new'),
]  # 如果rir为dict序列化，length可缺省

rir = dict(
    angle_min=0,
    angle_max=180,
    angle_start=5,
    angle_end=185,
    angle_step=10,
)

clean = dict(
    src_num = [
        dict(prob=0.6, val=1),
        dict(prob=1.1, val=2),  # 第二个clean data用于模拟方向扰动
    ],
    src_condition=[
        dict(
            path_condition=[
                dict(prob=1.1, key='clean'),
            ],
            use_vad_merge=False,  # 可缺省，值为False
            length=None,  # 可缺省，None为使用general里的max_speech_time
            in_dtype='np.int16',  # 可缺省，值为np.int16
            out_dtype='np.float32',  # 可缺省，值为np.float32
        ),
        dict(
            path_condition=[
                dict(prob=1.1, key='clean'),
            ],
            use_vad_merge=False,  # 可缺省，值为False
            length='random.randint(128000//4, 128000*3//4)',  # 可缺省，None为使用general里的max_speech_time
            in_dtype='np.int16',  # 可缺省，值为np.int16
            out_dtype='np.float32',  # 可缺省，值为np.float32
        ),
    ],
    rir_condition=[
        dict(
            type='withoutref',  # withref: 依赖参考角度； withoutref: 不依赖参考角度
            path_condition=[
                dict(prob=1.1, key='direct')
            ],
            direction_cfg=[
                dict(prob=0.45, choices=[8, 9]),
                dict(prob=0.65, choices=[7, 10]),
                dict(prob=0.8, choices=[5, 6, 11, 12]),
                dict(prob=0.9, choices=[3, 4, 13, 14]),
                dict(prob=1.1, choices=[0, 1, 2, 15, 16, 17]),
            ],  
            angle_variance=0.4,  # 可缺省, 值为0, 只在'withoutref'下提供
            angle_diffuse_degree=15.0,  # 可缺省, 值为0, 只在'withoutref'下提供
        ),
        dict(
            type='withref',  # withref: 依赖参考角度； withoutref: 不依赖参考角度
            path_condition=[
                dict(prob=1.1, key='direct')
            ],
            direction_cfg=[
                dict(prob=1.1, low=0, high=1),  # low, high all included
            ],  
            max_angle2ref = 20.0,  # 离ref角度最小间隔角度, 可缺省，值为0, 只在'withref'下提供
        ),
    ]
)

disturb = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=0.1, key='disturb'),
            dict(prob=0.4, key='music'),
            dict(prob=1.1, key='clean'),
        ],
    ),
    rir_condition=dict(
        type='withref',
        path_condition=[
            dict(prob=1, key='direct')
        ],
        direction_cfg=[
            dict(prob=0.5, low=2, high=4),
            dict(prob=0.85, low=4, high=5),
            dict(prob=1.1, low=5, high=10000),
        ],
        min_angle2ref = 25.0,  # 离ref角度最小间隔角度, 可缺省，值为0, 只在'withref'下提供
    ),
    snr_min=-5,
    snr_max=15,
)

noise = [
    dict(prob=0.6, key='noise_diffuse_equ'),
    dict(prob=1.1, key='noise_diffuse_rir')
]

noise_diffuse_equ = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1.1, key='noise'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        path_condition=[
            dict(prob=1.1, key='diffuse_equation')
        ],
        direction_cfg=[],
    ),
    snr_min=0,
    snr_max=25,
)

noise_diffuse_rir = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1.1, key='disturb'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        type='withoutref',
        path_condition=[
            dict(prob=1.1, key='diffuse'),
        ],
        direction_cfg=[],
    ),
    snr_min=0,
    snr_max=25,
)

colornoise = dict(
    type="correlated",  # 'correlated': 使用direct rir生成；'uncorrelated': 直接生成多通道不相关
    beta_min=1.4,  # colornoise 参数, 可缺省，值为1.4
    beta_max=2.0,  # colornoise 参数, 可缺省，值为2.0
    rir_condition=dict(
        direction_cfg=[],
    ),
    snr_min=12,
    snr_max=22,
    in_dtype='np.float32',
    out_dtype='np.float32'
)


