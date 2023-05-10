"simulator config for ssl with 2mic"
# pylint: disable=line-too-long
general = dict(
    wave_key="waveform",
    noise_disturb_and_flag=False, # 是否同时加干扰和扩散噪声
    array_type="linear",
    radius=0.03, #环阵生效，阵列半径
    mic_distance=0.027, #线阵生效，mic间距
    min_limit=1e-5,
    sampling_rate=16000,
    frame_size=160,
    agc_min=300,
    agc_max=6000,
    max_src_num=2,
    disturb_ratio=0.6,
    noise_ratio=0.85,
    colornoise_ratio=1.0,
    mic_num=2,  # rir_mic_num
    rir_num=18,
    top_db=40,# vad merge的参数
    use_outside_data='false', # 'false': complete decouple; 'partial': use waveform, if not enough in length, added by clean_data; 'true': use input waveform
    waveform_decouple_partial_axis=0,  # 在partial下指定的waveform通道, 可缺省，值为0
    max_speech_time=6,
)

falcon_reader = dict(
    mem_shared=True, # 多进程是否使用共享内存以减少内存占用, 若为True, 则底层会共享数据表单信息, 可缺省, 默认值为True
    chunk_size=8, # 为加速数据读取采取批读取方式, 此代表单次read操作读取的单个数据切片的长度, 可缺省, 默认值为10
    parrallel_chunk_num=5, # 一次读取操作所读取的数据切片的数量, 可缺省, 默认值为16
    io_thread_num=3, # 多线程读取采用的线程数, 可缺省, 默认值为8
    fd_cache_size=1024, # 单进程内句柄缓存上限, 可缺省, 默认值为256
)

volume_rand = dict(
    amp_low=0.8,
    amp_high=1.3,
    variance=0.2,
) #每个通道之间能量扰动的配置

audio_data = [ 
    # dict(key="clean", path='/opt/tiger/workspace/dataset/TrainSet/clean_aishell_1ch_16k'),
    # dict(key="noise", path='/opt/tiger/workspace/dataset/TrainSet/DNS_lark_music_noise'),
    dict(key="clean", path='/mnt/bd/speech-se/clean_aishell_1ch_16k'),
    dict(key="disturb", path='/mnt/bd/speech-se/disturb/{}{}_500h_audio_1ch_16k'.format('dou', 'yin')),
    dict(key="noise", path='/mnt/bd/speech-se/noise/DNS_lark_music_noise'),
    dict(key="music", path='/mnt/bd/speech-se/music/music_{}{}'.format('dou', 'yin')),
    dict(key="colornoise", path='/mnt/bd/speech-falcon-lq/litianyu.y/data/se/colornoise_1.4_2.0_128000_50000')
]
rir_data = [
    dict(key="direct", prefix='rir_data', length=8192, unserialize_type='numpy', path='/mnt/bd/speech-se/rir_dolphin/list_split8192_pack_new'),
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
        dict(prob=1.1, val=1)
    ],
    src_condition=dict(
        path_condition=[
            dict(prob=1.1, key='clean'),
        ],
        use_vad_merge=False,  # 可缺省，值为False
        length=None,  # 可缺省，None为使用general里的max_speech_time
        in_dtype='np.int16',  # 可缺省，值为np.int16
        out_dtype='np.float32',  # 可缺省，值为np.float32
    ),
    rir_condition=dict(
        type='withoutref',  # withref: 依赖参考角度； withoutref: 不依赖参考角度
        path_condition=[
            dict(prob=1.1, key='direct')
        ],
        direction_cfg=[],  # 空列表默认所有角度均匀采样
        angle_variance=0.0,  # 可缺省, 值为0, 只在'withoutref'下提供
        angle_diffuse_degree=0,  # 可缺省, 值为0, 只在'withoutref'下提供
    )
)

disturb = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=0.1, key='disturb'),
            dict(prob=1.1, key='clean'),
        ],
    ),
    rir_condition=dict(
        type='withref',
        path_condition=[
            dict(prob=1, key='direct')
        ],
        direction_cfg=[
            dict(prob=0.2, low=2, high=3),  # low, high all included
            dict(prob=0.4, low=4, high=5),  
            dict(prob=1.1, low=6, high=18),  
        ],
        min_angle2ref = 25.0,  # 离ref角度最小间隔角度, 可缺省，值为0, 只在'withref'下提供
    ),
    snr_min=-5,
    snr_max=15,
)

noise = [
    dict(prob=0.4, key='noise_diffuse_equ'),
    dict(prob=1.1, key='noise_direct_rir')
]

noise_diffuse_equ = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=0.5, key='noise'),
            dict(prob=0.8, key='music'),
            dict(prob=1.1, key='disturb'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        path_condition=[
            dict(prob=1.1, key='diffuse_equation')
        ],
        direction_cfg=[],
    ),
    snr_min=5,
    snr_max=20,
)

noise_direct_rir = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1.1, key='noise'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        type='withref',
        path_condition=[
            dict(prob=1.1, key='direct'),
        ],
        direction_cfg=[
            dict(prob=1.1, low=2, high=18),  # low, high all included
        ],
        min_angle2ref = 25.0,  # 离ref角度最小间隔角度, 可缺省，值为0, 只在'withref'下提供
    ),
    snr_min=5,
    snr_max=20,
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


