"simulator config"
# pylint: disable=line-too-long
general = dict(
    wave_key="waveform",
    random_seed=233,
    noise_disturb_and_flag=False, # 是否同时加干扰和扩散噪声
    array_type="circular",
    radius=0.03, #环阵生效，阵列半径
    mic_distance=0.05, #线阵生效，mic间距
    min_limit=1e-5,
    sampling_rate=16000,
    frame_size=160,
    agc_min=300,
    agc_max=30000,
    max_src_num=2,
    disturb_ratio=0,
    noise_ratio=1,
    mic_num=6,  # rir_mic_num
    rir_num=36,
    max_angle=36,
    top_db=40,# vad merge的参数
    waveform_decouple_type='pass',#是否采用随机加载的干净语音（和asr的语音不同) 'total': complete decouple; 'partial': use waveform, if not enough in length, added by clean_data; 'pass': use waveform regardless of length
    max_speech_time=4,
)

volume_rand = dict(
    amp_low=0.84,
    amp_high=1.19,
    variance=0.2,
) #每个通道之间能量扰动的配置

audio_data = [ 
    dict(key="clean", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanwenzhi/dns_dataset/DNSSpeech_1ch_16k'),
    dict(key="disturb", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanwenzhi/dns_dataset/DNSNoise_1ch_16k'),
    dict(key="noise", path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanwenzhi/dns_dataset/DNSNoise_1ch_16k'),
]
rir_data = [
    dict(key="direct", prefix='cicular_6mic_h_rir_', path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanwenzhi/dolphin_rir'),
    dict(key="diffuse", prefix='cicular_6mic_h_rir_', path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/fanwenzhi/dolphin_rir'),
]

rir = dict(
    angle_min=0,
    angle_max=360,
    angle_start=5,
    angle_end=365,
    angle_step=10,
)

clean = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1, key='clean'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        type='withoutref',
        path_condition=[
            dict(prob=1, key='direct')
        ],
        direction_cfg=[
            dict(prob=1.1, choices=list(range(36))),
        ],
        angle_variance=0.0,#没用到
        angle_diffuse_degree=0,#没用到
    )
)

disturb = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=0.1, key='disturb'),
            dict(prob=1, key='clean'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        type='withref',
        path_condition=[
            dict(prob=1, key='direct')
        ],
        direction_cfg=[
            dict(prob=1, low=2, high=36),
        ],
        min_angle2ref = 25.0,
    ),
    snr_min=0,
    snr_max=15,
)

noise = dict(
    src_condition=dict(
        path_condition=[
            dict(prob=1, key='noise'),
        ],
        use_vad_merge=False,
    ),
    rir_condition=dict(
        type='withoutref',
        path_condition=[
            dict(prob=0.5, key='diffuse'),
            dict(prob=1.0, key='direct')
        ],
        direction_cfg=[],
    ),
    snr_min=10,
    snr_max=25,
)


