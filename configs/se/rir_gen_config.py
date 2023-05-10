"""rir generator config"""
# pylint: disable=invalid-name
gpu_num = 1  # if zero, use cpurir, else use gpurir, default to zero
gpu_idx_list = [7]  # no need to provide if gpu_num is zero

general = dict(
    rir_num_total=100,  # total rir number
    rir_length=2048,  # rir length
    block_num=2,  # block num
    array_type="circular",  # calculate array mapping
    mic_distance=0.03,  # calculate array mapping
    radius=0.035,  # calculate array mapping
    d_min=0.5,  # min distance between the source/microphone and room boundary
    d_max=2.0,  # max distance between the source/microphone and room boundary
    fs=16000,  # fs
    order=-1,  # -1 equals maximum reflection order!
    orientation=[0, 0],  # Microphone orientation (rad)
    hp_filter=1,  # enable highpass filter, not supported in gpurir
    random_seed=777,  # random seed
    start_seed=0,  # start seed
    pool_num=0,  # num of pool
    angle_removed=[],  # angles been removed from angle list
    angle_start=5,  # inital angle start, angle = list(range(5, 185, 10))
    angle_end=365,  # inital angle end, angle = list(range(5, 185, 10))
    angle_step=10,  # inital angle interval, angle = list(range(5, 185, 10))
    index_range=160,  # index range
    angle_hwid=5.0,  # angle scope = [angle_mid - angle_hwid, angle_mid + angle_hwid]
    attenuation1=0.1,  # uniform_sample(0.1, 0.4)
    attenuation2=0.1,  # uniform_sample(0.1, 0.4)
    decay_mode1=10,  # np.random.randint(low=10, high=11)
    decay_mode2=10,  # np.random.randint(low=10, high=11)
)

# Decide to generate directed rir or diffused rir ("diffused_rir", "diffused_rir")
mode = "directed_rir"

room = dict(
    sound_speed=340,  # sound speed (m/s)
    dim=3,  # room dimention
    length_min=3,  # min room length
    length_max=7,  # max room length
    width_min=3,  # min room width
    width_max=5,  # max room width
    height_min=2.5,  # min room height
    height_max=4.0,  # max room height
    rev_time_min=0.05,  # max value of the room rt60
    rev_time_max=0.8,  # min value of the room rt60
)

source = dict(
    max_sou=2,  # source number
    source_delta=0.2,  # ?
    height_min=0.5,  # min source height
    height_max=1.8,  # max source height
    delta_min=-0.3,
    delta_max=0.3,
)

worker = dict(
    num=5,  # worker num
    index=1,  # worker index
)

mic = dict(
    num=6,  # microphone num
    space=0.055,  # ?
    space_delta=0.2,  # minimum value between the source/microphon and room boundary
    height_min=0.5,  # min microphone height
    height_max=1.8,  # max microphone height
    mtype="cardioid",
    # Type of microphone for cpu (omnidirectional/cardioid/subcardioid/hypercardioid/bidirectional)
    # Type of microphone for gpu ("omni", "homni", "card", "subcard", "hypcard", "bidir")
)

save = dict(
    path='/opt/tiger/workspace/dummy_rir_6mic/circular_6mic_',  # HDFS or local dir to store output
)
