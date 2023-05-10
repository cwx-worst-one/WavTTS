"""
    TUTORIAL:

    To generate rir data, you need:

        1. configure a config file, see example `rir_gen_config.py`.

            general config, room config, mic config, save config... should be configured.

        2. Get config: (use dolphin core.utils.Config to parse config file).

            cfg = Config.fromfile('rir_gen_config.py')

        3. Select a generator and init it, e.g. generate directed rir data.

            generator = Generator(cfg)

        4. generate rir data.

            generator.generate()

    To add a new generation method, you need:

        1. Create a new generator inherited from `BaseGenerator`, e.g. DiffusedGenerator.

            class DiffusedGenerator(BaseGenerator):
                # ...

        2. rewrite `generate_rir_block` method.

            class DiffusedGenerator(BaseGenerator):

                def generate_rir_block(self, angle, block_idx):
                    # ...
                    # your generation method implemented here.
                    # howerver, rir_h and directions must returned by this method.
                    # see function `generate_rir_block` comments for more details.
                    return rir_h, directions

        3. if your generation method requires a new source build method, you should:

            1. Create a new source factory inherited from `BaseSourceFactory`, e.g. DiffusedSource.

                class DiffusedSource(BaseSourceFactory):
                    # ...

            2. rewrite `product_source` method.

                class DiffusedSource(Source):

                    def product_source(self, room, angle):
                        # ...
                        # your source build method implemented here.
                        # this method should return a source instance and mic instance.
                        # see function comments for more details.
                        return source, mic

        4. if your generation method requires a new disturb build method, you should:

            1. Create a new disturb list factory inherited from `BaseDisturbListFactory`,
               e.g. DisturbListFactory.

                class DisturbListFactory(BaseDisturbListFactory):
                    # ...

            2. rewrite `product_disturb_list` method.

                class DiffusedDisturb(Disturb):

                    def product_disturb_list(self, room, mic, source):
                        # ...
                        # your disturb list build method implemented here.
                        # this method should return a DisturbList instance.
                        # see function comments for more details.
                        return DisturbList

    To use gpurir, needs to install gpurir package according to
    https://github.com/DavidDiazGuerra/gpuRIR
"""
# pylint: disable = assignment-from-no-return
# pylint: disable = no-self-use
# pylint: disable=no-name-in-module
# pylint: disable=line-too-long
import os
import abc
import math
from math import pi
from multiprocessing import Pool
import pickle
import torch
from tqdm import tqdm
import numpy as np
from pyrirgen import generateRir  # pylint: disable = import-error
from dataloader import FalconWriter
from core.utils import Config


def rand():
    """rand"""
    return np.random.rand()


def uniform_sample(a, b):
    """uniform sample"""
    return a + (b - a) * rand()


def randn():
    """randn"""
    tmp = np.random.randn()
    return np.clip(tmp, -1, 1)


class Room:
    """Room item"""

    def __init__(self, length, width, height, r_min):
        '''init.'''
        self.length = length
        self.width = width
        self.height = height
        self.size = [length, width, height]
        self.r_min = r_min


class RoomFactory:
    """Room Factory"""

    def __init__(self, cfg):
        '''init.'''
        self.cfg = cfg

    def product_room(self, cfg=None):
        """produce a Romm according to config"""
        if cfg is None:
            cfg = self.cfg
        cfg = cfg.get('room', {})
        while True:
            length = uniform_sample(cfg.get('length_min', 3), cfg.get('length_max', 7))
            width = uniform_sample(cfg.get('width_min', 3), cfg.get('width_max', 5))
            height = uniform_sample(cfg.get('height_min', 2.5), cfg.get('height_max', 4.0))
            volume = length * width * height
            area = 2 * (length * width + width * height + length * height)
            r_min = 24 * volume * np.log(10) / area / cfg.get('sound_speed', 340)
            if r_min < (0.8 * cfg.get('rev_time_max', 0.8)):
                return Room(length, width, height, r_min)


class Mic:
    """Mic item"""

    def __init__(self, x_center, y_center, z_center, cord, phi, x_axis, y_axis):
        '''init.'''
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.center = np.array([x_center, y_center, z_center])
        self.cord = cord
        self.phi = phi
        self.x_axis = x_axis
        self.y_axis = y_axis


class MicFactory:
    """Mic factory"""

    def __init__(self, cfg):
        '''init.'''
        self.cfg = cfg

    def _circular_array_mapping(self, radius):
        """calculate circular array mapping"""
        assert radius is not None
        mic_cfg = self.cfg.get('mic', {})
        mic_num = mic_cfg.get('num', 2)
        res = []
        for i in range(mic_num):
            res.append(
                [
                    radius * np.cos(2 * np.pi * i / mic_num),
                    radius * np.sin(2 * np.pi * i / mic_num),
                    0,
                ]
            )
        return np.array(res)

    def _linear_array_mapping(self, mic_distance):
        """calculate linear array mapping"""
        assert mic_distance is not None
        mic_cfg = self.cfg.get('mic', {})
        mic_num = mic_cfg.get('num', 2)
        res = []
        for i in range(mic_num):
            if mic_num % 2 == 0:
                res.append([-mic_distance / 2 + i * mic_distance, 0, 0])
            else:
                res.append([-mic_distance * (mic_num // 2) + i * mic_distance, 0, 0])
        return np.array(res)

    def calculate_array_mapping(self):
        """calculate array mapping according to config"""
        gen_cfg = self.cfg.get('general', {})
        array_type = gen_cfg.get('array_type', 'linear')
        radius = gen_cfg.get('radius', 0.05)
        mic_distance = gen_cfg.get('mic_distance', 0.05)
        if array_type == "circular":
            return self._circular_array_mapping(radius)
        if array_type == "linear":
            return self._linear_array_mapping(mic_distance)
        raise Exception("array_type not supported.")

    def product_mic(self, room):
        """produce a Mic according to room and config"""
        mic_cfg = self.cfg.get('mic', {})
        space_delta = mic_cfg.get('space_delta', 0.2)
        x_center = uniform_sample(space_delta, room.length - space_delta)
        y_center = uniform_sample(space_delta, room.width - space_delta)
        z_center = uniform_sample(mic_cfg.get('height_min', 0.5), mic_cfg.get('height_max', 1.8))
        mic_center = np.array([x_center, y_center, z_center])

        phi = 2 * pi * rand()
        rotate = [
            [math.cos(phi), -math.sin(phi), 0],
            [math.sin(phi), math.cos(phi), 0],
            [0, 0, 1],
        ]
        rotate_z = np.array(rotate).T
        array_mapping = self.calculate_array_mapping()
        shift = array_mapping.dot(rotate_z)
        cord = np.array([mic_center + shift[i, :] for i in range(mic_cfg.get('num', 2))])
        x_axis_tmp = np.array([1, 0, 0])
        x_axis = x_axis_tmp.dot(rotate_z)
        y_axis_tmp = np.array([0, 1, 0])
        y_axis = y_axis_tmp.dot(rotate_z)
        return Mic(x_center, y_center, z_center, cord, phi, x_axis, y_axis)


class Source:
    """Source item"""

    def __init__(self, cord, dist_sou, direction, orientation):
        '''init.'''
        self.cord = cord
        self.dist_sou = dist_sou
        self.direction = direction
        self.orientation = orientation


class BaseSourceFactory(metaclass=abc.ABCMeta):
    """Base source factory"""

    @abc.abstractmethod
    def product_source(self, room, mic):
        """produce a Source item"""


class SourceFactory(BaseSourceFactory):
    """Source factory"""

    def __init__(self, cfg):
        '''init.'''
        self.cfg = cfg

    def calculate_orientation(self, mic):
        """calculate orientation.

        Args:
            mic: mic instance.

        Return:
            orientation according to given mic.
        """
        mic_cfg = self.cfg.get('mic', {})
        mic_type = mic_cfg.get('mtype', 'omnidirectional')
        mic_num = mic_cfg.get('num', 2)
        if mic_type == 'omnidirectional':
            return [0, 0]
        orientation = [[2 * np.pi * i / mic_num, 0] for i in range(mic_num)]
        for idx in range(mic_num):
            orientation[idx][0] = orientation[idx][0] + mic.phi
            if orientation[idx][0] > 2 * pi:
                orientation[idx][0] = orientation[idx][0] - 2 * pi
        return orientation

    def product_source(self, room, angle):
        """Produce a source according to given room.

        Note that in the process of generating source,
        a mic instance will be generated also.

        Args:
            room: a room instance.
            angle: the mid angle of source range [low, high].

        Return:
            source: a source instance.
            mic: a mic instance.
        """
        gen_cfg = self.cfg.get('general', {})
        src_cfg = self.cfg.get('source', {})
        source_delta = src_cfg.get('source_delta', 0.2)
        delta_min = src_cfg.get('delta_min', -0.3)
        delta_max = src_cfg.get('delta_max', 0.3)
        d_min = gen_cfg.get('d_min', 0.5)
        d_max = gen_cfg.get('d_max', 2.0)
        angle_low = angle - gen_cfg.get('angle_hwid', 5.0)
        angle_high = angle + gen_cfg.get('angle_hwid', 5.0)
        while True:
            mic = MicFactory(self.cfg).product_mic(room)
            source_x = uniform_sample(source_delta, room.length - source_delta)
            source_y = uniform_sample(source_delta, room.width - source_delta)
            source_z = mic.z_center + uniform_sample(delta_min, delta_max)
            cord = np.array([source_x, source_y, source_z])
            dist_sou = dist = np.linalg.norm(cord - mic.center)
            if d_min < dist < d_max:
                sou_vec = cord - mic.center
                x_mapping = (sou_vec * mic.x_axis).sum()
                y_mapping = (sou_vec * mic.y_axis).sum()
                direction = math.atan2(y_mapping, x_mapping) * 180 / pi
                if direction < 0:
                    direction += 360
                if angle_low < direction < angle_high:
                    orientation = self.calculate_orientation(mic)
                    return Source(cord, dist_sou, direction, orientation), mic


class Disturb:
    """Disturb item"""

    def __init__(self, cord, direction, rev_time_disturb):
        '''init.'''
        self.cord = cord
        self.direction = direction
        self.rev_time_disturb = rev_time_disturb


class DisturbList:
    """Disturb list item"""

    def __init__(self, cord_list, direction_list, rev_time_disturb_list, rev_time_source):
        '''init.'''
        self.cord_list = cord_list
        self.direction_list = direction_list
        self.rev_time_disturb_list = rev_time_disturb_list
        self.rev_time_source = rev_time_source


class BaseDisturbListFactory(metaclass=abc.ABCMeta):
    """Base disturb factory"""

    @abc.abstractmethod
    def product_disturb_list(self, room, mic, source):
        """produce a disturb according to room, mic, source ..."""


class DisturbListFactory(BaseDisturbListFactory):
    """Disturb factory"""

    def __init__(self, cfg):
        '''init.'''
        self.cfg = cfg

    def product_disturb(self, room, mic, source, rev_time_source):
        """produce a disturb instance.

        Args:
            room: room instance.
            mic: mic instance.
            source: source instance.
            rev_time_source: given rev_time_source.

        Return:
            a Disturb instance.
        """
        gen_cfg = self.cfg.get('general', {})
        src_cfg = self.cfg.get('source', {})
        source_delta = src_cfg.get('source_delta', 0.2)
        source_height_min = src_cfg.get('height_min', 0.5)
        source_height_max = src_cfg.get('height_max', 1.8)
        d_min = gen_cfg.get('d_min', 0.5)
        d_max = gen_cfg.get('d_max', 2.0)
        cord = None
        direction = None
        while True:
            disturb_x = uniform_sample(source_delta, room.length - source_delta)
            disturb_y = uniform_sample(source_delta, room.width - source_delta)
            disturb_z = uniform_sample(source_height_min, source_height_max)
            cord = np.array([disturb_x, disturb_y, disturb_z])
            dist = np.linalg.norm(cord - mic.center)
            if d_min < dist < d_max * 2.0:
                sou_vec = cord - mic.center
                x_mapping = (sou_vec * mic.x_axis).sum()
                y_mapping = (sou_vec * mic.y_axis).sum()
                direction = math.atan(y_mapping / x_mapping) * 180 / pi
                if direction < 0:
                    direction += 180
                if y_mapping < 0:
                    direction += 180
                break
        if source.dist_sou < dist:
            rev_time_disturb = rev_time_source + rev_time_source * (rand() / 50)
        else:
            rev_time_disturb = rev_time_source - rev_time_source * (rand() / 50)
            if rev_time_disturb < room.r_min:
                rev_time_disturb = rev_time_source
        return Disturb(cord, direction, rev_time_disturb)

    def product_disturb_list(self, room, mic, source):
        """produce disturb list.

        Produce a disturb list (may only has one disturb)
        according to given room, mic and source.

        Args:
            room: given room instance.
            mic: given mic instance.
            source: given source instance.

        Return:
            DisturbList instance.
        """
        room_cfg = self.cfg.get('room', {})
        src_cfg = self.cfg.get('source', {})
        max_source = src_cfg.get('max_sou', 2)
        rev_time_max = room_cfg.get('rev_time_max', 0.8)
        disturb_cord_list = []
        direction_list = []
        rev_time_disturb_list = []
        rev_time_source = uniform_sample(room.r_min, rev_time_max)
        for _ in range(1, max_source):
            disturb = self.product_disturb(room, mic, source, rev_time_source)
            disturb_cord_list.append(disturb.cord)
            direction_list.append(disturb.direction)
            rev_time_disturb_list.append(disturb.rev_time_disturb)
        return DisturbList(
            disturb_cord_list, direction_list, rev_time_disturb_list, rev_time_source
        )


class BaseGenerator:
    """Base generator"""

    def __init__(self, cfg):
        '''init.'''
        self.cfg = cfg.cfg_dict
        self.rir_type = self.cfg.get('mode', 'directed_rir')
        self.gen_cfg = self.cfg.get('general', {})
        self.pool_num = self.gen_cfg.get('pool_num', 0)
        self.block_num = self.gen_cfg.get('block_num', 1)
        self.gpu_num = self.cfg.get('gpu_num', 0)
        if self.gpu_num > 0:
            print("Use GPURIR")
            # time.sleep(10)
            self.system_gpu_num = torch.cuda.device_count()
            assert self.gpu_num > 0 and self.gpu_num <= self.system_gpu_num, (
                "While using gpurir, gpu_num must be positive and "
                f"no greater than than the system gpu number {self.system_gpu_num}"
            )
            self.use_gpu = True
            self.gpu_idx = self.cfg.get('gpu_idx_list', [])
            if len(self.gpu_idx) > 0:
                assert self.gpu_num == len(
                    self.gpu_idx
                ), "If provided, length of gpu_idx_list must be equal to gpu_num"
                assert (
                    max(self.gpu_idx) < self.system_gpu_num
                ), "max gpu_idx exceeds current available gpu system"
            else:
                self.gpu_idx = list(range(self.gpu_num))
        else:
            print("Use CPURIR")
            # time.sleep(10)
            self.use_gpu = False

    def set_random_seed(self, seed=None):
        """set random seed"""
        if seed is None:
            seed = self.gen_cfg.get('random_seed', 0)
        np.random.seed(seed)

    def save_rir_block(self, writer, h_rir, directions, block_idx):
        """save block using falcon writer"""
        keys = []
        vals = []
        for i in range(h_rir.shape[0]):
            val = {}
            val['direction'] = directions[i]
            val['rir'] = h_rir[i]
            vals.append(pickle.dumps(val))
            keys.append(f'block{block_idx}_id{i}')
            if len(keys) > 5000:
                writer.write_many(keys, vals)
                keys = []
                vals = []
        if len(keys) > 0:
            writer.write_many(keys, vals)

    def calculate_angle_list(self):
        """Calculate inital mid-angle list to generate rir"""
        gen_cfg = self.cfg.general
        angle_start = self.gen_cfg.get('angle_start', 5)
        angle_end = self.gen_cfg.get('angle_end', 185)
        angle_step = self.gen_cfg.get('angle_step', 10)
        angle_list = list(range(angle_start, angle_end, angle_step))
        for angle in gen_cfg.angle_removed:
            angle_list.remove(angle)
        return angle_list

    def generate_rir_block_cpu(self, angle, block_idx):  # pylint: disable = no-self-use
        """Generate rir data by block using cpu.

        Should implemented by specifical rir generator.

        Args:
            angle: mid-angle of rir expected.

        Return:
            h_rir: [rir_num, max_sou, mic_num, rir_length].
            directions: [rir_num, max_sou].
            (rir_num = rir_num_total // block_num)
        """
        raise NotImplementedError

    def generate_rir_block_gpu(self, angle, block_idx):  # pylint: disable = no-self-use
        """Generate rir data by block using gpu.

        Should implemented by specifical rir generator.

        Args:
            angle: mid-angle of rir expected.

        Return:
            h_rir: [rir_num, max_sou, mic_num, rir_length].
            directions: [rir_num, max_sou].
            (rir_num = rir_num_total // block_num)
        """
        raise NotImplementedError

    def generate(self):
        """Generator entry."""
        save_cfg = self.cfg.get('save', {})
        save_path = save_cfg.get('path')
        assert save_path is not None, "save path can't be None"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.set_random_seed()
        angle_list = self.calculate_angle_list()
        if self.use_gpu:
            gen_func = getattr(self, "generate_rir_block_gpu")
        else:
            gen_func = getattr(self, "generate_rir_block_cpu")
        # pylint:disable=consider-using-with
        for angle in angle_list:
            file_path = save_path + "h_rir_" + str(angle)
            writer = FalconWriter(file_path, 4 * 1024**3)
            if self.pool_num > 0:
                po = Pool(self.pool_num)
            res_list = []
            for block_idx in range(self.block_num):
                if self.pool_num > 0:
                    res = po.apply_async(
                        gen_func,
                        (
                            angle,
                            block_idx,
                        ),
                    )
                    res_list.append([res, block_idx])
                else:
                    h_rir, directions = gen_func(angle, block_idx)
                    self.save_rir_block(writer, h_rir, directions, block_idx)

            if self.pool_num > 0:
                po.close()
                po.join()
                for res in res_list:
                    h_rir, directions = res[0].get()
                    block_idx = res[1]
                    self.save_rir_block(writer, h_rir, directions, block_idx)

            writer.flush()


class Generator(BaseGenerator):
    """Directed rir generator"""

    def generate_rir_block_gpu(self, angle, block_idx):
        """Generate directed rir data by block using gpu.

        Args:
            angle: mid-angle of rir expected.

        Return:
            h_rir: [rir_num, max_sou, mic_num, rir_length].
            directions: [rir_num, max_sou].
            (rir_num = rir_num_total // block_num)
        """
        # pylint:disable=too-many-branches
        # pylint:disable=too-many-locals
        cur_pid = os.getpid()
        cur_gpu_idx = cur_pid % self.gpu_num
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_idx[cur_gpu_idx])
        import gpuRIR  # pylint: disable=redefined-outer-name, reimported, import-outside-toplevel, import-error

        self.set_random_seed(angle + block_idx)
        gen_cfg = self.cfg.get('general', {})
        src_cfg = self.cfg.get('source', {})
        mic_cfg = self.cfg.get('mic', {})
        room_cfg = self.cfg.get('room', {})
        fs = gen_cfg.get('fs', 16000)
        rir_num_total = gen_cfg.get('rir_num_total', 100)
        rir_length = gen_cfg.get('rir_length', 8192)
        mic_num = mic_cfg.get('num', 2)
        max_source = src_cfg.get('max_sou', 2)
        mic_type = mic_cfg.get('mtype', 'omni')
        tmax = rir_length / fs
        if mic_type == "ominidirectional":
            mic_type = "omni"
        elif mic_type == "cardioid":
            mic_type = "card"
        elif mic_type == "subcardioid":
            mic_type = "subcard"
        elif mic_type == "hypercardioid":
            mic_type = "hypcard"
        elif mic_type == "bidirectional":
            mic_type = "bidir"

        rir_num = int(rir_num_total / self.block_num)
        h_rir = np.zeros([rir_num, max_source, mic_num, rir_length])
        directions = np.zeros([rir_num, max_source])

        for rir_id in tqdm(range(rir_num)):
            room = RoomFactory(self.cfg).product_room()
            source, mic = SourceFactory(self.cfg).product_source(room, angle)
            disturb_list = DisturbListFactory(self.cfg).product_disturb_list(room, mic, source)
            nb_img = gpuRIR.t2n(tmax, room.size)
            source.orientation = np.array(source.orientation)
            if source.orientation.ndim == 1:
                if source.orientation[0] == 0 and source.orientation[1] == 0:
                    mic_orientation = None
                else:
                    mic_orientation = np.tile(mic_orientation, (2, 1))
            else:
                mic_orientation = np.stack(
                    [
                        np.cos(source.orientation[:, 1]) * np.cos(source.orientation[:, 0]),
                        np.cos(source.orientation[:, 1]) * np.sin(source.orientation[:, 0]),
                        np.sin(source.orientation[:, 1]),
                    ],
                    axis=-1,
                )
            beta_source = gpuRIR.beta_SabineEstimation(
                room.size, disturb_list.rev_time_source
            )  # Reflection coefficients
            beta_dist_list = []
            for idx in range(max_source - 1):
                beta_dist_list.append(
                    gpuRIR.beta_SabineEstimation(room.size, disturb_list.rev_time_disturb_list[idx])
                )  # Reflection coefficients

            dist_rir_list = []
            src_rir = gpuRIR.simulateRIR(
                room.size,
                beta_source,
                source.cord[None, :],
                mic.cord,
                nb_img,
                tmax,
                fs,
                mic_pattern=mic_type,
                orV_rcv=mic_orientation,
                c=room_cfg.get('sound_speed', 340),
            )
            src_rir = src_rir.squeeze(0)

            for idx in range(max_source - 1):
                rir = gpuRIR.simulateRIR(
                    room.size,
                    beta_dist_list[idx],
                    disturb_list.cord_list[idx][None, :],
                    mic.cord,
                    nb_img,
                    tmax,
                    fs,
                    mic_pattern=mic_type,
                    orV_rcv=mic_orientation,
                    c=room_cfg.get('sound_speed', 340),
                )
                dist_rir_list.append(rir.squeeze(0))

            h_rir[rir_id] = np.array([src_rir] + dist_rir_list).astype(np.float32)
            directions[rir_id, 0] = source.direction
            for i in range(1, max_source):
                directions[rir_id, i] = disturb_list.direction_list[i - 1]
        return h_rir, directions

    def generate_rir_block_cpu(self, angle, block_idx):
        """Generate directed rir data by block using cpu.

        Args:
            angle: mid-angle of rir expected.

        Return:
            h_rir: [rir_num, max_sou, mic_num, rir_length].
            directions: [rir_num, max_sou].
            (rir_num = rir_num_total // block_num)
        """

        self.set_random_seed(angle + block_idx)
        gen_cfg = self.cfg.get('general', {})
        src_cfg = self.cfg.get('source', {})
        mic_cfg = self.cfg.get('mic', {})
        room_cfg = self.cfg.get('room', {})
        rir_num_total = gen_cfg.get('rir_num_total', 100)
        rir_length = gen_cfg.get('rir_length', 8192)
        mic_num = mic_cfg.get('num', 2)
        max_source = src_cfg.get('max_sou', 2)
        array_type = gen_cfg.get('array_type', 'linear')
        mic_type = mic_cfg.get('mtype', 'omnidirectional')

        rir_num = int(rir_num_total / self.block_num)
        h_rir = np.zeros([rir_num, max_source, mic_num, rir_length])
        directions = np.zeros([rir_num, max_source])

        for rir_id in tqdm(range(rir_num)):
            room = RoomFactory(self.cfg).product_room()
            source, mic = SourceFactory(self.cfg).product_source(room, angle)
            disturb_list = DisturbListFactory(self.cfg).product_disturb_list(room, mic, source)

            dist_rir_list = []
            if array_type == "circular" and mic_type != "omnidirectional":
                src_rir = []
                for idx in range(mic_num):
                    rir = generateRir(
                        room.size,
                        source.cord.tolist(),
                        mic.cord.tolist()[idx],
                        fs=gen_cfg.get('fs', 16000),
                        reverbTime=disturb_list.rev_time_source,
                        orientation=source.orientation[idx],
                        isHighPassFilter=gen_cfg.get('hp_filter', 1),
                        nDim=room_cfg.get('dim', 3),
                        nOrder=gen_cfg.get('order', -1),
                        nSamples=rir_length,
                        micType=mic_type,
                    )
                    src_rir.append(rir)
                src_rir = np.stack(src_rir, axis=0)

                for src_idx in range(max_source - 1):
                    dst_rir = []
                    for idx in range(mic_num):
                        rir = generateRir(
                            room.size,
                            disturb_list.cord_list[src_idx].tolist(),
                            mic.cord.tolist(),
                            fs=gen_cfg.get('fs', 16000),
                            reverbTime=disturb_list.rev_time_disturb_list[src_idx],
                            orientation=source.orientation[idx],
                            isHighPassFilter=gen_cfg.get('hp_filter', 1),
                            nDim=room_cfg.get('dim', 3),
                            nOrder=gen_cfg.get('order', -1),
                            nSamples=rir_length,
                            micType=mic_type,
                        )
                        dst_rir.append(rir)
                    dst_rir = np.stack(dst_rir, axis=0)
                    dist_rir_list.append(dst_rir)
            else:
                src_rir = generateRir(
                    room.size,
                    source.cord.tolist(),
                    mic.cord.tolist(),
                    fs=gen_cfg.get('fs', 16000),
                    reverbTime=disturb_list.rev_time_source,
                    orientation=source.orientation,
                    isHighPassFilter=gen_cfg.get('hp_filter', 1),
                    nDim=room_cfg.get('dim', 3),
                    nOrder=gen_cfg.get('order', -1),
                    nSamples=rir_length,
                    micType=mic_type,
                )
                for src_idx in range(max_source - 1):
                    dst_rir = generateRir(
                        room.size,
                        disturb_list.cord_list[src_idx].tolist(),
                        mic.cord.tolist(),
                        fs=gen_cfg.get('fs', 16000),
                        reverbTime=disturb_list.rev_time_disturb_list[src_idx],
                        orientation=source.orientation,
                        isHighPassFilter=gen_cfg.get('hp_filter', 1),
                        nDim=room_cfg.get('dim', 3),
                        nOrder=gen_cfg.get('order', -1),
                        nSamples=rir_length,
                        micType=mic_type,
                    )
                    dist_rir_list.append(dst_rir)

            h_rir[rir_id] = np.array([src_rir] + dist_rir_list).astype(np.float32)
            directions[rir_id, 0] = source.direction
            for src_idx in range(1, max_source):
                directions[rir_id, src_idx] = disturb_list.direction_list[src_idx - 1]
        return h_rir, directions


if __name__ == "__main__":
    generator = Generator(
        Config.fromfile('/opt/tiger/workspace/code/dolphin/configs/se/rir_gen_config.py')
    )
    generator.generate()
