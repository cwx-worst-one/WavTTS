"""feature input processor"""
# pylint: disable=abstract-method,too-many-branches
import math
from absl import logging
from core.solutions.inference.penguin.core.processor.processor import Processor
from core.solutions.inference.penguin.core.register import Registers


@Registers.processor.register('feature_input')
class FeatureInput(Processor):
    """feature input processor"""

    def __init__(self, config):
        '''init'''
        super().__init__()
        self.first_ = True
        self.last_ = False
        self.padding_frame_ = []
        self.config = config
        self.buffer_ = []
        if self.config.hop_frames == 1:
            self.padding_num_ = self.config.stack_frames / 2
        elif self.config.padding_of_first_packet > 0:
            self.padding_num_ = self.config.padding_of_first_packet
        else:
            self.padding_num_ = 0
        self.zero_frame_ = [0.0] * self.config.fbank_dim
        resize(self.buffer_, self.padding_num_, self.zero_frame_)
        self.prepare_padding()
        self.use_domain_id_ = self.config.use_domain_id
        if self.use_domain_id_:
            domain_num = self.config.domain_num
            domain_id = self.config.domain_id
            self.domain_feature = [0] * domain_num
            self.domain_feature[domain_id] = 1

    def __call__(self, data):
        '''call'''
        return self.process(data)

    def add_domain_feature(self, data):
        '''add domain feat'''
        data_frames = len(data)
        for i in range(data_frames):
            data[i].extend(self.domain_feature)

    def get_one_segment_features(self, data, target_frames):
        '''get_one_segment_features'''
        buffered_frames = len(self.buffer_)
        data_frames = len(data)
        if buffered_frames + data_frames < target_frames + self.config.stack_frames - 1:
            if data_frames > 0:
                self.buffer_.extend(data)
                data.clear()
            return None

        stacked_feature = []
        for i in range(target_frames):
            src = []
            for j in range(self.config.stack_frames):
                if i + j < buffered_frames:
                    src.append(self.buffer_[i + j])
                else:
                    src.append(data[i + j - buffered_frames])
            stacked_feature.append(src)

        buffered_left = buffered_frames - target_frames * self.config.hop_frames
        if buffered_left > 0:
            self.buffer_ = self.buffer_[-buffered_left:]
        else:
            self.buffer_.clear()
            del data[:-buffered_left]
        return stacked_feature

    def pack_one_segment(self, featrue, frames, last, encoder_datas, valid_frames):
        '''pack_one_segment'''
        batch = 1
        sign = 0
        if self.first_ and last:
            sign = 3
        elif self.first_:
            sign = 1
        elif last:
            sign = 2
        else:
            sign = 0
        if self.config.hop_frames > 1:
            frames = self.config.stack_frames
        encoder_data = [featrue, sign, batch, frames, valid_frames]
        encoder_datas.append(encoder_data)

    def process_impl(self, ori_data):
        '''process_impl'''
        target_frames = 0
        encoder_datas = []
        frames = len(ori_data)
        # used only in cif
        valid_frames = self.config.hop_frames / self.config.down_sample

        if self.config.seg_frames >= 1000:
            if frames < self.config.down_sample:
                return encoder_datas
            self.pack_one_segment(ori_data, frames, True, encoder_datas, valid_frames)
            return encoder_datas

        while True:
            target_frames = self.config.first_seg_size if self.first_ else self.config.seg_frames
            featrue = self.get_one_segment_features(ori_data, target_frames)
            if featrue:
                self.pack_one_segment(featrue, target_frames, False, encoder_datas, valid_frames)
                if self.first_:
                    self.first_ = False
            else:
                break

        if self.first_ and len(self.buffer_) <= self.padding_num_:
            return encoder_datas

        last_valid_frames = len(self.buffer_) // self.config.down_sample
        if last_valid_frames > valid_frames:
            last_valid_frames -= valid_frames

        for _ in range(self.config.extra_frames):
            self.buffer_.append(self.padding_frame_)

        resize(self.buffer_, len(self.buffer_) + self.padding_num_, self.zero_frame_)

        if self.config.hop_frames > 1:
            resize(self.buffer_, len(self.buffer_) + self.config.hop_frames, self.zero_frame_)

        remain_len = len(self.buffer_)
        total_required_seg = (
            math.ceil(1.0 * remain_len / self.config.down_sample) * self.config.down_sample
        )

        if total_required_seg == 0:
            total_required_seg = self.config.down_sample

        if not self.config.tail_seg_padding:
            target_frames = min(target_frames, total_required_seg)

        if remain_len > target_frames:
            featrue = self.get_one_segment_features(ori_data, target_frames)
            if featrue:
                self.pack_one_segment(featrue, target_frames, False, encoder_datas, valid_frames)
                self.first_ = False
                if self.config.tail_seg_padding:
                    target_frames = self.config.seg_frames
                else:
                    target_frames = total_required_seg - target_frames

        size = target_frames + self.config.stack_frames - 1
        # TODO(mingzou): avoid padding last segment if left buffers are small.
        # if len(self.buffer_) <= self.padding_num_ * 2:
        resize(self.buffer_, size, self.zero_frame_)
        featrue = self.get_one_segment_features(ori_data, target_frames)
        self.pack_one_segment(featrue, target_frames, True, encoder_datas, last_valid_frames)
        return encoder_datas

    def process(self, data, **_kwargs):
        '''process'''
        self.reset()
        if self.use_domain_id_:
            self.add_domain_feature(data[1])
        res = self.process_impl(data[1])
        return data[0], res

    def prepare_padding(self):
        '''padding'''
        self.padding_frame_ = [self.config.padding_value] * self.config.fbank_dim
        if self.config.global_cmvn_file_path == "":
            return
        try:
            with open(self.config.global_cmvn_file_path, 'r') as f:
                line = f.readline()
                i = 0
                while line:
                    parts = line.split()
                    if len(parts) < 2:
                        logging.error("Wrong cmvn file: %s , please check!", line)
                    mean = float(parts[0])
                    var_re = float(parts[1])
                    self.padding_frame_[i] = (self.config.padding_value - mean) * var_re
                    i = i + 1
                    line = f.readline()
                while i < self.config.fbank_dim:
                    self.padding_frame_[i] = 0.0
                    i += 1
        except IOError:
            logging.error("cmvn file not exist: %s!", self.config.global_cmvn_file_path)

    def reset(self):
        '''reset'''
        self.first_ = True
        self.last_ = False
        self.buffer_ = []
        self.zero_frame_ = [0.0] * self.config.fbank_dim
        resize(self.buffer_, self.padding_num_, self.zero_frame_)
        self.prepare_padding()


def resize(buffer, size, data):
    '''resize data'''
    length = len(buffer)
    if length < size:
        for _ in range(int(size - length)):
            buffer.append(data)
    else:
        del buffer[size:]
