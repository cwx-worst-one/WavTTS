"""penguin audio encoder processor"""
# pylint: disable=too-many-branches
import numpy as np
from core.solutions.inference.penguin.core.processor.message import Message
from core.solutions.inference.penguin.core.factory import Factory
from core.solutions.inference.penguin.core.register import Registers
from core.solutions.inference.penguin.core.processor.processor import Processor


@Registers.processor.register('encoder')
class Encoder(Processor):
    """Audio Encoder Processor"""

    def __init__(self, config, encoder_name='encoder'):
        '''init'''
        super().__init__()
        self.config = config
        self.inference = Factory.get_inference(encoder_name, config)
        self.use_las_rescore = bool(config.use_las_rescore)
        self.use_ce_timestamp_conf = bool(config.use_ce_timestamp_conf)
        self.use_las_g2p = bool(config.use_las_g2p)
        self.use_cascaded_encoders = bool(config.use_cascaded_encoders)
        self.dense_cif_units = config.get('dense_cif_units', 0)
        self.cif_threshold = config.get('cif_threshold', 1.0)
        self.encoder_type = config.encoder_type
        self.streaming = config.seg_frames < 1000

    def __call__(self, data):
        '''call'''
        return self.process(data)

    def get_result(self, message):
        '''get_result'''
        input_name = self.inference.get_input()
        input_data = []
        for name in input_name:
            input_data.append(message[name])
        encoder_embedding = self.inference.run(input_data=input_data)
        return encoder_embedding

    def process(self, data):
        '''process'''
        cif_encoder = False
        if "cif" in self.encoder_type:
            cif_encoder = True
            ctc_outputs = []
            cif_decoder_in = []
            accum = 0.0
            state = np.zeros((1, 1, self.dense_cif_units)).astype(np.float32)
            dims = []
            dims.append(state.shape[0])
            dims.append(state.shape[2])

        wav_name = data[0]
        encoder_datas = data[1]
        encoder_input = self.get_input_all()
        if len(encoder_input) != 1:
            global_state_in = np.zeros((1, encoder_input[1].shape[-1])).astype(np.float32)
        decoder_in = []
        if self.use_las_rescore or self.use_las_g2p or self.use_cascaded_encoders:
            assert (
                len(self.get_output()) == 3
            ), 'encoder must produce backbone_out for las rescore or las g2p or cascaded encoders'
            twopass_encoder_in = []
        ce_encoder_in = []
        for encoder_data in encoder_datas:
            feature = np.array(encoder_data[0]).astype(np.float32)
            sign = encoder_data[1]
            batch = encoder_data[2]
            frames = encoder_data[3]
            valid_frames = encoder_data[4]
            if cif_encoder and self.streaming:
                feature = feature.reshape((batch, frames, encoder_input[0].shape[-2], 1))
            else:
                feature = feature.reshape((batch, frames, encoder_input[0].shape[-1]))
            sign = np.array([sign]).astype(np.int32)
            encoder_msg = (
                Message(
                    {
                        encoder_input[0].name: feature,
                        encoder_input[1].name: global_state_in,
                    }
                )
                if len(encoder_input) != 1
                else Message({encoder_input[0].name: feature})
            )
            if len(encoder_input) == 3:
                encoder_msg.set(encoder_input[2].name, sign)
            result = self.get_result(encoder_msg)
            name = self.get_output()
            if len(encoder_input) != 1:
                if self.use_las_rescore or self.use_las_g2p or self.use_cascaded_encoders:
                    global_state_in = result[name[2]]
                    if self.use_cascaded_encoders:
                        twopass_encoder_in.append((result[name[1]], sign))
                    else:
                        twopass_encoder_in.append(result[name[1]])
                elif cif_encoder:
                    global_state_in = result[name[3]]
                else:
                    global_state_in = result[name[1]]
            decoder_in.append(result[name[0]])
            if self.use_ce_timestamp_conf:
                if len(encoder_input) != 1:
                    global_state_in = result[name[2]]
                if cif_encoder:
                    ce_encoder_in.append(result[name[0]])
                else:
                    ce_encoder_in.append(result.backbone_out)
            if cif_encoder:
                cif_output = result[name[1]]
                encoder_out = np.split(result[name[0]], result[name[0]].shape[1], axis=1)
                sum_len = cif_output.shape[1]
                valid_frames = sum_len // 2 if self.streaming else sum_len

                item = []
                for i in range(valid_frames):
                    remain = self.cif_threshold - accum
                    if cif_output[0][i] + accum >= self.cif_threshold:
                        remain = 1.0 - accum
                        state = state + encoder_out[i] * remain
                        state = state.reshape(dims)
                        item.append(state)
                        accum = cif_output[0][i] - remain
                        state = encoder_out[i] * accum
                    else:
                        state = state + encoder_out[i] * cif_output[0][i]
                        accum = accum + cif_output[0][i]
                cif_decoder_in.append(item)
                if len(result) > 3:
                    ctc_outputs.append(result[name[2]])
        if cif_encoder:
            if accum > 0.5:
                state = state.reshape(dims)
                if len(cif_decoder_in) > 0:
                    cif_decoder_in[-1].append(state / accum)
                else:
                    cif_decoder_in.append([state / accum])
            return (wav_name, cif_decoder_in, ctc_outputs, ce_encoder_in)

        if self.use_las_rescore or self.use_las_g2p or self.use_cascaded_encoders:
            return (wav_name, decoder_in, twopass_encoder_in)
        if self.use_ce_timestamp_conf:
            return (wav_name, decoder_in, ce_encoder_in)
        return (wav_name, decoder_in)


@Registers.processor.register('encoder_channel2')
class EncoderChannel2(Encoder):
    """Audio Encoder Processor for dual channel"""

    def __init__(self, config):
        '''init'''
        super().__init__(config, 'encoder_channel2')


@Registers.processor.register('nc_encoder')
class NoncausalEncoder(Encoder):
    def __init__(self, config, encoder_name='nc_encoder'):
        '''init'''
        self.config = config
        super().__init__(config, encoder_name)
        self.nc_first_seg_size = config.nc_first_seg_size
        self.nc_seg_frames = config.nc_seg_frames

    def __call__(self, data):
        '''call'''
        return self.process(data)

    def process(self, data):
        '''process'''
        wav_name = data[0]
        encoder_datas = data[2]
        nc_decoder_in = []
        encoder_input = self.get_input_all()
        global_state_in = np.zeros((1, encoder_input[1].shape[-1])).astype(np.float32)

        first = True
        accum_time_len = 0
        while len(encoder_datas) > 0:
            encoder_data = encoder_datas.pop(0)
            c_backbone_out = encoder_data[0]  # (batch, frames, dim)
            sign = encoder_data[1]
            if accum_time_len > 0:
                nc_backbone_in = np.hstack((next_nc_backbone_in, c_backbone_out))  # concat
            else:
                nc_backbone_in = c_backbone_out
            next_nc_backbone_in = np.zeros([0, 0])
            # get this seg's time length
            accum_time_len = nc_backbone_in.shape[1]
            if (first and len(encoder_datas) > 0 and accum_time_len < self.nc_first_seg_size) or (
                not first and len(encoder_datas) > 0 and accum_time_len < self.nc_seg_frames
            ):
                # wait for enough frames
                next_nc_backbone_in = nc_backbone_in
                continue
            elif first:
                if len(encoder_datas) == 0:
                    # no more frames, run as nonstreaming
                    sign = np.array([3]).astype(np.int32)
                elif accum_time_len >= self.nc_first_seg_size:
                    # has enough frames now, run as first
                    sign = np.array([1]).astype(np.int32)
                    # cut nc_backbone_in into nc_backbone_in and next_nc_backbone_in
                    nc_backbone_in, next_nc_backbone_in = np.split(
                        nc_backbone_in, np.array([self.nc_first_seg_size]), axis=1
                    )
                first = False
            else:
                if len(encoder_datas) == 0:
                    # no more frames, run as last
                    sign = np.array([2]).astype(np.int32)
                elif accum_time_len >= self.nc_seg_frames:
                    # has enough frames now, run as middle
                    sign = np.array([0]).astype(np.int32)
                    # cut nc_backbone_in into nc_backbone_in and next_nc_backbone_in
                    nc_backbone_in, next_nc_backbone_in = np.split(
                        nc_backbone_in, np.array([self.nc_seg_frames]), axis=1
                    )

            encoder_msg = Message(
                {
                    encoder_input[0].name: nc_backbone_in,
                    encoder_input[1].name: global_state_in,
                    encoder_input[2].name: sign,
                }
            )
            accum_time_len = next_nc_backbone_in.shape[1]
            result = self.get_result(encoder_msg)
            name = self.get_output()
            global_state_in = result[name[1]]
            nc_decoder_in.append(result[name[0]])

        return (wav_name, nc_decoder_in)
