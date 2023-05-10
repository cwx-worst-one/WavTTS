'''
fbank feature process.
'''
import random
import numpy as np
import torch
import torchaudio
from core.extensions import PantherLogFbank
from core.extensions import dynamic_cmvn
from core.utils.wav_util import get_frames_len
from .kaldi_fbank import SpeedFbank
from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class SpeedKaldiFbank:
    '''compute kaldi fbank feature with torchaudio 1.7 speedup.

    support batch version and non batch version.
    for non batch version:
        the input waveform only support [channels, num_samples] shape
        if channel_num == 1, returns [num_frames, num_mel_bins]
        else if channel_num > 1, returns [channels, num_frames, num_mel_bins]
    for batch version:
        the input waveform only support
        [bsz, num_samples] or [bsz, channels, num_samples] shape
        if channel_num == 1, returns [bsz, num_frames, num_mel_bins]
        else if channel_num > 1, returns [bsz, channels, num_frames, num_mel_bins]
    '''

    def __init__(
        self,
        in_key='waveform',
        out_key='fbank',
        mask_key='src_mask',
        use_energy=False,
        dither=1.0,
        dynamic_dither=False,
        device='cpu',
        out_numpy=True,
        fbank_dim=None,
        low_freq=20.0,
        high_freq=0.0,
    ):
        '''init'''
        self.in_key = in_key
        self.out_key = out_key
        self.mask_key = mask_key
        self.use_energy = use_energy
        self.dither = dither
        self.dynamic_dither = dynamic_dither
        self.out_numpy = out_numpy
        self.device = device
        self.fbank_dim = fbank_dim
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fns = dict()

    def get_fn(self, num_mel_bins, sample_rate):
        '''get reused fn, key by num_mel_bins.'''
        if num_mel_bins not in self.fns:
            fn = SpeedFbank(
                self.device,
                num_mel_bins=num_mel_bins,
                sample_frequency=sample_rate,
                use_energy=self.use_energy,
                low_freq=self.low_freq,
                high_freq=self.high_freq,
            )
            self.fns[num_mel_bins] = fn
        return self.fns[num_mel_bins]

    def __call__(self, item, **_kwargs):
        '''call for fbank extraction'''
        # pylint:disable=too-many-branches
        if item is None or self.in_key not in item:
            return item
        waveform = item[self.in_key]
        channels = waveform.shape[-2]
        num_samples = waveform.shape[-1]
        # non batch
        if self.out_numpy:
            assert waveform.ndim == 2, (
                f"SpeedKaldiFbank is processing the waveform with shape {waveform.shape}"
                " which does not meet the requirements [channels, num_samples]"
            )
        else:  # batch
            assert waveform.ndim in (2, 3), {
                f"SpeedKaldiFbank is processing the batch waveform with shape {waveform.shape}"
                " which does not meet the requirements [batch_size, num_samples]"
                " or [batch_size, channels, num_samples]"
            }
            if waveform.ndim == 2:
                bsz, channels = channels, 1
            else:
                bsz, channels, _ = waveform.shape
                waveform = waveform.reshape(-1, num_samples)

        sample_rate = item.get('sample_rate', 16000)
        if self.fbank_dim is not None:
            num_mel_bins = self.fbank_dim
        else:
            if sample_rate == 8000:
                num_mel_bins = 60
            elif sample_rate == 16000:
                num_mel_bins = 80
            else:
                return None

        if self.dynamic_dither and self.dither > 0:
            dither = random.uniform(self.dither / 2.0, self.dither)
        else:
            dither = self.dither
        if isinstance(waveform, np.ndarray):
            waveform = torch.Tensor(waveform)
        waveform = waveform.to(self.device)

        # fn may resue.
        fn = self.get_fn(num_mel_bins, sample_rate)
        fbank = fn(waveform, dither=dither)

        if not self.out_numpy:  # batch
            new_shape = [bsz, channels] + list(fbank.shape[-2:])
            fbank = fbank.reshape(new_shape)
        if channels == 1:
            fbank = fbank.squeeze(-3)

        if self.out_numpy:
            fbank = fbank.numpy().astype(np.float32)
        elif len(fbank.shape) == 3 and self.mask_key in item:  # batch mode
            mask = item[self.mask_key]
            fbank *= mask.unsqueeze(-1)
        item[self.out_key] = fbank
        item['length'] = item[self.out_key].shape[0]
        return item


@PREPROCESS.register_module()
class KaldiFbank(SpeedKaldiFbank):
    '''compute kaldi fbank feature'''


@PREPROCESS.register_module()
class LogFbank:
    '''logfbank realize in python_speech_features'''

    def __init__(
        self,
        in_key='waveform',
        out_key='fbank',
        winlen=0.025,
        winstep=0.01,
        highfreq=None,
        lowfreq=0.0,
        preemph=0.97,
        nfft=512,
        window_type='hamming',
        fbank_dim=40,
    ):
        '''init'''
        self.in_key = in_key
        self.out_key = out_key
        self.winlen = winlen
        self.winstep = winstep
        self.highfreq = highfreq
        self.lowfreq = lowfreq
        self.preemph = preemph
        self.nfft = nfft
        self.window_type = window_type
        self.fbank_dim = fbank_dim
        self.fns = dict()

    def get_fn(self, num_mel_bins, sample_rate):
        '''get reused fn, key by num_mel_bins.'''
        if num_mel_bins not in self.fns:
            fn = PantherLogFbank(
                winlen=self.winlen,
                winstep=self.winstep,
                highfreq=self.highfreq,
                lowfreq=self.lowfreq,
                preemph=self.preemph,
                samplerate=sample_rate,
                nfilt=num_mel_bins,
                nfft=self.nfft,
                window_type=self.window_type,
            )
            self.fns[num_mel_bins] = fn
        return self.fns[num_mel_bins]

    def __call__(self, item, **_kwargs):
        '''call for logfbank extraction'''
        if item is None or self.in_key not in item:
            return item
        waveform = item[self.in_key]
        sample_rate = item.get('sample_rate', 16000)
        if self.fbank_dim is not None:
            num_mel_bins = self.fbank_dim
        else:
            num_mel_bins = 26
            return None
        fn = self.get_fn(num_mel_bins, sample_rate)
        fbank = fn(torch.Tensor(waveform))  # TODO
        item[self.out_key] = fbank.numpy()
        item['length'] = item[self.out_key].shape[0]
        return item


@PREPROCESS.register_module()
class CMVN:
    '''cmvn(Cepstral Mean and Variance Normalization).

    .. note::
        item[key] = (item[key] - meam) * var

    Args:

        - key(str): a key of item data, tell us which value to normalize.
        - mask_key(str): the key of mask
        - cmvn_mean(numpy.ndarray): mean value
        - cmvn_var(numpy.ndarray): var value
        - item_data(dict): input data.

    Returns:

        -dict: normalized data.
    '''

    def __init__(self, key=None, mask_key='src_mask', cmvn_mean=None, cmvn_var=None):
        '''init.
        Args:
            key(str): a key of item data, tell us which value to normalize.
            mask_key(str): the key of mask
            cmvn_mean(numpy.ndarray): mean value
            cmvn_var(numpy.ndarray): var value
        '''
        self.key = key
        self.mask_key = mask_key
        self.mean = cmvn_mean
        self.var = cmvn_var

    def __call__(self, item_data, **_kwargs):
        '''do item data normalize.
        Args:
            item_data(dict): input data.

        Returns:
            dict: normalized data.
        '''
        if item_data is None or self.key not in item_data:
            return item_data

        data = item_data[self.key]
        dim = data.shape[-1]
        if len(data.shape) == 3:  # batch version
            self.to_tensor(data)
            mean = self.mean[:dim].reshape(1, 1, -1)
            var = self.var[:dim].reshape(1, 1, -1)
        else:
            mean = self.mean[:dim]
            var = self.var[:dim]
        data -= mean
        data *= var

        if len(data.shape) == 3 and self.mask_key in item_data:
            mask = item_data[self.mask_key]
            data *= mask.unsqueeze(-1)
        item_data[self.key] = data

        return item_data

    def to_tensor(self, data):
        '''convert mean and var from numpy array to torch tensor.'''
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.from_numpy(self.mean).type_as(data).to(data.device)
            self.var = torch.from_numpy(self.var).type_as(data).to(data.device)


@PREPROCESS.register_module()
class UtteranceCmvn(CMVN):
    """utterance-based cmvn(Cepstral Mean and Variance Normalization).

    .. note::
        item[key] = (item[key] - utt_meam) * utt_var
        only support fbank in [T, N]

    Args:

        - key(str): a key of item data, tell us which value to normalize.
        - item_data(dict): input data.

    Returns:

        -dict: normalized data.
    """

    def __init__(self, key=None, mask_key='src_mask', cmvn_mean=None, cmvn_var=None, eps=1e-12):
        super().__init__(key, mask_key, cmvn_mean, cmvn_var)
        self.eps = eps

    def __call__(self, item_data, **kwargs):
        """do item data normalize.
        Args:
            item_data(dict): input data.

        Returns:
            dict: normalized data.
        """
        if item_data is None or self.key not in item_data:
            return item_data
        fbank = item_data[self.key]
        ndim = fbank.ndim
        time_axis = 0
        assert ndim == 2  # only support fbank of shape [T, N]
        mean = fbank.mean(time_axis, keepdims=True)
        var = fbank.var(time_axis, keepdims=True)
        out_fbank = (fbank - mean) / (var + self.eps)
        item_data[self.key] = out_fbank
        return item_data


@PREPROCESS.register_module()
class DynamicCmvn:
    '''dynamic cmvn

    Args:
        - key: a key of item data, tell us which value to nromalize.
        - cmvn_type: use which cmvn function
        - norm_var: use or not the var
        - center: whether or not put it in the center
        - min_cmn_window: minimun window size
        - cmn_window: normal window size
        - item_data(dict): input data.

    Returns:

        - dict: normalize data
    '''

    def __init__(
        self,
        key=None,
        cmvn_type='sliding_dynamic_C',
        norm_var=False,
        center=False,
        min_cmn_window=100,
        cmn_window=300,
    ):
        '''init.
        Args:
            key: a key of item data, tell us which value to nromalize.
            cmvn_type: use which cmvn function
            norm_var: use or not the var
            center: whether or not put it in the center
            min_cmn_window: minimun window size
            cmn_window: normal window size
        '''
        self.key = key
        self.cmvn_type = cmvn_type
        self.norm_var = norm_var
        self.center = center
        self.min_cmn_window = min_cmn_window
        self.cmn_window = cmn_window
        self.mask_key = 'frames_len'

    def sliding_dynamic_cmvn(self, in_feat):
        '''sliding dynamic cmvn'''
        # pylint:disable=too-many-branches
        out_feat = np.zeros(in_feat.shape)
        n_frames, dim = in_feat.shape
        last_window_start = -1
        last_window_end = -1
        cur_sum = np.zeros((dim,))
        cur_sumsq = np.zeros((dim,))
        for t in range(n_frames):
            if self.center:
                window_start = t - (self.cmn_window // 2)
                window_end = window_start + self.cmn_window
            else:
                window_start = t - self.cmn_window
                window_end = t + 1
            if window_start < 0:
                window_end -= window_start
                window_start = 0
            if not self.center:
                if window_end > t:
                    window_end = max(t + 1, self.min_cmn_window)
            if window_end > n_frames:
                window_start -= window_end - n_frames
                window_end = n_frames
                # pylint:disable=consider-using-max-builtin
                if window_start < 0:
                    window_start = 0
            if last_window_start == -1:
                cur_sum += np.sum(in_feat[window_start:window_end, :], axis=0)
                if self.norm_var:
                    cur_sumsq += np.sum(np.square(in_feat[window_start:window_end, :]), axis=0)
            else:
                if window_start > last_window_start:
                    assert window_start == last_window_start + 1
                    cur_sum -= in_feat[last_window_start, :]
                    cur_sumsq -= np.square(in_feat[last_window_start])
                if window_end > last_window_end:
                    assert window_end == last_window_end + 1
                    cur_sum += in_feat[last_window_end, :]
                    cur_sumsq += np.square(in_feat[last_window_end, :])
            window_frames = window_end - window_start
            last_window_start = window_start
            last_window_end = window_end
            out_feat[t] = in_feat[t] - 1.0 / window_frames * cur_sum
            if self.norm_var:
                if window_frames == 1:
                    out_feat[t] = 0
                else:
                    var = 1.0 / window_frames * cur_sumsq
                    var -= np.square(1.0 / window_frames * cur_sum)
                    var[var < 1e-12] = 1e-12
                    var = np.sqrt(var)
                    out_feat[t] /= var
        return out_feat

    def __call__(self, item_data, **_kwargs):
        '''do item data normalize
        Args:
            item_data(dict): input data.

        returns:
            dict: normalize data
        '''
        if item_data is None:
            return None
        fbank = item_data[self.key]
        ndim = fbank.ndim
        if ndim == 4:  # batch multi channels
            bsz, channels, num_frames, feat_dim = fbank.shape
            fbank = fbank.reshape(bsz * channels, num_frames, feat_dim)

        if self.cmvn_type == 'sliding_dynamic':
            fbank = self.sliding_dynamic_cmvn(fbank)
        elif self.cmvn_type == 'sliding_dynamic_C':
            if self.mask_key in item_data:
                mask_list = item_data[self.mask_key]
            else:
                mask_list = None
            fbank = dynamic_cmvn(
                fbank,
                self.center,
                self.norm_var,
                self.cmn_window,
                self.min_cmn_window,
                mask_list,
            )
        if ndim == 4:
            fbank = fbank.reshape(bsz, channels, num_frames, feat_dim)
        item_data[self.key] = fbank
        return item_data


@PREPROCESS.register_module()
class AppendFrames:
    '''append frames to a fbank seq.

    Args:

        - frame: number of frames to append
        - key: whcih value to append
        - value: -15 is the approximate fbank value of silence
        - seg_frames: the num of frames of an audio segment, by default,\
        0 when training and 20 when inference
        - fix_frames: append to the same fixed frames
        - first_seg_frames: streaming model first segment size
        - item_data(dict): input data.

    Returns:

        - dict: appended data
    '''

    def __init__(
        self, frame=0, key='fbank', value=-15, seg_frames=0, fix_frames=0, first_seg_frames=0
    ):
        '''init
        Args:
            frame: number of frames to append
            key: whcih value to append
            value: -15 is the approximate fbank value of silence
            seg_frames: the num of frames of an audio segment, by default,
                        0 when training and 20 when inference
            fix_frames: append to the same fixed frames
            first_seg_frames: streaming model first segment size
        '''
        self.frames = frame
        self.key = key
        self.value = value
        self.seg_frames = seg_frames
        self.fix_frames = fix_frames
        self.first_seg_frames = first_seg_frames

    def __call__(self, item_data, **_kwargs):
        '''do append frames
        Args:
            item_data(dict): input data.

        Returns:
            dict: appended data
        '''
        if item_data is None or self.key not in item_data:
            return item_data

        data = item_data[self.key]
        dim = data.shape[1]

        if self.frames > 0:
            padding_feat = np.full((self.frames, dim), self.value, dtype=np.float32)
            data = np.concatenate((data, padding_feat), axis=0)

        if self.seg_frames > 0:
            if data.shape[0] < self.first_seg_frames:
                padding_frames = self.first_seg_frames - data.shape[0]
            else:
                remain = (data.shape[0] - self.first_seg_frames) % self.seg_frames
                padding_frames = (self.seg_frames - remain) if remain != 0 else 0
            if padding_frames > 0:
                zeros_v = np.zeros((padding_frames, dim), dtype=np.float32)
                data = np.concatenate((data, zeros_v), axis=0)

        if self.fix_frames > 0:
            now_len = data.shape[0]
            if now_len < self.fix_frames:
                padding_feat = np.full((now_len, dim), self.value, dtype=np.float32)
                data = np.concatenate((data, padding_feat), axis=0)

        item_data[self.key] = data
        item_data['length'] = data.shape[0]
        return item_data


@PREPROCESS.register_module()
class AppendFeatureDim:
    '''append zeros to expand the dim

    Args:

        - dim: number of dims to append
        - key: whcih value to append
        - item_data(dict): input data.

    Returns:

        - dict: appended data
    '''

    def __init__(self, total_dim=80, key='fbank'):
        '''init
        Args:
            dim: number of dims to append
            key: whcih value to append
        '''
        self.total_dim = total_dim
        self.key = key

    def __call__(self, item_data, **_kwargs):
        '''do append dim
        Args:
            item_data(dict): input data.

        Returns:
            dict: appended data
        '''
        if item_data is None or self.key not in item_data:
            return item_data

        data = item_data[self.key]
        (frame, dim) = data.shape
        if dim == self.total_dim:
            return item_data
        zeros = np.zeros((frame, self.total_dim - dim), dtype=np.float32)
        item_data[self.key] = np.concatenate((data, zeros), axis=1)

        return item_data


@PREPROCESS.register_module()
class ShiftFeature:
    '''shift feature by a random offset

    Args:

        - max_shift: max value of shift
        - key: which value to shift
        - item_data(dict): input data.

    Returns:

        - dict: normalized data.
    '''

    def __init__(self, max_shift=0, key='fbank'):
        '''init.
        Args:
            max_shift: max value of shift
            key: which value to shift
        '''
        self.key = key
        self.max_shift = max_shift

    def __call__(self, item_data, **_kwargs):
        '''shift feature
        Args:
            item_data(dict): input data.

        Returns:
            dict: normalized data.
        '''
        if item_data is None:
            return None

        if self.max_shift == 0:
            return item_data

        shift = random.randint(0, self.max_shift)
        if shift == 0:
            return item_data

        data = item_data[self.key]
        item_data[self.key] = data[shift:]
        item_data['length'] = item_data[self.key].shape[0]

        return item_data


@PREPROCESS.register_module()
class AlignFeature:
    '''remove some frames overflow real ce label

    Args:

        - key: which key compared with ce_key
        - ce_key: which ce_key compared with key
        - item_data:input data

    Returns:

        - dict: normalized data.
    '''

    def __init__(self, key='fbank', ce_key='ce_label'):
        '''init.'''
        self.key = key
        self.ce_key = ce_key

    def __call__(self, item_data, **_kwargs):
        '''remove feature'''
        if item_data is None or self.key not in item_data or self.ce_key not in item_data:
            return None

        data = item_data[self.key]
        ce_label = item_data[self.ce_key]
        data_length = data.shape[0]
        ce_length = len(ce_label)

        if data_length > ce_length:
            data = data[:ce_length, :]
            item_data[self.key] = data
        elif data_length < ce_length:
            ce_label = ce_label[:data_length]
            item_data[self.ce_key] = ce_label

        item_data['length'] = item_data[self.key].shape[0]

        return item_data


@PREPROCESS.register_module()
class SplitFeature:
    '''split long feature into multi pieces.

    Args:

        - splited_len(int): feature length of processed feature.
        - feature_key(str): key name of feature to split.\
        feature should be a numpy array.\
        and only the first dim will be split.
        - max_pieces(int): max pieces of splited features; -1 means no limit.
        - dim(int): which dim to split.

    Returns:

        - item: splited data.
    '''

    def __init__(self, splited_len, in_key='src', out_key='src', max_pieces=-1, dim=0):
        '''
        Args:
            splited_len(int): feature length of processed feature.
            feature_key(str): key name of feature to split.
                              feature should be a numpy array.
                              and only the first dim will be split.
            max_pieces(int): max pieces of splited features; -1 means no limit.
            dim(int): which dim to split.
        '''
        self.splited_len = splited_len
        self.max_pieces = max_pieces
        self.in_key = in_key
        self.out_key = out_key
        self.dim = dim

    def __call__(self, item, **_kwargs):
        '''do call.'''
        if item is None or self.in_key not in item:
            return None
        feature = item[self.in_key]
        splited = []
        start = 0
        end = feature.shape[self.dim]
        slc = [slice(None)] * len(feature.shape)
        while start + self.splited_len < end and (
            self.max_pieces == -1 or len(splited) < self.max_pieces
        ):
            slc[self.dim] = slice(start, start + self.splited_len)
            splited.append(feature[slc])
            start += self.splited_len
        item[self.out_key] = splited
        return item


@PREPROCESS.register_module()
class CutFeature:
    '''
    remove frames more than threshold.
    '''

    def __init__(self, max_len, key='src', random_cut=True):
        '''
        init.
        '''
        self.max_len = max_len
        self.key = key
        self.random_cut = random_cut

    def __call__(self, item_data, **_kwargs):
        '''
        Cut feature randomly when feature is longer than max_len.
        '''
        if item_data is None:
            return None
        data = item_data[self.key]
        now_len = data.shape[0]
        if now_len > self.max_len:
            start = random.randint(0, now_len - self.max_len + 1) if self.random_cut else 0
            data = data[start : start + self.max_len]
            item_data[self.key] = data

        return item_data


@PREPROCESS.register_module()
class SelfSplicing:
    '''remove frames more than threshold.

    .. note::

        | note that if use GPU Fbank, it needs [[audios]], while for CPU Fbank needs [audios],
        | So, need output_dim to fit it.

    .. note::

        Splicing wav to target length, it can be understand as a sub-data-augment
    '''

    def __init__(
        self,
        max_len=400,
        key='waveform',
        random_clip=True,
        vad_key='vad',
        sr_key='sample_rate',
        fake_vad=False,
    ):
        '''
        init.

        note that if use GPU Fbank, it needs [[audios]], while for CPU Fbank needs [audios],
        So, need output_dim to fit it.

        '''
        self.max_len = max_len
        self.random_clip = random_clip
        self.vad_key = vad_key
        self.sr_key = sr_key
        self.key = key
        self.fake_vad = fake_vad

    def __call__(self, item, **_kwargs):
        '''
        Splicing wav to target length, it can be understand as a sub-data-augment
        '''
        if item is None:
            return None
        frames_num = get_frames_len(item[self.key].shape[1])  # all wav frames num

        if self.fake_vad or self.vad_key not in item:
            vad_list = [1] * frames_num
        else:
            vad_list = item[self.vad_key].astype(dtype=int).tolist()
        valid_len = sum(vad_list)  # valid frames num
        assert frames_num == len(vad_list), "vad list length is not equal with all frames num"
        if valid_len > self.max_len:
            if self.random_clip:
                st_pos = random.randint(0, valid_len - self.max_len)
            else:
                st_pos = 0
            ed_pos = min(st_pos + self.max_len - 1, frames_num - 1)

            valid_num = 0
            for frame_idx in range(0, frames_num):
                valid_num += vad_list[frame_idx]
                if valid_num - 1 < st_pos or valid_num - 1 > ed_pos:
                    vad_list[frame_idx] = 0
        item[self.vad_key] = np.array(vad_list)
        return item


@PREPROCESS.register_module()
class VADFilter:
    """
    Using VAD to Filter Valid Frames
    """

    def __init__(
        self, key='feature', vad_key='vad', input_type='fbank', sample_rate=16000, frame_shift=10.0
    ):
        '''init.'''
        self.key = key
        self.vad_key = vad_key
        self.input_type = input_type
        self.sample_rate = sample_rate
        self.frame_shift_sample = int(frame_shift * sample_rate / 1000)

    def __call__(self, batch_data, **_kwargs):
        '''
        in:
        batch_data
        '''
        valid_len = batch_data['valid_len']
        features = batch_data[self.key]
        vads = batch_data[self.vad_key]

        if self.input_type == 'fbank':
            masks = torch.zeros(
                vads.size(0),
                valid_len,
                dtype=batch_data['mask'].dtype,
                device=batch_data['mask'].device,
            )
            batch_data[self.key], batch_data['mask'] = VADFilter.split_tensor(
                features, vads, valid_len, masks
            )
        elif self.input_type == 'wav':
            # This may discard the last few samples, but I think it doesn't matter.
            valid_wav_len = valid_len * self.frame_shift_sample
            masks = torch.zeros(
                features.size(0),
                valid_wav_len,
                dtype=batch_data['mask'].dtype,
                device=batch_data['mask'].device,
            )

            # wav_vads has 'almost' the same length of the original waveform
            wav_vads = (
                vads.unsqueeze(-1).repeat((1, 1, self.frame_shift_sample)).view(vads.size(0), -1)
            )
            batch_data[self.key], batch_data['mask'] = VADFilter.split_tensor(
                features, wav_vads, valid_wav_len, masks, input_type='wav'
            )
        return batch_data

    @staticmethod
    def split_tensor(features, vads, valid_len, masks, input_type='fbank'):
        '''
        get split valid feature tensor and valid frames length
        '''
        utt_num = vads.size(0)
        if input_type == 'fbank':
            dim = features.size(2)
            out_data = torch.zeros(
                utt_num, valid_len, dim, device=features.device, dtype=features.dtype
            )
        elif input_type == 'wav':
            features = features.squeeze(1)
            out_data = torch.zeros(utt_num, valid_len, device=features.device, dtype=features.dtype)
        # iter each datas
        for utt in range(utt_num):
            # vad:
            #    size: 1 * frames_len
            #    data: 0 / 1 --> valid / invalid frame
            # frames:
            #    size: frames_num * features
            #    data: float gpu tensor
            # frame's index whose vad == 1, len is valid_len
            vad_list = torch.nonzero(vads[utt]).reshape(-1)
            # filter valid index
            valid_features = features[utt][vad_list]
            vad_len = len(vad_list)
            masks[utt][:vad_len] = 1
            pad_shape = valid_len - vad_len
            if pad_shape > 0:
                if input_type == 'fbank':
                    pad_features = torch.zeros(
                        pad_shape, dim, device=features.device, dtype=features.dtype
                    )
                elif input_type == 'wav':
                    pad_features = torch.zeros(
                        pad_shape, device=features.device, dtype=features.dtype
                    )
                out_data[utt] = torch.cat((valid_features, pad_features), 0)
            else:
                out_data[utt] = valid_features
        return out_data, masks


@PREPROCESS.register_module()
class ReshapeAcousticFeat:
    '''reshape the acoustic feature to (num_frames, feat_dim)'''

    def __init__(self, key='frames', out_key='fbank', feat_dim=80):
        '''init'''
        self.key = key
        self.out_key = out_key
        self.feat_dim = feat_dim

    def __call__(self, item, **_kwargs):
        '''__call__'''
        if item is None or self.key not in item:
            return item
        feat = item.pop(self.key).reshape(-1)
        feat_len = feat.shape[0]
        if feat_len % self.feat_dim != 0:
            return None
        feat = feat.reshape(-1, self.feat_dim)
        item[self.out_key] = feat
        return item


@PREPROCESS.register_module()
class KaldiMFCC:
    '''compute kaldi mfcc feature'''

    def __init__(self, in_key='waveform', out_key='mfcc', out_numpy=True, need_length=False):
        '''init'''
        self.in_key = in_key
        self.out_key = out_key
        self.out_numpy = out_numpy
        self.need_length = need_length

    def __call__(self, item, **_kwargs):
        '''call for mfcc extraction'''
        if item is None or self.in_key not in item:
            return item
        waveform = item[self.in_key]
        sample_rate = item.get('sample_rate', 16000)

        if isinstance(waveform, np.ndarray):
            waveform = torch.Tensor(waveform)

        mfccs = torchaudio.compliance.kaldi.mfcc(
            waveform=waveform,
            sample_frequency=sample_rate,
            use_energy=False,
        )  # (time, freq)
        mfccs = mfccs.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1).contiguous()  # (time, freq)
        if self.out_numpy:
            concat = concat.numpy().astype(np.float32)
        item[self.out_key] = concat
        if self.need_length:
            item['length'] = concat.shape[0]
        return item
