'''
data augmentations.
'''

import inspect
import os
import time
import copy
from core.utils import logging
from core.utils.dist_util import get_local_rank
from .backup import Backup
from .label import (
    PunctuationFilter,
    BPE,
    SpaceAdd,
    Char2Syllable,
    Syllable2Phone,
    Text2Char,
    SpaceAddForZhLabel,
    ExtractRareWord,
    CodeSwitchTagAdd,
    MergeSplitedLabel,
    EmotionLabelParser,
    TimestampParser,
    MaxLengthCut,
    W2vPhone2charLabel,
    List2Str,
    Textchar2Index,
    BertTokenizer,
    ConcateEnLetters,
    DomainAdd,
    LangParser,
    DialogHistToContext,
    Char2Phone,
    PhonePredict,
    SubtituteWordInMap,
    MaskLabelByLang,
)
from .fbank import (
    KaldiFbank,
    SpeedKaldiFbank,
    KaldiMFCC,
    LogFbank,
    CMVN,
    AppendFrames,
    CutFeature,
    AppendFeatureDim,
    ShiftFeature,
    AlignFeature,
    SplitFeature,
    SelfSplicing,
    DynamicCmvn,
    VADFilter,
    ReshapeAcousticFeat,
)
from .filter import (
    LabelLengthFilter,
    TagFilter,
    LengthFilter,
    BadNumeralsFilter,
    SpecialTokenFilter,
    UttidKeywordsFilter,
    EerFilter,
    SilenceRatioFilter,
    WavDurationFilter,
)
from .key import AddKey
from .parser import PickleParser, NumpyParser, ProtoParser, DecordRaw
from .label import DictTrans, LabelMap, LabelParser, AlignZhG2PLabel
from .decompress import LilcomDecompress
from .wav import (
    WavParser,
    SimpleWavParser,
    CalculateFrameLength,
    VolumePerturbation,
    SpeedPerturbation,
    WavResample,
    RandomWavResample,
    AddNoise,
    ModifyProsody,
    AddNoiseFalcon,
    TempoPerturbation,
    AppendSilence,
    AppendAudio,
    WavConvert,
    WavNorm,
    WavCrop,
)
from .rir import AddRIR, GetRirData, RirDataCollate, GPURir
from .block_dropout import block_dropout_fn as block_dropout, BlockDropout
from .freq_mask import freq_mask, FreqMask, BatchFreqMask, FreqMaskCollate
from .time_mask import (
    time_mask,
    TimeMask,
    DynamicTimeMask,
    BatchTimeMask,
    BatchDynamicTimeMask,
    TimeMaskCollate,
)
from .time_warp import time_warp, TimeWarp
from .draw_batch import (
    ListCollate,
    RefLabelCollate,
    RareWordsCollate,
    InferenceRareWordsCollate,
    FbankCollate,
    CeLabelCollate,
    CharCollate,
    PreCharCollate,
    LMCharCollate,
    LMPreCharCollate,
    SidCollate,
    EerCollate,
    WaveformCollate,
    BinaryTargetCollect,
    TimeRange2FrameTarget,
    StressFeatCollate,
    EmotionFeatCollate,
    TimestampCollate,
    WaveformCollateNoSplit,
    WaveformNarrowCollate,
    OnehotLabelCollate,
    LengthRandomClipCollate,
    MultiChannelWaveformCollate,
    MultiChannelWaveformCollateNoSplit,
    FlagCollate,
    DomainCollate,
    LangCollate,
    ContextMakePairsCollate,
    ComputeW2vMask,
    PhoneCollate,
)
from .merge_feats import MergeStressFeats, CMNutt, AlignStressFeature
from .nnbeam_simulator import (
    DataCollateNNbeam,
    DataCollateNNbeamInfer,
    DataGeneratorNNbeam,
    CalculateFrameLengthFake,
    MultiChannelModuleSimu,
)
from .text_preprocess import RandomTemplateTextBuilder, MergeCode
from .preprocess import PREPROCESS
from .se.simulator import LoadSimulatorData, SimulatorWaveformCollate, SimulatorModule


__all__ = [
    'build_item_augmentation',
    'build_draw_batch_fn',
]


class Compose:
    """Composes several transforms together."""

    profile_env_name = 'DOLPHIN_DATA_TRANSFORM_PROFILE'

    def __init__(self, transforms):
        '''init.'''

        self.transforms = transforms
        self.profile_num = 0
        self.skip_num = dict()
        self.time_record = dict()
        self.start = dict()
        self.profilling = os.getenv(Compose.profile_env_name) == '1'
        self.skip_warning_freq = int(os.environ.get('SKIP_WARNING_FREQ', '10000'))
        self.log_num = 300
        self.compose_type = 'item'
        for transform in self.transforms:
            name = transform.__class__.__name__
            self.skip_num[name] = 0
            self.time_record[name] = 0

    def data_check_start(self, name):
        '''
        data check start
        Record every 1000 steps
        '''
        if self.profilling and self.profile_num > self.log_num:
            self.start[name] = time.time()

    def data_check_end(self, data, name, pid):
        '''data check end
        Record every 1000 steps, and the first 100 steps will not be record

        Return Value:
            True : means data is None
            False: means data is OK
        '''
        if self.profilling and self.profile_num > self.log_num:
            self.time_record[name] += time.time() - self.start[name]
            if self.profile_num % self.log_num == 0:
                logging.warning(
                    "Dolphin Dataset Profiler: rank: %d, pid: %d,"
                    "%s transform : %s, total used time : %f ms",
                    get_local_rank(),
                    pid,
                    self.compose_type,
                    name,
                    self.time_record[name] * 1000,
                )
                self.time_record[name] = 0

        if data is None:
            if self.skip_num[name] % self.skip_warning_freq == 100:
                logging.warning(
                    "rank: %d, pid: %d, item transform %s has filter %d datas",
                    get_local_rank(),
                    pid,
                    name,
                    self.skip_num[name],
                )
            self.skip_num[name] += 1
            return True
        return False

    def __call__(self, item, *args, **kwargs):
        '''call func.'''
        for t in self.transforms:
            item_name = t.__class__.__name__
            self.data_check_start(item_name)
            item = t(item, *args, **kwargs)
            if self.data_check_end(item, item_name, kwargs.get('pid', 0)):
                return None
        self.profile_num += 1
        return item

    def __repr__(self):
        '''string repr.'''
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class BatchCompose(Compose):
    """Composes draw batch funtions."""

    def __init__(self, transforms):
        super().__init__(transforms)
        self.log_num = 30
        self.compose_type = 'batch'

    def __call__(self, bucket_list, **kwargs):
        '''call func.'''
        batch_out = {}

        for collate_fn in self.transforms:
            batch_name = collate_fn.__class__.__name__
            self.data_check_start(batch_name)
            collate_fn(bucket_list, batch_out)
            if self.data_check_end(batch_out, batch_name, kwargs.get('pid', 0)):
                return None
        self.profile_num += 1
        return batch_out


def get_aug_fn(aug_cfg, meta_data=None):
    '''get augmentation object from config.
    Args:
        aug_cfg(dict): augmentation config, must have `type`.

    Returns:
        callable object: augmenation object.
    '''
    aug_cfg_bak = copy.deepcopy(aug_cfg)
    cls_name = aug_cfg_bak.pop('type')
    cls = PREPROCESS.get(cls_name)
    args = inspect.getfullargspec(cls.__init__).args
    if meta_data:
        if 'bpe_code' in args:
            meta_data['bpe_code'] = meta_data.get('total.code')
        for key in args:
            if key in meta_data.keys() and key not in aug_cfg_bak.keys():
                aug_cfg_bak[key] = meta_data[key]
    obj = cls(**aug_cfg_bak)
    return obj


def build_item_augmentation(aug_cfg_list, meta_data=None):
    '''get augmentation object from config.
    Args:
        aug_cfg_list(list): augmentation config list.

    Returns:
        Composed callable object: augmenation objects.
    '''
    aug_list = []

    for aug_cfg in aug_cfg_list:
        aug_fn = get_aug_fn(aug_cfg, meta_data=meta_data)
        aug_list.append(aug_fn)

    return Compose(aug_list)


def build_draw_batch_fn(cfgs, meta_data=None):
    '''
    build draw batch function.
    Args:
        cfgs(list): collate function list

    Return:
        callable object: draw batch fn.
    '''
    collate_list = []

    for cfg in cfgs:
        collate_fn = get_aug_fn(cfg, meta_data=meta_data)
        collate_list.append(collate_fn)

    return BatchCompose(collate_list)


def build_device_augmentation(aug_cfg_list, meta_data=None):
    '''get device augmentation object from config.
    Args:
        aug_cfg_list(list): augmentation config list.

    Returns:
        Composed callable object: augmenation objects.
    '''
    if not aug_cfg_list:
        return None

    aug_list = []
    for aug_cfg in aug_cfg_list:
        aug_fn = get_aug_fn(aug_cfg, meta_data=meta_data)
        aug_list.append(aug_fn)
    return Compose(aug_list)
