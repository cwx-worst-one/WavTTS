'''
merge feature process.
'''
import numpy as np
from .preprocess import PREPROCESS

SIL_LABEL = 0


@PREPROCESS.register_module()
class MergeStressFeats:
    '''merge multiple input_features'''

    def __init__(self, in_key, out_key='src'):
        '''init'''
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, item, **_kwargs):
        '''call for merge features'''
        # pylint:disable=too-many-branches,no-member,use-implicit-booleaness-not-comparison
        if item is None:
            return item
        for key in self.in_key:
            if key not in item:
                return item

        src_feat = []
        for key in self.in_key:
            if key == 'f0':
                f0 = item[key].astype(np.float32)
                if src_feat == []:
                    src_feat = f0
                else:
                    f0 = make_equal_frames(f0, f0.shape[0], src_feat.shape[0])
                    src_feat = np.concatenate((src_feat, f0), axis=1)
            elif key == 'mcc':
                mcc = item[key].astype(np.float32)
                mcc = mcc[:, 0]
                mcc = mcc.reshape(-1, 1)
                if src_feat == []:
                    src_feat = mcc
                else:
                    mcc = make_equal_frames(mcc, mcc.shape[0], src_feat.shape[0])
                    src_feat = np.concatenate((src_feat, mcc), axis=1)
            elif key == 'sp':
                sp = item[key].astype(np.float32)
                if src_feat == []:
                    src_feat = sp
                else:
                    sp = make_equal_frames(sp, sp.shape[0], src_feat.shape[0])
                    src_feat = np.concatenate((src_feat, sp), axis=1)
            elif key == 'mel':
                mel = item[key].astype(np.float32)
                if src_feat == []:
                    src_feat = mel
                else:
                    mel = make_equal_frames(mel, mel.shape[0], src_feat.shape[0])
                    src_feat = np.concatenate((src_feat, mel), axis=1)
            elif key == 'duration':
                dur = item[key].astype(np.float32)
                if len(dur.shape) == 1:
                    dur = dur.reshape(-1, 1)
                if src_feat == []:
                    src_feat = dur
                else:
                    dur = make_equal_frames(dur, dur.shape[0], src_feat.shape[0])
                    src_feat = np.concatenate((src_feat, dur), axis=1)

        item[self.out_key] = src_feat
        item['length'] = item[self.out_key].shape[0]

        return item


def make_equal_frames(in_features, in_frame_number, ref_frame_number):
    '''make the input feat with the same length as the ref feat'''

    target_features = np.zeros((ref_frame_number, in_features.shape[1]))
    if in_frame_number >= ref_frame_number:
        target_features[0:ref_frame_number,] = in_features[
            0:ref_frame_number,
        ]
    elif in_frame_number < ref_frame_number:
        tile_frames = np.tile(in_features[-1, :], (ref_frame_number - in_frame_number, 1))
        target_features = np.concatenate((in_features, tile_frames), axis=0)

    return target_features


@PREPROCESS.register_module()
class CMNutt:
    '''
    mean normalization for f0 at utterance level.
    '''

    def __init__(self, key='f0', dim=-1):
        '''init
        key='f0', dim=-1
        key='mcc', dim=0
        '''
        self.key = key
        self.dim = dim

    def __call__(self, item, **_kwargs):
        '''call for fbank extraction'''
        if item is None:
            return None

        data = item[self.key]
        feat = data[:, self.dim]
        mean = np.mean(feat)
        data[:, self.dim] -= mean
        item[self.key] = data

        return item


@PREPROCESS.register_module()
class AlignStressFeature:
    '''align the stress feature and stress label'''

    def __init__(self, key='src', ce_key='stress_label'):
        '''init.'''
        self.key = key
        self.ce_key = ce_key

    def __call__(self, item_data, **_kwargs):
        '''remove feature'''
        if item_data is None:
            return None

        data = item_data[self.key]
        ce_label = item_data[self.ce_key]
        data_length = data.shape[0]
        ce_length = len(ce_label)

        if data_length > ce_length:
            tile_frames = np.tile(SIL_LABEL, (data_length - ce_length,))
            item_data[self.ce_key] = np.concatenate((ce_label, tile_frames))

        elif data_length < ce_length:
            ce_label = ce_label[:data_length]
            item_data[self.ce_key] = ce_label

        item_data['length'] = item_data[self.key].shape[0]

        return item_data
