import numpy as np
import torch

from .datablock import DataBlock


class Masking(DataBlock):
    """
    Apply masking to generate the x_ctx and l_ctx according to voicebox (https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 9, section "Training")
    1. Drop the audio sequence x with p=0.3 (set x_ctx to be 0)
    2. Drop the duration sequence l with p=0.2 (set l_ctx=0)
    3. If the sequence is not dropped, U[70, 100]% audio segment is dropped and U[10,100]% duration sequence is dropped
        Note: only one segment is dropped in one utt
    """

    def __init__(self, 
                 p_drop_x=0.3, 
                 p_drop_l=0.2, 
                 p_drop_audio_frames=(0.7, 1.0), 
                 p_drop_from_0_audio_frame = None,
                 st_drop_audio_frame = 0,
                 p_drop_duration_frames=(0.1, 1.0), 
                 p_drop_from_0_duration_frame = None,
                 st_drop_duration_frame = 0,
                 p_conditional_drop=0.0,
                 padding_value=0.0,
                 mask_use_alignment=True
                 ):
        """
        args:
            p_drop_x: the probability of dropping the whole audio sequence
            p_drop_l: the probability of dropping the whole duration sequence
            p_conditional_drop: the probability of dropping the whole (audio sequence, per-frame phone transcription) or (duration sequence, phone sequence) 
            p_drop_audio_frames: the range of the proportion of dropped audio frames
            p_drop_from_0_audio_frame: the probability of dropping from the 0th audio frame
            st_drop_audio_frame: the start frame of the dropped segment other than the 0th frame
            p_drop_duration_frames: the range of the proportion of dropped duration frames
            p_drop_from_0_duration_frame: the probability of dropping from the 0th duration frame
            st_drop_duration_frame: the start frame of the dropped segment other than the 0th frame
        """
        super().__init__()
        self._output_iter = iter([])
        self._p_drop_x = p_drop_x
        self._p_drop_l = p_drop_l
        self._p_drop_audio_frames = p_drop_audio_frames
        self._p_drop_from_0_audio_frame = p_drop_from_0_audio_frame
        self._st_drop_audio_frame = st_drop_audio_frame
        self._p_drop_duration_frames = p_drop_duration_frames
        self._p_drop_from_0_duration_frame = p_drop_from_0_duration_frame
        self._st_drop_duration_frame = st_drop_duration_frame
        self._p_conditional_drop = p_conditional_drop
        self.padding_value = padding_value
        self.mask_use_alignment = mask_use_alignment

    def __next__(self):
        return next(self._output_iter)

    def reset(self):
        self._output_iter = iter(self._masking_iterator())

    def _masking_iterator(self):
        """
        Create an iterator to iterate over incoming examples
        and mask the audio frames or duration frames
        """
        try:
            example = self.next_input()
            while True:
                example = self.masking(example)
                yield example
                example = self.next_input()
        except StopIteration:
            return
    
    def _mask_audio_frames(self, x, l=None):
        """
        Takes in a feature for 1 utterance and mask the audio frames
        All frames that aligned to a phone are either entirelly masked or unmasked if l is given
        args:
            x: the feature for 1 utterance, a numpy array of shape (T, D)
            l: the duration sequence, a numpy array of shape (T, )
        """
        num_frames = x.shape[0]
        if self._p_drop_from_0_audio_frame is not None and np.random.rand() < self._p_drop_from_0_audio_frame:
            drop_start = 0
            num_frames_drop = np.random.randint(int(num_frames*self._p_drop_audio_frames[0]), int(num_frames*self._p_drop_audio_frames[1]))
        else:
            assert num_frames - int(num_frames*self._p_drop_audio_frames[0]) > self._st_drop_audio_frame, "The start frame of the dropped segment is too late"
            while True:
                num_frames_drop = np.random.randint(int(num_frames*self._p_drop_audio_frames[0]), int(num_frames*self._p_drop_audio_frames[1]))
                if num_frames-num_frames_drop > self._st_drop_audio_frame:
                    drop_start = np.random.randint(self._st_drop_audio_frame, num_frames-num_frames_drop)
                    break
        # check the start and end frames of the dropped segment, make sure all frames aligned to a phone are either entirely masked or unmasked
        if l is not None:
            frame_idx = 0
            for frames in l:
                if drop_start >= frame_idx and drop_start < frame_idx+frames:
                    drop_start = frame_idx
                drop_end = drop_start + num_frames_drop
                if drop_end >= frame_idx and drop_end < frame_idx+frames:
                    drop_end = frame_idx+frames
                    num_frames_drop = drop_end - drop_start
                    break
                frame_idx += frames
        x_ctx = torch.concatenate((x[:drop_start, :], torch.ones((num_frames_drop, x.shape[1]), dtype=x.dtype).to(x.device)*self.padding_value, x[drop_start+num_frames_drop:, :]), axis=0)

        x_ctx_mask = torch.concatenate((torch.zeros((drop_start), dtype=bool), torch.ones((num_frames_drop), dtype=bool), torch.zeros((num_frames-drop_start-num_frames_drop), dtype=bool)), axis=0) # true for dropped frames
        return x_ctx, x_ctx_mask
    
    def _mask_duration_frames(self, l):
        """
        Takes in a feature for 1 utterance and mask the duration frames
        """
        num_frames = l.shape[0]
        if self._p_drop_from_0_duration_frame is not None and np.random.rand() < self._p_drop_from_0_duration_frame:
            drop_start = 0
            num_frames_drop = np.random.randint(int(num_frames*self._p_drop_duration_frames[0]), int(num_frames*self._p_drop_duration_frames[1]))
        else:
            assert num_frames - int(num_frames*self._p_drop_duration_frames[0]) > self._st_drop_duration_frame, "The start frame of the dropped segment is too late"
            while True:
                num_frames_drop = np.random.randint(int(num_frames*self._p_drop_duration_frames[0]), int(num_frames*self._p_drop_duration_frames[1]))
                if num_frames-num_frames_drop > self._st_drop_duration_frame:
                    drop_start = np.random.randint(self._st_drop_duration_frame, num_frames-num_frames_drop)
                    break
        l_ctx = np.concatenate((l[:drop_start], np.zeros((num_frames_drop), dtype=l.dtype), l[drop_start+num_frames_drop:]), axis=0)
        # get the mask for the dropped segment
        l_ctx_mask = np.concatenate((np.zeros((drop_start), dtype=bool), np.ones((num_frames_drop), dtype=bool), np.zeros((num_frames-drop_start-num_frames_drop), dtype=bool)), axis=0) # true for dropped frames
        return l_ctx, l_ctx_mask

    def masking(self, example, mask_feature="mel"):
        """
        Takes in an example and mask the audio frames or duration frames
        """

        assert mask_feature in ["mel", "bn"]
        
        if np.random.rand() < self._p_conditional_drop:
            # drop all the conditonal inputs of the voicebox model
            if mask_feature in example:
                # drop audio, alignment, and codec if it exists
                example[f'{mask_feature}_ctx'] = np.zeros_like(example[mask_feature])
                example[f'ctx_mask'] = np.ones((example[mask_feature].shape[0]), dtype=bool)
        else:        
            if mask_feature in example:
                x = example[mask_feature]
                if np.random.rand() > self._p_drop_x:
                    if self.mask_use_alignment:
                        x_ctx, x_ctx_mask = self._mask_audio_frames(x, example.get('duration_ori', None))
                    else:
                        x_ctx, x_ctx_mask = self._mask_audio_frames(x)
                else:
                    x_ctx = torch.ones_like(x) * self.padding_value
                    x_ctx_mask = torch.ones((x.shape[0]), dtype=bool) # true for dropped frames
                example[f'{mask_feature}_ctx'] = x_ctx
                example[f'ctx_mask'] = x_ctx_mask
            
        return example












########################## 备份，回退  ##########################

# class Masking(DataBlock):
#     """
#     Apply masking to generate the x_ctx and l_ctx according to voicebox (https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 9, section "Training")
#     1. Drop the audio sequence x with p=0.3 (set x_ctx to be 0)
#     2. Drop the duration sequence l with p=0.2 (set l_ctx=0)
#     3. If the sequence is not dropped, U[70, 100]% audio segment is dropped and U[10,100]% duration sequence is dropped
#         Note: only one segment is dropped in one utt
#     """

#     def __init__(self, 
#                     p_drop_x=0.3, 
#                     p_drop_l=0.2, 
#                     p_drop_audio_frames=(0.7, 1.0), 
#                     p_drop_from_0_audio_frame = None,
#                     st_drop_audio_frame = 0,
#                     p_drop_duration_frames=(0.1, 1.0), 
#                     p_drop_from_0_duration_frame = None,
#                     st_drop_duration_frame = 0,
#                     p_conditional_drop=0.0,
#                     min_ctx_frame=240,  # 至少保留约3s（若音频长度小于6s时，至少保留音频的一半，可以低于3s）
#                  ):
#         """
#         args:
#             p_drop_x: the probability of dropping the whole audio sequence
#             p_drop_l: the probability of dropping the whole duration sequence
#             p_conditional_drop: the probability of dropping the whole (audio sequence, per-frame phone transcription) or (duration sequence, phone sequence) 
#             p_drop_audio_frames: the range of the proportion of dropped audio frames
#             p_drop_from_0_audio_frame: the probability of dropping from the 0th audio frame
#             st_drop_audio_frame: the start frame of the dropped segment other than the 0th frame
#             p_drop_duration_frames: the range of the proportion of dropped duration frames
#             p_drop_from_0_duration_frame: the probability of dropping from the 0th duration frame
#             st_drop_duration_frame: the start frame of the dropped segment other than the 0th frame
#         """
#         super().__init__()
#         self._output_iter = iter([])
#         self._p_drop_x = p_drop_x
#         self._p_drop_l = p_drop_l
#         self._p_drop_audio_frames = p_drop_audio_frames
#         self._p_drop_from_0_audio_frame = p_drop_from_0_audio_frame
#         self._st_drop_audio_frame = st_drop_audio_frame
#         self._p_drop_duration_frames = p_drop_duration_frames
#         self._p_drop_from_0_duration_frame = p_drop_from_0_duration_frame
#         self._st_drop_duration_frame = st_drop_duration_frame
#         self._p_conditional_drop = p_conditional_drop
#         self._min_ctx_frame = min_ctx_frame

#     def __next__(self):
#         return next(self._output_iter)

#     def reset(self):
#         self._output_iter = iter(self._masking_iterator())

#     def _masking_iterator(self):
#         """
#         Create an iterator to iterate over incoming examples
#         and mask the audio frames or duration frames
#         """
#         try:
#             example = self.next_input()
#             while True:
#                 example = self.masking(example)
#                 yield example
#                 example = self.next_input()
#         except StopIteration:
#             return
    
#     def _mask_audio_frames(self, x, l=None):
#         """
#         Takes in a feature for 1 utterance and mask the audio frames
#         All frames that aligned to a phone are either entirelly masked or unmasked if l is given
#         args:
#             x: the feature for 1 utterance, a numpy array of shape (T, D)
#             l: the duration sequence, a numpy array of shape (T, )
#         """
#         num_frames = x.shape[0]
#         if self._p_drop_from_0_audio_frame is not None and np.random.rand() < self._p_drop_from_0_audio_frame:
#             drop_start = 0
#             num_frames_drop = np.random.randint(int(num_frames*self._p_drop_audio_frames[0]), int(num_frames*self._p_drop_audio_frames[1]))
#         else:
#             assert num_frames - int(num_frames*self._p_drop_audio_frames[0]) > self._st_drop_audio_frame, "The start frame of the dropped segment is too late"
#             while True:
#                 num_frames_drop = np.random.randint(int(num_frames*self._p_drop_audio_frames[0]), int(num_frames*self._p_drop_audio_frames[1]))
#                 total_frames = x.shape[0]
#                 ctx_frames = min(self._min_ctx_frame, total_frames//2) # 至少保留一半的帧
#                 if total_frames - num_frames_drop < ctx_frames:  
#                     num_frames_drop = total_frames - ctx_frames

#                 if num_frames-num_frames_drop > self._st_drop_audio_frame:
#                     drop_start = np.random.randint(self._st_drop_audio_frame, num_frames-num_frames_drop)
#                     break
#         # check the start and end frames of the dropped segment, make sure all frames aligned to a phone are either entirely masked or unmasked
#         if l is not None:
#             frame_idx = 0
#             for frames in l:
#                 if drop_start >= frame_idx and drop_start < frame_idx+frames:
#                     drop_start = frame_idx
#                 drop_end = drop_start + num_frames_drop
#                 if drop_end >= frame_idx and drop_end < frame_idx+frames:
#                     drop_end = frame_idx+frames
#                     num_frames_drop = drop_end - drop_start
#                     break
#                 frame_idx += frames
#         torch.concatenate
#         x_ctx = torch.concatenate((x[:drop_start, :], torch.zeros((num_frames_drop, x.shape[1]), dtype=x.dtype).to(x.device), x[drop_start+num_frames_drop:, :]), axis=0)
#         x_ctx_mask = torch.concatenate((torch.zeros((drop_start), dtype=bool), torch.ones((num_frames_drop), dtype=bool), torch.zeros((num_frames-drop_start-num_frames_drop), dtype=bool)), axis=0) # true for dropped frames
#         return x_ctx, x_ctx_mask, drop_start, num_frames_drop
    

#     def masking(self, example):
#         """
#         Takes in an example and mask the audio frames or duration frames
#         """
#         if 'mel' not in example and 'duration' not in example:
#             raise ValueError("Masking requires either feature or duration in the example")
        
#         if np.random.rand() < self._p_conditional_drop:
#             # drop all the conditonal inputs of the voicebox model
#             if 'mel' in example:
#                 # drop audio, alignment, and codec if it exists
#                 example['mel_ctx'] = np.zeros_like(example['mel'])
#                 example['mel_ctx_mask'] = np.ones((example['mel'].shape[0]), dtype=bool)
#                 if 'alignment' in example:
#                     example['alignment'] = np.zeros_like(example['alignment'])
#                 if 'codec' in example:
#                     example['codec'] = np.zeros_like(example['codec'])
            
#             # TODO
#             # if 'duration' in example:
#             #     # drop duration and phone_sequence
#             #     example['duration_ctx'] = np.zeros_like(example['duration'])
#             #     example['duration_ctx_mask'] = np.ones((example['duration'].shape[0]), dtype=bool)
#             #     example['phone_sequence'] = np.zeros_like(example['phone_sequence'])
#         else:        
#             if 'mel' in example:
#                 x = example['mel']
#                 x_ctx = torch.zeros_like(x)
#                 x_ctx_mask = torch.ones((x.shape[0]), dtype=bool) # true for dropped frames
#                 drop_start = 0
#                 num_frames_drop = 0
#                 if np.random.rand() > self._p_drop_x:
#                     x_ctx, x_ctx_mask, drop_start, num_frames_drop = self._mask_audio_frames(x, example.get('duration_ori', None))
#                 example['mel_ctx'] = x_ctx
#                 example['mel_ctx_mask'] = x_ctx_mask
#                 example['drop_start'] = drop_start
#                 example['num_frames_drop'] = num_frames_drop
            
#             # TODO
#             # if 'duration' in example:
#             #     l = example['duration']
#             #     l_ctx = np.zeros_like(l)
#             #     l_ctx_mask = np.ones((l.shape[0]), dtype=bool) # true for dropped frames
#             #     if np.random.rand() > self._p_drop_l:
#             #         l_ctx, l_ctx_mask = self._mask_duration_frames(l)
#             #     example['duration_ctx'] = l_ctx
#             #     example['duration_ctx_mask'] = l_ctx_mask

#         return example

