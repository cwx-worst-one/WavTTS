import numpy as np

from .datablock import DataBlock
import os

class GhostSilenceInsert(DataBlock):
    """
    Insert ghost silence between words into the per-phone duration sequence and also the phone sequence 
    according to voicebox (https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 9, section "Training")
    """
    def __init__(self):
        super().__init__()
        self._output_iter = iter([])
        # read ./phones.txt and get a dictionary of phone id to phone name
        self.phone_dict = {}
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        phone_file = os.path.join(cur_dir, 'phones.txt')
        with open(phone_file, 'r') as f:
            for line in f:
                phone_name, phone_id = line.strip().split()
                self.phone_dict[int(phone_id)] = phone_name

    def __next__(self):
        return next(self._output_iter)

    def reset(self):
        self._output_iter = iter(self._ghost_silence_iterator())

    def _ghost_silence_iterator(self):
        """
        Create an iterator to iterate over incoming examples
        and mask the audio frames or duration frames
        """
        try:
            example = self.next_input()
            while True:
                y, l = self.insert_ghost_silence(example['phone_sequence'], example['duration'])
                assert len(l) == len(y), "The length of the duration sequence and the phone sequence should be the same"
                example['duration'] = l
                example['phone_sequence'] = y
                yield example
                example = self.next_input()
        except StopIteration:
            return
    
    def _is_ghost_silence(self, pre_phone, cur_phone):
        """
        check if a ghost silence should be inserted between two phones
        """
        SIL_ids = [1,2,3,4,5]
        # if either of the two phones is a silence phone, no ghost silence should be inserted
        if pre_phone in SIL_ids or cur_phone in SIL_ids:
            return False
        pre_phone_name = self.phone_dict[pre_phone]
        cur_phone_name = self.phone_dict[cur_phone]
        # if the previous phone is an end phone and the current phone is a begin phone, ghost silence should be inserted
        if pre_phone_name.endswith('_E') or pre_phone_name.endswith('_S'):
            if cur_phone_name.endswith('_B') or cur_phone_name.endswith('_S'):
                return True
        return False

    def insert_ghost_silence(self, y, l):
        """
        Insert ghost silence between words into the phone sequence, and also the per-phone duration sequence
        args:
            y: the phone sequence, a numpy array of int
            l: the per-phone duration sequence, a numpy array of int
        return:
            y: the phone sequence with ghost silence inserted, a numpy array of int
            l: the per-phone duration sequence with 0 ghost silence phone duration, a numpy array of int
        """
        y_new = [y[0]]
        l_new = [l[0]]

        for i in range(1, len(y)):
            if self._is_ghost_silence(y[i-1], y[i]):
                y_new.append(1) # 1 stands for SIL
                l_new.append(0)
            y_new.append(y[i])
            l_new.append(l[i])
        return np.array(y_new), np.array(l_new, dtype=np.float32)
        



