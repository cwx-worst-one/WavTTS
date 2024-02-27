import numpy as np

from .datablock import DataBlock


class PerPhoneDurationGen(DataBlock):
    """
    Generate per-phone duration sequence and the phone sequence based on the frame-level phone transcript 
    according to voicebox (https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 9, section "Training")
    """
    def __init__(self):
        super().__init__()
        self._output_iter = iter([])

    def __next__(self):
        return next(self._output_iter)

    def reset(self):
        self._output_iter = iter(self._phone_duration_iterator())
    
    def _phone_duration_iterator(self):
        """
        Create an iterator to iterate over incoming examples
        and mask the audio frames or duration frames
        """
        try:
            example = self.next_input()
            while True:
                l, l_transform, y = self.per_phone_duration_gen(example['alignment'])
                assert len(l) == len(y), "The length of the duration sequence and the phone sequence should be the same"
                example['duration'] = l_transform
                example['duration_ori'] = l
                example['phone_sequence'] = y
                yield example
                example = self.next_input()
        except StopIteration:
            return
    
    def per_phone_duration_gen(self, z):
        """
        Generate the per-phone duration sequence and the phone sequence based on the frame-level phone transcript
        args:
            z: the frame-level phone transcript, a list of phone ids(int)
        return:
            l: the per-phone duration sequence, a numpy array of int
            y: the phone sequence, a numpy array of int
        """
        l = []
        phones = []
        prev_phone = z[0]
        count = 0
        for phone in z:
            if phone == prev_phone:
                count += 1
            else:
                l.append(count)
                phones.append(prev_phone)
                prev_phone = phone
                count = 1
        l.append(count)
        phones.append(prev_phone)

        # apply transformation to the duration sequence
        # refer to https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 22, section "A.3 Data transformation"
        l_new = []
        for x in l:
            x_low = x - 0.5
            x_high = x + 0.5
            # uniformly sample a number from [x_low, x_high)
            x_new = np.random.uniform(x_low, x_high)
            # tranform to log scale
            x_new = np.log(1 + x_new, dtype=np.float32)
            l_new.append(x_new)

        return np.array(l), np.array(l_new), np.array(phones)



