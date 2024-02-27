import numpy as np
from scipy.interpolate import interp1d

from .datablock import DataBlock


class AudioPhoneSync(DataBlock):
    """
    Sync the audio frames and the phone frames
    """
    def __init__(self):
        super().__init__()
        self._output_iter = iter([])

    def __next__(self):
        return next(self._output_iter)

    def reset(self):
        self._output_iter = iter(self._audio_phone_sync_iterator())
    
    def _audio_phone_sync_iterator(self):
        """
        Create an iterator to iterate over incoming examples
        and mask the audio frames or duration frames
        """
        try:
            example = self.next_input()
            while True:
                if 'audio' in example:
                    audio_frames = example['audio'].shape[0]
                else:
                    # NOTE: this calculation is not accurate, since there is also padding happened during force alignment
                    # TODO: figure out how padding is used in force alignment, and how to calculate the number of audio frames from the alignment frames
                    audio_frames = int((len(example['alignment']) * 30 / 1000 * 24000+386*2-1024)//256 + 1)
                if 'alignment' in example:
                    z = self.audio_phone_sync(example['alignment'], audio_frames)
                    example['alignment'] = z
                if 'codec' in example:
                    c = self.audio_code_sync(example['codec'], audio_frames)
                    example['codec'] = c
                yield example
                example = self.next_input()
        except StopIteration:
            return
    
    def audio_phone_sync(self, z, num_audio_frames):
        """
        use interpolation with nearest to sync the phone frames with the audio frames
        args:
            z: the frame-level phone transcript, a list of phone ids(int), M frames
            num_audio_frames: the number of audio frames
        return:
            z: the frame-level phone transcript, a list of phone ids(int), synced with the audio frames
        """
        # Create an interpolation function based on the old labels
        N = num_audio_frames
        f = interp1d(np.linspace(0, 1, len(z)), z, kind='nearest')

        # Use this function to generate labels for the new frames
        z_sync = f(np.linspace(0, 1, N))
        return z_sync.astype('int32')
    
    def audio_code_sync(self, c, num_audio_frames):
        """
        use interpolation with nearest to sync the codec code frames with the audio frames
        args:
            c: the frame-level codec code, a numpy array of code ids(int), (num_codebooks, M frames)
            num_audio_frames: the number of audio frames
        return:
            c: the frame-level codec code, a list of code ids(int), synced with the audio frames
        """
        # Create an interpolation function based on the old labels
        N = num_audio_frames
        f = interp1d(np.linspace(0, 1, c.shape[-1]), c, kind='nearest')

        # Use this function to generate labels for the new frames
        c_sync = f(np.linspace(0, 1, N))
        return c_sync.astype('int32').transpose() # [N frames, num_codebooks]
        
        





