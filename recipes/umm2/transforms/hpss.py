import torch
import torch.nn as nn
import torch.nn.functional as F


class HPSS(nn.Module):
    """
    Wrap hpss.

    Args and Returns --> see `hpss`.
    """

    def __init__(self, kernel_size=31, power=2.0, hard=False, mask_only=False):
        super(HPSS, self).__init__()
        self.kernel_size = kernel_size
        self.power = power
        self.hard = hard
        self.mask_only = mask_only

    def forward(self, mag_specgrams):
        return hpss(mag_specgrams, self.kernel_size, self.power, self.hard, self.mask_only)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(kernel_size={}, power={}, hard={}, mask_only={})'.format(
                   self.kernel_size, self.power, self.hard, self.mask_only)


def hpss(mag_specgrams, kernel_size=31, power=2.0, hard=False, mask_only=False):
    """
    A function that performs harmonic-percussive source separation.
    Original method is by Derry Fitzgerald
    (https://www.researchgate.net/publication/254583990_HarmonicPercussive_Separation_using_Median_Filtering).

    Args:
        mag_specgrams (Tensor): any magnitude spectrograms in batch, (not in a decibel scale!)
            in a shape of (batch, ch, freq, time)

        kernel_size (int or (int, int)): odd-numbered
            if tuple,
                1st: width of percussive-enhancing filter (one along freq axis)
                2nd: width of harmonic-enhancing filter (one along time axis)
            if int,
                it's applied for both perc/harm filters

        power (float): to which the enhanced spectrograms are used in computing soft masks.

        hard (bool): whether the mask will be binarized (True) or not

        mask_only (bool): if true, returns the masks only.

    Returns:
        ret (Tuple): A tuple of four

            ret[0]: magnitude spectrograms - harmonic parts (Tensor, in same size with `mag_specgrams`)
            ret[1]: magnitude spectrograms - percussive parts (Tensor, in same size with `mag_specgrams`)
            ret[2]: harmonic mask (Tensor, in same size with `mag_specgrams`)
            ret[3]: percussive mask (Tensor, in same size with `mag_specgrams`)
    """

    def _enhance_either_hpss(mag_specgrams_padded, out, kernel_size, power, which, offset):
        """
        A helper function for HPSS

        Args:
            mag_specgrams_padded (Tensor): one that median filtering can be directly applied

            out (Tensor): The tensor to store the result

            kernel_size (int): The kernel size of median filter

            power (float): to which the enhanced spectrograms are used in computing soft masks.

            which (str): either 'harm' or 'perc'

            offset (int): the padded length

        """
        if which == 'harm':
            for t in range(out.shape[3]):
                out[:, :, :, t] = torch.median(mag_specgrams_padded[:, :, offset:-offset, t:t + kernel_size], dim=3)[0]

        elif which == 'perc':
            for f in range(out.shape[2]):
                out[:, :, f, :] = torch.median(mag_specgrams_padded[:, :, f:f + kernel_size, offset:-offset], dim=2)[0]
        else:
            raise NotImplementedError('it should be either but you passed which={}'.format(which))

        if power != 1.0:
            out.pow_(power)
        # end of the helper function

    eps = 1e-6

    if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int)):
        raise TypeError('kernel_size is expected to be either tuple of input, but it is: %s' % type(kernel_size))
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    pad = (kernel_size[0] // 2, kernel_size[0] // 2,
           kernel_size[1] // 2, kernel_size[1] // 2,)

    harm, perc, ret = torch.empty_like(mag_specgrams), torch.empty_like(mag_specgrams), torch.empty_like(mag_specgrams)
    mag_specgrams_padded = F.pad(mag_specgrams, pad=pad, mode='reflect')

    _enhance_either_hpss(mag_specgrams_padded, out=perc, kernel_size=kernel_size[0], power=power, which='perc',
                         offset=kernel_size[1] // 2)
    _enhance_either_hpss(mag_specgrams_padded, out=harm, kernel_size=kernel_size[1], power=power, which='harm',
                         offset=kernel_size[0] // 2)

    if hard:
        mask_harm = harm > perc
        mask_perc = harm < perc
    else:
        mask_harm = (harm + eps) / (harm + perc + eps)
        mask_perc = (perc + eps) / (harm + perc + eps)

    if mask_only:
        return None, None, mask_harm, mask_perc

    return mag_specgrams * mask_harm, mag_specgrams * mask_perc

# def pss_src(x, kernel_size=31, power=2.0, hard=False):
#     """perform percusive source separation using `hpss()`.
#     x: (batch, time)"""
#     n_fft = 1024
#     hop_length = 256
#     x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length)
#     x_mag = x_stft.pow(2).sum(-1).unsqueeze(1)  # add channel dim
#     _, _, _, mask_perc = hpss(x_mag, kernel_size, power, hard, mask_only=True)
#     mask_perc.squeeze_(1).unsqueeze_(3)  # remove channel, add last dim for complex
#     x_perc = time_freq.istft(x_stft * mask_perc, hop_length=hop_length, length=x.shape[1])
#     return x_perc


class Onset_Strength(nn.Module):
    def __init__(
            self,
            n_fft: int = 2048,
            hop_length: int = 512,
            lag: int = 1,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.lag = lag

    def forward(self, S: torch.tensor):
        # Compute difference to the reference, spaced by lag
        onset_env = S[..., self.lag:] - S[..., :-self.lag]

        # Discard negatives (decreasing amplitude)
        onset_env = torch.maximum(torch.tensor(0.0), onset_env)
        
        # Aggregate along channels (features)
        onset_env = torch.mean(onset_env, len(onset_env.shape)-2)

        # compensate for lag
        pad_width = self.lag
        
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += self.n_fft // (2 * self.hop_length)
        onset_env = torch.nn.functional.pad(onset_env, (int(pad_width), 0), mode="constant")

        # Trim to match the input duration
        onset_env = onset_env[..., : S.shape[-1]]

        return onset_env