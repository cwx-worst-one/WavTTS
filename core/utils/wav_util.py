"""
wav_util:
Provide conversion interface from wav to frames
"""
from core.utils import logging


def get_wav_len(frames_len, sample_rate=16000, frame_length=25.0, frame_shift=10.0):
    '''
    Given the frame length, calculate the length of the corresponding waveform
    frames-len -> waveform-len
    '''
    if frames_len <= 0:
        logging.warning(
            "WARNING: The parameter you passed in (%d) is a non positive number", int(frames_len)
        )
        return 0
    frame_length_sample = frame_length * sample_rate / 1000
    frame_shift_sample = frame_shift * sample_rate / 1000
    return int((frames_len - 1) * frame_shift_sample + frame_length_sample)


def get_frames_len(wav_len, sample_rate=16000, frame_length=25.0, frame_shift=10.0):
    '''
    Given the wav length, calculate the length of the corresponding frames
    waveform-len -> frames-len
    '''
    frame_length_sample = frame_length * sample_rate / 1000
    frame_shift_sample = frame_shift * sample_rate / 1000

    if wav_len <= frame_length_sample:
        logging.warning(
            "WARNING: The parameter you passed in (%d) is less than the shortest frame length(%d)",
            int(wav_len),
            int(frame_length_sample),
        )
        return 0
    return int(((wav_len - frame_length_sample) / frame_shift_sample) + 1)


def wav_start_pos(frame_idx, sample_rate=16000, frame_length=25.0, frame_shift=10.0):
    '''
    Given the sequence number of the frame, get the starting position in the waveform
    frames_idx -> start_pos in waveform
    '''
    if frame_idx < 0:
        logging.warning(
            "WARNING: The parameter you passed in (%d) is a negative number",
            frame_idx,
        )
        return -1
    _frame_length_sample = frame_length * sample_rate / 1000
    frame_shift_sample = frame_shift * sample_rate / 1000
    return int(frame_idx * frame_shift_sample)
