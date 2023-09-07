import os
import random
import shlex
import subprocess
import tempfile

import numpy as np


def decode_binary_to_bytes(audio_binary, sample_rate=16000):
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "-",
    ]
    p = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = p.communicate(audio_binary, timeout=120)
    if p.returncode != 0:
        return None
    return out


def _make_ffmpeg_options(sampling_rate: float = 0.0, mono: bool = False):
    # downmixing option
    channel_cmd = "-ac 1 " if mono else "-ac 2 "
    # downsampling option
    resampling_cmd = f"-ar {str(sampling_rate)}" if sampling_rate else ""

    return channel_cmd + resampling_cmd


def convert_audio_ffmpeg_bytes_to_bytes(
    input_bytes: bytes, sampling_rate: int = 0, mono: bool = False
) -> bytes or None:
    """Resamples audio using ffmpeg. Requires the `moov` atom to be at the
    start of the file (for .m4a AAC files). If this cannot be guaranteed,
    use `convert_audio_ffmpeg_bytes_to_bytes_with_remux_retry_with_remux_retry()`.

    Args:
        input_bytes (bytes): Input audio as bytes
        sample_rate (float): Target sample rate
        mono (bool): Convert the input to mono if True. Convert the input to stereo if False.

    Returns:
        out (bytes): Processed audio
    """
    cmd = f"ffmpeg -vn -i /dev/stdin \
        {_make_ffmpeg_options(sampling_rate, mono)} -f wav -"

    cp = subprocess.run(
        shlex.split(cmd),
        input=input_bytes,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    if "invalid data found when processing input" in str(cp.stderr).lower():
        out = decode_binary_to_bytes(input_bytes, sampling_rate)
        if out is None:
            raise RuntimeError(cp.stderr)
    else:
        out = cp.stdout

    return out


def convert_audio_ffmpeg_bytes_to_bytes_with_remux_retry(
    input_bytes: bytes, sampling_rate: float = 0.0, mono: bool = False
) -> bytes or None:
    """Resamples audio using ffmpeg. If ffmpeg fails, the audio will be saved to a
    file then used as an input to ffmpeg. Use this when handling .m4a AAC audio where the
    `moov` atom cannot be guaranteed to be at the start of the file.

    See: https://stackoverflow.com/questions/57958480/ffmpeg-cant-stream-aac-files-from-stdin#:~:text=A%20.,A%20m4a%2Fmp4%20can%20not

    Args:
        input_bytes (bytes): Input audio as bytes
        sample_rate (float): Target sample rate
        mono (bool): Convert the input to mono if True. Convert the input to stereo if False.

    Returns:
        out (bytes): Processed audio
    """
    try:
        out = convert_audio_ffmpeg_bytes_to_bytes(input_bytes, sampling_rate, mono)
    except RuntimeError:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, "temp"), "wb") as f:
                f.write(input_bytes)

            temp_out_file = os.path.join(tmpdirname, "temp_out.wav")

            cmd = f"ffmpeg -vn -i {os.path.join(tmpdirname, 'temp')} \
                {_make_ffmpeg_options(sampling_rate, mono)} {temp_out_file}"

            subprocess.run(shlex.split(cmd), check=True)

            with open(temp_out_file, "rb") as f:
                out = f.read()

    return out


def adjust_audio_length(
    waveform: np.ndarray,
    target_duration_sec: float,
    sampling_rate: float,
    random_crop: bool = True,
):
    """Adjust the length of waveform (numpy array).
    Trim off the end if it's too long, add zeros if it's too short.

    Args:
        waveform: Either a one-dimensional numpy array (frames) or two-dimensional
        (frames x channels) numpy array
        target_duration_sec (float): target length of the audio in seconds
        sampling_rate (float): sample rate of the input audio
        random_crop (bool): Randomly crop longer sequences when the target sequence is shorter than the input.
        If False, sample from the start of the sequence and truncate the end.

    Returns:
        waveform: length-adjusted audio.
        padding_mask: one dimensional mask of padding added to input
                      (target_duration_sec * sampling_rate,)
    """
    if sampling_rate <= 0:
        raise RuntimeError("Sample rate must be positive")

    is_1d = False
    if len(waveform.shape) == 1:
        is_1d = True
        waveform = np.reshape(waveform, (-1, 1))

    num_channels = waveform.shape[1]
    if len(waveform.shape) != 2 or num_channels > 2:
        raise RuntimeError(
            f"Waveform has incorrect shape: {waveform.shape}. It should be a 2D array (frames x channels) ."
        )

    desired_samples = int(target_duration_sec * sampling_rate)
    padding_mask = np.zeros((desired_samples, num_channels))
    if len(waveform) < desired_samples:
        original_num_samples = waveform.shape[0]
        waveform = np.pad(
            waveform, pad_width=((0, desired_samples - len(waveform)), (0, 0))
        )
        padding_mask = np.logical_not(
            np.logical_or(
                np.zeros(len(waveform)),
                np.pad(
                    np.ones(original_num_samples),
                    pad_width=(desired_samples - original_num_samples, 0),
                ),
            )
        ).squeeze()

    audio_length = len(waveform)
    if random_crop:
        ix = random.randint(0, audio_length - desired_samples)
    else:
        ix = 0

    waveform = waveform[ix : ix + desired_samples, :]
    padding_mask = padding_mask[ix : ix + desired_samples]

    if is_1d:
        waveform = np.squeeze(waveform)

    return waveform, 1 * padding_mask


def chunk_audio(
    np_audio: np.array, sampling_rate: int, chunk_duration_in_s: float, num_chunks: int
):
    """Returns multiple chunks from equally distributed positions.
    If the model's receptive field is 30 seconds and number of chunks are 8, the output shape is (8, 30 * sample_rate).
    Note: No padding will be added to the original audio

    Args:
        array (np.array): numpy array of audio
        chunk_duration_in_s (float): length in seconds of each chunk
        num_chunks (int): number of chunks
        sampling_rate (int): sampling rate of original audio

    Returns:
        (np.array, np.array): Returns chunked array with output shape (num_chunks, chunk_duration_in_s * sampling_rate) and empty padding_mask
    """
    len_chunk = chunk_duration_in_s * sampling_rate
    if len_chunk >= len(np_audio):
        raise ValueError("len_chunk must be less than the length of np_audio")
    hop = (len(np_audio) - len_chunk) // (num_chunks - 1)
    if hop <= 0:
        raise ValueError("hop size must be larger than 0")
    if len(np_audio.shape) > 1:
        chunked_array = []
        for channel in range(np_audio.shape[-1]):
            chunked_array.append(
                np.array(
                    [
                        np_audio[int(i * hop) : int(i * hop + len_chunk), channel]
                        for i in range(num_chunks)
                    ]
                )
            )
        # reorder so channel last
        chunked_array = np.moveaxis(chunked_array, 0, -1)
    else:
        chunked_array = np.array(
            [
                np_audio[int(i * hop) : int(i * hop + len_chunk)]
                for i in range(num_chunks)
            ]
        )
    padding_mask = np.zeros_like(chunked_array)
    return chunked_array, padding_mask
