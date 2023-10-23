import shlex
import subprocess


def _make_ffmpeg_options(sampling_rate: float = 0.0, mono: bool = False):
    # downmixing option
    channel_cmd = "-ac 1 " if mono else "-ac 2 "
    # downsampling option
    resampling_cmd = f"-ar {str(sampling_rate)}" if sampling_rate else ""

    return channel_cmd + resampling_cmd


def convert_audio_ffmpeg_path_to_bytes(
    path, sampling_rate: float = 0.0, mono: bool = False
) -> bytes or None:
    """Resamples audio using ffmpeg. Requires the `moov` atom to be at the
    start of the file (for .m4a AAC files). If this cannot be guaranteed,
    use `convert_audio_ffmpeg_bytes_to_bytes_with_remux_retry_with_remux_retry()`.

    Args:
        path: Input audio file path
        sample_rate (float): Target sample rate
        mono (bool): Convert the input to mono if True.
                     Convert the input to stereo if False.

    Returns:
        out (bytes): Processed audio
    """
    cmd = f'ffmpeg -vn -i "{path}" \
        {_make_ffmpeg_options(sampling_rate, mono)} -f wav -'

    cp = subprocess.run(
        shlex.split(cmd),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    if "invalid data found when processing input" in str(cp.stderr).lower():
        raise RuntimeError(cp.stderr)

    return cp.stdout


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
        shlex.split(cmd), input=input_bytes, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )

    if "invalid data found when processing input" in str(cp.stderr).lower():
        out = decode_binary_to_bytes(input_bytes, sampling_rate)
        if out is None:
            raise RuntimeError(cp.stderr)
    else:
        out = cp.stdout

    return out


def decode_binary_to_bytes(audio_binary, sample_rate=16000):
    ffmpeg_command = ["ffmpeg", "-i", "pipe:", "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", "-"]
    p = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = p.communicate(audio_binary, timeout=120)
    if p.returncode != 0:
        return None
    return out
