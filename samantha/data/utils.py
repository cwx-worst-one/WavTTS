from samantha.data.audio_utils import convert_audio, normalize_audio
from samantha.data.av_audio import audio_read


def read_audio(
    fp: str,
    target_sr: int,
    normalize_loudness: bool,
    loudness_headroom_db: float = 16.0,
    rms_headroom_db: float = 18.0,
    peak_clip_headroom_db: float = 1.0,
):
    audio, sr = audio_read(fp)

    audio = convert_audio(audio, sr, target_sr, 2)

    if normalize_loudness:
        audio = normalize_audio(
            audio,
            normalize=True,
            strategy="loudness",
            peak_clip_headroom_db=peak_clip_headroom_db,
            rms_headroom_db=rms_headroom_db,
            loudness_headroom_db=loudness_headroom_db,
            loudness_compressor=False,
            sample_rate=sr,
            log_clipping=True,
        )
    audio = audio[None]
    return audio, target_sr
