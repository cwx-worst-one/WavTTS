import io
import json
import logging
import subprocess

import euler
import numpy as np
from scipy.io.wavfile import write

euler.install_thrift_import_hook()
import base_thrift
import sami_thrift

# SAMI service client
CLIENT = euler.Client(
    sami_thrift.SamiService,
    target="sd://lab.sami.gateway?cluster=release_thrift",
    timeout=1200,
)
ACCESS_KEY = "ATUBJrWuzl"

## Get audio url using GeneralAudioFetch service
## with retry once


def _get_audio_url(meta_song_id):
    payload = json.dumps(
        {
            "url": str(meta_song_id),
            "url_type": "song_id",
            "extra": {"request_type": "audio_url"},
        }
    )
    url = ""
    response = ""
    try:
        request = sami_thrift.InvokeRequest(
            Base=base_thrift.Base(),
            access_key=ACCESS_KEY,
            method="GeneralAudioFetch",
            payload=payload,
            version="v4",
        )
        response = CLIENT.Invoke(request)
        url = json.loads(response.payload)["extra"]["url"]
    except:
        logging.error(meta_song_id, url, payload, response)
    return url


def get_audio_url(meta_song_id):
    url = _get_audio_url(meta_song_id)
    if url == "":
        # retry once
        url = _get_audio_url(meta_song_id)
    return url


## Get AudioBytes


def get_audio_bytes(url, sample_rate=24000):
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        url,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]
    p = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = p.communicate()
    if p.returncode != 0:
        return None
    npy = np.frombuffer(out, dtype=np.int16)
    wavform = io.BytesIO()
    write(wavform, sample_rate, npy)
    return {"wav": wavform.getvalue(), "npy": npy}


## Get AudioMetrics Result


def get_metric_by_audio(audio_bytes):
    payload = json.dumps(
        {
            "extra": {
                "cutoff_freq": True,
                "phase_check": True,
                "rms_stats": True,
                "clipping": True,
                "loudness": True,
            }
        }
    )
    metric = {}
    response = ""
    try:
        request = sami_thrift.InvokeRequest(
            Base=base_thrift.Base(),
            access_key=ACCESS_KEY,
            method="AudioMetrics",
            payload=payload,
            data=audio_bytes,
            version="v4",
        )
        response = CLIENT.Invoke(request)
        metric = json.loads(response.payload)
    except:
        logging.error(payload, response)
    return metric


def is_audio_metrics_good(audio_metrics):
    # Clipping
    clip = audio_metrics.get("clipping", {})
    if clip.get("rate", 0) >= 5e-5:
        return False, "clipping"
    for ch in ["left", "right"]:
        if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
            return False, "clipping"
    # Loudness
    loudness = audio_metrics.get("loudness", {})
    if (
        loudness.get("integrated_loudness", -7) > -5
        or loudness.get("max_mom_loud", -7) >= 0
        or loudness.get("max_short_term_loud", -7) >= 0
    ):
        return False, "loudness"
    # RMS stats
    rms_stats = audio_metrics.get("rms_stats", {})
    if rms_stats.get("peak", 0) > 3:
        return False, "rms_stats"
    for ch in ["left", "right"]:
        if (
            rms_stats.get(f"{ch}_total", -10) > -5
            or rms_stats.get(f"{ch}_total", -10) < -40
            or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
        ):
            return False, "rms_stats"
    # Cutoff frequency
    cutoff_freq = audio_metrics.get("cutoff_frequency", {})
    for ch in ["left", "right"]:
        if (
            cutoff_freq.get(f"rel_{ch}", 48000) < 15000
            and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
            and cutoff_freq.get(f"band_std_{ch}", 10) < 5
        ):
            return False, "cutoff_frequency"
    # Phase
    phase = audio_metrics.get("phase_check", {})
    if (
        phase.get("has_phase_issue", False)
        or abs(phase.get("rms_downmix_diff", 0.1)) > 3
    ):
        return False, "phase_check"
    return True, None


def is_audio_good(audio_bytes):
    metric = get_metric_by_audio(audio_bytes)
    result, reason = is_audio_metrics_good(metric)
    return result, metric


## Get VAD Result


def get_vad_by_audio(audio_bytes):
    payload = json.dumps(
        {
            "extra": {
                "begin_smooth_window_ms": 500,
                "begin_smooth_voice_proportion": 0.5,
                "end_smooth_window_ms": 500,
                "end_smooth_silence_proportion": 0.9,
                "voice_max_seconds": 25,
                "likelihood_threshold": 0.5,
                "enable_dynamic_smooth": False,
            },
            "model": "v2",
        }
    )
    vad = {}
    response = ""
    try:
        request = sami_thrift.InvokeRequest(
            Base=base_thrift.Base(),
            access_key=ACCESS_KEY,
            data=audio_bytes,
            method="VAD",
            payload=payload,
            version="v4",
        )
        response = CLIENT.Invoke(request)
        vad = json.loads(response.payload)
    except:
        logging.error(payload, response)
    return vad
