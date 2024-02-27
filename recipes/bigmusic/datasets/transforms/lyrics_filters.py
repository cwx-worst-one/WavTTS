from typing import Tuple
import numpy as np
import re

# Filters
def is_audio_metrics_good(metadata, lyrics) -> Tuple[bool, str]:
    audio_metrics = metadata.get('audio_metrics', {})
    # Clipping
    clip = audio_metrics.get("clipping", {})
    if clip.get("rate", 0) >= 5e-5:
        return False
    for ch in ["left", "right"]:
        if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
            return False
    # Loudness
    loudness = audio_metrics.get("loudness", {})
    if (
        loudness.get("integrated_loudness", -7) > -5
        or loudness.get("max_mom_loud", -7) >= 0
        or loudness.get("max_short_term_loud", -7) >= 0
    ):
        return False
    # RMS stats
    rms_stats = audio_metrics.get("rms_stats", {})
    if rms_stats.get("peak", 0) > 3:
        return False
    for ch in ["left", "right"]:
        if (
            rms_stats.get(f"{ch}_total", -10) > -5
            or rms_stats.get(f"{ch}_total", -10) < -40
            or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
        ):
            return False
    # Cutoff frequency
    cutoff_freq = audio_metrics.get("cutoff_frequency", {})
    for ch in ["left", "right"]:
        if (
            cutoff_freq.get(f"rel_{ch}", 48000) < 15000
            and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
            and cutoff_freq.get(f"band_std_{ch}", 10) < 5
        ):
            return False
    # Phase
    phase = audio_metrics.get("phase_check", {})
    if (
        phase.get("has_phase_issue", False)
        or abs(phase.get("rms_downmix_diff", 0.1)) > 3
    ):
        return False
    return True


def is_valid_lyrics(metadata, lyrics, confidence_threshold=0.8):
    if lyrics is None or len(lyrics) == 0:
        return False
    confidences = []
    for utterance in lyrics:
        if 'confidence' in utterance:
            confidence = float(utterance["confidence"])
        elif 'additions' in utterance:
            confidence = float(utterance["additions"]["confidence"])
        else:
            # some lyrics may not have confidence (force alignment). return True
            return True
        if confidence == 0:
            continue
        confidences.append(confidence)
    if len(confidences) == 0:
        return False
    return np.array(confidences).mean() > confidence_threshold

def is_valid_language(metadata, lyrics): 
    return 'final_language' in metadata and metadata['final_language'] != 'English'

def is_invalid_song(metadata, lyrics, confidence_threshold):
    # Metadata filtering
    # (AS) TODO: figure out a better way for english and cn to coexist. However, it should have already been filtered out on dataset
    # if is_valid_language(metadata, lyrics): 
    #     return True, "Filter invalid final_language"
    
    if not is_audio_metrics_good(metadata, lyrics): 
        return True, "Filter invalid audio metrics"
    
    # Song title filtering
    if not is_valid_song_name(metadata, lyrics):
        return True, "Filter live/instrumental song names."

    if not is_valid_lyrics(metadata, lyrics, confidence_threshold=confidence_threshold):
        return True, "Filter low confidence lyrics."
    
    
    return False, "Passed filters"


def is_valid_song_name(metadata, lyrics):
    def _is_live(song_name):
        if re.search(r'\(live\b', song_name):
            return True
        if re.search(r'- live\b', song_name):
            return True
        if re.search(r'\/ live\b', song_name):
            return True
        if re.search(r'\blive at ', song_name):
            return True
        if re.search(r'\blive from ', song_name):
            return True
        if 'recorded live ' in song_name:
            return True
        if ' live radio ' in song_name:
            return True
        if ' live lounge' in song_name:
            return True
        if ' live sets' in song_name:
            return True
        if ' live session' in song_name:
            return True
        if ' live performance' in song_name:
            return True
        if 'concert' in song_name:
            return True
        # chinese
        if '现场' in song_name: # live
            return True
        if '音乐会' in song_name: # concert
            return True
        return False

    def _is_instrumental(song_name):
        if 'instrumental' in song_name:
            return True
        return False

    def _is_remix(song_name):
        if 'remix' in song_name:
            return True
        if re.search(r'\bmix\b', song_name):
            return True
        if '混音' in song_name: # remix
            return True
        return False

    if 'meta_song_title' not in metadata or 'meta_album_title' not in metadata: return True
    song_name = metadata['meta_song_title']
    album_name = metadata['meta_album_title']
    if not album_name: album_name = ''
    full_name = (str(song_name) + ' ' + str(album_name)).lower()
    if _is_live(full_name): return False
    if _is_instrumental(full_name): return False
    # if _is_remix(full_name): return False # let's keep remixes for now
    return True
