import glob

import braceexpand


def concatenate_dataset(split):
    cat_urls = []
    urls = []
    if split == "train":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/ballroom_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/beatles_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/hainsworth_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/harmonix_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/hjdb_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/karaoke_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/rwc_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/simac_beat_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/smc_beat_24kHz/train/*.tar"
        )
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/gtzan_beat_24kHz/validation/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/clip500_beat_24kHz/test/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/bytebeat_beat_24kHz/test/*.tar"
        )
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/gtzan_beat_24kHz/validation/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/clip500_beat_24kHz/test/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/bytebeat_beat_24kHz/test/*.tar"
        )
        print("%d tar files exist for test." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls


def concatenate_vocal_dataset(split):
    cat_urls = []
    urls = []
    if split == "train":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/karaoke_beat_vocal_24kHz/train/*.tar"
        )
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/MCC126_beat_24kHz/validation/*.tar"
        )
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/beat/MCC32_beat_24kHz/test/*.tar"
        )
        print("%d tar files exist for test." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls
