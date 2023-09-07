import glob

import braceexpand


def concatenate_dataset(split):
    cat_urls = []
    urls = []
    if split == "train":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/mirst500_v2m_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/ksong_v2m_24kHz/train/*.tar"
        )
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/mirst500_v2m_24kHz/test/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/ksong_v2m_24kHz/test/*.tar"
        )
        # urls += glob.glob("/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/mirst500_v2m_24kHz/validation/*.tar")
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/mirst500_v2m_24kHz/test/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/vocal2midi/ksong_v2m_24kHz/test/*.tar"
        )
        print("%d tar files exist for test." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls
