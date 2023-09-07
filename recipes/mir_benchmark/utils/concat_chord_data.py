import glob

import braceexpand


def concatenate_dataset(split):
    cat_urls = []
    urls = []
    if split == "train":
        # urls += glob.glob("/mnt/bn/audio-diffusion/mir_benchmark/chord/billboard_truth_chord_24kHz/train/*.tar")
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/billboard_chord_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/isophonic_chord_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/leadsheet_chord_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/rwc_chord_24kHz/train/*.tar"
        )
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/uspop_chord_24kHz/train/*.tar"
        )
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        # urls += glob.glob("/mnt/bn/audio-diffusion/mir_benchmark/chord/jaychou_chord_24kHz/validation/*.tar")
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/leadsheet_chord_24kHz/validation/*.tar"
        )
        # urls += glob.glob("/mnt/bn/audio-diffusion/mir_benchmark/chord/pop909_chord_24kHz/validation/*.tar")
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        # urls += glob.glob("/mnt/bn/audio-diffusion/mir_benchmark/chord/jaychou_chord_24kHz/validation/*.tar")
        urls += glob.glob(
            "/mnt/bn/audio-diffusion/mir_benchmark/chord/leadsheet_chord_24kHz/validation/*.tar"
        )
        # urls += glob.glob("/mnt/bn/audio-diffusion/mir_benchmark/chord/pop909_chord_24kHz/validation/*.tar")
        print("%d tar files exist for test." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls
