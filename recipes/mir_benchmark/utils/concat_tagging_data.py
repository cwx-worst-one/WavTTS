import glob

import braceexpand


def concatenate_instrument_dataset(split):
    cat_urls = []
    urls = []
    with open(
        "/mnt/bn/audio-diffusion/data/karaoke_for_singsong_mulan149.mir+metadata.txt"
    ) as f:
        lines = f.readlines()
    hdfs_paths = ["pipe:hdfs dfs -cat %s" % line.strip() for line in lines]
    if split == "train":
        urls = hdfs_paths[:122]
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls = hdfs_paths[122:]
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls = hdfs_paths[122:]
        print("%d tar files exist for validation." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls


def concatenate_genre_dataset(split):
    cat_urls = []
    urls = []

    if split == "train":
        urls = glob.glob(
            "/mnt/bn/transcription-49c6ce75/unified_tagging/genre34/*/train/*.tar"
        )
        # urls = glob.glob("/mnt/bn/transcription/unified_tagging/genre34/*/train/*.tar")
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls = glob.glob(
            "/mnt/bn/transcription-49c6ce75/unified_tagging/genre34/*/validation/*.tar"
        )
        # urls = glob.glob("/mnt/bn/transcription/unified_tagging/genre34/*/validation/*.tar")
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls = glob.glob(
            "/mnt/bn/transcription-49c6ce75/unified_tagging/genre34/*/validation/*.tar"
        )
        # urls = glob.glob("/mnt/bn/transcription/unified_tagging/genre34/*/validation/*.tar")
        print("%d tar files exist for validation." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls


def concatenate_vocal_dataset(split):
    cat_urls = []
    urls = []

    if split == "train":
        urls = glob.glob(
            "/mnt/bn/transcription-49c6ce75/unified_tagging/vocal/vocal_24kHz/train/*.tar"
        )
        # urls = glob.glob("/mnt/bn/transcription/unified_tagging/vocal/vocal_24kHz/train/*.tar")
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls = glob.glob(
            "/mnt/bn/transcription-49c6ce75/unified_tagging/vocal/vocal_24kHz/validation/*.tar"
        )
        # urls = glob.glob("/mnt/bn/transcription/unified_tagging/vocal/vocal_24kHz/validation/*.tar")
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls = glob.glob(
            "/mnt/bn/transcription-49c6ce75/unified_tagging/vocal/vocal_24kHz/validation/*.tar"
        )
        # urls = glob.glob("/mnt/bn/transcription/unified_tagging/vocal/vocal_24kHz/validation/*.tar")
        print("%d tar files exist for validation." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls


def concatenate_multi_dataset(split):
    cat_urls = []
    urls = []
    with open(
        "/mnt/bn/audio-diffusion/data/karaoke_for_singsong_mulan149.mir+metadata.txt"
    ) as f:
        lines = f.readlines()
    instrument_hdfs_paths = ["pipe:hdfs dfs -cat %s" % line.strip() for line in lines]

    if split == "train":
        urls += instrument_hdfs_paths[:122]
        for i in range(10):
            urls += list(
                glob.glob(
                    "/mnt/bn/transcription/unified_tagging/incoming/sami_ai/datasets/genre34_24kHz_%d/train/*.tar"
                    % i
                )
            )
        urls += list(
            glob.glob(
                "/mnt/bn/transcription/unified_tagging/vocal/vocal_24kHz/train/*.tar"
            )
        )
        print("%d tar files exist for training." % len(urls))
    elif split == "validation":
        urls += instrument_hdfs_paths[122:]
        urls += list(
            glob.glob(
                "/mnt/bn/transcription/unified_tagging/incoming/sami_ai/datasets/genre34_24kHz_99/validation/*.tar"
            )
        )[:2]
        urls += list(
            glob.glob(
                "/mnt/bn/transcription/unified_tagging/vocal/vocal_24kHz/validation/*.tar"
            )
        )
        print("%d tar files exist for validation." % len(urls))
    elif split == "test":
        urls += instrument_hdfs_paths[122:]
        urls += list(
            glob.glob(
                "/mnt/bn/transcription/unified_tagging/incoming/sami_ai/datasets/genre34_24kHz_99/validation/*.tar"
            )
        )[:2]
        urls += list(
            glob.glob(
                "/mnt/bn/transcription/unified_tagging/vocal/vocal_24kHz/validation/*.tar"
            )
        )
        print("%d tar files exist for validation." % len(urls))
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls
