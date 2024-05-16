import argparse
import logging
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool

from lightning_fabric.utilities.cloud_io import get_filesystem

from samantha.dataio.parquet import ParquetWriter
from samantha.dataio.utils import parquet_reader
from scripts.utils.bigspeech import get_partition

logger = logging.getLogger(__name__)
# for randomizing manager server port
multiprocessing.util.abstract_sockets_supported = False
# for multiple filesystem instance
multiprocessing.set_start_method("spawn", force=True)


def process_one(url, output_url, fs):
    writer = ParquetWriter(filename=output_url, filesystem=fs, verbose=False)

    for item in parquet_reader(url, fs=fs, need_group_no=False):
        writer.write({"ori_audio": item["audio"], "uttid": item["uttid"]})
    writer.close()


def main(args):
    header = "hdfs://haruna"
    src_ds = [
        "ds=tts_en_S11labs-794speaker_P1",
        "ds=tts_en_Sbigspeech_P1",
        "ds=tts_en_Sdemo-speaker_P1",
        "librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_gt_10s",
        "librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_5_10s",
        "tts_Len_Sresso-podcast_F1.0_D10-60_P1",
        "tts_Len_Sresso-podcast_F2.0_D10-60_P1",
        "tts_Len_Sresso-podcast_F2.0_D5-10_P1",
        "ds=tts_zh_Sbigspeech_P1",
        "tts_Lmand_Sfanqie_F5.1_D0-10_P1",
        "fanqie_filter_v5.1/gt10",
        "tts_Lmand_Sxiaoyuzhou_F6.1_D0-10_P1",
        "tts_Lmand_Sxiaoyuzhou_F6.1_D10-100_P1",
        "tts_Lmand_Sximalaya_F6.1_D0-10_P1",
        "tts_Lmand_Sximalaya_F6.1_D10-100_P1",
    ]
    tgt_ds = [
        "tts_Len_S11labs-794speaker-diff-iter4_P1",
        "tts_Len_Sbigspeech-diff-iter4_P1",
        "tts_Len_Sdemo-speaker-diff-iter4_P1",
        "tts_Len_Slibrivox-librilight-diff-iter4_F5.0_D10-60_P1",
        "tts_Len_Slibrivox-librilight-diff-iter4_F5.0_D5-10_P1",
        "tts_Len_Sresso-podcast-diff-iter4_F1.0_D10-60_P1",
        "tts_Len_Sresso-podcast-diff-iter4_F2.0_D10-60_P1",
        "tts_Len_Sresso-podcast-diff-iter4_F2.0_D5-10_P1",
        "tts_Lmand_Sbigspeech-diff-iter4_P1",
        "tts_Lmand_Sfanqie-diff-iter4_F5.1_D0-10_P1",
        "tts_Lmand_Sfanqie-diff-iter4_F5.1_D10-60_P1",
        "tts_Lmand_Sxiaoyuzhou-diff-iter4_F6.1_D0-10_P1",
        "tts_Lmand_Sxiaoyuzhou-diff-iter4_F6.1_D10-100_P1",
        "tts_Lmand_Sximalaya-diff-iter4_F6.1_D0-10_P1",
        "tts_Lmand_Sximalaya-diff-iter4_F6.1_D10-100_P1",
    ]

    for sd, td in zip(src_ds, tgt_ds):
        src_path = (
            f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigTTS/{sd}"
        )
        if "librilight" in sd or sd == "fanqie_filter_v5.1/gt10":
            src_path = (
            f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/BigTTS/{sd}"
        )
        tgt_path = f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigTTS/{td}/features/vc_paired_1.0"
        filesystem = get_filesystem(src_path)
        if filesystem.exists(tgt_path):
            logger.warning(f"skip {tgt_path=}")
            continue
        if not filesystem.exists(src_path):
            logger.error(f"{src_path=} not exists")
            continue

        logger.info(f"start {src_path} -> {tgt_path}")
        partitions, suffix = get_partition(src_path, filesystem)
        url_pattern = f"{src_path}/data/{'/'.join(['*'] * len(partitions))}/*.{suffix}"

        logger.info(f"{url_pattern=}")
        urls = filesystem.glob(url_pattern)
        output_urls = [
            url.replace(f"{src_path[len(header):]}/data", tgt_path[len(header) :])
            for url in urls
        ]

        logger.info(f"{len(urls)=}")
        worker = Pool(args.num_workers)

        progress = []
        for url, output_url in zip(urls, output_urls):
            progress.append(
                worker.apply_async(
                    func=process_one,
                    args=(url, output_url, filesystem),
                    error_callback=lambda x: logger.error(x),
                )
            )

        worker.close()
        [e.get() for e in tqdm(progress)]
        worker.join()

        logger.info(f"done {src_path} -> {tgt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()
    main(args)
