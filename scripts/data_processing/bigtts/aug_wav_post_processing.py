import argparse
import json
import logging
import multiprocessing
import os
import re
import tqdm
from multiprocessing.pool import ThreadPool

from bytedance import easycycle
from lightning_fabric.utilities.cloud_io import get_filesystem

logger = logging.getLogger(__name__)
# for randomizing manager server port
multiprocessing.util.abstract_sockets_supported = False
# for multiple filesystem instance
multiprocessing.set_start_method("spawn", force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=int, default=None)
    parser.add_argument("--num_worker", type=int, default=10)
    parser.add_argument("--aug_wav_version", type=str, required=True)
    parser.add_argument("--extra_features", type=str, nargs="+", default=[])
    parser.add_argument("--formal_run", action="store_true", default=False)
    parser.add_argument("--copy_index", action="store_true", default=False)
    parser.add_argument("--copy_wav", action="store_true", default=False)
    parser.add_argument("--copy_features", action="store_true", default=False)
    parser.add_argument("--register_dataset", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    paths = easycycle.get_dataset_collection_info_v2(args.dataset_id)["origin"]["paths"]
    if not args.formal_run:
        dataset_name_mapping = {}
        for path in paths:
            assert "/BigTTS/" in path["index"], path
            dn, bn = path["index"].split("/BigTTS/")
            ds_name = bn.split("/index_")[0]
            dataset_name_mapping[ds_name] = ds_name
            print(ds_name, path["index"])

        print(json.dumps(dataset_name_mapping, indent=2, ensure_ascii=False))

    else:
        AUG_WAV_VERSION = args.aug_wav_version
        suffix = "-vc-taozi"
        # get this name mapping from dry run

        # vc_taozi en: aug wav 1.12
        # dataset_name_mapping = {
        #     "5wh_selectfrom200wh_for_token_validation": f"5wh_selectfrom200wh_for_token_validation{suffix}",
        #     "librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_gt_10s": f"tts_Len_Slibrilight{suffix}_F1.2_D10-60",
        #     "tts_Len_Sresso-podcast_F2.0_D10-60_P1": f"tts_Len_Sresso-podcast{suffix}_F2.0_D10-60_P1",
        #     "ds=tts_en_Slibrivox-bc13-seg_P1": f"tts_Len_Slibrivox-bc13-seg{suffix}_P1",
        #     "BC2013_mos3.9_sim0.0_snr7_rms-13_asr0.8_internal_1_5_10s": f"tts_Len_Sbc2013{suffix}_F1.1_D5-10s",
        #     "ds=tts_en_S11labs-794speaker_P1": f"tts_Len_S11labs-794speaker{suffix}_P1",
        #     "librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_5_10s": f"tts_Len_Slibrilight{suffix}_F1.2_D5-10",
        #     "tts_Len_S11labs-rp2900_P1": f"tts_Len_S11labs-rp2900{suffix}_P1",
        #     "tts_Len_Sresso-podcast_F2.0_D5-10_P1": f"tts_Len_Sresso-podcast{suffix}_F2.0_D5-10_P1",
        #     "BC2013_mos3.9_sim0.0_snr7_rms-13_asr0.8_internal_1_gt_10s": f"tts_Len_Sbc2013{suffix}_F1.1_D10-60",
        #     "ds=tts_en_Sbigspeech_P1": f"tts_Len_Sbigspeech{suffix}_P1",
        #     "ds=tts_en_Sdemo-speaker_P1": f"tts_Len_Sdemo-speaker{suffix}_P1",
        #     "ds=tts_en_S11labs-general_P1": f"tts_Len_S11labs-general{suffix}_P1",
        #     "tts_Len_Sresso-podcast_F1.0_D10-60_P1": f"tts_Len_Sresso-podcast{suffix}_F1.0_D10-60_P1",
        # }

        # vc_taozi zh: aug wav 1.11
        dataset_name_mapping = {
            "fanqie_filter_v5.1/gt10": f"tts_Lmand_Sfanqie{suffix}_F5.1_D10-60_P1",
            "tts_Lmand_Sximalaya_F6.1_D0-10_P1": f"tts_Lmand_Sximalaya{suffix}_F6.1_D0-10_P1",
            "tts_Lmand_Sfanqie_F5.1_D0-10_P1": f"tts_Lmand_Sfanqie{suffix}_F5.1_D0-10_P1",
            "tts_Lmand_Sxiaoyuzhou_F6.1_D0-10_P1": f"tts_Lmand_Sxiaoyuzhou{suffix}_F6.1_D0-10_P1",
            "tts_Lmand_Sximalaya_F6.1_D10-100_P1": f"tts_Lmand_Sximalaya{suffix}_F6.1_D10-100_P1",
            "tts_Lmand_Sxiaoyuzhou_F6.1_D10-100_P1": f"tts_Lmand_Sxiaoyuzhou{suffix}_F6.1_D10-100_P1",
            "ds=tts_zh_Sbigspeech_P1": f"tts_Lmand_Sbigspeech{suffix}_P1"
        }
        hdfs_head = "hdfs://haruna"
        hdfs_head_len = len(hdfs_head)
        pool = ThreadPool(args.num_worker)
        new_datasets = []
        rets = []
        for idx, path in enumerate(paths):
            fs = get_filesystem(path["index"])

            assert AUG_WAV_VERSION in path, f"{AUG_WAV_VERSION} not exists"
            assert "/BigTTS/" in path["index"]
            dn, bn = path["index"].split("/BigTTS/")
            new_dn = dn.replace("/data/data_store", "/data_store")
            ds_name = bn.split("/index_")[0]
            new_ds_name = dataset_name_mapping[ds_name]

            old_root = os.path.join(dn, "BigTTS", ds_name)
            new_root = os.path.join(new_dn, "BigTTS", new_ds_name)

            if args.copy_index or args.copy_wav or args.copy_features:
                assert not fs.exists(new_root), f"{new_root} already exists"

            print(f"idx: {old_root=}, {new_root=}")

            index_version = re.findall(r".*(index_\d+).*", path["index"])[0]

            new_datasets.append({"dataset_name": new_ds_name, "hdfs_addr": new_root})

            # copy index
            if args.copy_index:
                index_urls = fs.glob(path["index"])
                print(f"{len(index_urls)=}")
                for i, src in enumerate(index_urls):
                    dst = (
                        src.replace(old_root[hdfs_head_len:], new_root[hdfs_head_len:])
                        .replace(index_version, "index_1")
                        .replace(f"{index_version}.parquet", ".parquet")
                    )
                    dst_dn = os.path.dirname(dst)
                    if i == 0:
                        fs.makedirs(dst_dn, exist_ok=True)
                    rets.append(
                        pool.apply_async(
                            func=fs.cp,
                            args=(src, dst),
                            kwds={"recursive": True},
                            error_callback=lambda x: print(
                                f"copy index error {src=}, {dst=}"
                            ),
                        )
                    )
                print(f"{len(rets)=}")

            # copy aug wav
            if args.copy_wav:
                aug_wav_urls = fs.glob(path[AUG_WAV_VERSION])
                print(f"{len(aug_wav_urls)=}")
                for i, src in enumerate(aug_wav_urls):
                    dst = src.replace(
                        old_root[hdfs_head_len:], new_root[hdfs_head_len:]
                    ).replace(f"features/{AUG_WAV_VERSION}", "data")
                    dst_dn = os.path.dirname(dst)
                    if i == 0:
                        fs.makedirs(dst_dn, exist_ok=True)
                    rets.append(
                        pool.apply_async(
                            func=fs.cp,
                            args=(src, dst),
                            kwds={"recursive": True},
                            error_callback=lambda x: print(
                                f"copy wav error {src=}, {dst=}"
                            ),
                        )
                    )
                print(f"{len(rets)=}")

            # copy extra features
            if args.copy_features:
                for feat in args.extra_features:
                    feat_urls = fs.glob(
                        path[AUG_WAV_VERSION].replace(AUG_WAV_VERSION, feat)
                    )
                    for i, src in enumerate(feat_urls):
                        dst = src.replace(
                            old_root[hdfs_head_len:], new_root[hdfs_head_len:]
                        )
                        dst_dn, dst_bn = os.path.dirname(dst), os.path.basename(dst)
                        ckpt_dn = os.path.join(dst_dn, "SUCCESS")
                        ckpt_url = os.path.join(ckpt_dn, dst_bn)
                        if i == 0:
                            fs.makedirs(ckpt_dn, exist_ok=True)
                        rets.append(
                            pool.apply_async(
                                func=fs.cp,
                                args=(src, dst),
                                kwds={"recursive": True},
                                error_callback=lambda x: print(
                                    f"copy feature error {feat=} {src=}, {dst=}"
                                ),
                            )
                        )
                        fs.touch(ckpt_url)
                    print(f"{len(rets)=}")
        print(f"{len(rets)=}")
        pool.close()
        [e.get() for e in tqdm.tqdm(rets)]
        pool.join()

        # register datasets
        if args.register_dataset:
            for dataset in new_datasets:
                easycycle.register_dataset(
                    biz_category="BigTTS", username="wangxin.colin", **dataset
                )
                print(f"registered {dataset}")
