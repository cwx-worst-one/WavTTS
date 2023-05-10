""" test data gen for ssl """
import os
from glob import glob
import pickle
from multiprocessing import Pool
import librosa
import webrtcvad  # pylint: disable=import-error
import numpy as np
import soundfile as sf
from dataloader import FalconWriter
from dataloader import merge

# pylint: disable=line-too-long


def webrtcvad_proc(data):
    """vad of webrtc"""
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    frame_len_samp = 160
    nf = int(data.shape[0] // frame_len_samp)
    res = np.zeros_like(data)
    for i in range(nf):
        vad_out = vad.is_speech(
            data[i * frame_len_samp : (i + 1) * frame_len_samp].tobytes(), 16000
        )
        res[i * frame_len_samp : (i + 1) * frame_len_samp] = int(vad_out)
    return res


def write_test_data(
    data_list,
    direction_list,
    target_root_dir,
    target_local_dir,
    output_dir,
    data_suffix,
    fs_ori,
    proc_seg_len,
    trim,
):

    """write test data"""
    # pylint:disable=too-many-locals

    out_name_all = []
    proc_seg_len = int(proc_seg_len * 16000)
    for idx, target_dataname in enumerate(data_list):
        print(f"processing {target_dataname}")
        target_dataname_base = target_dataname.rsplit(f'.{data_suffix}', 1)[0]
        key_namebase = target_dataname_base
        target_direction = direction_list[idx]
        # set in/out path
        if target_root_dir.startswith('hdfs'):
            target_root_local_dir = os.path.join(target_local_dir, 'origin')
            ori_data_path = os.path.join(target_root_dir, target_dataname)
            if not os.path.exists(ori_data_path):
                print(
                    f"Origin data in {ori_data_path} will be downloaded to {target_root_local_dir}"
                )
                os.system(f"hdfs dfs -get {ori_data_path} {target_root_local_dir}")
        else:
            target_root_local_dir = target_root_dir
        target_data_path = os.path.join(target_root_local_dir, target_dataname)
        out_data_path = os.path.join(output_dir, key_namebase)
        out_name_all.append(out_data_path)
        writer = FalconWriter(out_data_path, 4 * 1024**3, 1)
        # read audio data
        target_wav, fs_ori_r = sf.read(target_data_path)
        assert fs_ori_r == fs_ori
        if fs_ori != 16000:
            target_wav = librosa.resample(target_wav, orig_sr=fs_ori, target_sr=16000)
        # apply vad
        if trim:
            target_wav_vad = target_wav[:, 0] * 32768
            target_wav_vad = target_wav_vad * 15000 / (np.max(np.abs(target_wav_vad)) + 1e-6)
            target_wav_vad = target_wav_vad.astype(np.int16)
            target_vad = webrtcvad_proc(target_wav_vad).astype(bool)
            target_wav = target_wav[target_vad, :]
        minlen = target_wav.shape[0]
        # segmentation and mix
        seg_num = int(minlen // proc_seg_len)
        new_keys = []
        new_vals = []
        for seg_idx in range(seg_num):
            cur_target = target_wav[seg_idx * proc_seg_len : (seg_idx + 1) * proc_seg_len, :]
            cur_target = cur_target * 32768
            if not trim:
                cur_target_wav_vad = (
                    cur_target[:, 0] * 15000 / (np.max(np.abs(cur_target[:, 0])) + 1e-6)
                )
                cur_target_wav_vad = cur_target_wav_vad.astype(np.int16)
                vad = webrtcvad_proc(cur_target_wav_vad).astype(bool)
            else:
                vad = np.ones(cur_target.shape[0], dtype=np.int16)
            cur_target = cur_target.astype(np.int16)
            # gen key/val
            new_keys.append(key_namebase + '_' + str(seg_idx).zfill(6))
            new_vals.append(
                pickle.dumps(
                    dict(
                        mc_waveform=cur_target.T,
                        direction=target_direction,
                        vad=vad[None, :].astype(np.int16),
                        length=int(cur_target.shape[0]),
                        wavname=key_namebase + '_' + str(seg_idx).zfill(6),
                    )
                )
            )
            if len(new_keys) == 100:
                writer.write_many(new_keys, new_vals)
                new_keys = []
                new_vals = []
        if len(new_vals) > 0:
            writer.write_many(new_keys, new_vals)
        writer.flush()
        print(f"Finished writing {out_data_path}")
    return out_name_all


def gen_ssl_test_data(
    target_root_dir=None,
    target_local_dir=None,
    output_dir=None,
    target_direction_list=None,
    data_suffix='wav',
    wav_basename='record-',
    proc_seg_len=1 * 60,
    fs_ori=16000,
    prefetch_worker_num=1,
    trim=False,
    out_dataname="test_data",
):
    """gen test data"""
    # pylint:disable=too-many-branches
    if output_dir.startswith('hdfs'):
        os.system(f"hdfs dfs -mkdir {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    if target_root_dir.startswith("hdfs"):
        assert target_local_dir is not None, "Must assign local_dir to put hdfs origin data"
        target_root_local_dir = os.path.join(target_local_dir, 'origin')
        os.makedirs(target_root_local_dir, exist_ok=True)

    if target_direction_list is not None:
        target_wav_num = len(target_direction_list)
        target_direction_list_int = []
        target_name_list = []
        for _, direction in enumerate(target_direction_list):
            if isinstance(direction, str):
                target_name_list.append(wav_basename + direction + "." + data_suffix)
            elif isinstance(direction, int):
                target_name_list.append(wav_basename + str(direction) + "." + data_suffix)
            elif isinstance(direction, float):
                target_name_list.append(wav_basename + str(int(direction)) + "." + data_suffix)
            else:
                raise ValueError("unrecognized data type in direction_list")
            target_direction_list_int.append(int(direction))
        new_target_root_dir = target_root_dir
    else:
        if target_root_dir.startswith('hdfs'):
            os.system(f"hdfs dfs -get {target_root_dir} {target_root_local_dir}")
            target_root_local_dir = os.path.join(
                target_root_local_dir, os.path.basename(target_root_dir)
            )
        target_name_list = [
            os.path.basename(x)
            for x in glob(os.path.join(target_root_local_dir, "*." + data_suffix))
        ]
        target_direction_list_int = [
            int(x.rsplit(f".{data_suffix}")[0].rsplit('-')[1]) for x in target_name_list
        ]
        new_target_root_dir = target_local_dir

    proc_list_all = target_name_list
    proc_direction_list_all = target_direction_list_int
    out_name_all = []
    if prefetch_worker_num > 0:
        prefetch_worker_num = min(prefetch_worker_num, target_wav_num)
        # pylint:disable=consider-using-with
        po = Pool(prefetch_worker_num)
        for pid in range(prefetch_worker_num):
            proc_list = proc_list_all[pid::prefetch_worker_num]
            proc_direction_list = proc_direction_list_all[pid::prefetch_worker_num]
            po.apply_async(
                write_test_data,
                (
                    proc_list,
                    proc_direction_list,
                    new_target_root_dir,
                    target_local_dir,
                    output_dir,
                    data_suffix,
                    fs_ori,
                    proc_seg_len,
                    trim,
                ),
                callback=out_name_all.extend,
            )
        po.close()
        po.join()
    else:
        out_name_all.extend(
            write_test_data(
                proc_list_all,
                proc_direction_list_all,
                new_target_root_dir,
                target_local_dir,
                output_dir,
                data_suffix,
                fs_ori,
                proc_seg_len,
                trim,
            )
        )
    merge(out_name_all, os.path.join(output_dir, out_dataname))


if __name__ == "__main__":
    args = dict(
        # dir path for target source, used in 'nonovlp' and 'ovlp' mode
        target_root_dir='/opt/tiger/workspace/dataset/TestSet/aishell3/h1/split_direction_2ch/disturb',
        # local dir path for target source, must be assigned when target_root_dir is hdfs path
        target_local_dir=None,
        # output dir path for test data, can be local or hdfs
        output_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/guzhaoyi/dolphin_test_data/ssl',
        out_dataname="test_data",  # test data output name, default 'test_data'
        # if assigned and target_root_dir is hdfs, enable multiprocess downloading hdfs origin data to target local dir, default None
        target_direction_list=list(np.linspace(20, 90, num=8, endpoint=True)),
        # must be assigned if either 'target_direction_list' or 'interf_direction_list' is assigned,
        # program will recognized audio file name as '{wav_basename}{direcion}' to download from hdfs, default 'record-'
        wav_basename='record-',
        data_suffix='wav',  # original audio file suffix, currently only support 'wav'
        # length of each test data, in seconds, default 60
        proc_seg_len=60,  # s
        fs_ori=16000,  # sampling rate of original audio data, default 16000
        prefetch_worker_num=0,  # zero, no multiprocessing; >0 number of processes, default 0
        trim=False,  # whether to trim original audio data, default False
    )
    gen_ssl_test_data(**args)
