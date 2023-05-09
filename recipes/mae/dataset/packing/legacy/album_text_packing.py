import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
import tarfile

download_queue = mp.Queue()
pack_text_queue = mp.Queue()


def download_process():
    while True:
        tar_file = download_queue.get()
        if tar_file is None:
            break
        # Download tar
        cmd = f"hdfs dfs -get {tar_file} tmp/"
        print(cmd)
        os.system(cmd)
        pack_text_queue.put(tar_file)


def pack_text_process(hdfs_output_dir):
    with open("album_data_dict.pkl", "rb") as f:
        df_dict = pickle.load(f)
    while True:
        print(pack_text_queue.qsize())
        tar_file = pack_text_queue.get()
        if tar_file is None:
            break
        # List filename in tar
        cmd = f"tar -tf tmp/{os.path.basename(tar_file)}"
        file_list = os.popen(cmd).read().split("\n")[:-1]
        os.unlink(f"tmp/{os.path.basename(tar_file)}")
        # <music_id>_<slice>.inst.npy -> <music_id>_<slice>.txt
        # Find text from meta csv using music_id
        # Write text to <music_id>_<slice>.txt
        # Add <music_id>_<slice>.txt to new tar
        with tarfile.open("meta/" + os.path.basename(tar_file), "w") as tar:
            for file in file_list:
                music_id = int(file.split("_")[0])
                new_file = file.replace(".inst.npy", ".meta.json")
                if str(music_id) not in df_dict:
                    meta = {"data_source": "unknown", "music_id": music_id}
                else:
                    d = df_dict[str(music_id)]
                    d["data_source"] = "album_review"
                    d["music_id"] = music_id
                    meta = d
                stream = io.BytesIO()
                stream.write(json.dumps(meta).encode())
                stream.seek(0)
                tarinfo = tarfile.TarInfo(new_file)
                tarinfo.size = len(stream.getbuffer())
                tar.addfile(tarinfo, fileobj=stream)
        if hdfs_output_dir is not None:
            cmd = (
                f"hdfs dfs -put meta/{os.path.basename(tar_file)} "
                f"{hdfs_output_dir} && rm meta/{os.path.basename(tar_file)}"
            )
            print(cmd)
            os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_id", type=int, default=7, help="bytedrive id")
    parser.add_argument(
        "--n_downloaders", type=int, default=40, help="num of download process"
    )
    parser.add_argument("--n_packers", type=int, default=10, help="num of pack process")
    parser.add_argument(
        "--hdfs_out_dir",
        default="/home/byte_speech_sv/mulan/short_form_uio/album_shards",
        help="hdfs output path",
    )

    args = parser.parse_args()

    drive_id = args.drive_id
    n_downloaders = args.n_downloaders
    n_packers = args.n_packers

    # List tar from hdfs://harunava/home/byte_speech_sv/mulan/short_form_uio/inst_shards
    base_path = "/home/byte_speech_sv/mulan/short_form_uio/inst_shards"
    cmd = f"hdfs dfs -ls {base_path}/shards_{drive_id}_*.tar"
    print(cmd)
    tar_list = os.popen(cmd).read().split("\n")
    tar_list = [x.split(" ")[-1] for x in tar_list if x != ""]
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("meta", exist_ok=True)

    processes = []
    for _ in range(n_downloaders):
        p = mp.Process(target=download_process)
        p.start()
        processes.append(p)

    pack_processes = []
    for _ in range(n_packers):
        p = mp.Process(target=pack_text_process, args=(args.hdfs_out_dir,))
        p.start()
        pack_processes.append(p)

    for tar_file in tar_list[:]:
        download_queue.put(tar_file)

    for _ in range(n_downloaders):
        download_queue.put(None)

    for p in processes:
        p.join()

    for _ in range(n_packers):
        pack_text_queue.put(None)

    for p in pack_processes:
        p.join()
