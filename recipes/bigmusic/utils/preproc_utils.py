import os
import itertools
import traceback
from typing import Dict, Any, Iterator

import numpy as np

from multiprocessing import Process, Queue, Pool, Manager
from recipes.bigmusic.datasets.symbolic_music.base import SymbolicMusicCodecBase
from recipes.bigmusic.datasets.symbolic_music.decorators import skip_keys
from recipes.bigmusic.utils.common_utils import TqdmWrapper


def _worker(
    dfs_dict: Dict[str, Any],
    codec: SymbolicMusicCodecBase,
    valid_perc: float,
    output_queue: Queue,
):
    # Perform the conversion
    remi_arr = codec.encode(dfs_dict)

    # Decide whether it's validation or training data
    assert len(remi_arr) > 0, "no remi array generated"
    output_queue.put((
        'valid' if np.random.rand() <= valid_perc else 'train',
        remi_arr.tobytes(order="C"),
        dfs_dict["key"],
        dfs_dict["index_dict"],
    ))


def _worker_wrapper(*args):
    try:
        # Call the actual worker function and get the result
        result = _worker(*args)
        return result
    except Exception as e:
        # Handle the exception (e.g., log or report it)
        error_message = f"Worker {os.getpid()} encountered an exception: {e}\n"
        error_message += traceback.format_exc()
        print(error_message)
        return None  # Return a sentinel or error code if needed


def _writer(output_dir: str, output_queue: Queue, count_start: int):
    train_file_path = os.path.join(output_dir, "train_text_document.bin")
    valid_file_path = os.path.join(output_dir, "valid_text_document.bin")
    preproc_record_path = os.path.join(output_dir, "preproc_record.txt")

    pbar = TqdmWrapper(verbose=True)
    
    with open(train_file_path, "ab") as train_file, \
        open(valid_file_path, "ab") as valid_file, \
        open(preproc_record_path, "a") as preproc_record_file:
        for i in itertools.count(start=count_start):
            message = output_queue.get()  # Get the message from the queue
            if message == "STOP":
                break  # Stop signal to end the writer process

            data_type, data, key, index_dict = message
            
            if data_type == "valid":
                valid_file.write(data)
            else:
                train_file.write(data)
            pbar.set_postfix({
                "i": i,
                **index_dict,
            })
            pbar.update()
            preproc_record_file.write(key + "\n")

        pbar.close()


def symbolic_preproc(
    output_dir,
    dfs_dict_iter: Iterator[Dict[str, Any]],
    codec: SymbolicMusicCodecBase,
    valid_perc: float = 0.01,
    seed: int = 42,
):
    np.random.seed(seed)
    train_file = open(os.path.join(output_dir, "train_text_document.bin"), "ab")
    valid_file = open(os.path.join(output_dir, "valid_text_document.bin"), "ab")
    preproc_record_path = os.path.join(output_dir, "preproc_record.txt")
    if os.path.exists(preproc_record_path):
        with open(preproc_record_path, "r") as f:
            processed_keys = set(l.strip() for l in f.readlines())
        print(f"{len(processed_keys)} samples already processed.")
    else:
        processed_keys = set()
    preproc_record_file = open(preproc_record_path, "a")
    pbar = TqdmWrapper(verbose=True)

    for i, dfs_dict in enumerate(
        skip_keys(keys=processed_keys)(lambda : dfs_dict_iter)()
    ):
        remi_arr = codec.encode(dfs_dict)
        if len(remi_arr) > 0:
            if np.random.rand() <= valid_perc:
                valid_file.write(remi_arr.tobytes(order="C"))
            else:
                train_file.write(remi_arr.tobytes(order="C"))
        preproc_record_file.write(dfs_dict["key"])
        preproc_record_file.write("\n")
        pbar.set_postfix({
            "i": int(i + len(processed_keys)),
            **dfs_dict["index_dict"],
        })
        pbar.update()

    pbar.close()
    preproc_record_file.close()
    train_file.close()
    valid_file.close()


def symbolic_preproc_mp(
    output_dir: str,
    dfs_dict_iter: Iterator[Dict[str, Any]],
    codec: SymbolicMusicCodecBase,
    valid_perc: float = 0.01,
    seed: int = 42,
    num_proc: int = 64,
):
    np.random.seed(seed)
    processed_keys = set()
    preproc_record_path = os.path.join(output_dir, "preproc_record.txt")
    if os.path.exists(preproc_record_path):
        with open(preproc_record_path, "r") as f:
            processed_keys = {line.strip() for line in f}
        print(f"{len(processed_keys)} samples already processed.")

    with Manager() as manager:
        # Create the queue for communication with the writer process
        output_queue = manager.Queue()

        # Start the writer process
        writer_process = Process(target=_writer, args=(output_dir, output_queue, len(processed_keys)))
        writer_process.start()

        # Create a Pool of worker processes
        with Pool(processes=num_proc) as pool:
            for i, dfs_dict in enumerate(
                skip_keys(keys=processed_keys)(lambda : dfs_dict_iter)()
            ):
                dfs_dict["index_dict"].update({"f": i})
                pool.apply_async(_worker_wrapper, (dfs_dict, codec, valid_perc, output_queue))

        # Signal the writer process to stop
        output_queue.put('STOP')
        writer_process.join()