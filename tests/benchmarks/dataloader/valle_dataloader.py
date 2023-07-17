import os
import time
import torch
import logging

from samantha.utils.hparams import DotDict
from recipes.valle.datasets.wds_dataload import WDSDataset, WDSCollator, dynamic_bucketizer

# mpirun -np $ARNOLD_WORKER_GPU python3 tests/benchmarks/dataloader/test_valle_dataloader.py
if __name__ == "__main__":
    max_steps = 20000
    hp = DotDict({
        "phone_tokens_num": 8000,
        "audio_tokens_num": 1024,
        "quality_check": False,
        "return_full_seq": True
    })

    dataset = WDSDataset(
        wds_lst="tests/benchmarks/configs/valle_path_list.lst",
        metaid2textid="recipes/valle/datasets/dict/metaid_to_textid.json",
        hp=hp,
        resampled=False
    )

    collate_fn = WDSCollator(tokenizer_pad=0., block_sparse=False)


    dataloader = torch.utils.data.DataLoader(
        dynamic_bucketizer(dataset, 140000, buffer_size=2000), 
        num_workers=4, 
        collate_fn=collate_fn,
        batch_size=1
        )
    worker_id = int(os.getenv('DMLC_WORKER_ID', '0'))
    worker_num = int(os.getenv('ARNOLD_WORKER_NUM', '1'))
    gpu_num = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', int(os.getenv('OMPI_COMM_WORLD_RANK', '0')) + worker_id * gpu_num))
    world_size = int(os.getenv('WORLD_SIZE', worker_num * gpu_num))

    last_time = time.time()
    cur_time = 0
    total_cnt = 0
    first_time = 0
    total_time = 0
    total_token_size = 0
    for idx, batch in enumerate(dataloader):
        cur_time = time.time() - last_time
        utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs = batch
        if idx == 0:
            first_time = cur_time
        else:
            total_cnt += 1
            total_time += cur_time
            total_token_size += seqs.shape[0] * seqs.shape[1]
        
        logging.error(f"rank: {rank}, world_size: {world_size}, idx: {idx}, time: {cur_time}, seq size: {seqs.shape}")

        if idx > max_steps:
            break
        last_time = time.time()
    logging.error(f"rank: {rank}, world_size: {world_size}, read first batch use time {first_time}s")

    logging.error(f"rank: {rank}, world_size: {world_size}, read {total_token_size} tokens, speed is {total_token_size / total_time} tokens/s")
    logging.error(f"rank: {rank}, world_size: {world_size}, read {total_cnt} batches use time {total_time}s, speed is {total_time/total_cnt}sec per batch")
