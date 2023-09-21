import os
import json
from collections import defaultdict

import pandas as pd
import pytest
import torchaudio
from tqdm import tqdm

from recipes.bigmusic.datasets.mix import SFTWebDataModule


# @pytest.mark.skip()
def test_sft_stats():
    # Used to generate report:
    # https://bytedance.sg.feishu.cn/docx/Jr2wdT9KaohMbaxz4q8lp5fRgLe

    num_workers = 0
    batch_size = 12
    pl_datamodule = SFTWebDataModule(
        buckets_in_sec=[10, 15, 20, 25, 30],
        batch_size=batch_size,
        shuffle_buffer_size=50,
        num_workers=num_workers,
    )

    train_loader = pl_datamodule.train_dataloader()
    n_iter = 20000
    
    df = []
    for batch_idx, batch in enumerate(tqdm(train_loader, total=n_iter)):
        if batch_idx == n_iter:
            break
            
        del batch["target_audio"]
        del batch["conditions"]

        d = [{} for _ in range(batch_size)]
        for k in batch.keys():
            for idx in range(len(batch[k])):
                if len(batch[k]):
                    d[idx][k] = batch[k][idx]

        df.extend(d)


        if batch_idx % 1000 == 0:
            df_save = pd.DataFrame(df)
            df_save.to_pickle(f"SFTWebDataModule-{batch_idx}.p")

    df = pd.DataFrame(df)
    df.to_pickle(f"SFTWebDataModule-{batch_idx}.p")

@pytest.mark.skip()
def test_sft_billboard_audio():
    # Used to generate report:
    # https://bytedance.sg.feishu.cn/sheets/P54JsfT8ahnqzetQrrBlz3Ydgka?sheet=Lw3Rd7

    batch_size = 12
    pl_datamodule = SFTWebDataModule(
        buckets_in_sec=[10, 15, 20, 25, 30],
        batch_size=batch_size,
        shuffle_buffer_size=50,
        num_workers=6,
    )
    train_dataloader = pl_datamodule.train_dataloader()

    n_batches = 4
    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
        if batch_idx == n_batches:
            break

        
        for b_idx in range(len(batch)):
            torchaudio.save(f"download/{batch_idx}-{b_idx}.wav", batch["target_audio"][b_idx], 24000)

            with open(f"download/{batch_idx}-{b_idx}.txt", "w") as f:
                json.dump(batch["style_metadata"][b_idx], f)

@pytest.mark.skip()
def test_sft_workers_shards():
    num_workers = 6
    batch_size = 12
    pl_datamodule = SFTWebDataModule(
        buckets_in_sec=[10, 15, 20, 25, 30],
        batch_size=batch_size,
        shuffle_buffer_size=50,
        num_workers=num_workers,
    )
    train_dataloader = pl_datamodule.train_dataloader()

    worker_shards = defaultdict(list)
    
    batches = []
    n_batches = 2000


    print(f"NODE_RANK: {os.getenv('NODE_RANK')}")
    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
        if batch_idx == n_batches:
            break

        for b_idx in range(batch_size):
            worker_id = batch["worker_id"][b_idx]
            shard = batch["shard"][b_idx]

            if worker_id is None:
                worker_id = -1

            worker_shards[worker_id].append(shard)

    breakpoint()
    # print(worker_shards)