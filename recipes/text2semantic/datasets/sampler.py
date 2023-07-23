from torch.utils.data import Dataset, DistributedSampler
import torch.distributed as dist
import os
import torch
import math
import random
from typing import Optional

# sampler for dynamic batch size
class DistributedBatchSamplerSimilarLength(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_total_tokens = 4000) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # rank = int(os.getenv("LOCAL_RANK", 0))
            rank = int(os.getenv("LOCAL_RANK", 0)) + int(os.getenv("ARNOLD_ID", 0)) * int(os.getenv("ARNOLD_WORKER_GPU", 1))
            print(f'##### rank: {rank}')
        print("rank: ", rank)
        print("num_replicas: ", num_replicas)
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_total_tokens = batch_total_tokens
        self.indices = [(i, seqlen) for i, seqlen in enumerate(dataset.seqlens)]
        # self.indices = [(i, wav_id.shape[0] + text_id.shape[0] + 2) for i, (wav_id, text_id) in enumerate(self.dataset)]
        self.max_num_samples = 0

    def __iter__(self):
        ### shuffle indices
        # print("self.indices: ", self.rank, self.indices)
        # print("self.seed + self.epoch: ", self.seed + self.epoch, self.seed, self.epoch)
        if self.shuffle:
            # deterministically shuffle based on time_now and seed
            g = torch.Generator()
            # time_now = int(time.time())
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # cur_indices = torch.tensor(self.indices)[indices].tolist()
        cur_indices = [self.indices[x] for x in indices]
        # print("cur_indices: ", self.rank, cur_indices)

        ### create pool of indices with similar lengths: batch_total_tokens * 10
        pooled_indices = []
        cache_size = self.batch_total_tokens * 10
        begin_index = 0
        acc_size = 0
        for i in range(len(cur_indices)):
            if cur_indices[i][1] > self.batch_total_tokens:
                continue
            acc_size += cur_indices[i][1]
            if acc_size > cache_size:
                pooled_indices.append(sorted(cur_indices[begin_index : i], key=lambda x: x[1]))
                begin_index = i
                acc_size = cur_indices[i][1]
        if begin_index < len(cur_indices):
            pooled_indices.append(sorted(cur_indices[begin_index : ], key=lambda x: x[1]))
        # len_pooled_indices = sum([len(pooled_indice) for pooled_indice in pooled_indices])
        # assert len_pooled_indices == len(cur_indices), (len_pooled_indices, len(cur_indices))

        ### split batch by batch_total_tokens
        batches_ = []
        for pooled_indice in pooled_indices:
            begin_index = 0
            acc_size = 0
            for i in range(len(pooled_indice)):
                acc_size += pooled_indice[i][1]
                if i > 0 and acc_size > self.batch_total_tokens:
                    batches_.append(pooled_indice[begin_index : i])
                    begin_index = i
                    acc_size = pooled_indice[i][1]
            if begin_index < len(pooled_indice):
                batches_.append(pooled_indice[begin_index : ])
        # print("frame num of each batch: ", len(batches_), [sum([x[1] for x in batch_]) for batch_ in batches_])
        print("avg batch size: ", sum([len(batch_) for batch_ in batches_]) / len(batches_))
        # print("each batch size: ", len(batches_), [len(batch_) for batch_ in batches_])
        # for batch_ in batches_:
        #     print(len(batch_))
        batches = [[x[0] for x in batch_] for batch_ in batches_]

        ### cal 'fake' number of batch for multi-gpu
        if self.drop_last and len(batches) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(batches) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(batches) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.max_num_samples = max(self.max_num_samples, self.num_samples)

        ### pad or drop
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(batches)
            if padding_size <= len(batches):
                batches += batches[:padding_size]
            else:
                batches += (batches * math.ceil(padding_size / len(batches)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            batches = batches[:self.total_size]
        assert len(batches) == self.total_size, (len(batches), self.total_size)

        # print("batches_all: ", self.rank, batches[0])

        ### subsample
        batches = batches[self.rank:self.total_size:self.num_replicas]
        assert len(batches) == self.num_samples
        # print("batches: ", self.rank, batches)

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        return self.max_num_samples
