import math

from torch.utils.data import SequentialSampler

from f5_tts.model.dataset import DynamicBatchSampler, load_dataset


train_dataset = load_dataset("LibriTTS_100_360_500", "char")
sampler = SequentialSampler(train_dataset)

gpus = 8
batch_size_per_gpu = 51200
max_samples_per_gpu = 64
max_updates = 400000    # 600k or 800k

batch_sampler = DynamicBatchSampler(
    sampler,
    batch_size_per_gpu,
    max_samples=max_samples_per_gpu,
    random_seed=666,
    drop_residual=False,
)

print(
    f"One epoch has {int(len(batch_sampler) / gpus)} updates if gpus={gpus}, with "
    f"batch_size_per_gpu={batch_size_per_gpu} (frames) & "
    f"max_samples_per_gpu={max_samples_per_gpu}."
)
print(
    f"If gpus={gpus}, for max_updates={max_updates} "
    f"should set epoch={math.ceil(max_updates / int(len(batch_sampler) / gpus))}."
)

# python utils/count_max_epoch_precise.py