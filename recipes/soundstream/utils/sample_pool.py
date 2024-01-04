import random
import torch
import numpy as np


class SamplePool:
    def __init__(
        self,
        data_samplerate=44100,
        cache_size=50,
        batch_size=10,
        length_samples=44100,
        silence_prob=0.05,
    ):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.length_samples = length_samples
        self.silence_prob = silence_prob
        self.data = []

    def _process_pool(self, batch):
        for audio in batch:
            self.data.append(audio)
            # remove
            if len(self.data) > self.cache_size:
                self.data.pop(0)

    def _batch_resample(self):
        new_batch = []
        for i in range(self.batch_size):
            # use silent audio
            if random.random() < self.silence_prob:
                new_batch.append(torch.torch.zeros_like(self.data[0][:, :self.length_samples]))
            else:
                random_audio = random.choice(self.data)
                start = random.randint(0, random_audio.shape[-1] - self.length_samples)
                new_batch.append(random_audio[:, start:start+self.length_samples])
        
        return torch.stack(new_batch)
        
    def process(self, batch):
        self._process_pool(batch)
        return self._batch_resample()

class OfflineFeatureSamplePool:
    def __init__(self, cache_size=50, batch_size=10, silence_prob=0.05,sample_duration=30, hop_size=5):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.silence_prob = silence_prob
        self.sample_duration = sample_duration
        self.hop_size=hop_size
        self.data = {}

    def _process_pool(self, batch):
        # batch: dict[torch.Tensor]
        for key, item in batch.items():
            assert isinstance(
                item, (torch.Tensor, list)
            ), f"unexpected type of batch data with key='{key}': {type(item)}"
            if key not in self.data:
                self.data[key] = item
            else:
                if isinstance(item, list):
                    self.data[key]+=list(item)
                elif isinstance(item, torch.Tensor):
                    self.data[key] = torch.cat([self.data[key], item], dim=0)
            n_data = min(len(self.data[key]), self.cache_size)
            self.data[key] = self.data[key][-n_data:]

    def _batch_resample(self):
        all_vocoder_embs = self.data["vocoder_embs"]
        all_condition_tokens = self.data["condition_tokens"]
        all_loudness = self.data["loudness"].detach().cpu().numpy() # b,n
        all_dur = np.asarray(self.data["duration"]) # b

        n_valid = np.minimum(
            np.sum((all_loudness>=-50) * ~np.isneginf(all_loudness),axis=1), # number of non-trim chunks
            (all_dur//self.hop_size+1).astype(np.int64) # number of no pad chunks  
        ).tolist()
        
        # print("[all_loudness]",all_loudness)
        # print("[all_dur]",all_dur)
        # print("[n_valid]",n_valid)
        # print("\n")
        b, n = all_vocoder_embs.shape[:2]

        silence_mask= np.random.rand(self.batch_size)<self.silence_prob # b
        data_indices=[]
        chunk_indices=[]
        for _ in range(self.batch_size):
            bidx = random.choice(list(range(b)))
            cidx = random.choice(list(range(n_valid[bidx])))
            data_indices.append(bidx)
            chunk_indices.append(cidx)

        def random_data(data: torch.Tensor, b_indices, n_indices, mask):
            picked_data = data[b_indices, n_indices].detach()
            picked_data[mask.nonzero()] *= 0
            return picked_data.squeeze(1)

        return {
            "condition_tokens": random_data(
                all_condition_tokens, data_indices, chunk_indices, silence_mask
            ),
            "vocoder_embs": random_data(
                all_vocoder_embs, data_indices, chunk_indices, silence_mask
            ),
        }

    def process(self, batch):
        self._process_pool(batch)
        return self._batch_resample()
