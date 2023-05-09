import cruise
import torch


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device()
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


class RandomDataloader:
    def __init__(
        self,
        n_steps: int = 10,
        batch_size: int = 32,
        seq_len: int = 1024,
        vocab_size: int = 50257,
    ) -> None:
        self.n_steps = n_steps
        self.cur_step = 0
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        self.cur_step = 0
        return self

    def __next__(self):
        if self.cur_step >= self.n_steps:
            raise StopIteration
        self.cur_step += 1
        return get_data(self.batch_size, self.seq_len, self.vocab_size)

    def __len__(self):
        return self.n_steps


class GPTDataModule(cruise.CruiseDataModule):
    def __init__(self, train_dataloader=RandomDataloader()):
        super().__init__()
        self._train_dataloader = train_dataloader

    def train_dataloader(self):
        return self._train_dataloader
