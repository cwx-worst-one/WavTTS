import torch


class CharacterTokenizer:
    def __init__(self) -> None:
        pass

    def setup(self, data: str) -> None:
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str, device) -> torch.Tensor:
        return torch.tensor([self.stoi[s] for s in text], dtype=torch.long)[
            None, ...
        ].to(device)

    def decode(self, tokens: torch.Tensor):
        return "".join([self.itos[int(i)] for i in tokens])

    def __call__(self, text: str, device: str) -> torch.Tensor:
        return self.encode(text, device=device)
