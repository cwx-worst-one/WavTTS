import hashlib
from typing import List, Optional, Tuple

import torch


def length_to_mask(
    lengths: torch.Tensor, max_len: Optional[int] = None
) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(
        final_length, 1
    )  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """
    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size


class NoopTokenizer:
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]
    """

    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def pad_token_id(self) -> int:
        return self.pad_idx

    def __call__(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.tensor(output, dtype=torch.long, device=device).unsqueeze(1)
        mask = length_to_mask(
            torch.tensor(lengths, dtype=torch.int32, device=device)
        ).int()
        return tokens, mask
