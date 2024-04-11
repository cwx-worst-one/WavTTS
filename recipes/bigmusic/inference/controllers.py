from typing import Optional, List

import torch
import numpy as np
from tqdm import tqdm

from recipes.bigmusic.inference.token_generator import TokenGenerator
from samantha.utils.ctiga.inference_params import InferenceParams


class ControllerBase:
    def __init__(
        self,
        token_gen: TokenGenerator,
    ):
        self.token_gen = token_gen
    
    @torch.inference_mode()
    def generate(
        self,
        input: Optional[List[int]] = None,
        **kwargs,
    ):
        raise NotImplementedError()


from recipes.bigmusic.datasets.symbolic_music.base_codec import BaseCodec
class FixedLengthLyrics2LeadsheetController(ControllerBase):
    """Assuming config.lyrics_seq_len and config.leadsheet_seq_len exists
    """
    def __init__(
        self,
        token_gen: TokenGenerator,
        codec: BaseCodec
    ):
        self.token_gen = token_gen
        self.codec = codec

    @torch.inference_mode()
    def generate(
        self,
        input: Optional[List[int]] = None,
        total_seq_len: Optional[int] = None
    ):
        total_seq_len = total_seq_len or (self.codec.config.leadsheet_seq_len + self.codec.config.lyrics_seq_len)
        # pad_index = self.codec.indexer["pad"]
        eos_index = self.codec.indexer["eos"]
        prefix_tokens = np.array([self.codec.indexer["bar"]])
        if input is not None:
            prefix_tokens = np.append(
                prefix_tokens,
                self.codec.chop_or_pad(input, self.codec.config.lyrics_seq_len),
            )
        self.token_gen.reset()

        input_arr = torch.tensor(prefix_tokens, device=self.token_gen.device)[None, ...]
        inference_params = InferenceParams(
            max_sequence_len=total_seq_len,
            max_batch_size=1,
        )
        # start = len(lyrics_tokens) if lyrics_tokens is not None else 0
        for _ in tqdm(range(self.codec.config.leadsheet_seq_len)):
            logits = self.token_gen.generate(input_arr=input_arr, inference_params=inference_params)
            next_index = self.token_gen.sample(logits)
            if next_index == eos_index:
                print("End token generated. Stop early.")
                break
            input_arr = torch.tensor(
                [[next_index]],
                dtype=torch.long,
                device=self.token_gen.device
            )


class FLEmbedsController(ControllerBase):
    """Infer for a fixed length given input
    """
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        stop_index: Optional[int] = None,
    ):
        total_seq_len = seq_len + inputs_embeds.size()[1]
        self.token_gen.reset()

        # input_arr = torch.tensor(input, device=self.token_gen.device)[None, ...]
        inference_params = InferenceParams(
            max_sequence_len=total_seq_len,
            max_batch_size=1,
        )
        # start = len(lyrics_tokens) if lyrics_tokens is not None else 0
        for _ in tqdm(range(seq_len)):
            logits = self.token_gen.generate(inputs_embeds=inputs_embeds, inference_params=inference_params)
            next_index = self.token_gen.sample(logits)
            if stop_index is not None and next_index == stop_index:
                print("End token generated. Stop early.")
                break
            next_input_ids = torch.tensor(next_index, device=self.token_gen.device)[None, ...]
            inputs_embeds = self.token_gen.pl_module.target_embedder.embedder(next_input_ids).unsqueeze(1)
            # input_arr = torch.tensor(
            #     [[next_index]],
            #     dtype=torch.long,
            #     device=self.token_gen.device
            # )