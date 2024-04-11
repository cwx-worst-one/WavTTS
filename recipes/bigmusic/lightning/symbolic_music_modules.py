import os

import torch
from torch import nn
import torch.nn.functional as F

from recipes.bigmusic.lightning.base_modules import BaseModule
cwd = os.getcwd()
curr_dir = os.path.dirname(__file__)
## Try to find samantha root dir based on your current dir
samantha_root_dir = os.path.join(curr_dir, "../../../")
os.chdir(samantha_root_dir)
from recipes.bigmusic.lightning.semantic_modules import SemanticModule
os.chdir(cwd)


class DebugModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        ## In order to make the default trainer and optimizer happy
        ## we must provide something that has a gradient
        self.param = nn.Linear(2, 2)
        self.vocab_size = vocab_size

    def forward(self, input_ids=None):
        batch_size, seq_len = input_ids.shape
        return {
            "logits": torch.rand(batch_size, seq_len, self.vocab_size),
        }


class Lyrics2SymbolicMusicModule(BaseModule):
    ## Lyrics -> Symbolic, Symbolic is target
    def prepare_training_inputs(self, batch):
        return batch


class LSAModule(SemanticModule):
    def prepare_target_inputs(self, batch):
        # Prepare audio target ids
        target_ids = self.target_embedder.tokenize(self.requires, batch['target_audio'], with_sos=False, with_eos=False).to(self.device)
        if 'target_tokens_length' in batch:
            target_lengths = batch['target_tokens_length'].to(self.device)
        else: # set to default batch length. Silence will happen before EOS
            batch_size, seq_len = target_ids.shape[:2]
            target_lengths = torch.full((batch_size,), fill_value=seq_len, dtype=torch.long, device=self.device)

        # Extract remi_leadsheet_ids with sos
        remi_leadsheet_ids = batch['remi_leadsheet_tokens'].to(self.device)
        # Offset remi_leadsheet_ids to avoid decoding issuse
        offset = self.extra_params.audio_codebook_size
        remi_leadsheet_ids = remi_leadsheet_ids + offset
        # Added remi_leadsheet_ids to target 
        target_lengths += remi_leadsheet_ids.shape[1]
        target_ids = torch.cat([remi_leadsheet_ids, target_ids], dim=1)

        target_ids = F.pad(target_ids, (1, 1)) # pad for extra sos/eos ids
        target_ids[:, 0] = self.target_embedder.sos_id # add SOS
        eos_indices = (target_lengths+1).unsqueeze(1) # set last index to EOS
        target_ids.scatter_(dim=1, index=eos_indices, value=self.target_embedder.eos_id)
        target_lengths = target_lengths + 2 # +2 for eos and sos

        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids,
            'token_seq_lengths': target_lengths
        }
