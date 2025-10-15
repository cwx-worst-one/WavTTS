import os
import random
from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torchaudio.functional import resample

from samantha.criterion.masked_loss import sequence_mask
from samantha.utils.hparams import DotDict
from samantha.models.ctiga.gpt import GPTLMHeadModel
import torch.nn.functional as F
from samantha.utils.ctiga.inference_params import InferenceParams


# @qinxin: remove R (codebook depth) related codes, and delete all self.embedder- and id-related definition in mtp_module,
#          as these will be defined in SemanticEmbModuleMtp.target_embedder,
#          [VQ: BestRQTokenEmbedder; RVQ: BestRQHierarchicalTokenEmbedder (TODO)]
class MultiTokenPredictionModule(torch.nn.Module):
    def __init__(
        self,
        vocab_size=32_768,
        embedding_dim=1024,
        pattern='delay',
        group=2,
        encoder='fc',
        decoder='fc',
        criterion=None,
    ):
        self.last_hidden_state = None
        self.group = int(group)
        self.pattern = pattern
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.null_id = self.vocab_size + 2 + 1  # sos_id, eos_id, null_id
        self.R = 1  # hierarchy = 1 for VQ tokens

        super().__init__()

        if self.pattern is None:
            self.pattern = 'parallel'
            self.group = 1
            encoder = 'skip'
            decoder = {'model_type': 'fc'}

        ## ======= Encoder ======== ##
        self.encoder_type = encoder
        if self.encoder_type == 'fc':    # concat -> fc
            self.encoder = nn.Sequential(
                nn.Linear(self.group * embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        elif self.encoder_type in ['skip', 'sum']:
            self.encoder = None # operation instead of layers
        else:
            raise NotImplementedError(f"Unsupported multi-token encoder {self.encoder_type}")
        
        ## ======= Decoder ======== ##
        self.decoder_type = decoder["model_type"]
        if self.decoder_type in ['fc']:    # concat -> fc
            self.decoder = None
        elif self.decoder_type in ['transformer', 'RQ_transformer', 'ARQ_transformer']:
            self.decoder = GPTLMHeadModel(decoder['hp'])
        elif self.decoder_type in ['context_fc']:
            self.decoder = nn.Sequential(
                nn.Linear(2 * embedding_dim, 2 * embedding_dim),    # context + cond
                nn.SiLU(),
                nn.Linear(2 * embedding_dim, decoder["output_dim"]),    # num_logits
            )
        else:
            raise NotImplementedError(f"Unsupported multi-token decoder {self.decoder_type}")

        ## ======= Criterion ======== ##
        self.criterion = None
        if criterion is not None:
            self.criterion = criterion()
    
    def ungroup_token(self, group_ids, sos_id, eos_id, delete_null=False, add_sos=True, delay_back=False):
        """
        Input:
            group_ids: [G, T//G]
        Return:
            target_ids: [T]
        """
        # [G, T] -- reorganize --> [T, G]
        if self.pattern == 'delay' and delay_back:
            group_ids = torch.stack([torch.roll(group_ids[d], -d) for d in range(self.group)])

        group_ids = group_ids.mT
        target_id = group_ids.reshape(-1)
        if add_sos: # add on time axis
            target_id = F.pad(target_id, (1, 0), 'constant', sos_id)
            
        if delete_null:
            target_id = target_id[target_id != self.null_id]
        return target_id
    

    def encode_token_emb(self, token_embeds):
        """
        token_embeds:
            - 3D: [G, T, D]
            - 4D: [B, G, T, D]
        """
        if token_embeds.ndim == 3:   # [G, T, D] => [T, D]
            if self.encoder_type == 'skip':   # [G, T, D] => [T, D]
                token_embeds = token_embeds[0]
            elif self.encoder_type == 'sum': # => [G, T, D] => [T, D]
                token_embeds = token_embeds.sum(dim=0)
            else:    # [G, T, D] => [T, G, D] => [T, G*D] => [T, D]
                token_embeds = self.encoder(token_embeds.transpose(1, 0).reshape(-1, self.group*self.embedding_dim))
        else:   # [B, G, T, D] => [B, T, D]
            if self.encoder_type =='skip':   # [B, T]
                token_embeds = token_embeds[:, 0]
            elif self.encoder_type =='sum':  # [B, T]
                token_embeds = token_embeds.sum(dim=1)
            else:   # [B, G, T, D] => [B, T, G, D] => [B, T, G*D] => [B, T, D]
                token_embeds = self.encoder(token_embeds.transpose(2, 1).reshape(token_embeds.shape[0], -1, self.group*self.embedding_dim))
        return token_embeds
 

    def group_token(self, this_token_ids, eos_id):
        """
        Input: 
            - this_token_ids: [T]
            - eos_id: int
        Return: 
            - group_token_ids: [G, T//G] (padding with eos_id)
        """
        if len(this_token_ids) % self.group != 0:
            this_token_ids = F.pad(this_token_ids, 
                            (0, self.group - len(this_token_ids) % self.group), 
                            'constant', eos_id)
        group_token_ids = this_token_ids.reshape(-1, self.group).mT
        
        if self.pattern == 'delay':
            group_token_ids = F.pad(group_token_ids, (self.group - 1, 0), 'constant', self.null_id)
            group_token_ids = torch.stack([torch.roll(group_token_ids[d], d + 1 - self.group) for d in range(self.group)])
            # add EOS tokens, this_token_ids: [G, T]
            if self.null_id in group_token_ids[:, -self.group:]:
                null_mask = group_token_ids[:, -self.group:] == self.null_id
                group_token_ids[:, -self.group:][null_mask] = eos_id

        # if this_token_ids.shape[1] == 0: 
            # this_token_ids = torch.ones(self.group, 1, dtype=torch.long, device=this_token_ids.device) * self.sos_id
        if sum(group_token_ids[:, -1] == eos_id) != self.group:
            group_token_ids = F.pad(group_token_ids, (0, 1), 'constant', eos_id)
        
        return group_token_ids


    def decode_hidden_state_inference(self, hidden_state, last_predict_token_emb=None, 
                                      frame_idx=None, decoder_embeds=None, cfg_path=None):
        """hidden_state: [B, T=1, D], last_predict_token_emb: [B, T=1, D]"""
        if self.decoder is None or self.decoder_type not in ['transformer', 'RQ_transformer', 'context_fc', 'ARQ_transformer']:
            return hidden_state
        model_input = {"inputs_embeds": None}
        batch_size = hidden_state.shape[0]

        def AR_decoder(model_input, hidden_state, last_predict_token_emb=None):
            batch_size = hidden_state.shape[0]
            if last_predict_token_emb is not None:
                model_input["inputs_embeds"] = torch.cat((last_predict_token_emb[:batch_size], hidden_state), 
                                                        dim=1)
            else:
                model_input["inputs_embeds"] = torch.cat((self.get_sos_embed(batch_size).to(self.decoder.lm_head.weight.dtype),
                                                        hidden_state), dim=1)
            decoder_output = self.decoder(**model_input, inference_params=self.inference_params,
                                        position_ids=None, last_token_only=False,)
            self.inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            # [B, T=2, G*D]
            target_logits = decoder_output.logits
            target_logits = target_logits[:, -1:]   # [B, T=1, G*D]
            return target_logits
        
        def context_fc(hidden_state, last_predict_token_emb):
            batch_size = hidden_state.shape[0]
            if last_predict_token_emb is None:
                decoder_input = torch.cat((hidden_state, self.get_sos_embed(batch_size)), dim=-1)
            else:
                decoder_input = torch.cat((hidden_state, last_predict_token_emb[:batch_size]), dim=-1)
            decoder_output = self.decoder(decoder_input)
            return decoder_output
        
        def RQ_transformer(model_input, hidden_state):
            print("RQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        def ARQ_transformer(model_input, hidden_state):
            print("ARQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        if frame_idx == 0:
            if self.decoder_type in ['transformer', 'RQ_transformer']:
                self.init_decoder(batch_size, decoder_embeds=decoder_embeds, cfg_path=cfg_path)
        
        if self.decoder_type == 'transformer':
            target_logits = AR_decoder(model_input, hidden_state, last_predict_token_emb)
        elif self.decoder_type == 'context_fc':
            target_logits = context_fc(hidden_state, last_predict_token_emb)

        return target_logits


    def decode_hidden_state(self, hidden_state, group_token_embeds, prefix_length, target_length=None,
                            group_token_ids=None, target_embedder=None):
        """
        This function is called during training (teacher-forcing) for AR-based decoder,
        for simplicity, the prefix length will be truncated.
        hidden_state: list [B] of [prefix_T + T//G, D] serves as condition to generate upcoming [group] tokens
        group_token_embeds: list [B] of [G, T//G, D]
        prefix_length: list [B] of int
        target_length: list [B] of int (!grouped_length!)
        group_token_ids: list [B] of [G, T//G]
        """
        def AR_decoder(hidden_state, token_embeds, prefix_length, target_length):
            B, T, D = hidden_state.size()
            hidden_state_expanded = hidden_state.unsqueeze(2)  # [B, T, 1, D]
            token_embeds_expanded = token_embeds.unsqueeze(2)  # [B, T, 1, D]
            # prefix_emb...; emb_sos; cond_{1~G},emb_{1~G}; cond_{G+1~2G},emb_{G+1~2G};...; cond_{T*G~(T+1)*G}
            group_embeds = torch.cat((token_embeds_expanded, hidden_state_expanded), dim=2) # [B, T, 2, D]

            # remove LM prefix (keep the sos_emb)
            group_embeds = [group_embeds[b][prefix_length[b]:].reshape(-1, D) for b in range(B)] # B x [2*T_b (interleaved style), D]
            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   

            # [B, 2T, G*D]
            target_logits = self.decoder(inputs_embeds=group_embeds).logits
            # [B, G, 2T, D]
            target_logits = torch.stack(torch.chunk(target_logits, chunks=self.group, dim=-1), dim=1)
            # outputs corresponding to cond_{1~G},emb_{1~G}; cond_{G+1~2G},emb_{G+1~2G};...; cond_{T*G~(T+1)*G}
            n_logits = target_logits.shape[-1]
            batch_target_logits = []
            for b in range(B):
                # shape: [G, 2(Tb), D], index: -1 to remove input shift (target_length - 1)
                this_target_logits = target_logits[b, :, :(target_length[b]-1)*2]
                this_target_logits = this_target_logits[:,1::2]  # [G, Tb, D]
                # G, 2T, n_logits => 2T, G, n_logits => (2T*G), n_logits
                _target_logits = this_target_logits.transpose(1, 0).reshape(-1, n_logits)
                batch_target_logits.append(torch.cat(
                    (torch.zeros(prefix_length[b], n_logits).to(_target_logits.device),
                    _target_logits,
                    ), dim=0))
            # [B, pT+T, n_logits]
            return batch_target_logits

        def RQ_transformer(
                hidden_state : List[torch.Tensor], # [1 (SOS) + Tb//G), D] * B => [Tb//G + 1 (EOS), D] * B
                token_embeds : List[torch.Tensor], # [G, Tb//G + 1 (EOS), D] * B
                # target_ids: List[torch.Tensor], #  [1 (SOS) + Tb + 2 or 3 (EOS)] * B
                prefix_length : torch.Tensor,
                target_length : torch.Tensor, # [1 (SOS) + Tb//G + 1 (EOS)] * B
                sos_embed     : torch.Tensor, # [1, D]
                ) -> List[torch.Tensor]: # [T//G + 1 (EOS), n_logits] * B
            """
            hidden_state (with prefix): [B, pT + T//G, D]
            """
            # hidden_state [prefix - 1 + 1(SOS) + T//G + 1(EOS), ...]
            # prefix_length: [prefix]
            # target_length [1(SOS) + T//G + 1(EOS)]
            # token_embeds [prefix + 1(SOS) + T//G , ...]
            # target_ids [G, 1(SOS) + T//G + 1(EOS)]
            B = len(hidden_state)
            group_embeds, batch_seq_lens = [], [0]
            # input: concatenation of [LM_cond, single_token_emb] (B*(T//G), time_steps=G, 2D)
            # output: token logits (B*(T//G), time_steps=G, n_logits)
            for b in range(B):
                # 1) prepare group token embeddings
                this_group_token_embeds = token_embeds[b].transpose(1, 0)   # [G, T//G, D] => [T//G, G, D]
                # shift group_token_embeds by 1: group_1,...,group_G => sos_embed, group_1,...,group_G-1
                this_group_token_embeds = torch.cat((
                                torch.cat([sos_embed]*this_group_token_embeds.shape[0], dim=0),
                                this_group_token_embeds[:, :-1]), dim=1)    # [T//G, G, D]
                
                # 2) prepare LM condition embeddings
                this_hidden_state = hidden_state[b][prefix_length[b]:prefix_length[b] + target_length[b] - 1]  # [T//G, D]
                this_group_hidden_state = this_hidden_state.unsqueeze(1).repeat(1, self.group, 1)   # [T//G, G, D]

                # 3) decoder inputs
                this_group_embeds = torch.cat((this_group_hidden_state, this_group_token_embeds), -1)   # [T//G, G, 2D]
                group_embeds.append(this_group_embeds)
                batch_seq_lens.append(this_group_token_embeds.shape[0]+batch_seq_lens[-1])

            group_embeds = torch.cat(group_embeds, 0)   # [B x {T//G}, G, 2D]
            target_logits = self.decoder(inputs_embeds=group_embeds).logits  # [B x {T//G}, G, n_logits]
            
            n_logits = target_logits.shape[-1]  # no RQ_output layer, for VQ only
            target_logits_list = []
            for b in range(B):
                # [{T//G}, G, n_logits] => [T, n_logits] (remove prefix)
                this_target_logits = target_logits[batch_seq_lens[b]:batch_seq_lens[b+1], :].reshape(-1, n_logits)
                target_logits_list.append(this_target_logits)
            return target_logits_list
        
        def ARQ_transformer(
                hidden_state : List[torch.Tensor], # [1 (SOS) + Tb//G), D] * B => [Tb//G + 1 (EOS), D] * B
                token_embeds : List[torch.Tensor], # [G, Tb//G + 1 (EOS), D] * B
                # target_ids: List[torch.Tensor], #  [1 (SOS) + Tb + 2 or 3 (EOS)] * B
                prefix_length : torch.Tensor,
                target_length : torch.Tensor, # [1 (SOS) + Tb//G + 1 (EOS)] * B
                sos_embed     : torch.Tensor, # [1, D]
                ) -> List[torch.Tensor]: # [T//G + 1 (EOS), n_logits] * B
            B = len(hidden_state)
            D = hidden_state[0].shape[-1]
            group_embeds, batch_seq_lens = [], [0]
            # prefix_emb...; emb_sos; cond_{1~G},emb_1,...,emb_G; 
            #                         cond_{G+1~2G},emb_{G+1}...,emb_{2G}; ...; 
            #                         cond_{T*G~(T+1)*G},emb_{T*G}...,emb_{(T+1)*G}
            for b in range(B):
                # 1) prepare group token embeddings
                this_group_token_embeds = token_embeds[b].transpose(1, 0)   # [G, T//G, D] => [T//G, G, D]

                # 2) prepare LM condition embeddings
                this_hidden_state = hidden_state[b][prefix_length[b]:prefix_length[b] + target_length[b] - 1]  # [T//G, D]
                this_group_hidden_state = this_hidden_state.unsqueeze(1)   # [T//G, 1, D] (exclude sos_embed)

                # 3) insert LM into group token embeddings
                assert this_group_token_embeds.shape[0] == this_group_hidden_state.shape[0]
                this_group_embeds = torch.cat((this_group_hidden_state, this_group_token_embeds), dim=1)   # [T//G, G+1, D]
                this_group_embeds = torch.cat((sos_embed.squeeze(0), this_group_embeds.reshape(-1, D)), dim=0) # 1+T//G*(G+1), D

                group_embeds.append(this_group_embeds)  
                batch_seq_lens.append(this_group_embeds.shape[0]+batch_seq_lens[-1])
            
            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   # [B, 1+(T//G)*(G+1), D]
            target_logits = self.decoder(inputs_embeds=group_embeds).logits     # [B, 1+(T//G)*(G+1), D]

            n_logits = target_logits.shape[-1]  # no RQ_output layer, for VQ only
            target_logits = target_logits[:, 1:].reshape(B, -1, self.group+1, n_logits)   # [B, T//G, G+1, N]
            batch_target_logits = []
            for b in range(B):
                # [T//G, G+1, N] => [T//G, G, N] => [T, N]
                this_target_logits = target_logits[b, :target_length[b]-1, :self.group].reshape(-1, n_logits)  # 
                batch_target_logits.append(this_target_logits)
            return batch_target_logits

        def context_fc_decoder(hidden_state, token_embeds, prefix_length, target_length):
            # remove LM prefix (keep the sos_emb)
            B = len(hidden_state)
            group_embeds = [torch.cat((hidden_state[b][prefix_length[b]:], token_embeds[b][prefix_length[b]:]), -1)
                             for b in range(B)] # B x [T_b, 2D (concat style)]
            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   
            
            target_logits = self.decoder(group_embeds)  # [B, T, N]
            target_logits = torch.stack(torch.chunk(target_logits, chunks=self.group, dim=-1), dim=1)   # [B, G, T//G, N]
            # outputs corresponding to cond_{1~G},emb_{1~G}; cond_{G+1~2G},emb_{G+1~2G};...; cond_{T*G~(T+1)*G}
            n_logits = target_logits.shape[-1]
            batch_target_logits = []
            for b in range(B):
                # [G, T//G, N] => [T//G, G, N] => [T, N]
                this_target_logits = target_logits[b, :, :target_length[b]-1].transpose(1, 0).reshape(-1, n_logits)   
                batch_target_logits.append(this_target_logits)
            return batch_target_logits

        assert len(hidden_state) == prefix_length.shape[0], f"batch size mismatch {len(hidden_state)=} {prefix_length.shape=}"
        assert len(hidden_state) == target_length.shape[0], f"batch size mismatch {len(hidden_state)=} {target_length.shape=}"
        assert len(hidden_state) == len(group_token_embeds), f"batch size mismatch {hidden_state.shape=} {len(group_token_embeds)=}"
        prefix_length = prefix_length.long()

        assert group_token_embeds is not None
        if self.pattern != 'parallel':
            raise NotImplementedError()
        
        sos_embed = target_embedder.get_sos_embed(1)
        if self.decoder_type == 'transformer':
            assert target_length is not None
            batch_target_logits = AR_decoder(hidden_state, group_token_embeds, prefix_length, target_length)
        elif self.decoder_type == 'context_fc':
            batch_target_logits = context_fc_decoder(hidden_state, group_token_embeds, prefix_length, target_length)
        elif self.decoder_type == 'RQ_transformer':
            batch_target_logits = RQ_transformer(
                    hidden_state=hidden_state, 
                    token_embeds=group_token_embeds, 
                    prefix_length=prefix_length, 
                    target_length=target_length, 
                    sos_embed=sos_embed,
                    )
        elif self.decoder_type == 'ARQ_transformer':
            batch_target_logits = ARQ_transformer(hidden_state, group_token_embeds, prefix_length, target_length, sos_embed)
        
        return batch_target_logits
    
    def enable_kv_cache(self, batch_size=None, target_embedder=None):
        self.decoder.use_flash_attn_kvcache = True # infer with flash attn
        self.decoder.transformer.use_flash_attn_kvcache = True # infer with flash attn

        if self.decoder_type == 'ARQ_transformer':
            batch_size = 2 if batch_size is None else batch_size
            self.inference_params = InferenceParams(
                max_sequence_len=int(8000/self.group*(self.group+1)), 
                max_batch_size=batch_size)
            sos_embed = target_embedder.get_sos_embed(batch_size).to(self.decoder.lm_head.weight.dtype)
            decoder_out = self.decoder(inputs_embeds=sos_embed, inference_params=self.inference_params,
                                        position_ids=None, last_token_only=False,)
            self.inference_params.sequence_len_offset += sos_embed.size(1)

    def predict(self, hidden_states, mtp_pattern, target_embedder, sample_single_token_func, use_controller_cfg, n_cfg_path, **kwargs):
        def RQ_transformer(hidden_state):
            batch_size, t, d = hidden_state.shape  # (cfg_batch_size + 1)
            # initialize input parameters
            decoder_output = None
            inference_params = InferenceParams(max_sequence_len=self.group+1, max_batch_size=batch_size)
            last_predict_token_emb = target_embedder.get_sos_embed(batch_size)
            pred_tokens = []
            for g in range(self.group):
                # [B, T=1, 2*D]
                inputs_embeds = torch.cat((hidden_state, last_predict_token_emb), dim=-1).reshape(batch_size*1, -1).unsqueeze(1)
                decoder_output = self.decoder(
                        inputs_embeds=inputs_embeds, 
                        inference_params=inference_params,
                        )
                inference_params.sequence_len_offset += inputs_embeds.size(1)
                target_logits = decoder_output.logits

                # [B, T=1, N] => [B, T=1]
                predict_token = sample_single_token_func(target_logits)
                last_predict_token_emb = target_embedder.embedder(predict_token)
                
                if use_controller_cfg:
                    last_predict_token_emb = last_predict_token_emb.repeat(n_cfg_path + 1, 1, 1)
                pred_tokens.append(predict_token)
            return torch.cat(pred_tokens, -1)   # [B, T=G]

        def ARQ_transformer(hidden_state):
            # cond_{1...G}, token_1, token_2, ..., token_G
            batch_size = hidden_state.shape[0]
            last_predict_token_emb = hidden_state   # [B, T=1, D]
            pred_tokens = []
            for g in range(self.group + 1):
                decoder_output = self.decoder(inputs_embeds=last_predict_token_emb, 
                                            inference_params=self.inference_params,
                                            position_ids=None, last_token_only=False,)
                self.inference_params.sequence_len_offset += last_predict_token_emb.size(1)
                if g == self.group: # only inputs the last predicted token (token_G) without obtaining its output
                    break
                target_logits = decoder_output.logits
                # [B, T=1, N] => [B, T=1]
                predict_token = sample_single_token_func(target_logits)
                last_predict_token_emb = target_embedder.embedder(predict_token)

                if use_controller_cfg:
                    last_predict_token_emb = last_predict_token_emb.repeat(n_cfg_path + 1, 1, 1)
                pred_tokens.append(predict_token)
            return torch.cat(pred_tokens, -1)   # [B, T=G]

        if mtp_pattern == 'parallel-rq':
            pred_tokens = RQ_transformer(hidden_states)
        elif mtp_pattern == 'parallel-arq':
            pred_tokens = ARQ_transformer(hidden_states)
        else:
            raise NotImplementedError()
        return pred_tokens


class HierarchicalTokenPredictionModule(MultiTokenPredictionModule):
    def __init__(self,
        vocab_size: int = 32_768,
        embedding_dim: int = 1024,
        pattern: str = '2D-delay',
        group: int = 2,
        encoder: str = 'fc',
        decoder: str = 'fc',
        codebook_depth: list[int] = [1, 1],
        criterion: Optional[nn.Module] = None,
        token_embed_dim: Optional[int] = None,
        quantizer_dropout: Optional[float] = 0.0,
        condition_dropout: Optional[float] = 0.0,
        condition_merge_method: Optional[str] = "concat",
        **kwargs,
    ):
        nn.Module.__init__(self)

        if isinstance(codebook_depth, list):
            self.R, self.input_R = codebook_depth[0], codebook_depth[1]
        else:
            self.R = codebook_depth
            self.input_R = self.R
        
        self.token_embed_dim = token_embed_dim
        self.condition_dropout = condition_dropout
        self.quantizer_dropout = quantizer_dropout
        self.group = int(group)
        self.pattern = pattern
        self.condition_merge_method = condition_merge_method
        self.embedding_dim = embedding_dim

        self.use_lm_head_output = kwargs.get("use_lm_head_output", False)
        if self.use_lm_head_output:
            self.lm_head_norm = nn.LayerNorm(embedding_dim)

        ## ======= Encoder ======== ##
        self.encoder_type = encoder
        # 2D Encoder [group x input_R]
        self.encoder = nn.Sequential(
            nn.Linear(token_embed_dim * self.group, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim))
        
        ## ======= Decoder ======== ##
        self.decoder_type = decoder["model_type"].split("#")[0]
        self.decoder_cinfo = None if len(decoder["model_type"].split("#")) < 2 else decoder["model_type"].split("#")[1]
        assert self.decoder_type in ["RQ_transformer", "ARQ_transformer", "context_RQ_transformer"]
        self.decoder = GPTLMHeadModel(decoder['hp'])
        self.decoder_hidden_dim = decoder['encoder_dim']

        ## ======= Embedding mapper ======== # 
        # RQ_input: embedding2latent ([1, 32]->[1, D]) for each input_R in decoder (input_R/R)
        self.RQ_input = nn.Linear(self.R * token_embed_dim, self.decoder_hidden_dim) #nn.ModuleList([]) 
        # RQ_output: latent2logits ([1, D]->[1, n_logits]) for each output_R (R)
        self.RQ_output = nn.ModuleList([])
        for r in range(self.R):
            self.RQ_output.append(nn.Sequential(
                nn.LayerNorm(self.decoder_hidden_dim),
                nn.Linear(self.decoder_hidden_dim, vocab_size + 2))) 
            
        ## ======= Criterion ======== ##
        self.criterion = None
        if criterion is not None:
            self.criterion = criterion()

        ## ======= Conditioning ====== #
        if self.condition_merge_method == "add":
            self.decoder_sos_embed = nn.Embedding(1, self.decoder_hidden_dim)
        
        if self.decoder_type == 'context_RQ_transformer':
            self.RQ_encoder = None
            if self.decoder_cinfo is None:
                self.RQ_encoder = nn.Linear(token_embed_dim, self.decoder_hidden_dim, bias=False)
            elif self.embedding_dim != self.decoder_hidden_dim:
                self.RQ_encoder = nn.Linear(self.embedding_dim, self.decoder_hidden_dim, bias=False)
        
    def group_token(self, this_token_ids, eos_id):
        """
        Input: 
            - this_token_ids: [T, R]
            - eos_id: int
        Return: 
            - group_token_ids: [G, T//G, R] (padding with eos_id)
        """
        if this_token_ids.shape[-1] > self.R:
            this_token_ids = this_token_ids[..., :self.R]

        if len(this_token_ids) % self.group != 0:
            this_token_ids = F.pad(this_token_ids, 
                                   (0, 0, 0, self.group - len(this_token_ids) % self.group), 
                                   'constant', eos_id)  
        group_token_ids = this_token_ids.reshape(-1, self.group, self.R) # [T//G, G, R]

        # pad full eos_id 
        if (group_token_ids[-1] == eos_id).sum() != (self.group * self.R):
            group_token_ids = F.pad(group_token_ids, (0, 0, 0, 0, 0, 1), 'constant', eos_id)
        
        return group_token_ids.transpose(1, 0)

    def encode_token_emb(self, group_token_embeds, eos_mask=None, eos_embeds=None):
        """
        Input: 
            - group_token_embeds: [*, G, T//G, token_embed_dim]
            - eos_mask:           [*, G, T//G, input_R]
            - eos_embeds:         [*, T//G, 1, D]
        Return: 
            - enc_token_embeds:   [*, T//G, D]
        """
        group_token_embeds = group_token_embeds.transpose(-3, -2)   # [*, T//G, G, token_embed_dim]

        # encoder operate on G: [*, T//G, G, 32] => [*, T//G, D]
        group_token_embeds = group_token_embeds.reshape(*group_token_embeds.shape[:-2], self.group * self.token_embed_dim)
        enc_token_embeds = self.encoder(group_token_embeds)   # [*, T//G, D]

        if eos_mask is not None and eos_embeds is not None:
            # replace eos embeds
            eos_mask = eos_mask.transpose(-3, -2)   # [*, T//G, G, input_R]
            eos_mask_t = (eos_mask[..., :, 0].sum(-1) > 0).unsqueeze(-1) # [*, T//G, 1]
            eos_embeds = eos_embeds.squeeze(-2) # [*, T//G, D]
            enc_token_embeds = eos_mask_t * enc_token_embeds + (~eos_mask_t) * eos_embeds # [*, T//G, D]
        return enc_token_embeds
    
    def ungroup_token(self, group_ids, sos_id, eos_id, delete_null=False, add_sos=True, delay_back=False):
        """
        Input:
            group_ids: [G, T//G, R]
        Return:
            target_ids: [T, R]
        """
        # [G, T//G, R] => [T//G, G, R] => [T, R]
        target_id = group_ids.transpose(1, 0).reshape(-1, self.R)   # [T, R]

        if add_sos: # add on time axis
            target_id = F.pad(target_id, (0, 0, 1, 0), 'constant', sos_id)

        if (target_id[-1] == eos_id).sum() != self.R:
            target_id[-1] = eos_id
        return target_id
    

    def decode_hidden_state_inference(self, hidden_state, last_predict_token_emb=None, 
                                      frame_idx=None, decoder_embeds=None, cfg_path=None):
        """hidden_state: [B, T=1, D], last_predict_token_emb: [B, T=1, D]"""
        if self.decoder is None or self.decoder_type not in ['transformer', 'RQ_transformer', 'context_fc', 'ARQ_transformer']:
            return hidden_state
        model_input = {"inputs_embeds": None}
        batch_size = hidden_state.shape[0]

        def AR_decoder(model_input, hidden_state, last_predict_token_emb=None):
            batch_size = hidden_state.shape[0]
            if last_predict_token_emb is not None:
                model_input["inputs_embeds"] = torch.cat((last_predict_token_emb[:batch_size], hidden_state), 
                                                        dim=1)
            else:
                model_input["inputs_embeds"] = torch.cat((self.get_sos_embed(batch_size).to(self.decoder.lm_head.weight.dtype),
                                                        hidden_state), dim=1)
            decoder_output = self.decoder(**model_input, inference_params=self.inference_params,
                                        position_ids=None, last_token_only=False,)
            self.inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            # [B, T=2, G*D]
            target_logits = decoder_output.logits
            target_logits = target_logits[:, -1:]   # [B, T=1, G*D]
            return target_logits
        
        def context_fc(hidden_state, last_predict_token_emb):
            batch_size = hidden_state.shape[0]
            if last_predict_token_emb is None:
                decoder_input = torch.cat((hidden_state, self.get_sos_embed(batch_size)), dim=-1)
            else:
                decoder_input = torch.cat((hidden_state, last_predict_token_emb[:batch_size]), dim=-1)
            decoder_output = self.decoder(decoder_input)
            return decoder_output
        
        def RQ_transformer(model_input, hidden_state):
            print("RQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        def ARQ_transformer(model_input, hidden_state):
            print("ARQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        if frame_idx == 0:
            if self.decoder_type in ['transformer', 'RQ_transformer']:
                self.init_decoder(batch_size, decoder_embeds=decoder_embeds, cfg_path=cfg_path)
        
        if self.decoder_type == 'transformer':
            target_logits = AR_decoder(model_input, hidden_state, last_predict_token_emb)
        elif self.decoder_type == 'context_fc':
            target_logits = context_fc(hidden_state, last_predict_token_emb)

        return target_logits

    def decode_hidden_state(self, hidden_state, group_token_embeds, prefix_length, target_length=None,
                            group_token_ids=None, target_embedder=None):
        """
        This function is called during training (teacher-forcing) for AR-based decoder,
        for simplicity, the prefix length will be truncated.
        hidden_state: list [B] of [prefix_T + T//G, D] serves as condition to generate upcoming [group] tokens
        group_token_embeds: list [B] of [G, T//G, token_embed_dim]
        group_token_ids: list [B] of [G, T//G, R]
        prefix_length: list [B] of int
        target_length: list [B] of int (!grouped_length!)
        """
        assert len(hidden_state) == prefix_length.shape[0], f"batch size mismatch {len(hidden_state)=} {prefix_length.shape=}"
        assert len(hidden_state) == target_length.shape[0], f"batch size mismatch {len(hidden_state)=} {target_length.shape=}"
        assert len(hidden_state) == len(group_token_embeds), f"batch size mismatch {hidden_state.shape=} {len(group_token_embeds)=}"
        # assert self.null_id == target_embedder.null_id, f"null_id mismatch {self.null_id=} {target_embedder.null_id=}"
        null_id = target_embedder.null_id
        prefix_length = prefix_length.long()
        
        def prepare_delayed_embeds(this_group_token_ids):
            """
            Input:
                - this_group_token_ids: [T//G, G, R]
            Return: 
                - this_decoder_input: [T//G, G+R-1, R, D]
            """
            # 1) apply delay pattern to token_ids 
            this_group_token_ids = F.pad(this_group_token_ids, (0, 0, self.R - 1, 0), 'constant', null_id) # [T//G, G+R-1, R]
            this_group_token_delayed_ids = this_group_token_ids.transpose(-2, -1)  # [T//G, R, G+R-1]
            this_group_token_delayed_ids = torch.stack([torch.roll(this_group_token_delayed_ids[:,d], d + 1 - self.R, dims=1) 
                                                        for d in range(self.R)], dim=1)  # [T//G, R, G+R-1]
            # 2) prepare decoder input embedding
            this_group_token_delayed_embeds = torch.stack([target_embedder.embedder[r](this_group_token_delayed_ids[...,r,:]) 
                                                           for r in range(self.R)], dim=-2)  # [T//G, G+R-1, R, 32]
            # 3) mask null_id embeds with full zeros
            null_id_mask = (this_group_token_delayed_ids.mT == null_id).unsqueeze(-1)   # [T//G, G+R-1, R, 1]
            this_RQ_input = this_group_token_delayed_embeds * (~null_id_mask).to(this_group_token_delayed_embeds)   # [T//G, G+R-1, R, 32]
            this_RQ_input = this_RQ_input.reshape(-1, self.group+self.R-1, self.R*self.token_embed_dim) # [T//G, G+R-1, R*32]
            this_decoder_input = self.RQ_input(this_RQ_input)  # [T//G, G+R-1, D]

            return this_decoder_input

        def prepare_context_embeds(this_group_token_ids):
            """
            Input:
                - this_group_token_ids: [T//G, G, R]
            Return: 
                - decoder_context_seq_len: int, length of context embeds
                - this_group_token_input: [1+T//G-1, decoder_context_seq_len, D]
            """
            this_group_token_embeds = torch.stack([target_embedder.embedder[r](this_group_token_ids[...,r]) 
                                                   for r in range(self.R)], dim=-2) # [T//G, G, R, 32]
            decoder_context_seq_len = self.group
            if self.decoder_cinfo is None:
                this_group_token_embeds = this_group_token_embeds.sum(-2)   # [T//G, G, 32]
                this_group_token_input = self.RQ_encoder(this_group_token_embeds) # [T//G, G, D]
            else:
                decoder_context_seq_len = 1
                if self.decoder_cinfo == 'R1':
                    this_group_token_embeds = this_group_token_embeds[...,0,:]   # [T//G, G, 32]
                else:
                    this_group_token_embeds = this_group_token_embeds.sum(-2)   # [T//G, G, 32]
                this_group_token_input = self.encoder(this_group_token_embeds.reshape(-1, self.group*self.token_embed_dim)) # [T//G, D]
                if self.RQ_encoder is not None:
                    this_group_token_input = self.RQ_encoder(this_group_token_input)
                this_group_token_input = this_group_token_input.unsqueeze(1)    # [T//G, 1, D]
            this_group_token_input = torch.cat((
                torch.zeros_like(this_group_token_input)[:1],
                this_group_token_input[:-1]), dim=0)    # [1+(T//G-1), G, D]
            return this_group_token_input, decoder_context_seq_len


        B = len(hidden_state)
        decoder_inputs, batch_seq_lens = [], [0]
        decoder_condition_len = 1   # by default: LM condition only
        decoder_delay_ptn_len = self.group + self.R - 1
        for b in range(B):
            # 1) prepare group token embeddings
            this_group_token_ids = group_token_ids[b].transpose(1, 0) # [T//G, G, R] no sos_id; with all eos_id
            # # apply delay pattern to token_ids 
            # this_group_token_ids = F.pad(this_group_token_ids, (0, 0, self.R - 1, 0), 'constant', null_id) # [T//G, G+R-1, R]
            # this_group_token_delayed_ids = this_group_token_ids.transpose(-2, -1)  # [T//G, R, G+R-1]
            # this_group_token_delayed_ids = torch.stack([torch.roll(this_group_token_delayed_ids[:,d], d + 1 - self.R, dims=1) 
            #                                             for d in range(self.R)], dim=1)  # [T//G, R, G+R-1]

            # # prepare decoder input embedding
            # this_group_token_delayed_embeds = torch.stack([target_embedder.embedder[r](this_group_token_delayed_ids[...,r,:]) 
            #                                                for r in range(self.R)], dim=-2)  # [T//G, G+R-1, R, 32]
            # # mask null_id embeds with full zeros
            # null_id_mask = (this_group_token_delayed_ids.mT == null_id).unsqueeze(-1)   # [T//G, G+R-1, R, 1]
            # this_RQ_input = this_group_token_delayed_embeds * (~null_id_mask).to(this_group_token_delayed_embeds)   # [T//G, G+R-1, R, 32]
            # this_RQ_input = this_RQ_input.reshape(-1, self.group+self.R-1, self.R*self.token_embed_dim) # [T//G, G+R-1, R*32]
            this_decoder_input = prepare_delayed_embeds(this_group_token_ids)

            # 2) prepare context embeds (if any)
            if self.decoder_type == 'context_RQ_transformer':
                this_context_input, decoder_context_len = prepare_context_embeds(this_group_token_ids)
                this_decoder_input = torch.cat((this_context_input, this_decoder_input), dim=1)   # [T//G, 1+G+R-1, D]
                decoder_condition_len = decoder_context_len + 1

            # 3) prepare LM condition embeddings
            this_hidden_states = hidden_state[b][prefix_length[b]:prefix_length[b] + (target_length[b]-1)].unsqueeze(1) # [T//G, 1, D]
            if self.condition_merge_method == "add":
                sos_embed = self.decoder_sos_embed(torch.LongTensor([0]).to(this_hidden_states.device)).unsqueeze(0).repeat(this_decoder_input.shape[0], 1, 1)            # [1, 1, D]
                this_decoder_input = this_hidden_states + torch.cat((sos_embed, this_decoder_input), dim=1)    # [T//G, 1+G+R-1, D]
            else:
                this_decoder_input = torch.cat((this_hidden_states, this_decoder_input), dim=1)   # [T//G, 1+G+R-1, D]

            # 3) determine decoder input according to decoder type
            if self.decoder_type in ["RQ_transformer", "context_RQ_transformer"]:
                this_decoder_input = this_decoder_input.reshape(-1, decoder_condition_len+decoder_delay_ptn_len, self.decoder_hidden_dim)[:,:-1]  # [T//G, G+R-1, D]
            elif self.decoder_type == "ARQ_transformer":
                this_decoder_input = this_decoder_input.reshape(-1, self.decoder_hidden_dim)[:,:-1]  # [(T//G)*(G+R)-1, D]

            decoder_inputs.append(this_decoder_input)
            batch_seq_lens.append(this_decoder_input.shape[0] + batch_seq_lens[-1])

        if self.decoder_type in ["RQ_transformer", "context_RQ_transformer"]:
            decoder_inputs = torch.cat(decoder_inputs, dim=0)    # [batch_axis: B*(T//G), time_axis: G+R-1, feat_axis: D]
        elif self.decoder_type == "ARQ_transformer":
            decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=0) # [batch_axis: B, time_axis: (T//G)*(G+R)-1, feat_axis: D]
        decoder_outputs = self.decoder(inputs_embeds=decoder_inputs).logits     

        RQ_outputs = torch.stack([self.RQ_output[r](decoder_outputs) for r in range(self.R)], dim=-2)   
        n_logits = RQ_outputs.shape[-1]

        batch_target_logits = []
        offset = decoder_condition_len - 1
        for b in range(B):
            if self.decoder_type in ["RQ_transformer", "context_RQ_transformer"]:
                this_RQ_outputs = RQ_outputs[batch_seq_lens[b]:batch_seq_lens[b+1]] # [Tb//G, G+R-1, R, N]
                this_target_logits = this_RQ_outputs.reshape(-1, decoder_condition_len+decoder_delay_ptn_len-1, self.R, n_logits)   # [Tb//G, G+R-1, R, N]=
            elif self.decoder_type == "ARQ_transformer":
                this_RQ_outputs = RQ_outputs[b, :target_length[b]-1] # [(T//G)*(G+R)-1, R, N]
                this_RQ_outputs_pad = F.pad(this_RQ_outputs, (0, 0, 0, 0, 0, 1), 'constant', 0) # [(T//G)*(G+R)-1+1, R, N]
                this_target_logits = this_RQ_outputs_pad.reshape(-1, self.group+self.R, self.R, n_logits) # [T//G, G+R, R, N]

            valid_logits = torch.stack([this_target_logits[:,r+offset:r+offset+self.group,r,:] for r in range(self.R)], dim=-2)   # [Tb//G, G, R, N]
            valid_logits = valid_logits.reshape(-1, self.R, n_logits)   # [Tb//G*G, R, N]
            batch_target_logits.append(valid_logits)    # [T, R, N]
        return batch_target_logits
    
    def initialize_valid_2Dmap(self,):
        map2D = torch.ones([self.group, self.R])
        map2D = F.pad(map2D, (0, 0, self.R - 1, 0), "constant", 0) # [G+R-1, R]
        map2D = map2D.mT    # [R, G+R-1]

        delayed_map2D = torch.stack([torch.roll(map2D[d], d + 1 - self.R, dims=0) for d in range(self.R)], dim=1)  # [G+R-1, R]
        self.delayed_map2D = delayed_map2D.long()   # [G+R-1, R]
        print("2D map is initialized", self.delayed_map2D)

    def enable_kv_cache(self, **kwargs):
        self.initialize_valid_2Dmap()
        super().enable_kv_cache(**kwargs)

    # TODO (@qinxin)
    def predict(self, hidden_states, mtp_pattern, target_embedder, use_controller_cfg, n_cfg_path, sample_single_token_func, **kwargs):
        def token_delay_back(pred_tokens):
            pred_tokens = pred_tokens.mT    # [..., R, G+R-1]
            pred_tokens = torch.stack([torch.roll(pred_tokens[...,r,:], self.R - 1 - r, dims=-1) for r in range(self.R)], dim=-1) # [..., G+R-1, R] 
            pred_tokens = pred_tokens[...,self.R - 1:,:]  # [B, 1, G, R]
            return pred_tokens


        def RQ_delay_transformer(hidden_state):
            batch_size, T, D = hidden_state.shape  # (cfg_batch_size + 1)
            G, R = self.group, self.R

            if self.condition_merge_method == "add":
                sos_embed = self.decoder_sos_embed(torch.LongTensor([0])).to(hidden_state.device).unsqueeze(0).repeat(hidden_state.shpae[0], 1, 1)            # [1, 1, D]
                last_RQ_input = hidden_state + sos_embed    # [B, T=1, D]
            else:
                last_RQ_input = hidden_state    # [B, T=1, D]

            # initialize input parameters
            inference_params = InferenceParams(max_sequence_len=G+R-1, max_batch_size=batch_size)
            pred_tokens = torch.ones([batch_size//(n_cfg_path+1), T, G + R - 1, R], dtype=torch.long).to(hidden_state.device) * -1
            for g in range(G + R - 1):
                # prepare decoder input [B, T=1, D]
                decoder_output = self.decoder(inputs_embeds=last_RQ_input, inference_params=inference_params)
                inference_params.sequence_len_offset += last_RQ_input.size(1)
                decoder_output = decoder_output.logits

                # [B, T=1, R, 32]
                this_RQ_output = torch.stack([self.RQ_output[r](decoder_output) for r in range(R)], dim=-2)
                this_token_embeds = torch.zeros(batch_size//(n_cfg_path+1), T, R, self.token_embed_dim).to(hidden_state.device)
                for r in range(R):
                    if self.delayed_map2D[g,r] > 0:
                        this_predict = sample_single_token_func(this_RQ_output[..., r, :], r_idx=r)
                        pred_tokens[...,g,r] = this_predict
                        this_token_embeds[..., r,:] = target_embedder.embedder[r](pred_tokens[...,g,r])

                # prepare last_predict_token_emb 
                last_RQ_input = self.RQ_input(this_token_embeds.reshape(-1, T, R*self.token_embed_dim))
                if self.condition_merge_method == "add":
                    last_RQ_input = hidden_state + last_RQ_input
                if use_controller_cfg:
                    last_RQ_input = last_RQ_input.repeat(n_cfg_path+1, 1, 1)
            return token_delay_back(pred_tokens)
        
        def context_RQ_delay_transformer(hidden_state, last_predict_token):
            """
            hidden_state: [B, T=1, D]
            last_predict_token: [B, T=1, G, R]
            Return: [B, decoder_context_len, D]
            """
            def prepare_context_embeds(last_predict_token, batch_size):
                if last_predict_token is None:
                    decoder_context_len = self.group if self.decoder_cinfo is None else 1
                    last_context_input = torch.zeros(batch_size, decoder_context_len, hidden_state.shape[-1]).to(hidden_state)
                else:
                    last_frame_token_emb = torch.stack([target_embedder.embedder[r](last_predict_token[...,r]) for r in range(self.R)], dim=-2) # [B, T=1, G, R, 32]
                    if self.decoder_cinfo is None:
                        last_frame_token_emb = last_frame_token_emb.sum(-2)
                        last_context_input = (self.RQ_encoder(last_frame_token_emb)).squeeze(1)   # [B, G, D]
                    else:
                        last_frame_token_emb = last_frame_token_emb[..., 0, :] if self.decoder_cinfo == 'R1' else last_frame_token_emb.sum(-2)
                        last_context_input = self.encoder(last_frame_token_emb.reshape(-1, self.token_embed_dim*self.group))
                        last_context_input = last_context_input.unsqueeze(1)   # [B, 1, D]
                        if self.RQ_encoder is not None:
                            last_context_input = self.RQ_encoder(last_context_input)

                    if last_context_input.shape[0] < batch_size:
                        last_context_input = last_context_input.repeat(n_cfg_path+1, 1, 1)  # [B, T=1, D]
                return last_context_input
            
            batch_size, T, D = hidden_state.shape
            G, R = self.group, self.R
            if self.condition_merge_method == "add":
                sos_embed = self.decoder_sos_embed(torch.LongTensor([0])).to(hidden_state.device).unsqueeze(0).repeat(hidden_state.shpae[0], 1, 1)            # [1, 1, D]
                last_RQ_input = hidden_state + sos_embed    # [B, T=1, D]
            else:
                last_RQ_input = hidden_state    # [B, T=1, D]
            last_context_input = prepare_context_embeds(last_predict_token, batch_size)
            last_RQ_input = torch.cat((last_RQ_input, last_context_input), dim=1) # [B, 1+G, D]

            inference_params = InferenceParams(max_sequence_len=2*G+R-1, max_batch_size=batch_size)
            pred_tokens = torch.ones([batch_size//(n_cfg_path+1), T, G + R - 1, R], dtype=torch.long).to(hidden_state.device) * -1
            for g in range(G + R - 1):
                # prepare decoder input [B, T=1, D]
                decoder_output = self.decoder(inputs_embeds=last_RQ_input, inference_params=inference_params)
                inference_params.sequence_len_offset += last_RQ_input.size(1)
                decoder_output = decoder_output.logits
                decoder_output = decoder_output[:, -1:]  # [B, T=1, D]

                # [B, T=1, R, 32]
                this_RQ_output = torch.stack([self.RQ_output[r](decoder_output) for r in range(R)], dim=-2)
                this_token_embeds = torch.zeros(batch_size//(n_cfg_path+1), T, R, self.token_embed_dim).to(hidden_state.device)
                for r in range(R):
                    if self.delayed_map2D[g,r] > 0:
                        this_predict = sample_single_token_func(this_RQ_output[..., r, :], r_idx=r)
                        pred_tokens[...,g,r] = this_predict
                        this_token_embeds[..., r,:] = target_embedder.embedder[r](pred_tokens[...,g,r])

                # prepare last_predict_token_emb 
                last_RQ_input = self.RQ_input(this_token_embeds.reshape(-1, T, R*self.token_embed_dim))
                if self.condition_merge_method == "add":
                    last_RQ_input = hidden_state + last_RQ_input
                if use_controller_cfg:
                    last_RQ_input = last_RQ_input.repeat(n_cfg_path+1, 1, 1)
                
            # delay back
            return token_delay_back(pred_tokens)

        def ARQ_delay_transformer(hidden_state):
            raise NotImplementedError("ARQ_delay_transformer is not implemented")
        
        if mtp_pattern == 'delay-2d-rq':
            pred_tokens = RQ_delay_transformer(hidden_states)
        elif mtp_pattern == 'delay-2d-arq':
            pred_tokens = ARQ_delay_transformer(hidden_states)
        elif mtp_pattern == 'delay-2d-context-rq':
            pred_tokens = context_RQ_delay_transformer(hidden_states, kwargs.get("last_predict_token", None))
        else:
            raise NotImplementedError()
        return pred_tokens
