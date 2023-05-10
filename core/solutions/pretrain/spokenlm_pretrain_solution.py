"""best-rq pretrain"""
import contextlib
import logging
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from core.models.pretrained.acoustic_tokenizer import *
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.pretrained.data2vec_model import PretrainedData2Vec
from core.criterions.criterion import Xentropy, SpokenLlmAffixXentropy
from core.solutions.base_solution import BaseSolution, register_solution
from core.criterions.spokenlm_e2e_criterion import SpokenLmE2eCriterion
from core.solutions.asr.base_rnnt_model import BaseRnntModel
from core.extensions import mpu


@register_solution("SpokenLmPretrainModel")
class SpokenLmPretrainModel(BaseSolution):
    """SpokenLmPretrainModel model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args

        # Configure
        self.token_combination_mode = args.get('token_combination_mode', "simple_asr")

        # Acoustic Tokenizer Frontend
        self.acoustic_front_end_module = None
        if eval(args.front_end_type) is not None:
            self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_tokenizer = eval(args.acoustic_tokenizer_type)(args)

        # TransformerLM Backbone
        if not args.acoustic_backbone_type == "TransformerBackbone":
            raise ValueError("Currently, we only support using TransformerBackbone as the LM.")
        if (not args.backbone_topology) or (not args.causal_transformer):
            raise ValueError(
                "backbone_topology is {}. ".format(args.backbone_topology)
                + "causal_transformer is {}. ".format(args.causal_transformer)
                + "Please ensure that the LM is casual."
            )
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)

        # Dictionary and Embeddings
        self.padding_idx = args.tgt_dict.pad()
        self.bos = args.tgt_dict.bos()
        self.eos = args.tgt_dict.eos()
        self.begin_of_speech_idx = args.tgt_dict.index("<speech>")
        self.end_of_speech_idx = args.tgt_dict.index("</speech>")

        self.total_vocab_size = args.tgt_vocab_size + args.acoustic_vocab_size
        self.embed_tokens = nn.Embedding(
            self.total_vocab_size, args.embedding_size
        )  # [0, tgt_vocab_size) for text, [tgt_vocab_size, total_vocab_size) for acoustic tokens.
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=args.embedding_size**-0.5)

        self.logits = nn.Linear(args.backbone_memory_size, self.total_vocab_size, bias=False)

        if self.token_combination_mode == "simple_asr":
            self.criterion = SpokenLlmAffixXentropy(self.args)
        else:
            self.criterion = Xentropy(self.args)

    def frontend(self, fbank, mask):
        '''frontend'''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, "BTN"

    def tokenize(self, inputs, input_masks):
        '''tokenize'''
        self.acoustic_tokenizer.eval()
        codes, x_masks = self.acoustic_tokenizer(inputs, input_masks)
        return codes, x_masks

    def get_acoustic_tokens(self, batch_data):
        '''A wrapper to get acoustic codes'''
        if 'waveform' in batch_data:
            src = batch_data['waveform']  # (B, T, ndim)
            src_mask = batch_data['wav_mask']
        else:
            src = batch_data['src']  # (B, T, ndim)
            src_mask = batch_data['src_mask']
        src, src_mask, src_shape = self.frontend(src, src_mask)
        acoustic_tokens, acoustic_token_masks = self.tokenize(src, src_mask)
        return acoustic_tokens, acoustic_token_masks

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''Backbone'''
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=True, frontend_shape=frontend_shape
        )
        return backbone_out

    def combine_tokens(self, batch_data, combination_mode="simple_asr", inference=False):
        if combination_mode == "simple_asr":
            """
            building token seqs like
            [<speech>, token1, token2, token3, ..., </speech>, <sos>, text1, text2, ..., <eos>, <pad>, <pad>]
            """
            acoustic_tokens, acoustic_token_masks = self.get_acoustic_tokens(batch_data)
            if inference:
                text_tokens = None
                text_token_masks = None
            else:
                text_tokens = batch_data['char']
                text_token_masks = batch_data['char_mask']

            device = acoustic_tokens.device
            batch_size = acoustic_tokens.shape[0]

            # Adding offsets
            acoustic_tokens = acoustic_tokens + self.args.tgt_vocab_size
            acoustic_token_lengths = acoustic_token_masks.sum(dim=1).long()

            if text_tokens is None:  # As sentinels
                text_tokens = torch.zeros(batch_size, 1, device=device).long() + self.padding_idx
                text_token_masks = torch.zeros(batch_size, 1, device=device).long()

            assert batch_size == text_tokens.shape[0]
            text_token_lengths = text_token_masks.sum(dim=1).long()
            # bos, eos, begin-of-speech, end-of-speech
            additional_token_nums = 3 if inference else 4
            max_length = (acoustic_token_lengths + text_token_lengths).max() + additional_token_nums

            tokens = torch.zeros(batch_size, max_length, device=device).long() + self.padding_idx

            for b in range(batch_size):
                tokens[b, 0] = self.begin_of_speech_idx
                tokens[b, 1 : 1 + acoustic_token_lengths[b]] = acoustic_tokens[
                    b, : acoustic_token_lengths[b]
                ]
                tokens[b, 1 + acoustic_token_lengths[b]] = self.end_of_speech_idx
            acoustic_token_part_masks = (tokens != self.padding_idx).long()

            for b in range(batch_size):
                tokens[b, 1 + acoustic_token_lengths[b] + 1] = self.bos
                tokens[
                    b,
                    1
                    + acoustic_token_lengths[b]
                    + 2 : 1
                    + acoustic_token_lengths[b]
                    + 2
                    + text_token_lengths[b],
                ] = text_tokens[b, : text_token_lengths[b]]
                if not inference:
                    tokens[b, 1 + acoustic_token_lengths[b] + 2 + text_token_lengths[b]] = self.eos
            token_masks = (tokens != self.padding_idx).long()
            text_token_part_masks = token_masks - acoustic_token_part_masks
            acoustic_token_part_with_bos_masks = (
                acoustic_token_part_masks + (tokens == self.bos).long()
            )

            data = {
                "tokens": tokens,
                "token_masks": token_masks,
                "acoustic_token_part_masks": acoustic_token_part_masks,
                "text_token_part_masks": text_token_part_masks,
                "acoustic_token_part_with_bos_masks": acoustic_token_part_with_bos_masks,
            }
        elif combination_mode == "simple_acoustic_token_pretraining":
            """
            building token seqs like
            [token1, token2, token3, ..., ]
            """
            acoustic_tokens, acoustic_token_masks = self.get_acoustic_tokens(batch_data)
            if inference:
                text_tokens = None
                text_token_masks = None
            else:
                text_tokens = batch_data['char']
                text_token_masks = batch_data['char_mask']

            device = acoustic_tokens.device
            batch_size = acoustic_tokens.shape[0]

            device = acoustic_tokens.device
            batch_size = acoustic_tokens.shape[0]

            # Adding offsets
            acoustic_tokens = acoustic_tokens + self.args.tgt_vocab_size
            acoustic_token_lengths = acoustic_token_masks.sum(dim=1).long()

            data = {"tokens": acoustic_tokens, "token_masks": acoustic_token_masks}
        # TODO(baiye): write a template based general combination mode.
        elif combination_mode == "text_modeling":
            """
            building token seqs like
            [<speech>, token1, token2, token3, ..., </speech>, <sos>, text1, text2, ..., <eos>, <pad>, <pad>]
            """
            raise NotImplementedError
        else:
            raise ValueError("Unknown combination_mode: {}".format(combination_mode))
        return data

    def forward(self, batch_data, inference=False):
        """forward"""

        token_data = self.combine_tokens(
            batch_data, combination_mode=self.token_combination_mode, inference=inference
        )

        tokens = token_data["tokens"]
        token_masks = token_data["token_masks"]
        if inference:
            return tokens, token_masks, acoustic_tokens, acoustic_token_masks
        ids = tokens[:, :-1]
        id_masks = token_masks[:, :-1]

        embeds = self.embed_tokens(ids)
        backbone_out = self.encoder_backbone(embeds, id_masks, frontend_shape="BTN")
        logits = self.logits(backbone_out)

        tgts = tokens[:, 1:]
        tgt_masks = token_masks[:, 1:]
        if self.token_combination_mode == "simple_asr":
            forward_out = self.criterion(
                logits=logits,
                targets=tgts,
                tgt_token_masks=tgt_masks,
                tgt_acoustic_token_masks=token_data["acoustic_token_part_masks"][:, 1:],
                tgt_text_token_masks=token_data["text_token_part_masks"][:, 1:],
            )
        else:
            forward_out = self.criterion(
                logits=logits, src_mask=id_masks, target=tgts, target_mask=tgt_masks
            )
        forward_out["token_num"] = token_masks.sum()
        return forward_out

    def forward_step(self, acoustic_tokens, acoustic_token_masks):
        embeds = self.embed_tokens(acoustic_tokens)
        backbone_out = self.encoder_backbone(embeds, acoustic_token_masks, frontend_shape="BTN")
        logits = torch.log_softmax(self.logits(backbone_out), dim=-1)
        return logits

    def set_num_updates(self, num_updates):
        '''set_num_updates'''
        self.num_updates = num_updates

    def init_beam_search(self, inference_cfg, lm_solution):
        '''beam search init'''
        self.beam_size = inference_cfg.get("beam_size", 10)
        return
        # self.beam_searcher = BaseBeamSearch(
        #     inference_cfg,
        #     self.predictor_module,
        #     self.jointer_module,
        #     self.criterion_module,
        #     lm_solution,
        # )

    @torch.no_grad()
    def greedy_inference(self, batch_data):
        '''
        Greedy inference
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        max_decoder_positions = 1000
        max_len_a = 1.0
        max_len_b = 10
        ############ Acoustic ############
        ids, id_masks, acoustic_tokens, acoustic_token_masks = self.forward(
            batch_data, inference=True
        )
        src_len = ids.shape[1]
        device = ids.device
        num_remaining_sent = bsz
        # max output length
        max_len = min(
            int(max_len_a * src_len + max_len_b),
            # exclude the EOS marker
            max_decoder_positions - 1,
        )
        # Output tokens
        tokens = torch.zeros(bsz, max_len + 2, device=device).long()
        tokens[:, 0] = self.bos
        output_mask = torch.ones(bsz, max_len + 2, device=device).float()
        for step in range(1, max_len):
            assert num_remaining_sent >= 0, "Error occured, num remaining sent < 0"
            if step > 1:
                new_token_data = self.combine_tokens(
                    acoustic_tokens,
                    acoustic_token_masks,
                    tokens[:, 1:step],
                    output_mask[:, 1:step],
                    combination_mode=self.token_combination_mode,
                    inference=True,
                )
                ids = new_token_data["tokens"]
                id_masks = new_token_data["token_masks"]
            lprobs = self.forward_step(ids, id_masks)
            new_lprobs = torch.zeros(bsz, lprobs.size(-1), device=device)
            for b in range(bsz):
                new_lprobs[b] = lprobs[b, (id_masks.sum(dim=-1) - 1)[b], :]
            lprobs = new_lprobs
            tokens[:, step] = torch.argmax(lprobs, dim=-1) * output_mask[:, step - 1]
            finished_idxs = (tokens[:, step] == self.eos).nonzero(as_tuple=True)[0]
            output_mask[finished_idxs, step:] = 0
            num_remaining_sent -= finished_idxs.size(0)
            if num_remaining_sent == 0:
                break
        # Remove bos at front
        output = tokens[:, 1:]
        # output_lengths = output_mask[:, 1:].sum(dim=-1)
        out_rlt_list = []
        for bid in range(bsz):
            hyp_token_list = output[bid].view(-1).cpu().tolist()
            out_rlt_list.append(hyp_token_list)
        return out_rlt_list

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        nbest=1,
        **_kwargs,
    ):
        '''
        Beam inference
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        max_decoder_positions = 1000
        max_len_a = 1.0
        max_len_b = 10
        ############ Acoustic ############
        ids, id_masks, acoustic_tokens, acoustic_token_masks = self.forward(
            batch_data, inference=True
        )
        running_size = bsz * self.beam_size

        src_len = ids.shape[1]
        device = acoustic_tokens.device
        ids = (
            ids.unsqueeze(1).repeat(1, self.beam_size, 1).view(running_size, src_len)
        )  # (B*N, src_len)
        id_masks = (
            id_masks.unsqueeze(1).repeat(1, self.beam_size, 1).view(running_size, src_len)
        )  # (B*N, src_len)
        acoustic_tokens = (
            acoustic_tokens.unsqueeze(1)
            .repeat(1, self.beam_size, 1)
            .view(running_size, acoustic_tokens.size(1))
        )  # (B*N, src_len)
        acoustic_token_masks = (
            acoustic_token_masks.unsqueeze(1)
            .repeat(1, self.beam_size, 1)
            .view(running_size, acoustic_token_masks.size(1))
        )  # (B*N, src_len)

        # max output length
        max_len = min(
            int(max_len_a * (src_len - 3) + max_len_b),
            # exclude the EOS marker
            max_decoder_positions - 1,
        )
        # Output tokens
        # tokens = torch.zeros(running_size, max_len + 2, device=device).long()
        # tokens[:, 0] = self.bos

        tokens = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(
            self.bos
        )  # (B*N, 1)
        # tokens_masks = torch.ones(running_size, max_len + 2, device=device).float()
        tokens_masks = torch.ones(running_size, 1, device=device).float()
        scores = torch.tensor([0.0] + [-float('inf')] * (self.beam_size - 1), dtype=torch.float)
        scores = scores.to(device).repeat([bsz]).unsqueeze(1).to(device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)

        # 2. Decoder forward step by step
        for step in range(1, max_len + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            if step > 1:
                new_token_data = self.combine_tokens(
                    acoustic_tokens,
                    acoustic_token_masks,
                    tokens[:, 1:],
                    tokens_masks[:, 1:],
                    combination_mode=self.token_combination_mode,
                    inference=True,
                )
                ids = new_token_data["tokens"]
                id_masks = new_token_data["token_masks"]
            # logp: (B*N, vocab)
            lprobs = self.forward_step(ids, id_masks)
            new_lprobs = torch.zeros(running_size, lprobs.size(-1), device=device)
            for b in range(running_size):
                new_lprobs[b] = lprobs[b, (id_masks.sum(dim=-1) - 1)[b], :]
            lprobs = new_lprobs
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = lprobs.topk(self.beam_size)  # (B*N, N)
            zero_mask = torch.zeros_like(end_flag, dtype=torch.bool)
            if self.beam_size > 1:
                unfinished = torch.cat((zero_mask, end_flag.repeat([1, self.beam_size - 1])), dim=1)
                finished = torch.cat((end_flag, zero_mask.repeat([1, self.beam_size - 1])), dim=1)
            else:
                unfinished = zero_mask
                finished = end_flag
            top_k_logp.masked_fill_(unfinished, -float('inf'))
            top_k_logp.masked_fill_(finished, 0)
            finished = end_flag.repeat([1, self.beam_size])
            top_k_index.masked_fill_(finished, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(bsz, self.beam_size * self.beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=self.beam_size)  # (B, N)
            # Update cache to be consistent with new topk scores / hyps
            # cache_index = (offset_k_index // self.beam_size).view(-1)  # (B*N)
            # base_cache_index = (torch.arange(bsz, device=device).view(
            #     -1, 1).repeat([1, self.beam_size]) * self.beam_size).view(-1)  # (B*N)
            # cache_index = base_cache_index + cache_index
            # cache = [torch.index_select(c, dim=0, index=cache_index) for c in cache]
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = (
                torch.arange(bsz, device=device).view(-1, 1).repeat([1, self.beam_size])
            )  # (B, N)
            base_k_index = base_k_index * self.beam_size * self.beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(
                top_k_index.view(-1), dim=-1, index=best_k_index
            )  # (B*N)
            best_hyps_index = best_k_index // self.beam_size
            last_best_k_hyps = torch.index_select(tokens, dim=0, index=best_hyps_index)  # (B*N, i)
            tokens = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            # import pdb
            # pdb.set_trace()
            end_flag = torch.eq(tokens[:, -1], self.eos).view(-1, 1)
            tokens_masks = torch.cat((tokens_masks, (~end_flag).float()), dim=1)  # (B*N, i+1)
            # tokens_masks = torch.cat((tokens_masks, torch.ones(running_size, 1, device=device).float()), dim=1)  # (B*N, i+1)

        # 3. Select best of best
        scores = scores.view(bsz, self.beam_size)
        # import pdb
        # pdb.set_trace()
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = (
            best_index + torch.arange(bsz, dtype=torch.long, device=device) * self.beam_size
        )
        best_hyps = torch.index_select(tokens, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        out_rlt_list = []
        for bid in range(bsz):
            hyp_token_list = best_hyps[bid].view(-1).cpu().tolist()
            out_rlt_list.append(hyp_token_list)
        return out_rlt_list


@register_solution("SpokenLmPretrainModelFinetune")
class SpokenLmPretrainModelFinetune(BaseRnntModel):
    """SpokenLmPretrainModel Finetune downstream task"""

    def __init__(self, args):
        """init"""
        super().__init__(args)
        self.acoustic_tokenizer = eval(args.acoustic_tokenizer_type)(args)
        # TODO(baiye): Supporting text tokens to support prompting
        self.embed_tokens = nn.Embedding(args.vocab_size, args.embedding_size)
        self.tokenizer_fix = args.get('tokenizer_fix', False)
        if self.tokenizer_fix:
            self.freeze_tokenizer()

    def freeze_tokenizer(self):
        self.acoustic_front_end_module.requires_grad_(False)
        self.acoustic_backbone_module.requires_grad_(False)
        self.acoustic_tokenizer.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)
        self.acoustic_front_end_module.train(False)
        self.acoustic_backbone_module.train(False)
        self.acoustic_tokenizer.train(False)
        self.embed_tokens.train(False)

    def tokenize(self, inputs, input_masks):
        codes, x_masks = self.acoustic_tokenizer(inputs, input_masks)
        return codes, x_masks

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''encoder backbone forward'''
        codes, code_masks = self.tokenize(inputs, mask)
        # ids = codes[:, :-1]
        # id_masks = code_masks[:, :-1]
        embeds = self.embed_tokens(codes)

        attn_mask = False
        if (
            self.args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.acoustic_backbone_module(
            embeds, code_masks, attn_mask=attn_mask, frontend_shape=frontend_shape
        )
        return backbone_out


@register_solution("SpokenLmTranslationModel")
class SpokenLmTranslationModel(BaseSolution):
    """SpokenLmTranslationModel model
    This cls directly train text codes
    """

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args

        # Acoustic Tokenizer Frontend
        self.acoustic_front_end_module = None
        if eval(args.front_end_type) is not None:
            self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_tokenizer = None
        if eval(args.acoustic_tokenizer_type) is not None:
            self.acoustic_tokenizer = eval(args.acoustic_tokenizer_type)(args)

        # TransformerLM Backbone
        if not args.acoustic_backbone_type == "TransformerBackbone":
            raise ValueError("Currently, we only support using TransformerBackbone as the LM.")
        if (not args.backbone_topology) or (not args.causal_transformer):
            raise ValueError(
                "backbone_topology is {}. ".format(args.backbone_topology)
                + "causal_transformer is {}. ".format(args.causal_transformer)
                + "Please ensure that the LM is casual."
            )
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)

        # Dictionary and Embeddings
        self.padding_idx = args.tgt_dict.pad()
        self.total_vocab_size = len(args.tgt_dict)

        self.embed_tokens = nn.Embedding(
            self.total_vocab_size, args.embedding_size
        )  # [0, tgt_vocab_size) for text, [tgt_vocab_size, total_vocab_size) for acoustic tokens.
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=args.embedding_size**-0.5)

        self.logits = nn.Linear(args.backbone_memory_size, self.total_vocab_size, bias=False)

        if getattr(self.args, 'loss_type', 'SpokenLlmAffixXentropy') == "SpokenLlmAffixXentropy":
            self.criterion = SpokenLlmAffixXentropy(self.args)
        else:
            self.criterion = Xentropy(self.args)

    def _get_prefix_and_affix_masks(self, tokens, token_masks):
        masks = torch.zeros_like(tokens, device=tokens.device).long()
        for eos_token in self.args.eos_tokens:
            masks |= (tokens == self.args.tgt_dict.index(eos_token)).long()
        first_eos_postion_idxs = masks.argmax(dim=1).long()
        not_exceed = ((first_eos_postion_idxs + 1) < masks.shape[1]).long()
        affix_mask_start_position_idxs = (
            first_eos_postion_idxs + 1
        ) * not_exceed + first_eos_postion_idxs * (1 - not_exceed)
        affix_mask_start_positions = F.one_hot(
            affix_mask_start_position_idxs.long(), masks.shape[1]
        ).long()
        affix_masks = torch.cumsum(affix_mask_start_positions, dim=1) * token_masks
        prefix_masks = (1 - affix_masks) * token_masks
        return prefix_masks, affix_masks

    def forward(self, batch_data, inference=False):
        """forward"""
        tokens = batch_data['tokens']
        token_masks = batch_data['tokens_mask']

        ids = tokens[:, :-1]
        id_masks = token_masks[:, :-1]
        tgts = tokens[:, 1:]
        tgt_masks = token_masks[:, 1:]

        embeds = self.embed_tokens(ids)
        backbone_out = self.encoder_backbone(embeds, id_masks, frontend_shape="BTN")
        logits = self.logits(backbone_out)

        if getattr(self.args, 'loss_type', 'SpokenLlmAffixXentropy') == "SpokenLlmAffixXentropy":
            prefix_masks, affix_masks = self._get_prefix_and_affix_masks(tgts, tgt_masks)
            forward_out = self.criterion(
                logits=logits,
                targets=tgts,
                prefix_masks=prefix_masks,
                affix_masks=affix_masks,
                masks=tgt_masks,
            )
        else:
            forward_out = self.criterion(
                logits=logits, src_mask=id_masks, target=tgts, target_mask=tgt_masks
            )

        forward_out["token_num"] = token_masks.sum()
        forward_out["data_num"] = token_masks.numel()
        forward_out["valid_token_ratio"] = (
            forward_out["token_num"].float() / forward_out["data_num"]
        )
        return forward_out

    def set_num_updates(self, num_updates):
        '''set_num_updates'''
        self.num_updates = num_updates

    def frontend(self, fbank, mask):
        '''frontend'''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, "BTN"

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''Backbone'''
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=True, frontend_shape=frontend_shape
        )
        return backbone_out


@register_solution("SpokenLmE2EPretrainModel")
class SpokenLmE2EPretrainModel(BaseSolution):
    """SpokenLmE2EPretrainModel model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args
        encoder_hidden_size = args.encoder_embed_dim
        decoder_hidden_size = args.backbone_memory_size

        # Speech Encoder part
        self.apply_mask = args.apply_mask
        self.w2v_model = eval(args.wav2vec_type)(args)
        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates

        self._register_load_state_dict_pre_hook(self._model_load_hook)
        self.w2v_logits_proj = nn.Linear(encoder_hidden_size, args.vocab_size, bias=False)

        # gumbel parameters
        self.init_temp = self.args.get('init_temp', 1.0)
        self.temp_decay_factor = self.args.get('temp_decay_factor', 0.999995)

        # Decoder-only part
        self.embed_tokens = nn.Linear(args.vocab_size, decoder_hidden_size, bias=False)
        self.lm_backbone_module = eval(args.lm_backbone_type)(args)
        self.lm_logits_proj = nn.Linear(decoder_hidden_size, args.vocab_size, bias=False)

        self.criterion = SpokenLmE2eCriterion(self.args)

        # Fix speech encoder to prevent information leaky
        if self.args.fix_w2v_model:
            self.w2v_model.requires_grad_(False)

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.num_updates = num_updates
        self.w2v_model.encoder.num_updates = num_updates

    @staticmethod
    def _model_load_hook(
        state_dict,
        _prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        """Transform bytespeech-chkpt to dolphin-chkpt"""
        old_state_dict = state_dict.copy()
        state_dict.clear()
        for name, param in old_state_dict.items():
            new_name = name
            if name.startswith('data2vec_model.'):
                new_name = name.replace('data2vec_model.', 'w2v_model.')

            state_dict[new_name] = param

    def forward(self, batch_data):
        """forward"""
        # Encoder Part
        padding_mask = (1 - batch_data['src_mask']).int().bool()
        w2v_args = {
            "batch_data": batch_data,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)
        spoken_tokens_logits = self.w2v_logits_proj(x)

        # Get Spoken Tokens
        torch.autograd.set_detect_anomaly(True)
        spoken_tokens_probs = spoken_tokens_logits.softmax(-1)
        if self.training:
            tau = self.init_temp * self.temp_decay_factor**self.num_updates
            spoken_tokens_one_hot = nn.functional.gumbel_softmax(
                spoken_tokens_logits, tau=tau, hard=True
            )
        else:
            index = spoken_tokens_probs.max(-1, keepdim=True)[1]
            spoken_tokens_one_hot = torch.zeros_like(spoken_tokens_probs).scatter_(-1, index, 1.0)

        spoken_tokens_inputs = spoken_tokens_one_hot[:, :-1]
        spoken_tokens_targets = spoken_tokens_probs.max(-1)[1][:, 1:]

        # Decoder Part
        embeds = self.embed_tokens(spoken_tokens_inputs)
        backbone_out = self.lm_backbone_module(
            embeds,
            (~padding_mask[:, :-1]).float(),
        )
        logits = self.lm_logits_proj(backbone_out)

        forward_out = self.criterion(
            logits=logits,
            logits_mask=~padding_mask[:, :-1],
            targets=spoken_tokens_targets,
            targets_mask=~padding_mask[:, 1:],
            spoken_tokens_probs=spoken_tokens_probs,
            spoken_tokens_mask=~padding_mask,
        )

        self.num_updates += 1

        return forward_out
