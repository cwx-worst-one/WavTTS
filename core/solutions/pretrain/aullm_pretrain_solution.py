"""best-rq pretrain"""

import string
import contextlib
import torch
from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from core.models.pretrained.data2vec_model import PretrainedData2Vec
from zhon.hanzi import punctuation  # pylint: disable=import-error
from transformers import (  # pylint: disable=unused-import
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BertModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BloomForCausalLM,
    GenerationConfig, # pylint: disable=no-name-in-module
)
from core.criterions.criterion import Xentropy
from core.solutions.inference.generate_lm_beam_search import GenerateBeamSearch
from core.solutions.base_solution import (
    register_solution,
)
from core.solutions.pretrain.spokenlm_pretrain_solution import SpokenLmE2EPretrainModel
from core.utils.misc import infer_text_format
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
)
from core.models.layers.cif import CifCalculator, CifWeightEstimator
from core.models.asr.cif_decoder import GenerateDecoder


@register_solution("AuLlmPretrainModel")
class AuLlmPretrainModel(SpokenLmE2EPretrainModel):
    """AuLlmE2EPretrainModel model"""

    def __init__(self, args):  # pylint: disable=too-many-branches
        """init"""
        super().__init__(args)
        self.args = args
        self.eos_id = args.get('eos_id', 2)
        self.dev_beam_size = args.get('dev_beam_size', 1)

        encoder_hidden_size = args.encoder_embed_dim
        decoder_hidden_size = args.hidden_size

        if self.args.get('add_cif_module', False):
            self.cif_weight_estimator = CifWeightEstimator(args)
            self.cif_calculator = CifCalculator(self.args)

        if self.args.use_continuous_rep:
            self.w2v_logits_proj = None

            # Project audio embeddings
            if self.args.get("use_llm", False):
                if self.args.get('add_cif_module', False):
                    self.w2v_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)

                    if self.args.fix_w2v_model:
                        self.w2v_proj.requires_grad_(False)
                        self.cif_weight_estimator.requires_grad_(False)

                if self.args.get("s2l_interface_type", "") == "transformer":
                    if self.args.get("only_load_s2l_structure", True):
                        transformer_config = AutoConfig.from_pretrained(
                            self.args.s2l_transformer_ckpt
                        )
                        self.s2l_transformer = AutoModel.from_config(transformer_config)
                        self.s2l_transformer.init_weights()  # make sure random init
                    else:
                        self.s2l_transformer = AutoModel.from_pretrained(
                            self.args.s2l_transformer_ckpt
                        )

                    if self.args.get("s2l_interface_gradient_checkpointing", False):
                        self.s2l_transformer.encoder.gradient_checkpointing = True

                    if self.args.get('add_cif_module', False):
                        self.w2v_proj2 = nn.Linear(
                            decoder_hidden_size, self.s2l_transformer.config.hidden_size
                        )
                    else:
                        self.w2v_proj = nn.Linear(
                            self.args.am_hidden_size, self.s2l_transformer.config.hidden_size
                        )
                    self.post_audio_proj = nn.Linear(
                        self.s2l_transformer.config.hidden_size, self.args.llm_hidden_size
                    )
                else:
                    if self.args.get('add_cif_module', False):
                        self.w2v_proj2 = nn.Linear(decoder_hidden_size, self.args.llm_hidden_size)
                    else:
                        self.w2v_proj = nn.Linear(
                            self.args.am_hidden_size, self.args.llm_hidden_size
                        )
            else:
                self.w2v_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)

                if self.args.get('add_cif_module', False):
                    if self.args.fix_w2v_model:
                        self.w2v_proj.requires_grad_(False)
                        self.cif_weight_estimator.requires_grad_(False)

        else:
            self.w2v_logits_proj = nn.Linear(
                encoder_hidden_size, args.gumbel_vocab_size, bias=False
            )
            self.speech_embed_tokens = nn.Linear(
                args.gumbel_vocab_size, decoder_hidden_size, bias=False
            )

            # Fix speech encoder & logits_proj
            if self.args.fix_w2v_model:
                self.w2v_model.requires_grad_(False)
                self.w2v_logits_proj.requires_grad_(False)

        self.embed_tokens = None
        self.lm_backbone_module = None
        self.lm_logits_proj = None
        self.text_embed_tokens = nn.Embedding(args.vocab_size, decoder_hidden_size)

        # Pre-set all decoder-related attributes to None
        self.lm_generate_decoder = None
        self.llm_decoder = None
        self.llm_tokenizer = None
        if self.args.get("use_llm", False):
            self.llm_decoder = BloomForCausalLM.from_pretrained(self.args.llm_dir)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.args.llm_dir)
            self.llm_decoder.requires_grad_(False)
            self.llm_decoder.transformer.gradient_checkpointing = True

            self.prompt = self.args.prompt
            self.prompt_tokens = self.llm_tokenizer(
                [self.prompt], return_tensors="pt", padding="longest"
            )  # only has two keys: attention_mask & input_ids
            self.prompt_length = self.prompt_tokens["attention_mask"].sum(-1)[0]
        else:
            self.lm_generate_decoder = GenerateDecoder(args, sign='gen_lm')

        self.criterion = Xentropy(self.args)

        if self.dev_beam_size > 1:
            args['beam_size'] = self.dev_beam_size
            self.init_beam_search(args)

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
            # for add_cif_module
            if name.startswith('w2v_model_proj.'):
                new_name = name.replace('w2v_model_proj.', 'w2v_proj.')

            state_dict[new_name] = param

    def train(self, mode: bool = True):
        '''
        Param will really be freezed by set param.requires_grad_(False).
        `with no_grad` may case param be updated by optimizer' weight_decay.
        What's more, we need to set module.eval() to fix some module buffer,
        such as BatchNorm.running_mean.
        '''
        super().train(mode)

        # control LLM parameters
        if self.args.get("use_llm", False):
            if self.args.get('unfreeze_llm', False):
                unfreezed_llm_layers = eval(self.args.unfreezed_llm_layers)
                for name, param in self.llm_decoder.named_parameters():
                    if "transformer.h" in name:
                        cur_layer_id = int(name.split("transformer.h.")[-1].split(".")[0])
                        if cur_layer_id in unfreezed_llm_layers:
                            param.requires_grad_(True)
                        else:
                            param.requires_grad_(False)
                    else:
                        param.requires_grad_(False)
            else:
                self.llm_decoder.requires_grad_(False)

    def collect_labels_str(self, label_seqs, lang=None):
        """build the strings for LLMs"""
        if lang == "cn":
            labels_str = [
                "".join(label_seq).strip() for label_seq in label_seqs
            ]  # This is the final labels
        elif lang == "en":
            labels_str = [
                " ".join(label_seq).strip() for label_seq in label_seqs
            ]  # This is the final labels
        else:
            ## For multilingual processing
            temp_labels = []
            for label_seq in label_seqs:
                new_label_seq = []
                for token in label_seq:
                    if not self.is_contains_chinese(token):
                        # for English
                        new_label_seq.append(" " + token + " ")
                    else:
                        # for Chinese
                        new_label_seq.append(token)
                temp_labels.append(new_label_seq)

            labels_str = [
                " ".join("".join(label_seq).split()).strip() for label_seq in temp_labels
            ]  # This is the final labels

        return labels_str

    def forward(self, batch_data):
        """forward"""
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches

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

        # Get audio embed
        if self.args.use_continuous_rep:
            audio_embeds = self.w2v_proj(x)
        else:
            spoken_tokens_logits = self.w2v_logits_proj(x)
            spoken_tokens_probs = spoken_tokens_logits.softmax(-1)
            # Get Spoken Tokens
            torch.autograd.set_detect_anomaly(True)
            index = spoken_tokens_probs.max(-1, keepdim=True)[1]
            spoken_tokens_one_hot = torch.zeros_like(spoken_tokens_probs).scatter_(-1, index, 1.0)
            audio_embeds = self.speech_embed_tokens(spoken_tokens_one_hot)
        audio_embeds_mask = (~padding_mask).float()

        # Get text
        targets = batch_data['char'].long()
        targets_mask = batch_data['char_mask'].float()

        if self.args.get('add_cif_module', False):
            frame_audio_embeds, not_padding = audio_embeds, audio_embeds_mask.int()
            a = self.cif_weight_estimator(frame_audio_embeds, not_padding)

            if self.args.get('add_scaled_cif', False):
                # CIF part
                (
                    audio_embeds,
                    audio_embeds_mask,
                    _,  # pylint: disable=unused-variable
                    _,
                    _,
                ) = self.cif_calculator(
                    frame_audio_embeds, not_padding, a, targets=targets, is_training=True
                )
                audio_embeds_mask = audio_embeds_mask.float()
            else:
                if self.args.use_tail_handling:
                    frame_audio_embeds = nn.functional.pad(frame_audio_embeds, [0, 0, 0, 1, 0, 0])
                    not_padding = nn.functional.pad(not_padding, [0, 1, 0, 0])
                    a = nn.functional.pad(a, [0, 1, 0, 0])

                # CIF part
                (
                    audio_embeds,
                    audio_embeds_mask,
                    sum_a, # pylint: disable=unused-variable
                    _,
                    _,
                ) = self.cif_calculator(frame_audio_embeds, not_padding, a)
                audio_embeds_mask = audio_embeds_mask.float()

        if self.args.get("use_llm", False):
            if self.args.get('add_cif_module', False):
                audio_embeds = self.w2v_proj2(audio_embeds)

            if self.args.get("s2l_interface_type", "") == "transformer":
                s2l_outputs = self.s2l_transformer(
                    inputs_embeds=audio_embeds, attention_mask=audio_embeds_mask
                )
                audio_embeds = self.post_audio_proj(s2l_outputs.last_hidden_state)

            # Prepare input strings
            labels_str = self.collect_labels_str(batch_data["label"])
            label_tokens = self.llm_tokenizer(
                labels_str,
                return_tensors="pt",
                padding="longest",
            ).to(audio_embeds.device)

            # Process inputs & targets
            prompt_tokens = self.prompt_tokens.to(audio_embeds.device)
            expanded_prompt_tokens = {
                k: v.repeat([label_tokens["input_ids"].size(0), 1])
                for k, v in prompt_tokens.items()
            }

            eos_col = (
                torch.ones([audio_embeds.size(0), 1], dtype=label_tokens["input_ids"].dtype).to(
                    audio_embeds.device
                )
                * self.llm_tokenizer.pad_token_id
            )
            targets = torch.cat(
                [expanded_prompt_tokens["input_ids"], label_tokens["input_ids"], eos_col], dim=1
            )
            add_mat = torch.nn.functional.one_hot(
                (targets != self.llm_tokenizer.pad_token_id).sum(-1), num_classes=targets.size(1)
            ).type_as(targets).to(audio_embeds.device) * (
                self.llm_tokenizer.eos_token_id - self.llm_tokenizer.pad_token_id
            )
            targets = targets + add_mat

            for k, v in label_tokens.items():
                prompt_part = expanded_prompt_tokens[k].to(audio_embeds.device)
                if k == "input_ids":
                    sos_col = self.llm_tokenizer.bos_token_id * torch.ones(
                        [v.size(0), 1], dtype=v.dtype
                    ).to(audio_embeds.device)
                else:
                    sos_col = torch.ones([v.size(0), 1], dtype=v.dtype).to(audio_embeds.device)
                label_tokens[k] = torch.cat([prompt_part, sos_col, v], dim=1)

            # Ignore unnecessary targets
            targets = targets.masked_fill(
                targets == self.llm_tokenizer.pad_token_id, -100
            )  # B x T_t
            targets[:, : self.prompt_length] = (
                torch.ones([targets.size(0), self.prompt_length])
                .type_as(targets)
                .to(audio_embeds.device)
                * -100
            )
            empty_speech_targets = (
                torch.ones([targets.size(0), audio_embeds.size(1)])
                .type_as(targets)
                .to(audio_embeds.device)
                * -100
            )  # B x T_a
            targets = torch.cat([empty_speech_targets, targets], dim=-1)  # B x (T_a + T_t)

            # Forward LLM
            inputs_embeds = self.llm_decoder.transformer.word_embeddings(label_tokens["input_ids"])
            inputs_embeds = torch.cat([audio_embeds, inputs_embeds], dim=1)  # B x T_all x C
            attention_mask = torch.cat([audio_embeds_mask, label_tokens["attention_mask"]], dim=-1)
            llm_outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            logits = llm_outputs.logits

            temp_targets = torch.where(
                targets != -100, targets, torch.tensor(0).to(audio_embeds.device)
            )
            forward_out = self.criterion(
                logits=logits,
                src_mask=attention_mask,
                target=temp_targets.long(),
                target_mask=(temp_targets != 0),
            )
        else:
            # add sos label to split audio and text
            sos_label = torch.ones(targets.shape[0], 1).long().cuda()
            targets = torch.cat([sos_label, targets], dim=1)
            text_embeds = self.text_embed_tokens(targets.reshape(-1)).reshape(
                *targets.shape, -1
            )  # B x (T + 1) x C
            text_embeds_mask = torch.cat(
                [sos_label.float(), targets_mask], dim=1
            )  # Start with an sos label

            # Concat audio & text embed
            embeds, embeds_mask, concat_targets = self.concat_audio_text_embed(
                audio_embeds, text_embeds, audio_embeds_mask, text_embeds_mask, targets
            )  # remove eos and concat audio first and text last, and prepare input mask

            # Decoder Part
            logits = self.lm_generate_decoder(embeds, None, is_training=True)  # B x (T_a + T_t) x V

            forward_out = self.criterion(
                logits=logits,
                src_mask=embeds_mask,
                target=concat_targets.long(),
                target_mask=(concat_targets != 0),
            )

        # For validation
        if not logits.requires_grad:
            if self.args.get("use_llm", False):
                (
                    total_error_count,
                    ins_error_count,
                    del_error_count,
                    sub_error_count,
                    total_count,
                ) = self.llm_inference(
                    audio_embeds=audio_embeds,
                    audio_embeds_mask=audio_embeds_mask,
                    tgt_strs=labels_str,
                    uttid=batch_data['uttid'],
                    beam_size=self.dev_beam_size,
                )
            else:
                (
                    total_error_count,
                    ins_error_count,
                    del_error_count,
                    sub_error_count,
                    total_count,
                ) = self.asr_inference(
                    audio_embeds,
                    audio_embeds_mask,
                    targets[:, 1:],
                    targets_mask,
                    batch_data['uttid'],
                    beam_size=self.dev_beam_size,
                )

            forward_out['cer'] = total_error_count
            forward_out['total_error_count'] = total_error_count
            forward_out['ins_error_count'] = ins_error_count
            forward_out['del_error_count'] = del_error_count
            forward_out['sub_error_count'] = sub_error_count
            forward_out['total_count'] = total_count

        return forward_out

    def concat_audio_text_embed(
        self, audio_embeds, text_embeds, audio_embeds_mask, text_embeds_mask, targets
    ):
        """concatenate audio and text embedding"""
        bsz, audio_len, embed_size = audio_embeds.shape
        bsz, text_len, embed_size = text_embeds.shape
        max_total_len = audio_len + text_len
        embeds_list = []
        embeds_mask_list = []
        concat_targets_list = []
        for i in range(bsz):
            cur_audio_len = audio_embeds_mask[i].sum(-1).int()
            cur_text_len = text_embeds_mask[i].sum(-1).int()

            cur_embeds = torch.cat(
                [audio_embeds[i, :cur_audio_len], text_embeds[i, : cur_text_len - 1]], dim=0
            )

            pad_len = max_total_len - (cur_audio_len + cur_text_len) + 1
            cur_embeds = torch.cat([cur_embeds, torch.zeros([pad_len, embed_size]).cuda()], dim=0)
            cur_embeds_mask = torch.cat(
                [torch.ones([cur_audio_len + cur_text_len - 1]), torch.zeros([pad_len])], dim=0
            ).cuda()
            cur_concat_targets = torch.cat(
                [
                    torch.zeros([cur_audio_len]).int().cuda(),
                    targets[i, 1:cur_text_len],
                    torch.zeros([pad_len]).int().cuda(),
                ],
                dim=0,
            )

            embeds_list.append(cur_embeds.float().unsqueeze(0))
            embeds_mask_list.append(cur_embeds_mask.int().unsqueeze(0))
            concat_targets_list.append(cur_concat_targets.int().unsqueeze(0))
        embeds_mask = torch.cat(embeds_mask_list, dim=0)
        max_valid_len = embeds_mask.sum(-1).max()

        embeds = torch.cat(embeds_list, dim=0)[:, :max_valid_len]
        embeds_mask = embeds_mask[:, :max_valid_len]
        concat_targets = torch.cat(concat_targets_list, dim=0)[:, :max_valid_len]

        return embeds, embeds_mask, concat_targets

    def inference(
        self,
        batch_data,
        mode='test',
        beam_size=1,
        nbest=1,  # pylint: disable=unused-argument
        language=None,
        filter_list=None,
        concate_en_letters=None,  # pylint: disable=unused-argument
        output_timestamp=None,  # pylint: disable=unused-argument
        output_preappear_metric=None,  # pylint: disable=unused-argument
    ):
        """inference"""

        # Encoder Part
        padding_mask = (1 - batch_data['src_mask']).int().bool()
        w2v_args = {
            "batch_data": batch_data,
            "padding_mask": padding_mask,
            "mask": False,
        }
        x, padding_mask = self.w2v_model.extract_features(**w2v_args)

        # Get audio embed
        if self.args.use_continuous_rep:
            audio_embeds = self.w2v_proj(x)
        else:
            spoken_tokens_logits = self.w2v_logits_proj(x)
            spoken_tokens_probs = spoken_tokens_logits.softmax(-1)
            # Get Spoken Tokens
            torch.autograd.set_detect_anomaly(True)
            index = spoken_tokens_probs.max(-1, keepdim=True)[1]
            spoken_tokens_one_hot = torch.zeros_like(spoken_tokens_probs).scatter_(-1, index, 1.0)
            audio_embeds = self.speech_embed_tokens(spoken_tokens_one_hot)
        audio_embeds_mask = (~padding_mask).float()

        if self.args.get('add_cif_module', False):
            frame_audio_embeds, not_padding = audio_embeds, audio_embeds_mask.int()
            a = self.cif_weight_estimator(frame_audio_embeds, not_padding)

            if self.args.use_tail_handling:
                frame_audio_embeds = nn.functional.pad(frame_audio_embeds, [0, 0, 0, 1, 0, 0])
                not_padding = nn.functional.pad(not_padding, [0, 1, 0, 0])
                a = nn.functional.pad(a, [0, 1, 0, 0])

            # CIF part
            (
                audio_embeds,
                audio_embeds_mask,
                _,
                _,
                _,
            ) = self.cif_calculator(frame_audio_embeds, not_padding, a)
            audio_embeds_mask = audio_embeds_mask.float()

        # Get text
        targets = batch_data['char'].long()
        targets_mask = batch_data['char_mask'].float()

        # get results
        if self.args.get("use_llm", False):
            if self.args.get('add_cif_module', False):
                audio_embeds = self.w2v_proj2(audio_embeds)

            if self.args.get("s2l_interface_type", "") == "transformer":
                s2l_outputs = self.s2l_transformer(
                    inputs_embeds=audio_embeds, attention_mask=audio_embeds_mask
                )
                audio_embeds = self.post_audio_proj(s2l_outputs.last_hidden_state)

            # get target strings
            labels_str = self.collect_labels_str(batch_data["label"])

            ret_tuples = self.llm_inference(
                audio_embeds=audio_embeds,  # B x T x C
                audio_embeds_mask=audio_embeds_mask,  # B x T
                tgt_strs=labels_str,
                uttid=batch_data['uttid'],
                beam_size=beam_size,
                mode=mode,
            )
        else:
            ret_tuples = self.asr_inference(
                audio_embeds,
                audio_embeds_mask,
                targets,
                targets_mask,
                batch_data['uttid'],
                beam_size=beam_size,
                mode=mode,
                language=language,
                filter_list=filter_list,
            )

        return ret_tuples

    def asr_inference(
        self,
        audio_embeds,
        audio_embeds_mask,
        targets,
        targets_mask,
        uttid,
        beam_size=1,
        mode='dev',
        language=None,
        filter_list=None,
    ):
        '''Au-llm Inference'''
        # get tokens ids
        if beam_size > 1:
            token_ids = self.lm_generate_beam_searcher(
                audio_embeds, audio_embeds_mask, self.text_embed_tokens
            )
        else:
            logits = self.lm_generate_decoder(
                audio_embeds, audio_embeds_mask, text_embed_lookup=self.text_embed_tokens
            )
            token_ids = logits.argmax(dim=-1)

        # get the text result for cer evaluation
        audio_length = audio_embeds_mask.sum(-1).int()
        token_ids_list = []

        for i in range(audio_embeds.shape[0]):
            cur_tokens_id = token_ids[i, audio_length[i] :]
            cur_pad_tokens = torch.zeros([audio_length[i]]).int().cuda()
            cur_tokens_id = torch.cat([cur_tokens_id, cur_pad_tokens], dim=0)
            token_ids_list.append(cur_tokens_id.unsqueeze(0))
        token_ids = torch.cat(token_ids_list, dim=0)

        target_lengths = targets_mask.sum(-1).int()
        hyp_strs, tgt_strs, ed_info_list, align_info_list = self.cal_edit_distance(
            uttid,
            token_ids,
            targets,
            target_lengths,
            language,
            filter_list,
        )
        if mode == 'test':
            return (
                hyp_strs,
                tgt_strs,
                ed_info_list,
                align_info_list,
                None,
            )
        (
            total_error_dist,
            ins_error_dist,
            del_error_dist,
            sub_error_dist,
            total_dist,
        ) = self.rlt_post_process(ed_info_list)
        return total_error_dist, ins_error_dist, del_error_dist, sub_error_dist, total_dist

    def llm_inference(
        self,
        audio_embeds,
        audio_embeds_mask,
        tgt_strs,
        uttid,
        temperature=1.0,
        top_p=0.75,
        top_k=40,
        beam_size=1,
        max_new_tokens=128,
        mode='dev',
    ):
        """LLM inference"""

        # prepare input_tokens
        prompt_tokens = self.prompt_tokens.to(audio_embeds.device)  # 1 x ?
        prompt_tokens = {
            k: v.repeat([audio_embeds.size(0), 1]) for k, v in prompt_tokens.items()
        }  # B x T_prompt
        input_tokens = dict()
        for k, v in prompt_tokens.items():
            if k == "input_ids":
                sos_col = self.llm_tokenizer.bos_token_id * torch.ones(
                    [v.size(0), 1], dtype=v.dtype
                ).to(
                    audio_embeds.device
                )  # B x 1
            else:
                sos_col = torch.ones([v.size(0), 1], dtype=v.dtype).to(audio_embeds.device)  # B x 1
            input_tokens[k] = torch.cat([v, sos_col], dim=1)  # B x T

        # build inputs_embeds with AM outputs, prompt and <bos>
        inputs_embeds = self.llm_decoder.transformer.word_embeddings(
            input_tokens["input_ids"]
        )  # B x T x llm_hidden_size
        inputs_embeds = torch.cat([audio_embeds, inputs_embeds], dim=1)
        input_attention_mask = torch.cat(
            [audio_embeds_mask, input_tokens["attention_mask"]], dim=-1
        )

        # begin inference
        generation_config = GenerationConfig(
            num_beams=beam_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            bos_token_id=self.llm_tokenizer.bos_token_id,
        )
        with torch.no_grad():
            gen_out = self.llm_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=input_attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        batch_results = gen_out.sequences
        hyp_strs = self.llm_tokenizer.batch_decode(
            batch_results, skip_special_tokens=True
        )  # batch_str is List[str]

        # get the text result for cer evaluation
        hyp_strs, tgt_strs, ed_info_list, align_info_list = self.cal_ed_for_llm_outputs(
            uttid=uttid, hyp_strs=hyp_strs, tgt_strs=tgt_strs
        )

        if mode == 'test':
            return (
                hyp_strs,
                tgt_strs,
                ed_info_list,
                align_info_list,
                None,
            )
        (
            total_error_dist,
            ins_error_dist,
            del_error_dist,
            sub_error_dist,
            total_dist,
        ) = self.rlt_post_process(ed_info_list)

        return total_error_dist, ins_error_dist, del_error_dist, sub_error_dist, total_dist

    def decode_ids(self, hyp):
        """decode ids"""
        hyp_token = []
        word = ''
        for idx, token_id in enumerate(hyp):
            token_id = token_id.item()
            if token_id in (0, 1):
                continue
            if self.args.id_map is not None:
                token = self.args.id_map[token_id]
            else:
                token = self.args.tgt_dict.string([token_id])
            if not token:
                continue
            word += token
            if token[-1] != '@' or idx == len(hyp) - 1:
                hyp_token.append(word)
                word = ''
        res_str = ' '.join(hyp_token).replace('@@', '').replace(" ' ", "'")
        return res_str

    @staticmethod
    def rlt_post_process(ed_info_list):
        """rlt post process"""
        total_dist = 0
        ins_error_dist = 0
        del_error_dist = 0
        sub_error_dist = 0
        total_error_dist = 0
        for ed_info in ed_info_list:
            total_error_dist += ed_info['ins_err'] + ed_info['del_err'] + ed_info['sub_err']
            ins_error_dist += ed_info['ins_err']
            del_error_dist += ed_info['del_err']
            sub_error_dist += ed_info['sub_err']
            total_dist += ed_info['ref_word_num']
        return total_error_dist, ins_error_dist, del_error_dist, sub_error_dist, total_dist

    def post_process_llm_str(self, input_str):
        """post process the strings of LLMs"""
        new_seq = []
        for seq in input_str.split():
            if self.is_contains_chinese(seq):
                new_seq = new_seq + list(seq)
            else:
                new_seq.append(seq)
        input_str = " ".join(new_seq)

        # remove Chinese and English punctuations
        input_str = input_str.translate(str.maketrans('', '', string.punctuation))
        input_str = input_str.translate(str.maketrans('', '', punctuation))
        input_str = " ".join(input_str.split())

        return input_str

    def cal_ed_for_llm_outputs(
        self,
        uttid,
        hyp_strs,
        tgt_strs,
    ):
        """calculate edit-distance for LLM outputs"""

        bsz = len(tgt_strs)
        ed_calculator = EditDistanceCalculator()

        new_hyp_strs = []
        new_tgt_strs = []
        ed_info_list = []
        align_info_list = []
        for bid in range(bsz):
            hyp_str = hyp_strs[bid]
            tgt_str = tgt_strs[bid]
            hyp_str = self.post_process_llm_str(hyp_str)
            tgt_str = self.post_process_llm_str(tgt_str)
            ed_info, align_info = ed_calculator.show_alignment(
                uttid[bid], tgt_str.split(), hyp_str.split()
            )
            ed_info_list.append(ed_info)
            align_info_list.append(align_info)
            new_hyp_strs.append(hyp_str)
            new_tgt_strs.append(tgt_str)

        return new_hyp_strs, new_tgt_strs, ed_info_list, align_info_list

    def cal_edit_distance(self, uttid, hyps, target, target_lengths, language, filter_list):
        '''cal edit distance'''
        bsz, _ = hyps.shape
        ed_calculator = EditDistanceCalculator()
        if language is not None:
            formator = TextFormator(language)
        ed_info_list = []
        align_info_list = []
        hyp_strs = []
        tgt_strs = []
        for bid in range(bsz):
            hyp_ids = hyps[bid]
            tgt_ids = target[bid][: target_lengths[bid]]

            # save until eos
            hyp_str = self.decode_ids(self.save_until_idx(hyp_ids, idx=self.eos_id))
            tgt_str = self.decode_ids(self.save_until_idx(tgt_ids, idx=self.eos_id))

            if language is not None:
                hyp_str = ' '.join(infer_text_format(hyp_str, filter_list, formator))
                tgt_str = ' '.join(infer_text_format(tgt_str, filter_list, formator))
            ed_info, align_info = ed_calculator.show_alignment(
                uttid[bid], tgt_str.split(), hyp_str.split()
            )
            ed_info_list.append(ed_info)
            align_info_list.append(align_info)
            hyp_strs.append(hyp_str)
            tgt_strs.append(tgt_str)
        return hyp_strs, tgt_strs, ed_info_list, align_info_list

    @staticmethod
    def save_until_idx(hyp, idx=1):
        '''save until some idx'''
        try:
            index = list(hyp).index(idx)
            return hyp[0:index]
        except ValueError:
            # No EOS_ID: return the array as-is.
            return hyp

    @staticmethod
    def is_contains_chinese(strs):
        """judge whether the input string has any Chinese character"""
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def init_beam_search(
        self,
        inference_cfg,
        lm_solution=None,
        fst_solution=None,
        domain_fst_solution=None,
        fst_dict=None,
    ):
        '''beam search init'''
        self.lm_generate_beam_searcher = GenerateBeamSearch(
            self.args,
            inference_cfg,
            self.lm_generate_decoder.logits_loop_decoder,
            lm_solution,
            fst_solution,
            domain_fst_solution,
            fst_dict,
        )
