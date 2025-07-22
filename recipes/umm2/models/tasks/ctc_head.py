import torch
from torch import nn
from torch.nn import functional as F

from recipes.umm2.models.base import BaseStage
from recipes.umm2.models.umm_fm import Conv2dSubsampling
import samantha
from types import SimpleNamespace
from mariana.models.audio.acoustic_head import (
    TimePatchHead,
)

class Downsampler(nn.Module):
    def __init__(self, hidden_size, downsample_rate, downsample_type):
        super().__init__()
        self.hidden_size = hidden_size
        self.downsample_type = downsample_type
        self.downsample_rate = downsample_rate
        self.hidden_size = hidden_size
    

        if downsample_type == "temporal":
            args = SimpleNamespace(patch_size=downsample_rate)
            self.downsample_layer = TimePatchHead(args)
            self.projection_layer = nn.Linear(
                self.hidden_size * downsample_rate, self.hidden_size
            )
        elif downsample_type == "conv":
            self.projection_layer = nn.Identity()
            if downsample_rate == 2: 
                self.downsample_layer = nn.Conv2d(
                                        in_channels=self.hidden_size, 
                                        out_channels=self.hidden_size, 
                                        kernel_size=(3, 1), 
                                        stride=(2, 1), 
                                        padding=(1, 0))
            elif downsample_rate == 4:
                self.downsample_layer = nn.Conv2d(
                                            in_channels=self.hidden_size, 
                                            out_channels=self.hidden_size, 
                                            kernel_size=(8, 1), 
                                            stride=(4, 1), 
                                            padding=(1, 0))
            else:
                raise NotImplementedError()

    def forward(self, x, x_mask):
        if self.downsample_type == "temporal":
            x, x_mask = self.downsample_layer(x, x_mask)
        elif self.downsample_type == "conv":
            x = x.permute(0, 2, 1).unsqueeze(-1)
            x = self.downsample_layer(x)
            x_mask = x_mask[:, ::self.downsample_rate]
        x = self.projection_layer(x)
        return x, x_mask


class CTC_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["token", "latent"], 
        provides=["text_ids", "ctc_out", "loss", "flops"],
        bypasses=[],
        task="ctc",
        loss_weight=1.0,
        lr_ratio=1.0,

    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio)

        self.config = config
        self.blank_id = config.ctc_blank_id
        self.reduction = config.ctc_loss_reduction
        self.loss_weight = loss_weight
        self.ignore_empty = config.ctc_ignore_empty

        if config.ctc_downsample:
            self.ctc_downsample = Downsampler(
                config.hidden_size, 
                config.ctc_downsample_rate, 
                config.ctc_downsample_type
            )
        else:
            self.ctc_downsample = None

        if config.get("head_output_norm", False):
            self.ctc_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.ctc_norm = None

        self.ctc_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )

    def get_metrics(self, ctc_logits, input_attn_mask, tokens, tokens_attn_mask, ignore_invalid: bool = True):
        """
        Calculates the CTC loss. Includes a switch to safely ignore samples 
        with empty targets (pure music) or to include them in the calculation.

        Args:
            ctc_logits (Tensor): The raw logits from the model.
            input_attn_mask (Tensor): The attention mask for the input sequence.
            tokens (Tensor): The target token IDs.
            tokens_attn_mask (Tensor): The attention mask for the target tokens.
            ignore_invalid (bool): If True, samples where target_length is 0 (invalid samples) will be
                                excluded from loss calculation. Defaults to True.
        """
        input_lengths = input_attn_mask.sum(-1).long()
        target_lengths = tokens_attn_mask.sum(-1).long()
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)

        if ignore_invalid:
            # --- Logic to IGNORE invalid samples (e.g., pure music) ---

            # 1. Create a boolean mask to identify samples that have actual text.
            valid_samples_mask = target_lengths > 0

            # 2. Handle the edge case where the entire batch consists of invalid samples.
            if not valid_samples_mask.any():
                return {"loss": (ctc_logits.sum() * 0.0)}

            # 3. Filter all inputs for the CTC loss function using the mask.
            valid_indices = valid_samples_mask.nonzero(as_tuple=True)[0]
            filtered_log_probs = log_probs[:, valid_indices, :]

            filtered_input_lengths = input_lengths[valid_indices]
            filtered_target_lengths = target_lengths[valid_indices]
            filtered_tokens = tokens[valid_indices]

            filtered_tokens_attn_mask = tokens_attn_mask[valid_indices]
            filtered_flattened_targets = filtered_tokens.masked_select(
                filtered_tokens_attn_mask.bool()
            )

            # Calculate CTC loss only on the filtered, valid data
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = self.ctc_loss_fn(
                    log_probs=filtered_log_probs,
                    targets=filtered_flattened_targets,
                    input_lengths=filtered_input_lengths,
                    target_lengths=filtered_target_lengths,
                )

        else:
            flattened_targets = tokens.masked_select(tokens_attn_mask.bool())

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = self.ctc_loss_fn(
                    log_probs=log_probs,
                    targets=flattened_targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                )

        return {"loss": ctc_loss}


    def get_metrics_utt_level(self, ctc_logits, utterances, input_dict):
        # ctc_logits: [B, T (frame_rate), C]
        # utterances: [B] x [U_b] x [C_b_u]
        # import pdb; pdb.set_trace()
        ctc_logits = ctc_logits.contiguous().float()
        B, T = ctc_logits.shape[:2]
        device = ctc_logits.device
        ctc_losses, valid_samples = 0, 0
        text_ids = []
        for b in range(B):
            this_utt = utterances[b]    # [C_b] x [T_b]
            start_frame = [round(utt["start_time"] * self.config.frame_rate) for utt in this_utt]
            end_frame = [round(utt["end_time"] * self.config.frame_rate) for utt in this_utt]
            if len(end_frame) >= 2 and T - end_frame[-2] < self.config.frame_rate:
                end_frame[-2] = T
                this_utt[-2]["token"] = torch.cat((this_utt[-2]["token"], this_utt[-1]["token"]), -1)
                del end_frame[-1]
            this_ctc_logits, this_text_ids = [], []
            for i in range(len(start_frame)):
                sf, ef = start_frame[i] if i > 0 else 0, end_frame[i] if i < len(start_frame) - 1 else T
                if ef - sf > 0:
                    this_text_ids.append(this_utt[i]["token"])
                    this_ctc_logits.append(ctc_logits[b][sf:ef])

            if len(this_text_ids) <= 0:
                continue

            concat_text_ids = torch.cat([tt for tt in this_text_ids], -1)
            text_ids.append(concat_text_ids)

            input_lengths = torch.tensor([x.size(0) for x in this_ctc_logits]).long().to(device)
            this_text_ids = torch.nn.utils.rnn.pad_sequence(this_text_ids, batch_first=True, padding_value=0).long().to(device)
            this_ctc_logits = torch.nn.utils.rnn.pad_sequence(this_ctc_logits, batch_first=True, padding_value=0).to(device)

            labels_mask = this_text_ids > 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = this_text_ids.masked_select(labels_mask)

            # CTCLoss doesn't support fp16
            log_probs = F.log_softmax(
                this_ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)  # [N, T, C] -> [T, N, C]
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = self.ctc_loss_fn(
                    log_probs, flattened_targets, input_lengths, target_lengths
                )
            print(this_ctc_logits.shape, this_text_ids.shape, ctc_loss / this_text_ids.shape[0], T)

            # note that CTC loss is weighted by the number of text tokens instead of the number of frames
            # which means loss can be very large for long utterances without text tokens
            # here I divide the loss by the number of frames to make it more comparable to other losses
            if this_ctc_logits.shape[1] > 2000:
                print("extreme value", this_utt)
                ctc_loss = ctc_loss / this_ctc_logits.shape[1]
                
            ctc_losses += (ctc_loss / this_text_ids.shape[0])
            valid_samples += 1

        param_loss = 0
        for p in self.parameters():
            param_loss += (p.pow(2).sum() * 0)
        if len(text_ids) <= 0:
            return {"loss": param_loss}, torch.zeros([B, 0]).to(device)

        ctc_losses = (ctc_losses + param_loss) / valid_samples
        text_ids = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0).long().to(device)
        return {"loss": ctc_losses}, text_ids


    def forward(self, input_dict):

        hidden_states = input_dict['latent']
        attn_mask = input_dict['attn_mask']
        tokens = input_dict['token']['input_ids']
        tokens_attn_mask = input_dict['token']['attention_mask']

        utterances = input_dict.get('utterances', None)

        flops = 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]

        # if self.ctc_downsample is not None:
        #    down_sample_input = hidden_states.permute(0, 2, 1).unsqueeze(-1)
        #    hidden_states = self.ctc_downsample(down_sample_input).squeeze(-1).permute(0, 2, 1)

        if self.ctc_downsample is not None:
            hidden_states, attn_mask = self.ctc_downsample(hidden_states, attn_mask)

        if self.ctc_norm is not None:
            hidden_states = self.ctc_norm(hidden_states)

        ctc_out = self.ctc_head(hidden_states)
        output_dict = {}

        if "token" not in input_dict and  utterances is not None:
            metric_dict, text_ids = self.get_metrics_utt_level(ctc_out,utterances, input_dict)
        else:
            metric_dict = self.get_metrics(ctc_out, attn_mask, tokens, tokens_attn_mask, self.ignore_empty)
                
        loss = metric_dict['loss'] * self.loss_weight

        output_dict.update({
            'ctc_out': ctc_out,
            'loss': loss,
            'flops': flops,
            'aux/loss_ctc': metric_dict['loss'],
            'aux/num_text_tokens': tokens_attn_mask.sum()
        })

        return output_dict
