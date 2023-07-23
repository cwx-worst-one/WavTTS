from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model as HFGPT
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

try:
    from flash_attn.models.gpt import GPTModel as FAGPT
except Exception:
    FAGPT = None


class LanguageModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(
        self,
        config,
        logit_num,
        use_cond=True,
        cond_dim=128,
        cond_type="linear",
        flash_attn=None,
    ):
        super().__init__(config)
        if use_cond:
            if cond_type == "tanh":
                self.cond_layer = nn.Sequential(
                    nn.Linear(cond_dim, config.n_embd),
                    nn.Tanh(),
                    nn.Linear(config.n_embd, config.n_embd),
                )
            else:
                self.cond_layer = nn.Sequential(nn.Linear(cond_dim, config.n_embd))

        if flash_attn:
            if FAGPT is None:
                raise EnvironmentError(
                    f"Detected the flash attn switch is on with arg: {flash_attn}, "
                    f"but found no flash attention package, please install it or check "
                    f"the env ``PYTHONPATH``."
                )
            flash_attn = flash_attn.lower()
            valid_flash = ["flash_attn", "flash_attn_cuda"]
            if flash_attn not in valid_flash:
                raise ValueError(
                    f"Expecting flash_attn is one of {valid_flash}, "
                    f"but got {flash_attn}"
                )
            if flash_attn == "flash_attn_cuda":
                config.use_flash_attn = True
            self.transformer = FAGPT(config)
        else:
            self.transformer = HFGPT(config)

        self.lm_head = nn.Linear(
            config.n_embd, logit_num, bias=False
        )  # quant_token_nums

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def wte(self, input_ids):
        return self.transformer.wte(input_ids)

    def condition(self, input_embed):
        return self.cond_layer(input_embed)

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""# noqa
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.is_flash_attn:
            return self.fa_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                return_dict=return_dict,
            )
        return self.hf_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def hf_forward(
        self, labels=None, **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        transformer_outputs = self.transformer(**kwargs)  # TODO: add location
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = self.compute_loss(logits=lm_logits, labels=labels)

        if not kwargs.pop("return_dict", False):
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def fa_forward(
        self, input_ids=None, inputs_embeds=None, return_dict=None, labels=None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        hidden_states = self.transformer(
            input_ids=input_ids, inputs_embeds=inputs_embeds
        )
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = self.compute_loss(logits=lm_logits, labels=labels)

        if not return_dict:
            output = (lm_logits, hidden_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss, logits=lm_logits, hidden_states=hidden_states
        )

    @property
    def is_flash_attn(self):
        return FAGPT is not None and isinstance(self.transformer, FAGPT)

    def compute_loss(self, logits, labels=None):
        if labels is None:
            return None
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            input=shift_logits.view(-1, shift_logits.size(-1)),
            target=shift_labels.view(-1),
        )
