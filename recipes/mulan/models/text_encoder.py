import os
import time
from typing import Optional

import tenacity
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, LlamaModel, LlamaForCausalLM, BitsAndBytesConfig, ClapModel
from peft import LoraConfig, get_peft_model, TaskType

from samantha.utils.flops_calculator import bert_calculator


def truncate_layers(llama_model, num_sub_layers: int = -1):
    # By default, keep the lower half of layers
    if num_sub_layers == -1:
        num_sub_layers = len(llama_model.layers) // 2
    llama_model.layers = nn.ModuleList(llama_model.layers[:num_sub_layers])


@tenacity.retry(
    stop=(tenacity.stop_after_delay(100) | tenacity.stop_after_attempt(3)), reraise=True
)
def load_llama(llama_path):
    model = LlamaModel.from_pretrained(llama_path).eval()
    return model


@tenacity.retry(
    stop=(tenacity.stop_after_delay(100) | tenacity.stop_after_attempt(3)), reraise=True
)
def load_t5(t5_path):
    from transformers.models.t5.modeling_t5 import T5EncoderModel
    model = T5EncoderModel.from_pretrained(t5_path).eval()
    return model
def load_t5_train(t5_path):
    from transformers.models.t5.modeling_t5 import T5EncoderModel
    model = T5EncoderModel.from_pretrained(t5_path)
    return model

@tenacity.retry(
    stop=(tenacity.stop_after_delay(100) | tenacity.stop_after_attempt(3)), reraise=True
)
def load_partial_llama(llama_path, local_rank, number_of_layers=-1):
    """Load a subset layers of llama model
    number_of_layers: keep the first `number_of_layers` layer, and discard other layers.
    """
    # hacky codes, avoid reading the ckpt at the same time
    time.sleep(15 * (local_rank % 4) + 5)
    print(f"Loading llama model for rank {local_rank}")
    model = LlamaForCausalLM.from_pretrained(llama_path).model.eval()
    print(
        f"===== Truncating llama, which originally has {len(model.layers)} layers ====="
    )
    truncate_layers(model, number_of_layers)
    print(
        f"===== The truncated llama model has {len(model.layers)} transformer layers ====="
    )
    return {"llama_model": model.to(f"cuda:{local_rank}")}


## Infer llama model


@torch.no_grad()
def extract_llama_embeds(
    model: LlamaModel,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor] = None,
    out_embed_layer: int = None,
) -> torch.Tensor:
    # Set the layer whose embed to be returned
    out_embed_layer = (
        len(model.layers) - 1 if out_embed_layer is None else out_embed_layer
    )

    batch_size, seq_length = input_ids.shape
    past_key_values_length = 0

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    inputs_embeds = model.embed_tokens(input_ids)
    attention_mask = model._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds
    for idx, decoder_layer in enumerate(model.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
        if idx == out_embed_layer:
            return hidden_states
    return hidden_states


@torch.no_grad()
def infer_llama_embeds(
    llama_model,
    input_ids,
    attention_mask,
    micro_batch: int = -1,
    output_embed_layer: int = -1,
):
    if micro_batch == -1:
        return extract_llama_embeds(
            model=llama_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            out_embed_layer=output_embed_layer,
        ).detach()
    else:
        text_embeds = []
        for xl in range(0, input_ids.shape[0], micro_batch):
            xr = min(xl + micro_batch, input_ids.shape[0])
            text_embeds.append(
                extract_llama_embeds(
                    model=llama_model,
                    input_ids=input_ids[xl:xr, ...],
                    attention_mask=attention_mask[xl:xr, ...],
                    out_embed_layer=output_embed_layer,
                ).detach()
            )
        return torch.cat(text_embeds, dim=0)


class LoraLLamaTextEncoder(nn.Module):
    def __init__(self, pretrained_model="llama-7b", emb_dim: int = 128):
        super(LoraLLamaTextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.text_linear = nn.Linear(4096, emb_dim)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=64, # TODO: configurable hyperparameter
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        quantize_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(
            pretrained_model,
            quantization_config=quantize_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ["LOCAL_RANK"])}
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.gradient_checkpointing_enable()

    def _infer_llama(self, input_ids, attention_mask, token_type_ids=None):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.to(attention_mask.device)
        last_hidden_state = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

        text_embeds = (
            torch.sum(attention_mask.unsqueeze(-1) * last_hidden_state, dim=1, keepdim=False)
                / torch.sum(attention_mask, dim=1, keepdim=True)
        )
        return text_embeds
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        text_output = self._infer_llama(input_ids, attention_mask, token_type_ids)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed

class MultiLoraLLamaTextEncoder(nn.Module):
    def __init__(self, pretrained_model="llama-7b", emb_dim: int = 128):
        super(LoraLLamaTextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.text_linear = nn.Linear(4096, emb_dim)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=32, # TODO: configurable hyperparameter
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        quantize_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(
            pretrained_model,
            quantization_config=quantize_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ["LOCAL_RANK"])}
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.gradient_checkpointing_enable()

    def _infer_llama(self, input_ids, attention_mask, token_type_ids=None):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.to(attention_mask.device)
        last_hidden_state = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

        text_embeds = (
            torch.sum(attention_mask.unsqueeze(-1) * last_hidden_state, dim=1, keepdim=False)
                / torch.sum(attention_mask, dim=1, keepdim=True)
        )
        return text_embeds
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        text_output = self._infer_llama(input_ids, attention_mask, token_type_ids)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed


class LLamaTextEncoder(nn.Module):
    def __init__(self, pretrained_model="llama-7b", emb_dim: int = 128):
        super(LLamaTextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.__dict__["llama_model"] = load_llama(pretrained_model)
        self.text_linear = nn.Linear(4096, emb_dim)
        self.micro_batch_size = 8
    
    @torch.no_grad()
    def _infer_llama(self, input_ids, attention_mask, token_type_ids):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.to(attention_mask.device)
        text_embeds = []
        for xl in range(0, input_ids.size(0), self.micro_batch_size):
            xr = min(xl + self.micro_batch_size, input_ids.size(0))
            i = input_ids[xl:xr, ...]
            m = attention_mask[xl:xr, ...]
            p = position_ids[xl:xr, ...]
            last_hidden_state = self.llama_model(
                input_ids=i,
                attention_mask=m,
                position_ids=p,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            text_embed = (
                torch.sum(m.unsqueeze(-1) * last_hidden_state, dim=1, keepdim=False)
                    / torch.sum(m, dim=1, keepdim=True)
            )
            text_embeds.append(text_embed)
        text_embed = torch.cat(text_embeds, dim=0)
        return text_embed
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        text_output = self._infer_llama(input_ids, attention_mask, token_type_ids)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed
    
    def manually_to_device(self, device):
        self.llama_model.to(device)


class T5TextEncoder(nn.Module):
    def __init__(self, pretrained_model="t5-xxl", emb_dim: int = 128):
        super(T5TextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.__dict__["t5_encoder_model"] = load_t5(pretrained_model)
        self.text_linear = nn.Linear(4096, emb_dim)
        self.micro_batch_size = 8
    
    @torch.no_grad()
    def _infer_t5(self, input_ids, attention_mask, token_type_ids):
        text_embeds = []
        for xl in range(0, input_ids.size(0), self.micro_batch_size):
            xr = min(xl + self.micro_batch_size, input_ids.size(0))
            i = input_ids[xl:xr, ...]
            m = attention_mask[xl:xr, ...]
            last_hidden_state = self.t5_encoder_model(
                input_ids=i,
                attention_mask=m
            ).last_hidden_state
            text_embed = (
                torch.sum(m.unsqueeze(-1) * last_hidden_state, dim=1, keepdim=False)
                    / torch.sum(m, dim=1, keepdim=True)
            )
            text_embeds.append(text_embed)
        text_embed = torch.cat(text_embeds, dim=0)
        return text_embed
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        text_output = self._infer_t5(input_ids, attention_mask, token_type_ids)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed
    
    def manually_to_device(self, device):
        self.t5_encoder_model.to(device)

class FinetuneT5TextEncoder(nn.Module):
    def __init__(self, pretrained_model="t5-3b", emb_dim: int = 128):
        super(FinetuneT5TextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.t5_encoder_model = load_t5_train(pretrained_model)
        self.text_linear = nn.Linear(1024, emb_dim)
        self.micro_batch_size = 8
        self.t5_encoder_model.gradient_checkpointing_enable() ##check

    
    def _infer_t5(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.t5_encoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  #[bsz, tsz, fsz]
        return last_hidden_state[:, 0, :] # TODO: check the first term of T5 tokenizer output

    
    def forward(self, input_ids, attention_mask, token_type_ids):
        text_output = self._infer_t5(input_ids, attention_mask, token_type_ids)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed
    
    def manually_to_device(self, device):
        self.t5_encoder_model.to(device)
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", emb_dim: int = 128, output_type="cls"):
        super(TextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.output_type = output_type
        self.text_model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.text_model.gradient_checkpointing_enable()
        if pretrained_model == 'bert-large-uncased':
            input_dim = 1024
        elif pretrained_model == "bert-base-multilingual-cased":
            input_dim = 768
        elif pretrained_model == "bert-base-chinese":
            input_dim = 768
        self.text_linear = nn.Linear(input_dim, emb_dim)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]
        if self.output_type == "cls":
            text_output = last_hidden_state[:, 0, :]
        else:
            text_output = last_hidden_state
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed
    
    def flops_fn(self, batch_size, seq_len):
        flops = 0
        # add bert flops
        bert_config = self.text_model.config
        flops += bert_calculator(
            bert_config.num_hidden_layers,
            bert_config.hidden_size,
            bert_config.intermediate_size,
            bert_config.vocab_size,
            seq_len,
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * bert_config.hidden_size * self.emb_dim
        return flops

class MultiLoraLLamaTextEncoder(nn.Module):
    def __init__(self, pretrained_model="llama-7b", emb_dim: int = 128):
        super(MultiLoraLLamaTextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.text_linear = nn.Linear(4096*33, emb_dim)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=32, # TODO: configurable hyperparameter
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        quantize_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(
            pretrained_model,
            quantization_config=quantize_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ["LOCAL_RANK"])}
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.gradient_checkpointing_enable()

    def _infer_llama(self, input_ids, attention_mask, token_type_ids=None):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.to(attention_mask.device)
        all_hidden_state = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
        text_embeds_all = []
        for i in range(len(all_hidden_state)):
            text_embeds = (
                torch.sum(attention_mask.unsqueeze(-1) * all_hidden_state[i], dim=1, keepdim=False)
                    / torch.sum(attention_mask, dim=1, keepdim=True)
            )
            text_embeds_all.append(text_embeds)
        bsz = text_embeds.shape[0]
        #feature_size = text_embeds.shape[1]
        #print(bsz, feature_size)         # （33, bs, 4096） 
        text_embeds_all = torch.stack(text_embeds_all).permute(1, 0, 2).reshape([bsz, -1]) # [bs, 33, 4096]
        return text_embeds_all
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        text_output = self._infer_llama(input_ids, attention_mask, token_type_ids)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed

class ClapEncoder(nn.Module):
    def __init__(self, pretrained_model="laion/clap-htsat-unfused", emb_dim: int = 128, output_type="cls"):
        super(ClapEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.text_model = ClapModel.from_pretrained(pretrained_model)
        self.text_linear = nn.Linear(512, emb_dim)
        self.output_type = output_type

    def _infer(self, input_ids, attention_mask):
        outputs = self.text_model.get_text_features(input_ids, attention_mask=attention_mask)
        return outputs

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        text_output = self._infer(input_ids=input_ids, attention_mask=attention_mask)
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed

    def manually_to_device(self, device):
        self.text_model.to(device)


def get_text_encoder(text_encoder="bert", emb_dim=128, output_type="cls", model_path=None):
    if text_encoder == "bert":
        return TextEncoder("bert-large-uncased", emb_dim, output_type)
    elif text_encoder == "multilingual":
        return TextEncoder("bert-base-multilingual-cased", emb_dim, output_type)
    elif text_encoder == "chinese":
        return TextEncoder("bert-base-chinese", emb_dim, output_type)
    elif text_encoder =="llama":
        assert model_path is not None
        return LLamaTextEncoder(model_path, emb_dim=emb_dim)
    elif text_encoder =="multi-llama-lora":
        assert model_path is not None
        return MultiLoraLLamaTextEncoder(model_path, emb_dim=emb_dim)
    elif text_encoder =="llama-lora":
        assert model_path is not None
        return LoraLLamaTextEncoder(model_path, emb_dim=emb_dim)
    elif text_encoder == "t5":
        return T5TextEncoder(model_path, emb_dim=emb_dim)
    elif text_encoder == "t5-finetune":
        return FinetuneT5TextEncoder(model_path, emb_dim=emb_dim)
    elif text_encoder == "clap":
        return ClapEncoder("laion/larger_clap_general", emb_dim=emb_dim)
    else:
        raise NotImplementedError