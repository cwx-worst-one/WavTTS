
import torch
import logging

import transformers as HF
from s3a.providers.ctiga.models.gpt import GPTLMHeadModel
from samantha.models.llama import Llama, LlamaConfig
from tests.model_test.models.tester import ModelTester
from s3a.providers.ctiga.models.llama import llama_config_to_gpt2_config


class LlamaModelTester(ModelTester):
    def __init__(self, args):
        super().__init__(args)
        self.model_dict = {'huggingface': HF.LlamaForCausalLM,
                           'janne': Llama,
                           'ctiga': GPTLMHeadModel
                           }
        self.llama_configs = {
            "300M": HF.LlamaConfig(
                num_hidden_layers=24, num_attention_heads=16, hidden_size=1024
            ),
            "7B": HF.LlamaConfig(
                num_hidden_layers=32, num_attention_heads=32, hidden_size=4096
            ),
            "13B": HF.LlamaConfig(
                num_hidden_layers=40, num_attention_heads=40, hidden_size=5120
            ),
            "30B": HF.LlamaConfig(
                num_hidden_layers=60, num_attention_heads=52, hidden_size=6656
            ),
            "65B": HF.LlamaConfig(
                num_hidden_layers=80, num_attention_heads=64, hidden_size=8192
            ),
        }

        self.unsupported_provider = []
        self.batch_size = args.benchmark_params.variables.batch_size
        self.max_seqlen = args.benchmark_params.variables.max_seqlen

    def load_from_pretrained(self, provider, checkpoint_path):
        if provider == 'huggingface':
            model = self.model_dict[provider].from_pretrained(checkpoint_path)
        else:
            logging.warning(
                f'{__name__}: {self.args.model_params.model} cannot load pretrained model for provider `{provider}`.')
            self.unsupported_provider.append(provider)
            return None
        return model

    def gen_train_config(self, provider):
        try:
            config = self.llama_configs[self.args.model_params.model_name]
        except KeyError:
            logging.warning(
                f"{__name__}: f'{self.args.model_params.model} doesn't support {self.args.model_params.model_name} model.")
            return None
        config.vocab_size = self.args.model_params.vocab_size
        N = self.args.model_params.N
        config.intermediate_size = (
            int(4 * config.hidden_size * 2 // 3) - 1) // N * N + N
        config.max_position_embeddings = self.args.model_params.max_position_embeddings
        config.torch_dtype = str(self.args.model_params.dtype)
        config.bos_token_id = self.args.model_params.bos_token_id
        config.eos_token_id = self.args.model_params.eos_token_id

        if provider == "huggingface":
            pass
        elif provider == "ctiga":
            config = llama_config_to_gpt2_config(config)
            config.use_flash_attn = True
            config.fused_bias_fc = True
            config.fused_mlp = False  # We don't have fused GatedMLP yet
            config.fused_dropout_add_ln = True
            config.residual_in_fp32 = True
            config.rotary_emb_compat = self.args.model_params.rotary_emb_compat
        elif provider == "janne":
            config = LlamaConfig(
                n_layer=config.num_hidden_layers,
                n_head=config.num_attention_heads,
                n_embd=config.hidden_size,
                vocab_size=config.vocab_size,
            )
        else:
            return None

        return config

    def gen_generate_config(self, provider):
        logging.warning(
            f'{__name__}: action `generate` for Llama model has not been implemented.')
        exit(0)

    def gen_model(self, provider):
        if provider not in self.model_dict.keys():
            self.unsupported_provider.append(provider)
            logging.warning(
                f'{__name__}: {self.args.model_params.model} does not support provider `{provider}`.')
            return None, 0

        try:
            checkpoint_path = self.args.model_params.from_pretrained[provider]
            if checkpoint_path != "":
                try:
                    logging.info(
                        f"load {self.args.model_params.model} {self.args.model_params.model_name} {provider} model from pretrained model.")
                    model = self.load_from_pretrained(provider, checkpoint_path)
                    model = model.cuda().to(dtype=self.args.model_params.dtype)
                except:
                    logging.error(
                        f'{__name__}: load pretrained {self.args.model_params.model} {provider} model failed.')
                    raise
            else:
                return None, 0
        except KeyError:
            logging.info(
                f"create {self.args.model_params.model} {self.args.model_params.model_name} {provider} model.")

            config = self.gen_config(provider)
            if config is None:
                logging.warning(
                    f'{__name__}: {self.args.model_params.model} create `{self.action}` config for provider `{provider}` failed.')
                self.unsupported_provider.append(provider)
                return None, 0

            model = self.model_dict[provider](config)
            model = model.cuda().to(dtype=self.args.model_params.dtype)

        model_size = self.get_model_size(provider, model)

        return model, model_size

    def gen_all_train_inputs(self, provider):
        if provider in self.unsupported_provider:
            return None
        else:
            if provider not in ["huggingface", "janne", "ctiga"]:
                logging.warning(
                    f'{__name__}: generate `{self.action}` inputs for provider `{provider}` is not implemented.')
                return None

        inputs = {}
        for msl in self.max_seqlen:
            inputs_ = {}
            for bs in self.batch_size:
                inputs__ = None
                input_ids = torch.randint(0, self.args.model_params.vocab_size,
                                          (bs, msl),
                                          dtype=torch.long, device='cuda')
                if provider == "huggingface":
                    inputs__ = {"input_ids": input_ids, "return_dict": False}
                elif provider == "janne":
                    inputs__ = {"input_ids": input_ids}
                elif provider == "ctiga":
                    inputs__ = {"input_ids": input_ids}
                inputs_[bs] = {"inputs": inputs__}
            inputs[msl] = inputs_

        return inputs

    def gen_all_inputs(self, provider):
        if self.action == 'train':
            return self.gen_all_train_inputs(provider)
        else:
            return None

    def gen_inputs(self, metric, provider):
        inputs = None

        if metric in ['time', 'gpu_memory']:
            inputs = self.gen_all_inputs(provider)
        else:
            logging.warning(
                f'{__name__}: generate inputs for metric `{metric}` is not implemented.')

        return inputs
