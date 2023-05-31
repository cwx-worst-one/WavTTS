import logging
import torch

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as GPT2LMHeadModelHF
from s3a.providers.ctiga.models.gpt import GPTLMHeadModel
from s3a.providers.xperf.models.gpt2 import FTGPT2Model
from tests.model_test.models.tester import ModelTester


class GPT2ModelTester(ModelTester):
    def __init__(self, args):
        super().__init__(args)
        self.model_dict = {'huggingface': GPT2LMHeadModelHF,
                           'xperf': FTGPT2Model,
                           'ctiga': GPTLMHeadModel
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
        config = GPT2Config(
            vocab_size=self.args.model_params.vocab_size,
            n_positions=6000,
            n_ctx=6000,
            n_embd=1024,
            n_layer=self.args.model_params.nlayer,
            n_head=self.args.model_params.nhead,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            summary_type='mean',
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            bos_token_id=1225,
            eos_token_id=1226,
        )

        if provider == "huggingface":
            pass
        elif provider == "xperf":
            config.__dict__["max_position_embeddings"] = 6000
            config.__dict__["n_embed"] = 1024
            config.__dict__["hidden_size"] = 1024
            config.__dict__["n_inner"] = None
            config.__dict__["use_ft_linear"] = self.args.benchmark_params.use_ft_linear
            config.__dict__[
                "use_ft_layernorm"] = self.args.benchmark_params.use_ft_layernorm
            config.__dict__[
                "use_ft_flash_attn"] = self.args.benchmark_params.use_ft_attn
            config.__dict__["scale_attn_weights"] = True
            config.__dict__["scale_attn_by_inverse_layer_idx"] = False
            config.__dict__[
                "gradient_checkpointing"] = self.args.benchmark_params.gradient_checkpointing
        elif provider == "ctiga":
            config.use_flash_attn = True
            config.fused_bias_fc = self.args.benchmark_params.fused_bias_fc
            config.fused_ft_kernel = self.args.benchmark_params.fused_ft_kernel
            config.fused_mlp = self.args.benchmark_params.fused_mlp
            config.fused_dropout_add_ln = self.args.benchmark_params.fused_dropout_add_ln
            config.residual_in_fp32 = self.args.benchmark_params.residual_in_fp32
        else:
            return None

        return config

    def gen_generate_config(self, provider):
        config = GPT2Config(
            vocab_size=self.args.model_params.vocab_size,
            n_positions=6000,
            n_ctx=6000,
            n_embd=1024,
            n_layer=self.args.model_params.nlayer,
            n_head=self.args.model_params.nhead,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            summary_type='mean',
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            bos_token_id=1225,
            eos_token_id=1226,
        )

        if provider == "huggingface":
            pass
        elif provider == "ctiga":
            config.use_flash_attn = True
            config.fused_bias_fc = self.args.benchmark_params.fused_bias_fc
            config.fused_ft_kernel = self.args.benchmark_params.fused_ft_kernel
            config.fused_mlp = self.args.benchmark_params.fused_mlp
            config.fused_dropout_add_ln = self.args.benchmark_params.fused_dropout_add_ln
            config.residual_in_fp32 = self.args.benchmark_params.residual_in_fp32
        else:
            return None

        return config

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
                        f"load {self.args.model_params.model} {provider} model from pretrained model.")
                    model = self.load_from_pretrained(provider, checkpoint_path)
                    model = model.cuda().to(dtype=self.args.model_params.dtype)
                except:
                    logging.error(
                        f'{__name__}: load pretrained {self.args.model_params.model} {provider} model failed.')
                    raise
            else:
                return None, 0
        except KeyError:
            logging.info(f"create {self.args.model_params.model} {provider} model.")

            config = self.gen_config(provider)
            if config is None:
                logging.warning(
                    f'{__name__}: {self.args.model_params.model} create `{self.action}` config for provider `{provider}` failed.')
                self.unsupported_provider.append(provider)
                return None, 0

            model = self.model_dict[provider](config)

            if self.action == 'train':
                if self.args.benchmark_params.gradient_checkpointing and provider != "xperf":
                    model.gradient_checkpointing_enable()
                model = model.cuda().to(dtype=self.args.model_params.dtype)
            elif self.action == 'generate':
                model = model.cuda().to(dtype=self.args.model_params.dtype)

        model_size = self.get_model_size(provider, model)

        return model, model_size

    def gen_all_generate_inputs(self, provider):
        if provider in self.unsupported_provider:
            return None
        else:
            if provider not in ["huggingface", "ctiga"]:
                logging.warning(
                    f'{__name__}: generate `{self.action}` inputs for provider `{provider}` is not implemented.')
                return None

        inputs = {}
        for msl in self.max_seqlen:
            inputs_ = {}
            for bs in self.batch_size:
                inputs__ = None
                input_ids = torch.randint(0, self.args.model_params.vocab_size, (
                    bs, 1), dtype=torch.long, device='cuda')
                if provider == "huggingface":
                    inputs__ = {"input_ids": input_ids, "max_length": msl,
                                "return_dict_in_generate": True, "output_scores": True}
                elif provider == "ctiga":
                    inputs__ = {"input_ids": input_ids,
                                "max_length": msl,
                                "fused_ft_kernel": self.args.benchmark_params.fused_ft_kernel,
                                "return_dict_in_generate": True,
                                "output_scores": True,
                                "timing": False}
                inputs_[bs] = {"inputs": inputs__}
            inputs[msl] = inputs_

        return inputs

    def gen_all_train_inputs(self, provider):
        if provider in self.unsupported_provider:
            return None
        else:
            if provider not in ["huggingface", "xperf", "ctiga"]:
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
                elif provider == "xperf":
                    inputs__ = {"input_ids": input_ids}
                elif provider == "ctiga":
                    inputs__ = {"input_ids": input_ids}
                inputs_[bs] = {"inputs": inputs__}
            inputs[msl] = inputs_

        return inputs

    def gen_all_inputs(self, provider):
        if self.action == 'generate':
            return self.gen_all_generate_inputs(provider)
        elif self.action == 'train':
            return self.gen_all_train_inputs(provider)

    def gen_inputs(self, metric, provider):
        inputs = None

        if metric in ['time', 'gpu_memory']:
            inputs = self.gen_all_inputs(provider)
        else:
            logging.warning(
                f'{__name__}: generate inputs for metric `{metric}` is not implemented.')

        return inputs
