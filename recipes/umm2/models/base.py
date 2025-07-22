import os
import logging
import torch
import torch.nn as nn

from typing import Dict, List, Union, Callable
from functools import partial
from hydra.utils import instantiate

from easydict import EasyDict
from mariana.models.audio.conformer import ConformerLayer
from mariana.models.audio.positional_encoding import RotaryPositionalEncoding
from recipes.umm2.models.umm_conformer import ConformerEncoderLayer

logger = logging.getLogger(__name__)


class BaseStage(nn.Module):
    r"""Base class for all stages. Provide a generic interface for data computing.

    :class:`~samantha.core.BaseStage` defines functions such as pre-processor,
    post-processor, encoder, decoder. It follows yaml-style format. Users can
    define what the function takes and provides after processing. A sample
    definition is as below:

    .. code-block:: yaml

        stages:
          - !new:recipes.sample_project.modules.model.CNNModelStage
             takes: [img_tensor]
             provides: [label]
             weight: 0.5

    Args:
        takes (List[str]): the names that this stage will take.
        provides (List[str]): the names that this stage will output.
        bypasses (List[str]): the names that this stage will bypass without being processed by the module.
        weight (float): weight for the external loss combination

    """

    def __init__(
            self, 
            takes: List[str], 
            provides: List[str], 
            bypasses: List[str] = [],
            task: str = None,
            loss_weight: float = None, 
            lr_ratio: float = 1.0,
            is_frozen: bool = False,
            config = None,
            *args, 
            **kwargs
        ):
        super().__init__()
        self.takes = set(takes)
        self.provides = set(provides) # loss and flops appended if fields exist
        self.bypasses = set(bypasses)
        self.task = task
        self.loss_weight = loss_weight
        self.lr_ratio = lr_ratio
        self.is_frozen = is_frozen

        if config is not None:
            conformer_config = EasyDict({
                # backbone
                'conformer_normalize_before': True,
                'conformer_attention_heads': config.num_attention_heads,
                # conformer_mask_topology: None
                'conformer_linear_units': config.intermediate_size,
                'conformer_num_blocks': config.num_hidden_layers,
                'conformer_dropout_rate': config.hidden_dropout,
                'conformer_positional_dropout_rate': 0.0,
                'conformer_attention_dropout_rate': config.attention_dropout,
                'conformer_positionwise_layer_type': 'linear',
                'conformer_activation_fn': 'gelu',
                'conformer_positionwise_conv_kernel_size': 1,
                'conformer_macaron_style': 1,
                'conformer_pos_enc_layer_type': 'rope',
                'conformer_selfattention_layer_type': 'early_rope_selfattn',
                'conformer_layer_order': 'mhsa_before_conv',
                'conformer_use_cnn_module': 1,
                'conformer_cnn_module': 'ConvolutionModule',
                'conformer_cnn_module_kernel': str(config.conv_depthwise_kernel_size),
                'conformer_cnn_norm_type': 'layer_norm',
                'conformer_layernorm_interval': 0,
                'conformer_weight_scale': 1.0,
                'conformer_half_pooling': 0,
                'backbone_memory_size': config.hidden_size,
                'dropout': 0.0,
                "squeeze_mem": True,
                'attn_amp_enable': True,
                'flash_attn': True,
                'conformer_mask_type': 'default',
            })
            
            if not config.use_fused_kernel:
                self.encoder_layers = nn.ModuleList(
                    [ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
                )
            else:
                self.encoder_layers = nn.Sequential(*[
                    ConformerLayer(conformer_config, None, i) for i in range(config.num_hidden_layers)
                ])

                self.pos_enc = RotaryPositionalEncoding(config.hidden_size / config.num_attention_heads, 
                    config, variant_type=2 if config.rope_enhance_pos > 0 else 0)
 
    def validattr(self, attr_name):
        return hasattr(self, attr_name) and getattr(self, attr_name) is not None

    def forward(self, data: Dict) -> Dict:
        r"""Perform forward based on takes and provides

        Args:
            data (dict): A dictionary contains all keys in ``takes``.

        Returns:
            A dictionary contains all keys in ``provides``.

        # """
        raise NotImplementedError(
            "Subclass must provide implementation of this functon"
        )

    
    # def _compute(self, *args):
    #     # do some awesome things
    #     raise NotImplementedError(
    #         "Subclass must provide implementation of this functon"
    #     )


class SpanModel(nn.Module):
    r"""Abstract module that collects head modules that source from the same input.

    .. code-block:: yaml

        spans:
            head_0: BaseStage ...
            head_1: BaseStage ...
            ...

    Args:
        input_names (List[str]): the names that this stage will take.
        output_names (List[str]): the names that this stage will output; 
            however, there are no constraints since they have been applied in each span model with BaseStage

    """
    def __init__(
        self,
        spans: Dict[str, Union[BaseStage, Callable]],
        input_names: List[str],
        output_names: List[str],
        bypass_names: List[str] = [],
    ):
        super(SpanModel, self).__init__()

        if not spans:
            raise ValueError("spans must be specified.")

        full_spans = {}
        for k, span in spans.items():
            if isinstance(span, partial):
                full_spans[k] = span()
            else:
                full_spans[k] = span

        if not input_names: 
            raise ValueError("input_names must be specified.")

        if not output_names:
            raise ValueError("output_names must be specified.")

        self.task_names = list(full_spans.keys())
        self.spans = nn.ModuleList([full_spans[k] for k in self.task_names])
        self.input_names = input_names
        self.output_names = output_names
        self.bypass_names = bypass_names

    def validattr(self, attr_name):
        return hasattr(self, attr_name) and getattr(self, attr_name) is not None

    def forward(self, batch):
        if isinstance(batch, dict):
            data = batch
        else:
            if not isinstance(batch, List):
                batch = [batch]
            data = dict(zip(self.input_names, batch))

        # output = {k: batch[k] for k in self.bypass_names}
        # we need all keys from batch
        output = batch  
        
        flops = data['flops'] if 'flops' in data else 0
        loss = 0
        for task, span_module in zip(self.task_names, self.spans):
            out = span_module.forward(data)

            if 'flops' in out:
                flops += out['flops']
            if 'loss' in out and out['loss'] is not None:
                loss += out['loss']

            output.update(out)
        
        output['flops'] = flops
        output['loss'] = loss

        return output 


class PipelineModel(nn.Module):
    def __init__(
        self,
        stages: List[Union[BaseStage, SpanModel, Callable]],
        input_names: List[str],
        output_names: List[str],
    ):
        super(PipelineModel, self).__init__()

        if not stages:
            raise ValueError("stages must be specified.")
        full_stages = []
        for stage in stages:
            if isinstance(stage, partial):
                full_stages.append(stage()) 
            else:
                full_stages.append(stage)

        if not input_names:
            raise ValueError("input_names must be specified.")

        if not output_names:
            raise ValueError("output_names must be specified.")

        self.stages = nn.ModuleList(full_stages)
        self.input_names = input_names
        self.output_names = output_names + ["loss", "flops"]


    def forward(self, batch):
        r"""Perform forward computation.

        This method will call :func:`forward <BaseStage.forward>` function of
        :attr:`stages <self.stages>` sequentially.

        Args:
            batch (List[Tensor]): input batch which contains
             ``len(input_names)`` tensors.

        """
        if isinstance(batch, dict):
            data = batch
        else:
            if not isinstance(batch, List):
                batch = [batch]
            data = dict(zip(self.input_names, batch))

        flops = 0
        loss = 0

        for i, stage in enumerate(self.stages):
            flops = flops + data['flops'] if 'flops' in data else flops
            loss =  loss + data['loss'] if 'loss' in data and data['loss'] is not None else loss 
            data =  stage.forward(data)

        data['flops'] = data['flops'] + flops if 'flops' in data else flops
        data['loss'] = data['loss'] + loss if 'loss' in data else loss

        #self.output_names += [k for k in data if k.startswith('loss_')]  
        #self.output_names += [k for k in data if k.startswith('aux/')]  

        # derpecated, the output_dict will include all keys in data and intermediate result
        # in pl_module log_dict_cached will filter out none-scalar values
        output_dict = {k: v for k, v in data.items()}
        
        return output_dict


class UMM_Pipeline(torch.nn.Module):
    
    def  __init__(
        self,
        config,
        backbone: Callable,
        spans: Dict[str, Union[BaseStage, Callable]]
    ):
        super().__init__()

        self.config = config
        self.backbone = instantiate(backbone)
        self.spans = nn.ModuleDict({
            name: instantiate(module_cfg)
            for name, module_cfg in spans.items()
        })

    def validattr(self, attr_name):
        return hasattr(self, attr_name) and getattr(self, attr_name) is not None


    def forward(self, batch):
        # torch.distributed.breakpoint(0)
        flops = 0 
        loss = 0

        backbone_output_dict = self.backbone(batch)
        flops += backbone_output_dict.get('flops', 0)

        batch.update(backbone_output_dict)

        task_outputs = {}
        for task_name, module in self.spans.items():
            task_outputs[task_name] = module(batch)

        
        # Flops and loss computation
        output_dict = {}

        for task_name, task_output in task_outputs.items():
            flops += task_output.get("flops", 0)
            weight = getattr(task_output, "loss_weight", 1.0)
            task_loss = task_output.get("loss", 0)
            output_dict[f"loss_{task_name}"] = task_loss
            loss += weight * task_loss

            for k, v in task_output.items():
                if isinstance(k, str) and k.startswith("aux/"):
                    output_dict[k] = v

        output_dict['flops'] = flops
        output_dict['loss'] = loss
   

        return output_dict
        