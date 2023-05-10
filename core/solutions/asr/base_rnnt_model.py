''' BaseRnntModel '''
# pylint: disable=unused-argument
# pylint: disable=too-many-lines
from collections import OrderedDict
import copy
import numpy as np
from packaging import version
import torch
from torch.nn import Linear
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.asr.las_decoder import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.inference import BaseInfer, INFERS, concat_global_states
from core.solutions.inference.utils import rnnt_rlt_neaten
from core.solutions.inference.rnnt_beam_search import (
    BatchBeamSearch,
    NonBatchBeamSearch,
)
from core.solutions.inference.infer import convert_to_np, mkdir_or_exist
from core.solutions.inference.rnnt_greedy_search import GreedySearch
from core.utils import FalconDict, get_rank
from core.extensions.panther_symbol import RnntJointerLMSymbolic
from core.models.layers.conformer import ConformerLayer

try:
    from panther_utilities.torch_symbolic.adaptive_softmax import PantherAdaptiveSoftmax
except ImportError:
    PantherAdaptiveSoftmax = None


def split_jointer_input(idx, step, **kwargs):
    '''split input for jointer and criterion.'''
    all_names = (
        'encoder_out',
        'predictor_out',
        'mtl_logits',
        'target_lengths',
        'adaptive_tgt_indices',
        'src_mask',
        'char',
        'char_mask',
        'backbone_mask',
        'target_lengths',
        'eos',
    )
    bsz = kwargs['encoder_out'].shape[0]
    start, stop = idx * step, min((idx + 1) * step, bsz)

    out = OrderedDict(jointer_scale=(stop - start) / bsz)
    for name in all_names:
        val = kwargs.get(name, None)
        if isinstance(val, (torch.Tensor, np.ndarray)):
            # all tensor shape[0] is bsz
            assert val.shape[0] == bsz
            val = val[start:stop]
        elif isinstance(val, (list, tuple)):
            val = val[idx]
        out[name] = val
    # fit for rnnt loss
    max_tgt_len = out['target_lengths'].max()
    out['char'] = out['char'][:, :max_tgt_len].contiguous()
    return out


def merge_jointer_output(outputs):
    '''merge multi jointer output.'''
    scale_names = ('backward_loss', 'loss', 'nll_loss', 'acc')
    all_names = (
        'utt_num',
        'backward_loss',
        'loss',
        'nll_loss',
        'acc',
        'frame_size',
        'tgt_size',
        'cer',
        'dist',
    )
    final_out = OrderedDict()
    for name in all_names:
        if name in scale_names:
            val = sum(out[name] * out['jointer_scale'] for out in outputs)
        else:
            val = sum(out[name] for out in outputs)
        final_out[name] = val
    # fake loss for loss.backward() in runner.
    final_out['backward_loss'] = final_out['backward_loss'].detach().requires_grad_()
    return final_out


# TODO: rename Exporter to Infer
class RnntPredictorExporter(BaseInfer):
    '''export rnnt predictor to onnx'''

    NAME = 'predictor'
    INPUTS = [
        FalconDict(name='prev_char', type=torch.int64, shape=['B']),
    ]
    OUTPUTS = [
        FalconDict(name='output', type=torch.float32, shape=['B', 1, -1]),
    ]

    def __init__(self, predictor_module, **kwargs):
        super().__init__(kwargs)
        self.predictor_module = predictor_module

        # other args
        self._convert_stream_flag = kwargs.get('predictor_convert_stream', True)
        self._export_stream_flag = kwargs.get('predictor_export_stream', False)
        self._reduced_embedding_flag = False
        if kwargs['predictor_type'] == 'ReducedEmbeddingPredictor':
            self._reduced_embedding_flag = True
        if self._export_stream_flag:
            self._convert_stream_flag = False
        if self._convert_stream_flag or self._export_stream_flag:
            # NOTE(liyong): compatible with unmodified module, remove this later.
            state_size = getattr(self.predictor_module, 'state_size', 0)
            self._inputs.append(
                FalconDict(name='global_state_in', type=torch.float32, shape=['Batch', state_size])
            )
            self._outputs.append(
                FalconDict(name='global_state_out', type=torch.float32, shape=['Batch', state_size])
            )

    def forward(self, prev_char, global_state_in=None, **_kwargs):
        '''
        Forward for RNN-T base module
        '''
        if global_state_in is None:
            prev_char = prev_char.unsqueeze(1)
            predictor_out, _ = self.predictor_module.forward(prev_char)
            return predictor_out
        predictor_out, states = self.predictor_module.forward_step(prev_char, global_state_in)
        predictor_out = predictor_out.unsqueeze(1)
        return predictor_out, states

    def forward_step(self, prev_char, global_state_in=None):
        '''
        forward step.

        Args:
          prev_char[torch.Tensor]: [Batch]
          global_state_in[torch.Tensor]: [Batch, state_size]
        '''
        pred_out, states = self.predictor_module.forward_step(prev_char, global_state_in)
        pred_out = pred_out.unsqueeze(1)
        states = concat_global_states(*states)
        return pred_out, states

    def sample_inputs(self):
        '''
        sample some inputs as example.
        may be used for jit trace or test.
        TODO: how to get real data.
        '''
        prev_char = self._generate_input_data(0, high=self._cfg.get('tgt_vocab_size', 1))
        if self._convert_stream_flag or self._export_stream_flag:
            global_state_in = self._generate_input_data(1)
            return prev_char, global_state_in
        return prev_char

    def sample_inputs_from_dataloader(self):
        '''
        sample some real data from dataloader.
        '''
        prev_char = []
        num_batch = 4
        for _ in range(num_batch):
            batch_data = self._data_loader.next()
            prev_char.append(batch_data['prev_char'].detach().cpu().numpy())
        datas = {'prev_char': prev_char}
        return datas

    def prepare_export(self):
        if self._reduced_embedding_flag and get_rank() == 0:
            if self.predictor_module.embed_tokens.__class__.__name__ == 'SlimModule':
                embed = torch.fake_quantize_per_tensor_affine(
                    self.predictor_module.embed_tokens.weight[0].contiguous(),
                    self.predictor_module.embed_tokens.pre_ops['QATQuantize']
                    .weight_pre_process_modules.get_scale()
                    .item()
                    / 127,
                    0,
                    -127,
                    127,
                )
            else:
                embed = self.predictor_module.embed_tokens.weight[0]
            embed = convert_to_np(embed)
            repeat_num = 4
            embed = np.tile(embed, repeat_num).astype(np.float32)
            mkdir_or_exist(self._onnx_dir)
            np.savetxt("{}/embed.txt".format(self._onnx_dir), embed, fmt='%f', delimiter='\n')


class RnntEncoderExporter(BaseInfer):
    '''export rnnt encoder to onnx'''

    NAME = 'encoder'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='output', type=torch.float32, shape=['B', 'T', -1]),
    ]
    ORIGIN_TEST_ATOL = 6e-4
    OPTIMIZED_TEST_ATOL = 6e-4

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        downsampling_size = kwargs.get('downsampling_size', 1)
        if downsampling_size > 1:
            self._stream_trial_t = [downsampling_size, 2 * downsampling_size, 3 * downsampling_size]
        self.acoustic_front_end_module = model.acoustic_front_end_module
        self.acoustic_backbone_module = model.acoustic_backbone_module
        self.acoustic_head_module = model.acoustic_head_module
        self.backbone_pool_module = model.backbone_pool_module
        self.encoder_backbone = model.encoder_backbone  # the encoder_backbone is function

        # fill out inputs and outputs info
        self._stack_frame = kwargs.get('onnx_stack_frame', 80)
        self._inputs[0].shape[2] = self._stack_frame  # eg: [B, T, 80] or [B, T, 320]
        self._with_mask = kwargs.get('onnx_with_mask', False)
        self._return_backbone = kwargs.get('return_backbone', False)
        self._convert_stream_flag = kwargs.get('encoder_convert_stream', True)
        self._with_mask = self._with_mask and not self._convert_stream_flag
        # streaming model does not support mask
        if self._with_mask:
            self._inputs.append(FalconDict(name='fbank_mask', type=torch.float32, shape=['B', 'T']))
            self._outputs.append(
                FalconDict(name='encoder_mask', type=torch.float32, shape=['B', 'T'])
            )
        if self._return_backbone:
            self._outputs.append(
                FalconDict(name='backbone_out', type=torch.float32, shape=['B', 'T', -1])
            )

        # other args
        if self._convert_stream_flag:
            # NOTE(liyong): compatible with unmodified module, remove this later.
            self.global_state_size = getattr(
                self.acoustic_front_end_module, 'state_size', 0
            ) + getattr(self.acoustic_backbone_module, 'state_size', 0)
            self._inputs.append(
                FalconDict(
                    name='global_state_in',
                    type=torch.float32,
                    shape=['Batch', self.global_state_size],
                )
            )
            self._inputs.append(
                FalconDict(
                    name='x_sign',
                    type=torch.int32,
                    shape=[1],
                )
            )
            self._outputs.append(
                FalconDict(
                    name='global_state_out',
                    type=torch.float32,
                    shape=['Batch', self.global_state_size],
                )
            )
        self._conv_w_list = kwargs.get('conv_w_list', None)

        # set stream mode if with dualmode and export stream model
        if hasattr(self.acoustic_backbone_module, 'set_stream_mode') and self._convert_stream_flag:
            self.acoustic_backbone_module.set_stream_mode(True)

    @staticmethod
    def convert_states(states):
        '''
        convert states
        lstm: list(tensor)
        dfsmn: tensor
        '''
        if isinstance(states, (np.ndarray, torch.Tensor)):
            return [states]
        if isinstance(states, (list, tuple)):
            return states
        return []

    def forward(self, fbank, fbank_mask=None, **_kwargs):
        '''
        Forward for RNN-T base module
        '''
        front_end_out, backbone_mask, frontend_shape = self.acoustic_front_end_module(
            fbank, fbank_mask
        )
        backbone_out = self.encoder_backbone(
            front_end_out, backbone_mask, frontend_shape=frontend_shape
        )
        if self.acoustic_head_module is not None:
            encoder_out = self.acoustic_head_module(backbone_out)
        else:
            return backbone_out
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
            if backbone_mask is not None:
                backbone_mask = backbone_mask[:, :-1][:, ::2]
        if backbone_mask is not None and self._return_backbone:
            return encoder_out, backbone_mask, backbone_out
        if backbone_mask is not None:
            return encoder_out, backbone_mask
        if self._return_backbone:
            return encoder_out, backbone_out
        return encoder_out

    def forward_step(
        self,
        fbank,
        fbank_mask=None,
        global_state_in=None,
        x_sign=None,
    ):
        '''
        Forward step for RNN-T base module
        '''

        front_end_state_size = self.acoustic_front_end_module.state_size
        backbone_state_size = self.acoustic_backbone_module.state_size
        assert global_state_in.shape[1] == front_end_state_size + backbone_state_size

        front_end_state_in = global_state_in[:, :front_end_state_size]
        (
            front_end_out,
            front_end_states,
            backbone_mask,
            frontend_shape,
        ) = self.acoustic_front_end_module.forward_step(
            fbank,
            fbank_mask,
            front_end_state_in,
            x_sign=x_sign,
        )

        backbone_out, backbone_states = self.acoustic_backbone_module.forward_step(
            front_end_out,
            backbone_mask,
            global_state_in[:, front_end_state_size:],
            x_sign=x_sign,
            frontend_shape=frontend_shape,
        )

        encoder_out = self.acoustic_head_module(backbone_out)
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
            if backbone_mask is not None:
                backbone_mask = backbone_mask[:, :-1][:, ::2]
        outs = [encoder_out]
        if backbone_mask is not None:
            outs.append(backbone_mask)
        if self._return_backbone:
            outs.append(backbone_out)

        front_end_states = self.convert_states(front_end_states)
        backbone_states = self.convert_states(backbone_states)
        global_state_out = concat_global_states(*front_end_states, *backbone_states)
        outs.append(global_state_out)
        return tuple(outs)

    def sample_inputs(self):
        '''
        sample some inputs as example.
        may be used for jit trace or test.
        TODO: how to get real data.
        '''
        fbank = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        datas = [fbank]
        if self._with_mask:
            fbank_mask = self._generate_input_data(1, method='ones', dynamic_axis=[1, 512])
            datas.append(fbank_mask)
        if self._convert_stream_flag:
            global_state_in = self._generate_input_data(0, dynamic_axis=[1, -1])
            datas.append(global_state_in)
        return tuple(datas)

    def sample_inputs_from_dataloader(self):
        '''
        sample some real data from dataloader.
        '''
        input_concat_size = self._cfg.get('input_concat_size')
        unfold = torch.nn.Unfold(
            (input_concat_size, 1), stride=(1, 1), padding=(input_concat_size // 2, 0)
        )
        fbank = []
        fbank_mask = []
        num_batch = 4
        for _ in range(num_batch):
            batch_data = self._data_loader.next()
            cur_fbank = batch_data['src']
            if self._stack_frame == 80:
                fbank.append(cur_fbank.detach().cpu().numpy())
            else:
                (bsz, _, _) = cur_fbank.size()
                unfold_fbank = (
                    (unfold(cur_fbank.unsqueeze(1)))
                    .contiguous()
                    .view(bsz, input_concat_size, -1, 80)
                )
                unfold_fbank = (
                    unfold_fbank.transpose(1, 2).contiguous().view(bsz, -1, input_concat_size * 80)
                )
                fbank.append(unfold_fbank.detach().cpu().numpy())
            cur_fbank_mask = batch_data['src_mask']
            unfold_fbank_mask = unfold(cur_fbank_mask.unsqueeze(1).unsqueeze(3))
            encoder_mask_out = unfold_fbank_mask[:, input_concat_size // 2, 1::2].contiguous()
            for _ in range(1, self._cfg.get('time_reduce_layer', 2)):
                encoder_mask_out = encoder_mask_out[:, 1::2].contiguous()
            fbank_mask.append(encoder_mask_out.detach().cpu().numpy())
        datas = {'fbank': fbank}
        if self._with_mask:
            datas['fbank_mask'] = fbank_mask
        return datas

    def get_module_map(self):
        '''
        Modify module_map to specify how to map module class to panther op
        '''
        for module in self.modules():
            # Conformer with Q/SVD cannot be exported as fused op
            # For Q in_proj is converted as SlimModule
            # For SVD in_proj is converted as Sequential
            if (
                isinstance(module, ConformerLayer)
                and not isinstance(module.self_attn.in_proj, Linear)
                and 'ConformerLayer' in self.module_map
            ):
                self.module_map.pop('ConformerLayer')
        return self.module_map


# NOTE: RnntJointerExporterForLM may merged with RnntJointerExporter
class RnntJointerExporterForLM(BaseInfer):
    '''export rnnt jointer to onnx'''

    NAME = 'jointer'
    INPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['batch', -1]),
        FalconDict(name='predictor_out', type=torch.float32, shape=['B', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='output', type=torch.float32, shape=['batch', -1]),
        FalconDict(name='new_score', type=torch.float32, shape=['batch', -1]),
    ]
    ORIGIN_TEST_ATOL = 2e2
    OPTIMIZED_TEST_ATOL = 2e2

    def __init__(self, jointer_module, criterion_module, **kwargs):
        super().__init__(kwargs)

        self.use_jointer_temperature = kwargs.get('use_jointer_temperature', False)

        self.jointer_module = jointer_module
        self.criterion_module = criterion_module

        # fill out missed shape info
        jointer_hidden_size = kwargs.get('jointer_hidden_size', -1)
        self._inputs[0].shape[1] = jointer_hidden_size
        self._inputs[1].shape[1] = jointer_hidden_size
        if self.use_jointer_temperature:
            self._inputs.append(FalconDict(name='temperature', type=torch.float32, shape=[1]))

    def prepare_export(self):
        '''prepare for onnx export.'''
        if self.criterion_module is not None:
            self.criterion_module.log_softmax_fc.combine_weight()
            self.criterion_module.log_softmax_fc.jit_init()

    def forward(self, encoder_out, predictor_out, temperature=1.0):
        '''
        Forward for RNN-T base module
        '''
        return RnntJointerLMSymbolic.apply(self, encoder_out, predictor_out, temperature)

    def sample_inputs(self):
        '''
        sample some inputs as example.
        may be used for jit trace or test.
        '''
        use_jointer_temperature = self._cfg.get('use_jointer_temperature', False)
        encoder_out = self._generate_input_data(0, dynamic_axis=1)
        predictor_out = self._generate_input_data(1, dynamic_axis=1)
        if use_jointer_temperature:
            temperature = self._generate_input_data(2, method='ones')
            return encoder_out, predictor_out, temperature
        return encoder_out, predictor_out


class RnntJointerExporter(BaseInfer):
    '''export rnnt jointer to onnx'''

    NAME = 'jointer'
    INPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', -1]),
        FalconDict(name='predictor_out', type=torch.float32, shape=['B', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='output', type=torch.float32, shape=['B', -1]),
    ]

    def __init__(self, jointer_module, criterion_module, **kwargs):
        super().__init__(kwargs)
        self.jointer_module = jointer_module
        self.criterion_module = criterion_module
        # pylint: disable=invalid-name
        self.use_non_combined_adaptive_softmax = kwargs.get(
            'use_non_combined_adaptive_softmax', False
        )
        self.use_jointer_temperature = kwargs.get('use_jointer_temperature', False)

        # fill out missed shape info
        jointer_hidden_size = kwargs.get('jointer_hidden_size', -1)
        self._inputs[0].shape[1] = jointer_hidden_size
        self._inputs[1].shape[1] = jointer_hidden_size
        if self.use_jointer_temperature:
            self._inputs.append(FalconDict(name='temperature', type=torch.float32, shape=[1]))

    def prepare_export(self):
        '''prepare for onnx export.'''
        if self.criterion_module is not None:
            self.criterion_module.log_softmax_fc.combine_weight()
            self.criterion_module.log_softmax_fc.jit_init()

    def forward(self, encoder_out, predictor_out, temperature=1.0):
        '''
        Forward for RNN-T base module
        '''
        jointer_out = self.jointer_module.forward_step(encoder_out, predictor_out)
        if self.criterion_module is not None:
            if (
                'RNNTAdaptiveSoftmax' in self.module_map
                and PantherAdaptiveSoftmax is not None
                and isinstance(self.criterion_module.log_softmax_fc, PantherAdaptiveSoftmax)
            ):
                # Use panther_utilities to export PantherAdaptiveSoftmax op
                setattr(
                    getattr(self.criterion_module.log_softmax_fc, '_torch_module'),
                    'use_non_combined_adaptive_softmax',
                    self.use_non_combined_adaptive_softmax,
                )
                jointer_out = self.criterion_module.log_softmax_fc(
                    jointer_out, temperature=temperature
                )
            else:
                if self.use_non_combined_adaptive_softmax:
                    jointer_out = self.criterion_module.log_softmax_fc.jit_forward_non_combined(
                        jointer_out, temperature
                    )
                else:
                    jointer_out = self.criterion_module.log_softmax_fc.jit_forward(
                        jointer_out, temperature
                    )
        return jointer_out

    def sample_inputs(self):
        '''
        sample some inputs as example.
        may be used for jit trace or test.
        TODO: how to get real data.
        '''
        use_jointer_temperature = self._cfg.get('use_jointer_temperature', False)
        encoder_out = self._generate_input_data(0, dynamic_axis=1)
        predictor_out = self._generate_input_data(1, dynamic_axis=1)
        if use_jointer_temperature:
            temperature = self._generate_input_data(2, method='ones')
            return encoder_out, predictor_out, temperature
        return encoder_out, predictor_out

    def sample_inputs_from_dataloader(self):
        '''
        sample some real data from dataloader.
        '''
        # For jointer on CPU, we use dynamic quantization
        # For CER robustness, use --solution.loose_quant 1 to skip quantize jointer
        if (
            not self._cfg.get('panther_use_gpu', True)
            and self.use_non_combined_adaptive_softmax
            and not self._cfg.get('loose_quant', False)
        ):
            self._dynamic_quant_flag = True


@register_solution("BaseRnntModel")
class BaseRnntModel(BaseSolution):
    '''
    Base model for RNN-T.
    - Encoder
        - frontend: VGGFrontEnd
        - backbone: TransformerBackbone, DFSMNBackbone
    - Predictor
    - Jointer
    - criterion
    '''

    def __init__(self, args):
        '''
        init function for RNN-T skeleton model.
        '''
        # pylint:disable=too-many-branches
        super().__init__()
        self.args = args
        self.pretrain_encoder = args.get('pretrain_encoder', False)
        if self.args.front_end_type == 'VGGFrontEnd':
            self.args.downsampling_size = 4
        self.acoustic_front_end_module = eval(args.front_end_type)(args)
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)
        self.acoustic_head_module = eval(args.head_type)(args)
        if not self.pretrain_encoder:
            self.predictor_module = eval(args.predictor_type)(args)
            self.jointer_module = eval(args.jointer_type)(args)
        if args.get('mtl_type', None) and self.training:
            self.mtl_type = args.get('mtl_type')
            self.mtl_module = eval(args.mtl_head)(args)
        else:
            self.mtl_type = None
            self.mtl_module = None
        self.backbone_pool_module = None
        if args.get('backbone_pool_type', None):
            if args.backbone_pool_type in ('AvgPoolTimeReduce', 'MaxPoolTimeReduce'):
                self.backbone_pool_module = eval(args.backbone_pool_type)()
            elif args.backbone_pool_type == 'LinearTimeReduce':
                self.backbone_pool_module = eval(args.backbone_pool_type)(
                    2, args.jointer_hidden_size
                )
            elif args.backbone_pool_type == 'Conv1dTimeReduce':
                self.backbone_pool_module = eval(args.backbone_pool_type)(args.jointer_hidden_size)
            self.args.downsampling_size *= 2
        self.do_inter_subsample = args.get('subsample_inter', False)
        if self.do_inter_subsample:
            self.args.downsampling_size *= 2
        self.criterion_module = eval(args.criterion_type)(args)
        self.update_steps = 0
        self.encoder_fix = args.get('encoder_fix', False)
        self.encoder_head_fix = args.get('encoder_head_fix', self.encoder_fix)
        self.adapter_finetuning = args.get('adapter_finetuning', not self.encoder_fix)
        self.adapter_finetuning_layernorm = args.get(
            'adapter_finetuning_layernorm', not self.encoder_fix
        )
        self.encoder_fix_steps = args.get('encoder_fix_steps', 0)
        assert (
            not self.encoder_fix or self.encoder_fix_steps == 0
        ), "cannot set encoder_fix and encoder_fix_steps both"
        self.jointer_fix = args.get('jointer_fix', False)
        self.predictor_fix = args.get('predictor_fix', False)
        if args.acoustic_backbone_type in (
            'OfflineTransformerBackbone',
            'TransformerBackbone',
            'RelTransformerBackbone',
            'EmformerBackbone',
        ):
            self.causal_transformer = args.get('causal_transformer', False)
        self.ilmt_weight = args.get('ilmt_weight', 0.0)
        self.limited_context = args.get('limited_context', None)
        self.train()

        # add las branch
        self.joint_las_weight = args.get('joint_las_weight', 0.0)
        if self.joint_las_weight > 0.0:
            self.init_las_decoder(args)

        # add lid branch
        self.lid_weight = args.get('lid_weight', 0.0)
        if self.lid_weight > 0.0:
            self.lid_head = LidHead(args)

        self.beam_searcher = None
        if not self.pretrain_encoder:
            self.greedy_searcher = GreedySearch(
                self.args, self.criterion_module, self.predictor_module, self.jointer_module
            )
        # excludes keys of model states for the initialization, split with ','
        # e.g.
        # --solution.exclude_load_keys jointer_module,mtl_module.head_fc_trans
        self.exclude_load_keys = args.get('exclude_load_keys', '')
        self._register_load_state_dict_pre_hook(self.model_param_filter_hook)

    def model_param_filter_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''
        load hook for exclude some submodules.
        '''
        exclude_load_keys = self.exclude_load_keys.split(',')
        for k in list(state_dict.keys()):
            for key in exclude_load_keys:
                filter_key = prefix + key
                if filter_key != "" and k.startswith(filter_key):
                    state_dict.pop(k)
                    break

    def train(self, mode: bool = True):
        '''
        Param will really be freezed by set param.requires_grad_(False).
        `with no_grad` may case param be updated by optimizer' weight_decay.
        What's more, we need to set module.eval() to fix some module buffer,
        such as BatchNorm.running_mean.
        '''
        super().train(mode)

        # encoder
        if self.encoder_fix:
            self.acoustic_front_end_module.requires_grad_(False)
            self.acoustic_backbone_module.requires_grad_(False)
            if self.backbone_pool_module is not None:
                self.backbone_pool_module.requires_grad_(False)
            self.acoustic_front_end_module.train(False)
            self.acoustic_backbone_module.train(False)
            if self.backbone_pool_module is not None:
                self.backbone_pool_module.train(False)
        # no head in best-rq
        if self.encoder_head_fix:
            self.acoustic_head_module.requires_grad_(False)
            self.acoustic_head_module.train(False)
        # adapter
        for name, param in self.acoustic_backbone_module.named_parameters():
            if self.adapter_finetuning_layernorm:
                if 'layer_norm' in name:
                    param.requires_grad = True

            if self.adapter_finetuning:
                if 'adapter' in name:
                    param.requires_grad = True
        # jointer
        if self.jointer_fix:
            self.jointer_module.requires_grad_(False)
            self.criterion_module.requires_grad_(False)
            self.jointer_module.train(False)
            self.criterion_module.train(False)
        # predictor
        if self.predictor_fix:
            self.predictor_module.requires_grad_(False)
            self.predictor_module.train(False)

    def init_las_decoder(self, args):
        '''init las decoder'''
        args = copy.deepcopy(args)
        args.update(args.las_args)
        if args.reorder_tgt_dict:
            self.bos = args.reorder_tgt_dict.bos()
            self.eos = args.reorder_tgt_dict.eos()
        else:
            self.bos = args.tgt_dict.bos()
            self.eos = args.tgt_dict.eos()

        self.las_decoder_type = args.las_decoder_type
        self.las_decoder_module = eval(args.las_decoder_type)(args)

    def init_beam_search(self, inference_cfg, lm_solution):
        '''beam search init'''
        if inference_cfg.get('output_timestamp', False):
            inference_cfg.nbest = 1
        if inference_cfg.get('output_streaming_stable_metric', False) and inference_cfg.get(
            'use_batch_beam', False
        ):
            raise RuntimeError(
                'output_streaming_stable_metric only works in non batch beam search!'
            )
        if not inference_cfg.get('use_batch_beam', False):
            self.beam_searcher = NonBatchBeamSearch(
                inference_cfg,
                self.predictor_module,
                self.jointer_module,
                self.criterion_module,
                lm_solution,
            )
        else:
            self.beam_searcher = BatchBeamSearch(
                inference_cfg,
                self.predictor_module,
                self.jointer_module,
                self.criterion_module,
                lm_solution,
            )

    def frontend(self, fbank, mask):
        '''
        frontend of RNN-T encoder
        '''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, "BTN"

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''
        Backbone model for RNN-T encoder
        '''
        attn_mask = False
        if (
            self.args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=attn_mask, frontend_shape=frontend_shape
        )
        return backbone_out

    def encoder(self, batch_data):
        '''
        Encoder Module in RNN-T
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        if 'domain' in batch_data:
            domain_emb = F.one_hot(batch_data['domain'], num_classes=self.args.domain_num).cuda()
            fbank = torch.cat(
                (fbank, domain_emb.unsqueeze(1).to(fbank.dtype).repeat(1, fbank.shape[1], 1)), dim=2
            )
        ############ Encoder ############
        if not self.training or self.update_steps <= self.encoder_fix_steps:
            with torch.no_grad():
                # front-end
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                # output (B, T, N)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
        else:
            with torch.enable_grad():
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
        # ctc mtl branch
        if self.mtl_module is not None:
            mtl_logits = self.mtl_module(encoder_backbone_out)
        else:
            mtl_logits = None
        # head
        encoder_out = self.acoustic_head_module(encoder_backbone_out)
        # backbone_pool_module
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
            batch_data["mtl_mask"] = backbone_mask.clone() if backbone_mask is not None else None
            backbone_mask = backbone_mask[:, :-1][:, ::2] if backbone_mask is not None else None
        if self.do_inter_subsample:
            backbone_mask = backbone_mask[:, :-1][:, ::2] if backbone_mask is not None else None
            batch_data["mtl_mask"] = backbone_mask.clone() if backbone_mask is not None else None
        if 'eos' in batch_data:
            batch_data['eos'] = batch_data['eos'] // self.args.downsampling_size
        trainable = encoder_out.requires_grad
        return encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out

    def las_decoder(self, batch_data, encoder_backbone_out, backbone_mask):
        '''
        Las Decoder Module
        '''
        target = batch_data['char']
        # add bos and eos in target for attention decoder input
        target_att = torch.where(target == 0, self.eos, target)
        target_att = F.pad(target_att, (1, 0), value=self.bos)
        target_mask = batch_data['char_mask']
        if 'mtl_mask' in batch_data:
            backbone_mask = batch_data['mtl_mask']
        if 'Transformer' in self.las_decoder_type:
            las_logits = self.las_decoder_module(
                encoder_backbone_out,
                backbone_mask,
                target_att,
                prev_tgt_mask=F.pad(target_mask, (1, 0), value=1),
            )
        else:
            las_logits = self.las_decoder_module(encoder_backbone_out, backbone_mask, target_att)
        return las_logits

    def predictor(self, batch_data, trainable=True, key='prev_char'):
        '''
        Predictor Module in RNN-T
        '''
        prev_char = batch_data[key]
        if self.limited_context is not None:
            limited_context = self.limited_context
            batch, time = prev_char.size()
            pad = F.pad(batch_data['char'], (limited_context - 1, 0, 0, 0)).type_as(
                batch_data['src']
            )
            prev_char_k = (
                F.unfold(pad.unsqueeze(1).unsqueeze(2), (1, limited_context - 1))
                .transpose(1, 2)
                .contiguous()
            )
            prev_char_k = F.pad(prev_char_k, (1, 0, 0, 0, 0, 0)).type_as(batch_data['char'])
            prev_char_k = prev_char_k.view(-1, limited_context).contiguous()
            predictor_out_k, _ = self.predictor_module(prev_char_k)
            predictor_out = predictor_out_k.view(batch, time, -1).contiguous()
        else:
            unk_idx = self.args.tgt_dict.index('<unk>')
            prev_char_drop = self.args.predictor_dropout_factor if trainable else 0
            predictor_out, _ = self.predictor_module(
                prev_char, prev_char_drop=prev_char_drop, unk_idx=unk_idx
            )
        return predictor_out

    def jointer(self, encoder, predictor, input_lengths, target_lengths):
        '''
        Jointer Module in RNN-T.
        '''
        jointer_out = self.jointer_module(encoder, predictor, input_lengths, target_lengths)
        return jointer_out

    def splitting_jointer_forward(self, batch_data, encoder_out, predictor_out, mtl_logits):
        '''
        do jointer and criterion with splitted batch.
        support large batch size of encoder, can imporve FPS a lot for some model.
        '''
        bsz = encoder_out.shape[0]
        split_num = min(self.args.jointer_split_num, bsz)
        step = (bsz + split_num - 1) // split_num
        split_num = (bsz + step - 1) // step

        forward_outs = []
        loss_scale = batch_data.get('loss_scale', 1.0)
        encoder_out_bak = encoder_out.detach().requires_grad_()
        predictor_out_bak = predictor_out.detach().requires_grad_()
        mtl_logits_bak = mtl_logits.detach().requires_grad_() if mtl_logits is not None else None
        for i in range(split_num):
            cur_batch = split_jointer_input(
                i,
                step,
                encoder_out=encoder_out_bak,
                predictor_out=predictor_out_bak,
                mtl_logits=mtl_logits_bak,
                **batch_data,
            )

            cur_jointer_out = self.jointer(
                cur_batch['encoder_out'],
                cur_batch['predictor_out'],
                None,
                cur_batch['target_lengths'],
            )

            # criterion for loss computation
            forward_out = self.criterion_module(
                cur_jointer_out,
                cur_batch,
                mtl_logits=cur_batch['mtl_logits'],
                mtl_type=self.mtl_type,
            )
            if (
                self.update_steps % self.args.cer_update_freq == 0
                or not cur_jointer_out.requires_grad
            ):
                target = cur_batch['char']
                forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                    cur_batch['encoder_out'], target, cur_batch
                )
            if self.training:
                scale = loss_scale * cur_batch['jointer_scale']
                (forward_out['backward_loss'] * scale).backward()
            forward_out['jointer_scale'] = cur_batch['jointer_scale']
            forward_outs.append(forward_out)
        if self.training:
            if mtl_logits_bak is not None:
                torch.autograd.backward(
                    [encoder_out, predictor_out, mtl_logits],
                    grad_tensors=[
                        encoder_out_bak.grad,
                        predictor_out_bak.grad,
                        mtl_logits_bak.grad,
                    ],
                )
            else:
                torch.autograd.backward(
                    [encoder_out, predictor_out],
                    grad_tensors=[encoder_out_bak.grad, predictor_out_bak.grad],
                )
        return merge_jointer_output(forward_outs)

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T base module
        '''
        if self.training:
            self.update_steps += 1
        ############ Encoder #############
        if 'encoder_out' in batch_data:
            encoder_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            trainable = batch_data['trainable']
            mtl_logits = batch_data['mtl_logits']
            encoder_backbone_out = batch_data['encoder_backbone_out']
        else:
            encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out = self.encoder(
                batch_data
            )
            batch_data['backbone_mask'] = backbone_mask
        if inference:
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out
        ############ Predictor ############
        predictor_out = self.predictor(batch_data, trainable)
        target_lengths = batch_data['target_lengths']

        ############ Jointer ############
        # special jointer for large batch support.
        if self.args.get('jointer_split_num', 1) > 1:
            return self.splitting_jointer_forward(
                batch_data, encoder_out, predictor_out, mtl_logits
            )

        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)
        if self.ilmt_weight > 0.0:
            ilmt_encoder_out = torch.zeros_like(encoder_out[:, 0:1, :])
            ilmt_jointer_out = self.jointer(ilmt_encoder_out, predictor_out, None, target_lengths)
            batch_data["ilmt_jointer_out"] = ilmt_jointer_out

        if self.joint_las_weight > 0.0:
            las_logits = self.las_decoder(batch_data, encoder_backbone_out, backbone_mask)
            batch_data["las_logits"] = las_logits

        if self.lid_weight > 0.0:
            lid_logits = self.lid_head(encoder_backbone_out.transpose(1, 2), backbone_mask)
            batch_data["lid_logits"] = lid_logits

        # criterion for loss computation
        if self.args.get('moe_args', None) is not None:
            batch_data['aux_loss'] = self.acoustic_backbone_module.aux_loss
        forward_out = self.criterion_module(
            jointer_out, batch_data, mtl_logits=mtl_logits, mtl_type=self.mtl_type
        )

        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )
        return forward_out

    @torch.no_grad()
    def greedy_inference(self, batch_data):
        '''
        Greedy inference
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        ############ Acoustic ############
        acoustic_out, _, _, _ = self.forward(batch_data, inference=True)

        greedy_infer_rlt = self.criterion_module.greedy_infer(
            acoustic_out, self.predictor_module, self.jointer_module
        )
        out_rlt_list = []
        for bid in range(bsz):
            hyp_token_list = greedy_infer_rlt[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list.append(hyp_token_list)
        return out_rlt_list

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        nbest=1,
        nbest_align_info=False,
        output_timestamp=False,
        prefetch=False,
        fixed_prefix=False,
        output_rnnt_confidence=False,
        endpoint=False,
        stable_metric_list=None,
        **_kwargs,
    ):
        '''
        Beam inference
        '''
        ############ Acoustic ############
        if 'encoder_out' in batch_data:
            # no more repetitive computation for efficiency
            acoustic_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            mtl_logits = batch_data['mtl_logits']
        else:
            acoustic_out, backbone_mask, mtl_logits, _ = self.forward(batch_data, inference=True)

        beam_infer_rlt, beam_infer_rlt_nbest = self.beam_searcher(
            acoustic_out,
            backbone_mask,
            stable_metric_list=stable_metric_list,
        )

        out_rlt_list, encoder_frames = self.process_beam_infer_rlt(
            batch_data,
            beam_infer_rlt,
            beam_infer_rlt_nbest,
            backbone_mask,
            nbest,
            nbest_align_info,
            output_timestamp,
            prefetch,
            fixed_prefix,
            output_rnnt_confidence,
            endpoint,
        )
        return out_rlt_list, mtl_logits, encoder_frames

    @staticmethod
    def process_beam_infer_rlt(
        batch_data,
        beam_infer_rlt,
        beam_infer_rlt_nbest,
        backbone_mask,
        nbest=1,
        nbest_align_info=False,
        output_timestamp=False,
        prefetch=False,
        fixed_prefix=False,
        output_rnnt_confidence=False,
        endpoint=False,
    ):
        '''
        Beam results process
        '''
        # pylint:disable=too-many-branches,too-many-locals
        fbank = batch_data['src']  # (B, T, ndim)
        bsz = fbank.shape[0]
        # return nbest align info
        if nbest_align_info:
            return beam_infer_rlt_nbest["nbest"], None
        out_rlt_list = {}
        # return timestamp
        if output_timestamp:
            out_rlt_list["timestamp"] = []
            for bid_rlt in beam_infer_rlt_nbest["nbest"]:
                out_rlt_list["timestamp"].append(bid_rlt[0][3][1:])

        out_rlt_list_nbest = []
        out_rlt_confidence_list = []
        out_rlt_list["inf_res"] = []
        # pylint: disable=too-many-nested-blocks
        for bid in range(bsz):
            hyp_token_list = beam_infer_rlt[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list)
            out_rlt_list["inf_res"].append(hyp_token_list)
            prefetch_frames = -1
            bid_rlt = beam_infer_rlt_nbest['nbest'][bid]
            if prefetch:
                final_best = beam_infer_rlt_nbest['prefetch'][bid][-1][1]
                if len(beam_infer_rlt_nbest['prefetch'][bid]) > 1:
                    for prefetch_rlt in beam_infer_rlt_nbest['prefetch'][bid][-2::-1]:
                        if prefetch_rlt[1] == final_best:
                            bid_rlt = prefetch_rlt[-1]
                            prefetch_frames = prefetch_rlt[0]
                            break
            if output_rnnt_confidence:
                out_rlt_confidence_list.append(bid_rlt[0][4][1:])
            if nbest > 1 and beam_infer_rlt_nbest is not None:
                cur_nbest = []
                if isinstance(bid_rlt, torch.Tensor):
                    num_beam = bid_rlt.size(0)
                else:
                    num_beam = len(bid_rlt)
                for beam_idx in range(num_beam):
                    beam_rlt = bid_rlt[beam_idx]
                    rlt_confidence = '0.0'
                    fst_score = 0.0
                    rlt_score = 0.0
                    if isinstance(beam_rlt, tuple):
                        rlt_score = beam_rlt[1]
                        if len(beam_rlt) >= 5:
                            fst_score = beam_rlt[2]
                            confidence = beam_rlt[4]
                            if len(confidence) > 1:
                                rlt_confidence = '|'.join([str(p) for p in confidence[1:]])
                        beam_rlt = beam_rlt[0]
                    cur_nbest.append(
                        (
                            rnnt_rlt_neaten(beam_rlt),
                            fst_score,
                            '{}|{}'.format(str(rlt_score), rlt_confidence),
                            prefetch_frames,
                        )
                    )

                out_rlt_list_nbest.append(cur_nbest)
        encoder_frames = backbone_mask.sum(dim=1).int().tolist()
        if nbest > 1 and beam_infer_rlt_nbest is not None:
            out_rlt_list["nbest"] = out_rlt_list_nbest
        if fixed_prefix:
            out_rlt_list["fixed_prefix"] = beam_infer_rlt_nbest["fixed_prefix"]
        if prefetch:
            out_rlt_list["prefetch"] = beam_infer_rlt_nbest["prefetch"]
        if endpoint:
            out_rlt_list["endpoint"] = beam_infer_rlt_nbest["endpoint"]
        if output_rnnt_confidence:
            out_rlt_list["confidence"] = out_rlt_confidence_list

        return out_rlt_list, encoder_frames

    def export(self, *_args, **kwargs):
        '''export onnx'''
        for infer_name in self._infer_names:
            INFERS[infer_name].export(**kwargs)

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        self._infer_names = []
        if self.pretrain_encoder:
            return
        infers = [
            RnntEncoderExporter(self, **self.args),
            RnntPredictorExporter(self.predictor_module, **self.args),
        ]
        export_jointer_for_lm = self.args.get('use_lm_jointer', False)
        if export_jointer_for_lm and version.parse(torch.__version__) < version.parse('1.7.0'):
            raise RuntimeError(
                'torch version is not sufficient for "use_lm_jointer = true", '
                'expected version >= 1.7.0, got: {}'.format(torch.__version__)
            )
        if export_jointer_for_lm:
            infers.append(
                RnntJointerExporterForLM(self.jointer_module, self.criterion_module, **self.args)
            )
        else:
            infers.append(
                RnntJointerExporter(self.jointer_module, self.criterion_module, **self.args)
            )

        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
