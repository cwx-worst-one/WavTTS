''' BaseRnntModel '''
# pylint: disable=unused-argument
# pylint:disable=too-many-lines
import contextlib
import copy
import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.inference.cif_beam_search import *

# pylint: disable=unused-import
from core.models.asr.acoustic_frontend import CifOriginalFrontend, Conv2dPooling
from core.models.layers.normalization import LayerNormNCHW, LayerNorm
from core.models.asr.acoustic_backbone import *
from core.models.asr.cif_decoder import CifDecoder
from core.models.layers.cif import CifCalculator, CifWeightEstimator
from core.models.pretrained.data2vec_model import *
from core.criterions.cif_criterion import *
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
)
from core.utils.misc import infer_text_format
from core.dataset.preprocess.label import ConcateEnLetters
from core.solutions.inference import BaseInfer, INFERS
from core.utils.dict import FalconDict
from core.utils import logging, get_rank
from core.solutions.asr.base_rnnt_model import RnntEncoderExporter

try:
    # pylint: disable=import-error, no-name-in-module
    import onnx
    import panther
    from panther.model_converter import onnx_process, onnx2panther

    class GlobalKVBaseExporter(BaseInfer):  # pylint: disable=abstract-method
        '''base exporter to merge global kv temsors'''

        def __init__(self, cfg):
            super().__init__(cfg)
            if get_rank() == 0:
                logging.info("Using Panther GlobalKVBaseExportor.")

        def _optimize_graph(self):
            '''graph optimization and test.'''
            if not self._optimize_graph_flag:
                return
            # step 1: graph optimization
            previous_file = self._onnx_files[-1]
            current_file = osp.join(self._onnx_dir, '{}_optimized.onnx'.format(self.NAME))
            os.system('rm -rf {}'.format(current_file))
            onnx2panther.from_onnx_model(
                previous_file,
                optimize_level=self._optimize_level,
                output_onnx_path=current_file,
                provider='CUDA',
                with_slim=self.with_slim,
            )
            self._onnx_files.append(current_file)
            # step 2: merge global KV
            tmp_file = osp.join(self._onnx_dir, '{}_runtimeopt.onnx'.format(self.NAME))
            previous_file = self._onnx_files[-1]
            current_file = osp.join(self._onnx_dir, '{}_mkv.onnx'.format(self.NAME))
            os.system('rm -rf {}'.format(current_file))

            try:
                sess_options = panther.SessionOptions()
                sess_options.graph_optimization_level = (
                    panther.GraphOptimizationLevel.PTH_ENABLE_BASIC
                )
                sess_options.optimized_model_filepath = tmp_file
                _ = panther.InferenceSession(previous_file, sess_options)
                model = onnx.load(tmp_file)
            except Exception as e:
                logging.warning('run panther optimization cuae Exception as %s', e)
                model = onnx.load(previous_file)
            # pylint: disable=no-member
            onnx_process.remove_domain(model.graph)
            onnx_process.add_decode_transformer_state(model.graph)
            onnx_process.add_decode_transformer_state_kvdim3(model.graph)
            onnx_process.remove_invalid_node(model.graph)
            onnx.save(model, current_file)
            os.system('rm -rf {}'.format(tmp_file))
            logging.info('success convert, save file to %s', current_file)
            if not osp.exists(current_file):
                raise RuntimeError(
                    'rank', get_rank(), self.NAME, 'failed on merge kv', current_file
                )
            self._onnx_files.append(current_file)

            infer_option = panther.SessionOptions()
            infer_option.intra_op_num_threads = 1
            infer_option.graph_optimization_level = panther.GraphOptimizationLevel.PTH_ENABLE_ALL
            sess = panther.InferenceSession(
                current_file,
                providers=self._providers,
                sess_options=infer_option,
            )
            self._inputs = [
                FalconDict(name=node.name, type=node.type, shape=node.shape)
                for node in sess.get_inputs()
            ]
            self._outputs = [
                FalconDict(name=node.name, type=node.type, shape=node.shape)
                for node in sess.get_outputs()
            ]

except Exception:
    GlobalKVBaseExporter = BaseInfer


class ChunkHopingEncoderExporter(GlobalKVBaseExporter):
    '''ChunkHopingEncoderExporter'''

    NAME = 'encoder'

    def __init__(self, frontend, backbone, cif_estimator, ctc_proj, **kwargs):
        '''__init__'''
        super().__init__(kwargs)
        self.frontend = frontend
        self.backbone = backbone
        self.cif_estimator = cif_estimator
        self.ctc_proj = ctc_proj
        self.args = FalconDict(self._cfg)
        self._inputs = [
            FalconDict(name='features_input', type=torch.float32, shape=['B', 'T', 80, 1]),
        ]
        self._outputs = [
            FalconDict(
                name='encoder_output', type=torch.float32, shape=['B', 'T', self.args.hidden_size]
            ),
        ]
        self.num_heads = self.args.num_heads
        total_key_depth = self.args.attention_key_channels or self.args.hidden_size
        total_value_depth = self.args.attention_value_channels or self.args.hidden_size
        self.key_depth = total_key_depth // self.num_heads
        self.value_depth = total_value_depth // self.num_heads
        if self.args.use_input_padding:
            self._inputs.append(
                FalconDict(name='features_mask', type=torch.float32, shape=['B', 'T']),
            )

        self.chunk_len = []
        for layer in range(self.args.num_encoder_layers):
            subsampling_time = 2**self.args.down_sample_conv_num_layers
            for ll in self.args.sa_pooling_layers:
                if layer >= ll:
                    subsampling_time *= 2

            if self.args.num_history_chunk:
                length = int(
                    (self.args.num_history_chunk * self.args.hop_size + subsampling_time - 1)
                    // subsampling_time
                )
            else:
                length = int((self.args.hop_size + subsampling_time - 1) // subsampling_time)
            self.chunk_len.append(length)

            self._inputs.extend(
                [
                    FalconDict(
                        name='layer{}_k'.format(layer),
                        type=torch.float32,
                        shape=['B', self.num_heads, length, self.key_depth],
                    ),
                    FalconDict(
                        name='layer{}_v'.format(layer),
                        type=torch.float32,
                        shape=['B', self.num_heads, length, self.value_depth],
                    ),
                ]
            )
            self._outputs.extend(
                [
                    FalconDict(
                        name='layer{}_k_o'.format(layer),
                        type=torch.float32,
                        shape=['B', self.num_heads, 'L', self.key_depth],
                    ),
                    FalconDict(
                        name='layer{}_v_o'.format(layer),
                        type=torch.float32,
                        shape=['B', self.num_heads, 'L', self.value_depth],
                    ),
                ]
            )

        self.use_conv_memory = (
            self.args.acoustic_backbone_type == 'CifConformerV2Encoder'
            and self.args.ch_conformer_add_conv_memory
        )
        if self.use_conv_memory:
            conv_memory_len = self.args.conformer_conv_width // 2
            for layer in range(self.args.num_encoder_layers):
                self._inputs.append(
                    FalconDict(
                        name='layer{}_conv_mem'.format(layer),
                        type=torch.float32,
                        shape=['B', conv_memory_len, self.args.hidden_size],
                    )
                )
                self._outputs.append(
                    FalconDict(
                        name='layer{}_conv_mem_o'.format(layer),
                        type=torch.float32,
                        shape=['B', conv_memory_len, self.args.hidden_size],
                    )
                )

        # cif_weight
        self._outputs.append(FalconDict(name='cif_weight', type=torch.float32, shape=['B', 'T']))
        # ctc
        if self.ctc_proj is not None:
            self._outputs.append(
                FalconDict(name='ctc_max_index', type=torch.int64, shape=['B', 'T'])
            )

    @torch.no_grad()
    def forward(self, features_input, *args, **kwargs):
        '''forward'''
        prev_kv_cache = None
        prev_conv_memory = None
        features_mask = None
        num_layers = self.args.num_encoder_layers
        if kwargs:
            if self.args.use_input_padding:
                features_mask = kwargs.get('features_mask')
            prev_kv_cache = [
                {
                    'len': self.chunk_len[layer],
                    'k': kwargs.get('layer{}_k'.format(layer)),
                    'v': kwargs.get('layer{}_v'.format(layer)),
                }
                for layer in range(num_layers)
            ]
            if self.use_conv_memory:
                prev_conv_memory = [
                    kwargs.get('layer{}_conv_mem'.format(layer)) for layer in range(num_layers)
                ]
        else:
            offset = 0
            if self.args.use_input_padding:
                features_mask = args[0]
                offset += 1
            prev_kv_cache = [
                {
                    'len': self.chunk_len[layer],
                    'k': args[offset + 2 * layer],
                    'v': args[offset + 2 * layer + 1],
                }
                for layer in range(num_layers)
            ]
            offset += 2 * num_layers
            if self.use_conv_memory:
                prev_conv_memory = [args[offset + layer] for layer in range(num_layers)]
        encoder_input, ignore_padding = self.frontend(features_input, features_mask)
        # skip dropout in export mode
        encoder_output, _, out_kv_cache = self.backbone(
            encoder_input,
            ignore_padding=ignore_padding,
            prev_kv_cache=prev_kv_cache,
            prev_conv_memory=prev_conv_memory,
        )
        outputs = [encoder_output]
        for cache in out_kv_cache:
            outputs.extend([cache['k'], cache['v']])
        if prev_conv_memory is not None:
            outputs.extend(prev_conv_memory)
        # cif_weight
        outputs.append(self.cif_estimator(encoder_output, None))
        # ctc
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(encoder_output)
            outputs.append(torch.argmax(ctc_logits, dim=-1, keepdim=False))
        return outputs

    def sample_inputs(self):
        '''sample_inputs'''
        features_input = self._generate_input_data(0, dynamic_axis=[1, 80, -1, -1])
        inputs = [features_input]
        if self.args.use_input_padding:
            inputs.append(self._generate_input_data(1, method='ones', dynamic_axis=[1, 80]))
        # kv data
        for _ in range(self.args.num_encoder_layers):
            idx = len(inputs)
            inputs.extend(
                [
                    self._generate_input_data(idx, dynamic_axis=[1, -1, -1, -1]),
                    self._generate_input_data(idx + 1, dynamic_axis=[1, -1, -1, -1]),
                ]
            )
        # conv memory data
        if self.use_conv_memory:
            idx = len(inputs)
            inputs.extend(
                [
                    self._generate_input_data(idx + i, method='zeros', dynamic_axis=[1, -1, -1])
                    for i in range(self.args.num_encoder_layers)
                ]
            )
        return tuple(inputs)


class CifDecoderLogitsExporter(GlobalKVBaseExporter):
    '''export model to compute decoder logits'''

    NAME = 'decoder'

    def __init__(self, decoder, **kwargs):
        '''init'''
        super().__init__(kwargs)
        self.decoder = decoder
        self.args = FalconDict(self._cfg)
        self._inputs = [
            FalconDict(name='prev_ids', type=torch.int32, shape=['B']),
            FalconDict(
                name='prev_cif_outputs', type=torch.float32, shape=['B', self.args.hidden_size]
            ),
            FalconDict(name='bias', type=torch.float32, shape=['B', 1, 1, 'L']),
            FalconDict(
                name='cur_cif_outputs', type=torch.float32, shape=['B', self.args.hidden_size]
            ),
        ]
        self._outputs = [
            FalconDict(name='decoder_output', type=torch.float32, shape=['B', self.args.vocab_size])
        ]
        total_key_depth = self.args.attention_key_channels or self.args.hidden_size
        total_value_depth = self.args.attention_value_channels or self.args.hidden_size
        self.num_layers = self.args.num_decoder_layers or self.args.num_hidden_layers
        for layer in range(self.num_layers):
            self._inputs.extend(
                [
                    FalconDict(
                        name='decoder_layer{}_k'.format(layer),
                        type=torch.float32,
                        shape=['B', 'T', total_key_depth],
                    ),
                    FalconDict(
                        name='decoder_layer{}_v'.format(layer),
                        type=torch.float32,
                        shape=['B', 'T', total_value_depth],
                    ),
                ]
            )
            self._outputs.extend(
                [
                    FalconDict(
                        name='decoder_layer{}_k_o'.format(layer),
                        type=torch.float32,
                        shape=['B', 'T', total_key_depth],
                    ),
                    FalconDict(
                        name='decoder_layer{}_v_o'.format(layer),
                        type=torch.float32,
                        shape=['B', 'T', total_value_depth],
                    ),
                ]
            )

    @torch.no_grad()
    def forward(self, prev_ids, prev_cif_outputs, bias, cur_cif_outputs, *args, **kwargs):
        '''forward'''
        atten_cache = dict()
        if kwargs:
            for layer in range(self.num_layers):
                atten_cache['decoder_layer_{}'.format(layer)] = {
                    'k': kwargs.get('decoder_layer{}_k'.format(layer)),
                    'v': kwargs.get('decoder_layer{}_v'.format(layer)),
                }
        else:
            for layer in range(self.num_layers):
                atten_cache['decoder_layer_{}'.format(layer)] = {
                    'k': args[2 * layer],
                    'v': args[2 * layer + 1],
                }
        logits, atten_cache = self.decoder.logits_decoder(
            prev_ids, prev_cif_outputs, cur_cif_outputs, bias, atten_cache
        )
        logits = log_prob_from_logits(logits * self.args.cif_temperature, dim=1)
        outputs = [logits]
        for layer in range(self.num_layers):
            cur_cache = atten_cache['decoder_layer_{}'.format(layer)]
            outputs.extend([cur_cache['k'], cur_cache['v']])
        return outputs

    def sample_inputs(self):
        '''sample_inputs'''
        prev_ids = self._generate_input_data(0, method='ones', dynamic_axis=[10])
        prev_cif_outputs = self._generate_input_data(1, dynamic_axis=[10, -1])
        bias = self._generate_input_data(2, dynamic_axis=[10, -1, -1, 2])
        cur_cif_outputs = self._generate_input_data(3, dynamic_axis=[10, -1])
        inputs = [prev_ids, prev_cif_outputs, bias, cur_cif_outputs]
        for _ in range(self.num_layers):
            idx = len(inputs)
            inputs.extend(
                [
                    self._generate_input_data(idx, dynamic_axis=[10, 1, -1]),
                    self._generate_input_data(idx + 1, dynamic_axis=[10, 1, -1]),
                ]
            )
        return tuple(inputs)


class CifEncoderExporter(RnntEncoderExporter):
    '''export cif encoder onnx'''

    ORIGIN_TEST_ATOL = 1e-4
    OPTIMIZED_TEST_ATOL = 2e-2

    def __init__(self, model, cif_estimator, ctc_proj, **kwargs):
        super().__init__(model, **kwargs)
        self.cif_estimator = cif_estimator
        self.ctc_proj = ctc_proj
        self.args = FalconDict(self._cfg)
        self._return_backbone = True
        # cif_weight
        self._outputs.append(FalconDict(name='cif_weight', type=torch.float32, shape=['B', 'T']))
        # ctc
        if self.ctc_proj is not None:
            self._outputs.append(
                FalconDict(name='ctc_max_index', type=torch.int64, shape=['B', 'T'])
            )

    def forward(self, fbank, fbank_mask=None, **_kwargs):
        '''
        Forward for RNN-T base module
        '''
        encoder_out = super().forward(fbank, fbank_mask=fbank_mask, **_kwargs)
        outputs = [encoder_out]
        # cif_weight
        outputs.append(self.cif_estimator(encoder_out, None))
        # ctc
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(encoder_out)
            outputs.append(torch.argmax(ctc_logits, dim=-1, keepdim=False))
        return outputs

    def sample_inputs(self):
        '''sample_inputs'''
        features_input = self._generate_input_data(0, dynamic_axis=[1, 80, -1, -1])
        inputs = [features_input]
        return tuple(inputs)


@register_solution("BaseCifModel")
class BaseCifModel(BaseSolution):
    '''
    Base model for CIF.
    '''

    def __init__(self, args, is_pretrain=False):
        '''
        init function for CIF skeleton model.
        '''
        super().__init__()
        # args means cfg.solution config block
        self.args = FalconDict(args)

        # Encoder part
        if self.args.data2vec_finetuning:
            pass
        else:
            self.front_end = (
                eval(args.front_end_type)(args)
                if self.args.get('front_end_type', None)
                else CifOriginalFrontend(args)
            )
            self.encoder_backbone = (
                eval(args.acoustic_backbone_type)(args)
                if self.args.get('acoustic_backbone_type', None)
                else CifSelfAttentionEncoder(args)
            )
        self.cif_weight_estimator = CifWeightEstimator(args)
        self.encoder_dropout = nn.Dropout(args.layer_prepostprocess_dropout, inplace=True)

        self.encoder_proj = None
        if not is_pretrain:
            if self.args.ctc_loss_on_encoder:
                self.encoder_proj = nn.Linear(
                    self.args.hidden_size, self.args.vocab_size, bias=False
                )
            # Cif Calculator
            self.cif_calculator = CifCalculator(self.args)

            # Decoder part
            if self.args.decoder_type == 'nar':
                raise NotImplementedError()

            self.decoder = CifDecoder(self.args)
            self.decoder_for_beam_search = self.decoder.logits_loop_decoder

        # criterion
        self.criterion = eval(args.criterion_type)(args)

        self.eos_id = args.get('eos_id', 2)
        self.pad_id = args.get('pad_id', 0)
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        '''
        initialize parameters in this module use xavier_uniform_
        '''
        init_scale = self.args.init_scale or 1.0
        gain = np.sqrt(init_scale)
        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, LayerNormNCHW, LayerNorm, CifDecoder)):
                # keep the default initializer of layer norm parameters:
                # ones_ for gamma/weight, zeros_ for beta/bias, uniform_ for emb_lookup
                continue
            for param in module.parameters(recurse=False):
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=gain)
                else:
                    limit = np.sqrt(3.0 * init_scale / float(param.numel()))
                    nn.init.uniform_(param, -limit, limit)

    def encoder(self, inputs, inputs_mask):
        '''
        Args:
          inputs: fbank feature inputs.
            [batch_size, input_length, fbank_dim, fbank_channel]

        Returns:
          encoder_output: Encoder representation.
            [batch_size, input_length/stride, hidden_size]
          not_padding: Used for loss computation and correct search length
            [batch_size, input_length/stride]
        '''
        # pylint:disable=too-many-branches,too-many-locals
        if self.args.use_fp16:
            if self.args.no_tensorcore:
                pass
            else:
                inputs_org_shape = inputs.shape
                need_tensor_core = ((inputs_org_shape[1] + 32 - 1) // 32) * 32
                need_pad_value = [0] * 2 * (inputs.dim() - 2) + [
                    0,
                    need_tensor_core - inputs_org_shape[1],
                ]
                inputs = F.pad(inputs, need_pad_value, mode='constant', value=0.0)

        if self.args.use_chunk_hopping:
            batch_size = inputs.shape[0]
            decode_length = inputs.shape[1]
            chunk_size = self.args.chunk_size
            hop_size = self.args.hop_size
            future_size = 0
            splice_length = 0
            subsampling_rate = 2 ** (
                self.args.down_sample_conv_num_layers + len(self.args.sa_pooling_layers)
            )

            encoder_outputs = []
            not_padding = []
            a = []

            if self.args.ch_type == 'fixed_future':
                future_size = self.args.ch_fixed_future_size
                splice_length = chunk_size - hop_size - future_size
                inputs = F.pad(inputs, (0, 0, 0, 0, splice_length, future_size))
                inputs_mask = F.pad(inputs_mask, (splice_length, future_size))
            else:
                raise ValueError("No such chunk_hopping type")

            prev_kv_cache = []
            for layer in range(self.args.num_encoder_layers):
                subsampling_time = 2**self.args.down_sample_conv_num_layers
                for ll in self.args.sa_pooling_layers:
                    if layer >= ll:
                        subsampling_time *= 2

                if self.args.num_history_chunk:
                    length = int(
                        (self.args.num_history_chunk * hop_size + subsampling_time - 1)
                        // subsampling_time
                    )
                else:
                    length = int((hop_size + subsampling_time - 1) // subsampling_time)

                prev_kv_cache.append({'len': length, 'k': None, 'v': None})

            prev_conv_memory = None
            if (
                self.args.acoustic_backbone_type == 'CifConformerV2Encoder'
                and self.args.ch_conformer_add_conv_memory
            ):
                prev_conv_memory = []
                for layer in range(self.args.num_encoder_layers):
                    assert self.args.conformer_conv_width > 0
                    prev_conv_memory.append(
                        torch.zeros(
                            batch_size,
                            self.args.conformer_conv_width // 2,
                            self.args.hidden_size,
                            dtype=torch.float32,
                            device='cuda',
                        )
                    )

            for start_pos in range(0, decode_length, hop_size):
                end_pos = np.minimum(start_pos + chunk_size, decode_length + chunk_size - hop_size)
                cur_inputs = inputs[:, start_pos:end_pos, :, :]
                cur_inputs_mask = inputs_mask[:, start_pos:end_pos]
                if self.args.last_chunk_pad:
                    pad_length = chunk_size - (end_pos - start_pos)
                    if pad_length > 0:
                        cur_inputs = F.pad(cur_inputs, (0, 0, 0, 0, 0, pad_length))
                        end_pos = start_pos + chunk_size
                        cur_inputs_mask = F.pad(cur_inputs_mask, (0, pad_length))
                cur_encoder_input, cur_ignore_padding = self.front_end(cur_inputs, cur_inputs_mask)
                cur_encoder_input = self.encoder_dropout(cur_encoder_input)
                cur_encoder_outputs, cur_not_padding, cur_key_value_cache = self.encoder_backbone(
                    cur_encoder_input,
                    ignore_padding=cur_ignore_padding,
                    prev_kv_cache=prev_kv_cache,
                    prev_conv_memory=prev_conv_memory,
                )
                cur_a = self.cif_weight_estimator(cur_encoder_outputs, cur_not_padding)

                if self.args.ch_type == 'fixed_future':
                    output_start_pos = int(np.ceil((0 + splice_length) / subsampling_rate))
                    output_end_pos = int(
                        np.ceil((end_pos - start_pos - future_size) / subsampling_rate)
                    )

                    if self.args.last_chunk_pad and self.args.end_problem_fix1:
                        output_end_pos = np.ceil(
                            (end_pos - start_pos - future_size - pad_length) / subsampling_rate
                        )
                else:
                    raise ValueError("No such chunk_hopping type")

                encoder_outputs.append(cur_encoder_outputs[:, output_start_pos:output_end_pos, :])
                not_padding.append(cur_not_padding[:, output_start_pos:output_end_pos])
                a.append(cur_a[:, output_start_pos:output_end_pos])
                prev_kv_cache = cur_key_value_cache

            encoder_outputs = torch.cat(encoder_outputs, dim=1)
            not_padding = torch.cat(not_padding, dim=1)
            a = torch.cat(a, dim=1)
        else:
            if self.args.front_end_type in (
                'Conv2dPooling'
            ) and self.args.acoustic_backbone_type in (
                'ConformerBackbone',
                'MaskedConformerBackbone',
            ):
                front_end_out, backbone_mask, frontend_shape = self.front_end(
                    inputs.squeeze(-1), inputs_mask
                )
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
                encoder_outputs, not_padding = encoder_backbone_out, backbone_mask.int()
            else:
                encoder_input, ignore_padding = self.front_end(inputs, inputs_mask)
                encoder_input = self.encoder_dropout(encoder_input)
                encoder_outputs, not_padding, _ = self.encoder_backbone(
                    encoder_input, ignore_padding=ignore_padding, prev_kv_cache=None
                )
            a = self.cif_weight_estimator(encoder_outputs, not_padding)
        return encoder_outputs, not_padding, a

    def forward(self, batch_data):
        '''
        Forward for CIF base module
        '''
        inputs = batch_data['src'].unsqueeze(-1)
        inputs_mask = batch_data['src_mask']
        targets = batch_data['char']
        hybrid_ce_targets = batch_data.get('ce_label', None)
        # inputs = batch_data[0]
        # targets = batch_data[1]
        # hybrid_ce_targets = None

        hparams = self.args
        # Encoder part
        encoder_outputs, not_padding, a = self.encoder(inputs, inputs_mask)

        # For calculating the CTC loss on the encoder
        logits_on_encoder = None
        not_padding_on_encoder = None
        if hparams.ctc_loss_on_encoder:
            logits_on_encoder = self.encoder_proj(encoder_outputs)
            not_padding_on_encoder = not_padding

        hybrid_ce_logits = None
        hybrid_ce_not_padding = None
        if hparams.use_hybrid_ce_loss or hparams.use_ce_pretrain:
            raise NotImplementedError()

        # CIF part
        (
            cif_outputs,
            not_padding_after_cif,
            sum_a,
            _,
            cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(encoder_outputs, not_padding, a, targets=targets, is_training=True)

        # Decoder part
        logits = self.decoder(cif_outputs, targets, is_training=True)

        # Loss part
        forward_out = self.criterion(
            logits,
            targets,
            inputs_mask,
            not_padding_after_cif,
            logits_on_encoder,
            not_padding_on_encoder,
            sum_a,
            hybrid_ce_logits,
            hybrid_ce_targets,
            hybrid_ce_not_padding,
        )

        # validation
        if not logits.requires_grad:
            (
                total_error_count,
                ins_error_count,
                del_error_count,
                sub_error_count,
                total_count,
            ) = self.inference(batch_data)
            forward_out['cer'] = total_error_count
            forward_out['total_error_count'] = total_error_count
            forward_out['ins_error_count'] = ins_error_count
            forward_out['del_error_count'] = del_error_count
            forward_out['sub_error_count'] = sub_error_count
            forward_out['total_count'] = total_count

        return forward_out

    @torch.no_grad()
    def fast_decode(self, inputs, inputs_mask):
        """
        fast_decode
        """
        hparams = self.args
        out_rlt_dict = {}
        # Encoder part
        encoder_outputs, not_padding, a = self.encoder(inputs, inputs_mask)

        if hparams.use_tail_handling:
            encoder_outputs = nn.functional.pad(encoder_outputs, [0, 0, 0, 1, 0, 0])
            not_padding = nn.functional.pad(not_padding, [0, 1, 0, 0])
            a = nn.functional.pad(a, [0, 1, 0, 0])

        if self.output_preappear_metric:
            logits_on_encoder = None
            not_padding_on_encoder = None
            if hparams.ctc_loss_on_encoder:
                logits_on_encoder = self.encoder_proj(encoder_outputs)
                not_padding_on_encoder = not_padding
            logits_on_encoder = logits_on_encoder * not_padding_on_encoder.unsqueeze(-1)
            out_rlt_dict['ctc_rlt'] = logits_on_encoder.argmax(dim=2, keepdim=False)

        # CIF part
        (
            cif_outputs,
            not_padding_after_cif,
            sum_a,  # pylint: disable=unused-variable
            _,
            cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(encoder_outputs, not_padding, a)

        if self.beam_size > 1:
            tokens = self.beam_searcher(cif_outputs, not_padding_after_cif)
        else:
            # Decoder part
            logits = self.decoder(cif_outputs)

            # argmax
            tokens = torch.argmax(logits, dim=-1)
            tokens = tokens * not_padding_after_cif

        if self.output_preappear_metric:
            out_rlt_dict['cif_boundary_marks'] = cif_dict['boundary_marks']
            out_rlt_dict['tokens'] = tokens

        if self.output_timestamp:
            bsz, _ = tokens.size()
            token_idxs = cif_dict['boundary_marks'].bool().nonzero(as_tuple=False)
            timestamp = [[] for _ in range(bsz)]
            for token_idx in token_idxs:
                timestamp[token_idx[0]].append(token_idx[1].item())
            out_rlt_dict['timestamp'] = timestamp
            out_rlt_dict['boundary_marks'] = token_idxs
            out_rlt_dict['tokens'] = tokens
        return tokens, out_rlt_dict

    @torch.no_grad()
    def inference(
        self,
        batch_data,
        mode='valid',
        beam_size=1,
        nbest=1,
        language=None,
        filter_list=None,
        concate_en_letters=None,
        output_timestamp=None,
        output_preappear_metric=None,
    ):
        '''CIF  Inference'''
        inputs = batch_data['src'].unsqueeze(-1)
        inputs_mask = batch_data['src_mask']

        targets = batch_data['char']
        targets_mask = batch_data['char_mask']

        self.beam_size = beam_size
        self.nbest = nbest
        self.output_timestamp = output_timestamp
        self.output_preappear_metric = output_preappear_metric

        token_ids, out_rlt_dict = self.fast_decode(inputs, inputs_mask)
        target_lengths = targets_mask.sum(-1).int()

        hyp_strs, tgt_strs, ed_info_list, align_info_list = self.cal_edit_distance(
            batch_data['uttid'],
            token_ids,
            targets,
            target_lengths,
            language,
            filter_list,
            concate_en_letters,
        )
        if mode == 'test':
            return (
                hyp_strs,
                tgt_strs,
                ed_info_list,
                align_info_list,
                out_rlt_dict,
            )
        (
            total_error_dist,
            ins_error_dist,
            del_error_dist,
            sub_error_dist,
            total_dist,
        ) = self.rlt_post_process(ed_info_list)
        return total_error_dist, ins_error_dist, del_error_dist, sub_error_dist, total_dist

    def cal_edit_distance(
        self, uttid, hyps, target, target_lengths, language, filter_list, concate_en_letters
    ):
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

            if self.args.feat_train_wav_eval:
                # (TODO) for aligning with tf v1.0 and v1.02 model, later delete
                refine_hyp_ids = []
                for cur_id in hyp_ids:
                    if cur_id == 1:
                        refine_hyp_ids.append(cur_id + 1)
                    elif cur_id > 1:
                        refine_hyp_ids.append(cur_id + 2)
                    else:
                        refine_hyp_ids.append(cur_id)
                hyp_ids = refine_hyp_ids

            if self.args.get('filter_special_marker', False):
                # align with penguin implementation
                vocab_shift = self.args.get('vocab_shift', 4)
                hyp_ids = [x for x in hyp_ids if x >= vocab_shift]
                tgt_ids = [x for x in tgt_ids if x >= vocab_shift]
                hyp_str = self.decode_ids(hyp_ids)
                tgt_str = self.decode_ids(tgt_ids)
            else:
                if self.args.get('save_until_pad', False):
                    hyp_ids = [x for x in hyp_ids if x != self.eos_id]
                    tgt_ids = [x for x in tgt_ids if x != self.eos_id]
                    hyp_str = self.decode_ids(self.save_until_idx(hyp_ids, idx=self.pad_id))
                    tgt_str = self.decode_ids(self.save_until_idx(tgt_ids, idx=self.pad_id))
                else:
                    hyp_str = self.decode_ids(self.save_until_idx(hyp_ids, idx=self.eos_id))
                    tgt_str = self.decode_ids(self.save_until_idx(tgt_ids, idx=self.eos_id))
            if concate_en_letters:
                hyp_str = ConcateEnLetters.concate_en_letters(hyp_str)
                tgt_str = ConcateEnLetters.concate_en_letters(tgt_str)
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

    @staticmethod
    def save_until_idx(hyp, idx=1):
        '''save until some idx'''
        try:
            index = list(hyp).index(idx)
            return hyp[0:index]
        except ValueError:
            # No EOS_ID: return the array as-is.
            return hyp

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

    def init_beam_search(
        self,
        inference_cfg,
        lm_solution,
    ):
        '''beam search init'''
        if inference_cfg.get('use_nonbatch_beam', False):
            self.beam_searcher = NonBatchBeamSearch(
                self.args,
                inference_cfg,
                self.decoder_for_beam_search,
                lm_solution,
            )
        else:
            self.beam_searcher = BaseBeamSearch(
                self.args,
                inference_cfg,
                self.decoder_for_beam_search,
                lm_solution,
            )

    def register_infers(self):
        '''register infer object for export'''
        if self.args.front_end_type in ('Conv2dPooling') and self.args.acoustic_backbone_type in (
            'ConformerBackbone',
            'MaskedConformerBackbone',
        ):
            self.acoustic_front_end_module = self.front_end
            self.acoustic_backbone_module = self.encoder_backbone
            self.acoustic_head_module = None
            self.backbone_pool_module = None
            # self.encoder_backbone = model.encoder_backbone
            infers = [
                CifEncoderExporter(self, self.cif_weight_estimator, self.encoder_proj, **self.args),
                CifDecoderLogitsExporter(self.decoder, **self.args),
            ]
        else:
            infers = [
                ChunkHopingEncoderExporter(
                    self.front_end,
                    self.encoder_backbone,
                    self.cif_weight_estimator,
                    self.encoder_proj,
                    **self.args
                ),
                CifDecoderLogitsExporter(self.decoder, **self.args),
            ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)


@register_solution("UniversalCifModel")
class UniversalCifModel(BaseCifModel):
    '''
    Dual mode traing for CIF
    '''

    def encoder(self, inputs, inputs_mask):
        '''
        Args:
          inputs: fbank feature inputs.
            [batch_size, input_length, fbank_dim, fbank_channel]

        Returns:
          encoder_output: Encoder representation.
            [batch_size, input_length/stride, hidden_size]
          not_padding: Used for loss computation and correct search length
            [batch_size, input_length/stride]
        '''
        if self.args.use_fp16:
            if self.args.no_tensorcore:
                pass
            else:
                inputs_org_shape = inputs.shape
                need_tensor_core = ((inputs_org_shape[1] + 32 - 1) // 32) * 32
                need_pad_value = [0] * 2 * (inputs.dim() - 2) + [
                    0,
                    need_tensor_core - inputs_org_shape[1],
                ]
                inputs = F.pad(inputs, need_pad_value, mode='constant', value=0.0)

        if self.args.front_end_type in ('Conv2dPooling') and self.args.acoustic_backbone_type in (
            'ConformerBackbone',
            'MaskedConformerBackbone',
        ):
            front_end_out, backbone_mask, frontend_shape = self.front_end(
                inputs.squeeze(-1), inputs_mask
            )

            encoder_backbone_out = self.encoder_backbone(
                front_end_out, backbone_mask, frontend_shape
            )
            encoder_outputs, not_padding = encoder_backbone_out, backbone_mask.int()
        else:
            raise RuntimeError('Unsupport front_end_type and acoustic_backbone_type')
        a = self.cif_weight_estimator(encoder_outputs, not_padding)

        return encoder_outputs, not_padding, a

    def forward(self, batch_data):
        '''
        Forward for CIF base module
        '''
        # pylint: disable=too-many-locals
        inputs = batch_data['src'].unsqueeze(-1)
        inputs_mask = batch_data['src_mask']
        targets = batch_data['char']
        # hybrid_ce_targets = batch_data.get('ce_label', None)
        # inputs = batch_data[0]
        # targets = batch_data[1]
        # hybrid_ce_targets = None
        hparams = self.args
        # Encoder part

        # forward stream mode
        self.encoder_backbone.set_stream_mode(True)
        stream_encoder_outputs, stream_not_padding, stream_a = self.encoder(inputs, inputs_mask)
        # forward nonstream mode
        self.encoder_backbone.set_stream_mode(False)
        encoder_outputs, not_padding, a = self.encoder(inputs, inputs_mask)

        # For calculating the CTC loss on the encoder
        logits_on_encoder = None
        not_padding_on_encoder = None
        stream_logits_on_encoder = None
        stream_not_padding_on_encoder = None
        if hparams.ctc_loss_on_encoder:
            stream_logits_on_encoder = self.encoder_proj(stream_encoder_outputs)
            stream_not_padding_on_encoder = stream_not_padding
            logits_on_encoder = self.encoder_proj(encoder_outputs)
            not_padding_on_encoder = not_padding

        # CIF part
        (
            cif_outputs,
            not_padding_after_cif,
            sum_a,
            _,
            cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(encoder_outputs, not_padding, a, targets=targets, is_training=True)

        (
            stream_cif_outputs,
            stream_not_padding_after_cif,
            stream_sum_a,
            _,
            stream_cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(
            stream_encoder_outputs, stream_not_padding, stream_a, targets=targets, is_training=True
        )

        # Decoder part
        logits = self.decoder(cif_outputs, targets, is_training=True)
        stream_logits = self.decoder(stream_cif_outputs, targets, is_training=True)

        # Loss part
        forward_out = self.criterion(
            stream_logits,
            logits,
            targets,
            inputs_mask,
            stream_not_padding_after_cif,
            not_padding_after_cif,
            stream_logits_on_encoder,
            logits_on_encoder,
            stream_not_padding_on_encoder,
            not_padding_on_encoder,
            stream_sum_a,
            sum_a,
        )

        # validation
        if not logits.requires_grad:
            # forward stream mode
            self.encoder_backbone.set_stream_mode(True)
            (
                total_error_count,
                ins_error_count,
                del_error_count,
                sub_error_count,
                total_count,
            ) = self.inference(batch_data)
            forward_out['stream_cer'] = total_error_count
            forward_out['stream_total_error_count'] = total_error_count
            forward_out['stream_ins_error_count'] = ins_error_count
            forward_out['stream_del_error_count'] = del_error_count
            forward_out['stream_sub_error_count'] = sub_error_count
            forward_out['stream_total_count'] = total_count
            # forward nonstream mode
            self.encoder_backbone.set_stream_mode(False)
            (
                total_error_count,
                ins_error_count,
                del_error_count,
                sub_error_count,
                total_count,
            ) = self.inference(batch_data)
            forward_out['nonstream_cer'] = total_error_count
            forward_out['nonstream_total_error_count'] = total_error_count
            forward_out['nonstream_ins_error_count'] = ins_error_count
            forward_out['nonstream_del_error_count'] = del_error_count
            forward_out['nonstream_sub_error_count'] = sub_error_count
            forward_out['nonstream_total_count'] = total_count

        return forward_out


@register_solution("Data2vecCifModel")
class Data2vecCifModel(BaseCifModel):
    '''
    CIF model with Data2vec Pretraining
    '''

    def __init__(self, args, is_pretrain=False):
        '''
        init function for CIF skeleton model.
        '''
        super().__init__(args, is_pretrain=is_pretrain)
        # args means cfg.solution config block
        self.args = FalconDict(args)
        self.num_updates = 0

        if self.args.data2vec_finetuning:
            # Encoder part
            self.apply_mask = args.apply_mask
            d = args.encoder_embed_dim

            self.w2v_model = eval(args.wav2vec_type)(args)
            self.final_dropout = nn.Dropout(args.final_dropout)
            self.freeze_finetune_updates = args.freeze_finetune_updates
            self.w2v_model_proj = nn.Linear(d, self.args.hidden_size)

        self._register_load_state_dict_pre_hook(self._model_load_hook)

        if self.args.fix_data2vec_model and self.args.data2vec_finetuning:
            self.w2v_model.requires_grad_(False)

        for name, param in self.w2v_model.named_parameters():
            if self.args.adapter_finetuning_layernorm:
                if 'layer_norm' in name:
                    param.requires_grad = True

            if self.args.adapter_finetuning:
                if 'adapter' in name:
                    param.requires_grad = True

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.num_updates = num_updates
        if self.args.data2vec_finetuning:
            self.w2v_model.encoder.num_updates = num_updates

    def _model_load_hook(
        self,
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
        if self.args.get('ema_prefix', None):
            old_state_dict = old_state_dict[self.args['ema_prefix']]
        for name, param in old_state_dict.items():
            new_name = name
            if name.startswith('data2vec_ctc_model.'):
                # bytespeech finetuned model
                new_name = name.replace('data2vec_ctc_model.', '')
            elif name.startswith('wav2vec_model.'):
                # bytespeech pretrained model
                new_name = name.replace('wav2vec_model.', 'w2v_model.')
            elif name.startswith('data2vec_model.'):
                if name.startswith('data2vec_model.h2l.'):
                    new_name = name.replace('data2vec_model.h2l.', 'w2v_model_proj.')
                elif name.startswith('data2vec_model.cif_weight_estimator.'):
                    new_name = name.replace(
                        'data2vec_model.cif_weight_estimator.', 'cif_weight_estimator.'
                    )
                else:
                    new_name = name.replace('data2vec_model.', 'w2v_model.')
            elif name.startswith('shared_encoder.d2v_model.'):
                # MOST pretrained model
                new_name = name.replace('shared_encoder.d2v_model.', 'w2v_model.')

            state_dict[new_name] = param

    def finetuning_encoder(self, batch_data):
        '''
        Args:
          inputs: fbank feature inputs.
            [batch_size, input_length, fbank_dim, fbank_channel]

        Returns:
          encoder_output: Encoder representation.
            [batch_size, input_length/stride, hidden_size]
          not_padding: Used for loss computation and correct search length
            [batch_size, input_length/stride]
        '''
        if self.args.data2vec_finetuning:
            padding_mask = (1 - batch_data['src_mask']).int().bool()

            w2v_args = {
                "batch_data": batch_data,
                "padding_mask": padding_mask,
                "mask": self.apply_mask and self.training,
            }
            ft = self.freeze_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                x, padding_mask = self.w2v_model.extract_features(**w2v_args)
        else:
            raise ValueError("Not support the fine-tuning of such pretrained model")

        x = self.final_dropout(x)

        if self.w2v_model_proj:
            x = self.w2v_model_proj(x)

        encoder_outputs, not_padding = x, (1 - padding_mask.int())

        if self.cif_weight_estimator is None:
            return encoder_outputs, not_padding, None

        a = self.cif_weight_estimator(encoder_outputs, not_padding)

        return encoder_outputs, not_padding, a

    @torch.no_grad()
    def fast_decode(self, inputs, inputs_mask, batch_data=None):
        """
        fast_decode
        """
        hparams = self.args
        # Encoder part
        if hparams.data2vec_finetuning:
            encoder_outputs, not_padding, a = self.finetuning_encoder(batch_data)
        else:
            encoder_outputs, not_padding, a = self.encoder(inputs, inputs_mask)

        if hparams.use_tail_handling:
            encoder_outputs = nn.functional.pad(encoder_outputs, [0, 0, 0, 1, 0, 0])
            not_padding = nn.functional.pad(not_padding, [0, 1, 0, 0])
            a = nn.functional.pad(a, [0, 1, 0, 0])

        # CIF part
        (
            cif_outputs,
            not_padding_after_cif,
            sum_a,  # pylint: disable=unused-variable
            _,
            cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(encoder_outputs, not_padding, a)

        if self.args.use_asr_nar_decoder:
            logits = self.decoder(cif_outputs, not_padding_after_cif)

            tokens = torch.argmax(logits, dim=-1)
            tokens = tokens * not_padding_after_cif
        else:
            if self.beam_size > 1:
                tokens = self.beam_searcher(cif_outputs, not_padding_after_cif)
            else:
                # Decoder part
                logits = self.decoder(cif_outputs)

                # argmax
                tokens = torch.argmax(logits, dim=-1)
                tokens = tokens * not_padding_after_cif
        return tokens

    @torch.no_grad()
    def inference(
        self,
        batch_data,
        mode='valid',
        beam_size=1,
        nbest=1,
        language=None,
        filter_list=None,
        concate_en_letters=None,
        output_timestamp=None,
        output_preappear_metric=None,
    ):
        '''CIF  Inference'''
        targets = batch_data['char']
        targets_mask = batch_data['char_mask']

        self.beam_size = beam_size
        self.nbest = nbest
        self.output_timestamp = output_timestamp
        self.output_preappear_metric = output_preappear_metric

        if self.args.data2vec_finetuning:
            token_ids = self.fast_decode(None, None, batch_data)
            out_rlt_dict = {}
        else:
            inputs = batch_data['src'].unsqueeze(-1)
            inputs_mask = batch_data['src_mask']
            token_ids = self.fast_decode(inputs, inputs_mask)
            out_rlt_dict = {}

        target_lengths = targets_mask.sum(-1).int()
        hyp_strs, tgt_strs, ed_info_list, align_info_list = self.cal_edit_distance(
            batch_data['uttid'],
            token_ids,
            targets,
            target_lengths,
            language,
            filter_list,
            concate_en_letters,
        )
        if mode == 'test':
            return (
                hyp_strs,
                tgt_strs,
                ed_info_list,
                align_info_list,
                out_rlt_dict,
            )
        (
            total_error_dist,
            ins_error_dist,
            del_error_dist,
            sub_error_dist,
            total_dist,
        ) = self.rlt_post_process(ed_info_list)
        return total_error_dist, ins_error_dist, del_error_dist, sub_error_dist, total_dist

    def forward(self, batch_data):
        '''
        Forward for CIF base module
        '''
        hparams = self.args
        org_batch_data = copy.deepcopy(batch_data)
        inputs_mask = batch_data['src_mask']
        targets = batch_data['char']
        hybrid_ce_targets = batch_data.get('ce_label', None)

        if hparams.data2vec_finetuning:
            encoder_outputs, not_padding, a = self.finetuning_encoder(batch_data)
        else:
            inputs = batch_data['src'].unsqueeze(-1)
            encoder_outputs, not_padding, a = self.encoder(inputs, inputs_mask)

        # For calculating the CTC loss on the encoder
        logits_on_encoder = None
        not_padding_on_encoder = None
        if hparams.ctc_loss_on_encoder:
            logits_on_encoder = self.encoder_proj(encoder_outputs)
            not_padding_on_encoder = not_padding

        hybrid_ce_logits = None
        hybrid_ce_not_padding = None
        if hparams.use_hybrid_ce_loss or hparams.use_ce_pretrain:
            raise NotImplementedError()

        # CIF part
        (
            cif_outputs,
            not_padding_after_cif,
            sum_a,
            _,
            cif_dict,  # pylint: disable=unused-variable
        ) = self.cif_calculator(encoder_outputs, not_padding, a, targets=targets, is_training=True)

        # Decoder part
        if self.args.use_asr_nar_decoder:
            logits = self.decoder(cif_outputs, not_padding_after_cif)
        else:
            logits = self.decoder(cif_outputs, targets, is_training=True)

        # Loss part
        forward_out = self.criterion(
            logits,
            targets,
            inputs_mask,
            not_padding_after_cif,
            logits_on_encoder,
            not_padding_on_encoder,
            sum_a,
            hybrid_ce_logits,
            hybrid_ce_targets,
            hybrid_ce_not_padding,
        )

        self.num_updates += 1

        # validation
        if not logits.requires_grad:
            (
                total_error_count,
                ins_error_count,
                del_error_count,
                sub_error_count,
                total_count,
            ) = self.inference(org_batch_data)
            forward_out['cer'] = total_error_count
            forward_out['total_error_count'] = total_error_count
            forward_out['ins_error_count'] = ins_error_count
            forward_out['del_error_count'] = del_error_count
            forward_out['sub_error_count'] = sub_error_count
            forward_out['total_count'] = total_count

            self.num_updates -= 1

        return forward_out


@register_solution("Data2vecCtcModel")
class Data2vecCtcModel(Data2vecCifModel):
    '''
    Data2vec model for CIF Comparison Experiments.
    '''

    def __init__(self, args):
        '''
        init function for CTC skeleton model.
        '''
        super().__init__(args)

        self.cif_weight_estimator = None
        # Cif Calculator
        self.cif_calculator = None
        self.decoder = None
        self.decoder_for_beam_search = None

    def forward(self, batch_data):
        '''
        Forward for CIF base module
        '''
        hparams = self.args
        org_batch_data = copy.deepcopy(batch_data)
        inputs_mask = batch_data['src_mask']
        targets = batch_data['char']

        if hparams.data2vec_finetuning:
            encoder_outputs, not_padding, _ = self.finetuning_encoder(batch_data)
        else:
            inputs = batch_data['src'].unsqueeze(-1)
            encoder_outputs, not_padding, _ = self.encoder(inputs, inputs_mask)

        # For calculating the CTC loss on the encoder
        logits_on_encoder = None
        not_padding_on_encoder = None
        if hparams.ctc_loss_on_encoder:
            logits_on_encoder = self.encoder_proj(encoder_outputs)
            not_padding_on_encoder = not_padding

        # Loss part
        forward_out = self.criterion(
            logits_on_encoder,
            targets,
            inputs_mask,
            None,
            logits_on_encoder,
            not_padding_on_encoder,
            None,
            None,
            None,
            None,
        )

        self.num_updates += 1

        # validation
        if not logits_on_encoder.requires_grad:
            (
                total_error_count,
                ins_error_count,
                del_error_count,
                sub_error_count,
                total_count,
            ) = self.inference(org_batch_data)
            forward_out['cer'] = total_error_count
            forward_out['total_error_count'] = total_error_count
            forward_out['ins_error_count'] = ins_error_count
            forward_out['del_error_count'] = del_error_count
            forward_out['sub_error_count'] = sub_error_count
            forward_out['total_count'] = total_count

            self.num_updates -= 1

        return forward_out

    def decode_ids(self, hyp):
        """decode ids"""
        if self.args.get('decode_char', False):
            words = ''
            for i in hyp:
                if i in (0, 1, 2, 3, 32):
                    continue
                if i < 0 or i > 32:
                    continue
                token = self.args.tgt_dict.string([i])
                if not token:
                    continue
                if token == '<space>':
                    token = ' '
                words += token
            return words

        hyp_token = []
        word = ''
        for idx, token_id in enumerate(hyp):
            token_id = token_id.item()
            if token_id in (0, 1, 2, 3):
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

    @torch.no_grad()
    def fast_decode(self, inputs, inputs_mask, batch_data=None):
        """
        fast_decode
        """
        hparams = self.args

        # Encoder part
        if hparams.data2vec_finetuning:
            encoder_outputs, not_padding, _ = self.finetuning_encoder(batch_data)
        else:
            encoder_outputs, not_padding, _ = self.encoder(inputs, inputs_mask)

        logits_on_encoder = None
        not_padding_on_encoder = None
        if hparams.ctc_loss_on_encoder:
            logits_on_encoder = self.encoder_proj(encoder_outputs)
            not_padding_on_encoder = not_padding
        logits_on_encoder = logits_on_encoder * not_padding_on_encoder.unsqueeze(-1)

        max_token_len = 0
        batch_size = logits_on_encoder.size(0)
        frame_length = logits_on_encoder.size(1)
        tokens = []

        if self.beam_size > 1:
            raise ValueError("Use greedy search for ctc evaluation")

        # use ctc greedy search
        ctc_output_tokens = logits_on_encoder.argmax(dim=2, keepdim=False)

        for bid in range(batch_size):
            # blk id
            prev_id = 0
            cur_tokens = []
            for t in range(frame_length):
                cur_id = ctc_output_tokens[bid][t]
                if prev_id == 0:
                    if cur_id != 0:
                        cur_tokens.append(cur_id)
                else:
                    if cur_id not in (prev_id, 0):
                        cur_tokens.append(cur_id)
                prev_id = cur_id
            max_token_len = len(cur_tokens) if len(cur_tokens) > max_token_len else max_token_len
            cur_tokens += [
                torch.tensor(0, device='cuda') for _ in range(frame_length - len(cur_tokens))
            ]
            cur_tokens = torch.Tensor(cur_tokens).unsqueeze(0).int()
            tokens.append(cur_tokens)
        tokens = torch.cat(tokens, axis=0)[:, :max_token_len]

        return tokens
