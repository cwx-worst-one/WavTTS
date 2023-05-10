''' RNN-T Las Rescore Training solution '''
import copy
from packaging import version

from core.models.asr.acoustic_backbone import *
from core.models.asr.las_decoder import *
from core.criterions import *
from core.solutions.base_solution import register_solution
from core.solutions.inference import INFERS, BaseInfer
from core.utils import FalconDict
from .base_rnnt_model import (
    BaseRnntModel,
    RnntEncoderExporter,
    RnntJointerExporterForLM,
    RnntPredictorExporter,
    RnntJointerExporter,
)
from .rnnt_twopass_model import (
    LasEncoderExporter,
    RnntLasRescoreModel,
)


class RnntLasDecoderExporter(BaseInfer):
    '''export las decoder to onnx'''

    NAME = 'las_decoder'
    INPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T']),
        FalconDict(name='char', type=torch.long, shape=['B', 'T']),
    ]
    OUTPUTS = [FalconDict(name='logits', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, args, decoder_module, **kwargs):
        kwargs['need_jit_script'] = True
        super().__init__(kwargs)
        self.encoder_proj_fc = torch.jit.trace(
            decoder_module.encoder_proj_fc.cuda(),
            torch.rand(1, 512, args.backbone_memory_size, device='cuda'),
        )
        self.embed_tokens = torch.jit.trace(
            decoder_module.embed_tokens.cuda(), torch.ones(1, 128, device='cuda').long()
        )
        self.attention = torch.jit.trace(
            decoder_module.attention.cuda(),
            (
                torch.rand(1, 512, args.backbone_memory_size, device='cuda'),
                torch.rand(1, 512, args.atten_hidden_size, device='cuda'),
                torch.rand(1, args.embedding_size, device='cuda'),
                torch.rand(1, args.decoder_lstm_hidden_size, device='cuda'),
            ),
        )
        self.lstm = [
            torch.jit.trace(
                decoder_module.lstm[0].cuda(),
                (
                    torch.rand(1, 1, args.decoder_input_size + args.embedding_size, device='cuda'),
                    (
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    ),
                ),
            )
        ]
        self.lstm += [
            torch.jit.trace(
                decoder_module.lstm[layer].cuda(),
                (
                    torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    (
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    ),
                ),
            )
            for layer in range(1, args.decoder_lstm_layer_num)
        ]
        self.lstm = nn.ModuleList(self.lstm).cuda()
        self.pred_fc = torch.jit.trace(
            decoder_module.pred_fc.cuda(),
            torch.rand(
                1,
                127,
                args.decoder_lstm_hidden_size + args.backbone_memory_size + args.embedding_size,
                device='cuda',
            ),
        )

        self.decoder_lstm_hidden_size = torch.tensor(
            decoder_module.decoder_lstm_hidden_size, device='cuda'
        )
        self.decoder_lstm_layer_num = torch.tensor(
            decoder_module.decoder_lstm_layer_num, device='cuda'
        )
        self.eval()

    def forward(self, encoder_out, char):
        '''
        Forward for RNN-T Las G2P decoder module
        '''
        encoder_proj = self.encoder_proj_fc(encoder_out)
        prev_emb = self.embed_tokens(char)

        # repeat encoder beams times for efficiency
        bsz = char.shape[0]
        encoder_out = encoder_out.repeat(bsz, 1, 1)
        encoder_proj = encoder_proj.repeat(bsz, 1, 1)

        lstms_state0 = torch.zeros(
            self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
        )
        lstms_state1 = torch.zeros(
            self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
        )

        total_concat_out = torch.zeros(
            bsz,
            1,
            prev_emb.shape[-1] + encoder_out.shape[-1] + self.decoder_lstm_hidden_size,
            device='cuda',
        )

        for token_idx in range(char.shape[1]):
            att_ctx = self.attention(
                encoder_out, encoder_proj, prev_emb[:, token_idx, :], lstms_state0[-1][0]
            )
            lstm_input = torch.cat([att_ctx, prev_emb[:, token_idx, :]], dim=1).unsqueeze(1)
            new_lstms_state0 = torch.zeros(1, 1, bsz, self.decoder_lstm_hidden_size, device='cuda')
            new_lstms_state1 = torch.zeros(
                self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
            )
            for index, layer in enumerate(self.lstm):
                lstm_out, (lstm_state0, lstm_state1) = layer(
                    lstm_input, (lstms_state0[index], lstms_state1[index])
                )
                lstm_input = lstm_out
                new_lstms_state0 = torch.cat([new_lstms_state0, lstm_state0.unsqueeze(0)])
                new_lstms_state1 = torch.cat([new_lstms_state1, lstm_state1.unsqueeze(0)])
            concat_out = torch.cat([prev_emb[:, token_idx, :], att_ctx, lstm_out[:, 0, :]], dim=1)
            total_concat_out = torch.cat([total_concat_out, concat_out.unsqueeze(1)], dim=1)
            lstms_state0 = new_lstms_state0[1:]
            lstms_state1 = new_lstms_state1[1:]

        logits = self.pred_fc(total_concat_out[:, 1:, :])
        return logits

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        las_encoder = self._generate_input_data(0, dynamic_axis=[1, 512])
        target_rnnt = self._generate_input_data(1, method='ones', dynamic_axis=[1, 128])
        datas = [las_encoder, target_rnnt]
        return tuple(datas)


@register_solution("RnntLasG2PModel")
class RnntLasG2PModel(BaseRnntModel):
    '''RNN-T LAS G2P Model'''

    def __init__(self, args):
        '''
        init function for RNNT & LAS G2P model
        '''
        super().__init__(args)
        self.las_added_backbone_type = args.get('las_added_backbone_type', None)
        if self.las_added_backbone_type is not None:
            self.las_backbone_final_norm = None
            las_args = copy.deepcopy(args)
            if 'TransformerBackbone' in args.las_added_backbone_type:
                las_args.backbone_topology = args.las_backbone_topology
                las_args.backbone_layer_num = len(eval(las_args.backbone_topology))
                las_args.dropout = args.las_dropout
                las_args.backbone_memory_size = args.las_backbone_memory_size
                las_args.backbone_hidden_size = args.las_backbone_hidden_size
                las_args.self_attn_heads = args.las_self_attn_heads
                las_args.self_attn_dropout = args.las_self_attn_dropout
                las_args.self_attn_activation_dropout = args.las_self_attn_activation_dropout
                las_args.self_attn_layer_norm_before = args.las_self_attn_layer_norm_before
                las_args.self_attn_activation_fn = args.las_self_attn_activation_fn
            elif args.las_added_backbone_type == 'LSTMPBackbone':
                las_args.backbone_layer_num = args.las_backbone_layer_num
                las_args.backbone_memory_size = args.las_backbone_memory_size
                las_args.backbone_hidden_size = args.las_backbone_hidden_size
                las_args.dropout = args.las_dropout
                las_args.backbone_bilstm = args.las_backbone_bilstm
                las_args.backbone_residual = args.las_backbone_residual
                las_args.backbone_mask = args.las_backbone_mask
            elif args.las_added_backbone_type == 'DFSMNBackboneLN':
                las_args.backbone_topology = args.las_backbone_topology
                las_args.dropout = args.las_dropout
                las_args.backbone_memory_size = args.las_backbone_memory_size
                las_args.backbone_hidden_size = args.las_backbone_hidden_size
            else:
                raise RuntimeError("las_added_backbone_type does not supported")
            self.las_acoustic_backbone_module = eval(args.las_added_backbone_type)(las_args)
        else:
            self.las_acoustic_backbone_module = None
        self.decoder_module = eval(args.las_decoder_type)(args)
        self.las_criterion_module = eval(args.las_criterion_type)(args)
        self._register_load_state_dict_pre_hook(RnntLasRescoreModel.compatible_load_hook)

    def prepare_rnnt_encoder_out(self, batch_data):
        '''get rnnt encoder out'''
        acoustic_out, backbone_mask, _, mtl_logits, encoder_backbone_out = self.encoder(batch_data)
        batch_data['encoder_out'] = acoustic_out
        batch_data['backbone_mask'] = backbone_mask
        batch_data['mtl_logits'] = mtl_logits
        batch_data['encoder_backbone_out'] = encoder_backbone_out

    def las_encoder(self, encoder_backbone_out, backbone_mask=None):
        '''
        additional las encoder layers above the rnnt encoder layers
        '''
        if self.las_acoustic_backbone_module is None:
            return encoder_backbone_out, backbone_mask
        encoder_out = self.las_acoustic_backbone_module(encoder_backbone_out, backbone_mask, "BTN")
        return encoder_out, backbone_mask

    def las_decoder(self, encoder_out, backbone_mask, batch_data):
        '''
        Decoder Module
        '''
        prev_char = batch_data['char']
        logits = self.decoder_module(encoder_out, backbone_mask, prev_char)
        src_mask = batch_data['src_mask']
        target_mask = batch_data['char_mask']
        target = batch_data['sy_label']
        forward_out = self.las_criterion_module(logits, src_mask, target, target_mask)
        return forward_out

    def las_forward(self, batch_data):
        '''
        Forward
        '''
        encoder_backbone_out = batch_data['encoder_backbone_out']
        backbone_mask = batch_data['backbone_mask']
        encoder_out, backbone_mask = self.las_encoder(encoder_backbone_out, backbone_mask)
        forward_out = self.las_decoder(encoder_out, backbone_mask, batch_data)
        return forward_out

    def g2p_greedy_inference(self, batch_data):
        '''
        greedy inference
        '''
        encoder_backbone_out = batch_data['encoder_backbone_out']
        backbone_mask = batch_data['backbone_mask']
        encoder_out, backbone_mask = self.las_encoder(encoder_backbone_out, backbone_mask)
        char = batch_data['char']
        logits = self.decoder_module(encoder_out, backbone_mask, char)
        hyps = logits.max(dim=-1)[1]
        clean_hyps = []
        char_length = batch_data['char_mask'].sum(1).int().tolist()
        for idx, hyp in enumerate(hyps.tolist()):
            clean_hyps.append(hyp[: char_length[idx]])
        return clean_hyps

    def register_infers(self):
        # pylint: disable=invalid-name
        self.args.setdefault("return_backbone", True)
        infers = [
            RnntPredictorExporter(self.predictor_module, **self.args),
            RnntEncoderExporter(self, **self.args),
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
            use_non_combined_adaptive_softmax = self.args.get(
                'use_non_combined_adaptive_softmax', False
            )
            infers.append(
                RnntJointerExporter(
                    self.jointer_module,
                    self.criterion_module,
                    use_non_combined_adaptive_softmax=use_non_combined_adaptive_softmax,
                    **self.args
                )
            )
        infers.append(LasEncoderExporter(self, **self.args))
        infers.append(RnntLasDecoderExporter(self.args, self.decoder_module, **self.args))

        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
