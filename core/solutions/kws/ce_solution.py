''' BaseCeModel and BaseCeSolution '''
import numpy as np
import torch.nn.functional as F

from core.models.kws.ce_encoder import *
from core.models.kws.criterion_head import *

# pylint: disable=unused-import
from core.models.kws.rnnt_encoder import ConvDFSMNEncoder as RNNTConvDFSMNEncoder

# pylint: disable=unused-import
from core.models.kws.rnnt_encoder import ConvNolowrankDFSMNEncoder as RNNTConvNolowrankDFSMNEncoder
from core.criterions import *
from core.utils import FalconDict
from core.solutions.inference import BaseInfer, INFERS
from core.solutions.base_solution import BaseSolution, register_solution


class KwsCeModelExporter(BaseInfer):
    '''export kws ce model to onnx'''

    NAME = 'kws_ce'
    INPUTS = [FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T'])]

    def __init__(self, model, **kwargs):
        '''
        init function
        '''
        super().__init__(kwargs)
        self.encoder = model.encoder
        self.args = model.args
        self.eval()
        fbank_dim = kwargs.get('fbank_dim')
        self._inputs[0].shape[2] = fbank_dim

    def forward(self, fbank):
        '''
        forward for kws ce model
        '''
        encoder_out, _, _, _ = self.encoder(fbank)
        output_layer_type = self.args.get('output_layer_type', 'softmax')
        if output_layer_type == 'softmax':
            encoder_out = F.softmax(encoder_out.float(), dim=-1)
        elif output_layer_type == 'log_softmax':
            encoder_out = F.log_softmax(encoder_out.float(), dim=-1)
        else:
            raise RuntimeError('not supported output layer type: %s' % (output_layer_type))
        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, method='ones', dynamic_axis=[1, 512, -1])
        return fbank


@register_solution("KwsCeModel")
class KwsCeModel(BaseSolution):
    '''
    KWS CE model.
    '''

    def __init__(self, args):
        '''
        init function for CE skeleton model.
        '''
        super().__init__()
        self.args = args
        self.encoder = eval(args.encoder_type)(args)
        self.criterion = eval(args.criterion_type)(args)

    def forward(self, batch_data):
        '''
        Forward for CE base module
        '''
        fbank = batch_data['src']
        fbank_mask = batch_data['src_mask']
        ce_label = batch_data['ce_label']
        encoder_out, _, encoder_mask, target = self.encoder(fbank, fbank_mask, ce_label)

        # criterion for loss computation
        forward_out = self.criterion(encoder_out, fbank_mask, target, encoder_mask)
        return forward_out

    def load_cmvn(self, mean, inv_std):
        '''
        Load cmvn for model export
        '''
        self.eval()
        mvn_concat_size = self.args.get('mvn_concat_size', 1)
        with torch.no_grad():
            self.encoder.mean.copy_(torch.tensor(np.tile(mean, mvn_concat_size)))
            self.encoder.inv_std.copy_(torch.tensor(np.tile(inv_std, mvn_concat_size)))

    def register_infers(self):
        infers = [KwsCeModelExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def export(self, *_args, **_kwargs):
        '''export onnx'''
        for infer_name in self._infer_names:
            INFERS[infer_name].export()


class KwsCeAdaptModelEncoderExporter(BaseInfer):
    '''export kws ce adapt model's encoder to onnx'''

    NAME = 'kws_ce_adapt_encoder'
    INPUTS = [FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, model, **kwargs):
        '''
        init function
        '''
        super().__init__(kwargs)
        self.encoder = model.encoder
        self.args = model.args
        self.eval()
        fbank_dim = kwargs.get('fbank_dim')
        self._inputs[0].shape[2] = fbank_dim
        if kwargs.get('selected_dfsmn_layer', -1) != -1:
            self._outputs.append(
                FalconDict(name='selected_encoder_out', type=torch.float, shape=['B', 'T', -1])
            )
        self.with_slim = kwargs.get('slim_quant_encoder')

    def forward(self, fbank):
        '''
        Forward for kws ce adapt model
        '''
        encoder_out, selected_encoder_out, _, _ = self.encoder(fbank)

        if selected_encoder_out is not None:
            return encoder_out, selected_encoder_out

        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, method='ones', dynamic_axis=[1, 512, -1])
        return fbank


class KwsCeAdaptModelHeadExporter(BaseInfer):
    '''export kws ce adapt model's head to onnx'''

    NAME = 'kws_ce_adapt_head'
    INPUTS = [FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='head_out', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.ce_head = model.ce_head
        self.args = model.args
        self.eval()
        head_input_size = kwargs.get('head_input_size')
        self._inputs[0].shape[2] = head_input_size
        self.with_slim = kwargs.get('slim_quant_head')

    def forward(self, encoder_out):
        '''
        Forward for kws ce adapt model
        '''
        head_out = self.ce_head(encoder_out)
        output_layer_type = self.args.get('output_layer_type', 'softmax')
        if output_layer_type == 'softmax':
            head_out = F.softmax(head_out.float(), dim=-1)
        elif output_layer_type == 'log_softmax':
            head_out = F.log_softmax(head_out.float(), dim=-1)
        else:
            raise RuntimeError('not supported output layer type: %s' % (output_layer_type))
        return head_out

    def sample_inputs(self):
        encoder_out = self._generate_input_data(0, method='ones', dynamic_axis=[1, 512, -1])
        return encoder_out


@register_solution("KwsCeAdaptModel")
class KwsCeAdaptModel(BaseSolution):
    '''
    KWS CE adaptation model.
    '''

    def __init__(self, args):
        '''
        init function for CE skeleton model.
        '''
        super().__init__()
        self.args = args
        self.encoder = eval(args.encoder_type)(args)
        self.ce_head = eval(args.head_type)(args)
        self.criterion = eval(args.criterion_type)(args)
        if self.args.freeze_encoder:
            print('freeze encoder')
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
        self.args['slim_quant_encoder'] = False
        self.args['slim_quant_head'] = False

    def forward(self, batch_data):
        '''
        Forward for CE base module
        '''
        if self.args.freeze_encoder:
            self.encoder.eval()

        fbank = batch_data['src']
        fbank_mask = batch_data['src_mask']
        ce_label = batch_data['ce_label']
        encoder_out, selected_encoder_out, encoder_mask, target = self.encoder(
            fbank, fbank_mask, ce_label
        )

        if self.args.head_use_selected_encoder_out:
            head_input = selected_encoder_out
        else:
            head_input = encoder_out
        head_out = self.ce_head(head_input)

        # criterion for loss computation
        forward_out = self.criterion(head_out, fbank_mask, target, encoder_mask)

        return forward_out

    def load_cmvn(self, mean, inv_std):
        '''
        Load cmvn for model export
        '''
        self.eval()
        mvn_concat_size = self.args.get('mvn_concat_size', 1)
        with torch.no_grad():
            self.encoder.mean.copy_(torch.tensor(np.tile(mean, mvn_concat_size)))
            self.encoder.inv_std.copy_(torch.tensor(np.tile(inv_std, mvn_concat_size)))

    def register_infers(self):
        infers = [
            KwsCeAdaptModelEncoderExporter(self, **self.args),
            KwsCeAdaptModelHeadExporter(self, **self.args),
        ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def export(self, *_args, **_kwargs):
        '''export onnx'''
        for infer_name in self._infer_names:
            INFERS[infer_name].export()
