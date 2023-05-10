"""BaseNNLMModel and BaseNNLMSolution
(Note luotong) This file is here only for training lstmlm,
will be deleted in next MR for refactor training.
"""
# pylint: disable=abstract-method
from core.models.lm.lstm_lm import *
from core.criterions import *
from core.models.utils import xavier_init
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.inference import BaseInfer, INFERS, concat_global_states
from core.utils import FalconDict


class LSTMLMExporter(BaseInfer):
    '''export lstm lm to onnx'''

    NAME = 'lstm_lm'
    INPUTS = [FalconDict(name='prev_tgt', type=torch.long, shape=['B'])]
    OUTPUTS = [FalconDict(name='lprobs', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, lstm_module, **kwargs):
        super().__init__(kwargs)
        self.lstm_module = lstm_module
        self._tgt_vocab_size = kwargs.get('tgt_vocab_size', 1)
        self._convert_stream_flag = kwargs.get('nnlm_convert_stream', True)
        if self._convert_stream_flag:
            state_size = getattr(self.lstm_module, 'state_size', 0)
            self._inputs.append(
                FalconDict(name='global_state_in', type=torch.float32, shape=['Batch', state_size])
            )
            self._outputs.append(
                FalconDict(name='global_state_out', type=torch.float32, shape=['Batch', state_size])
            )

    def forward(self, prev_tgt):
        '''forward
        Args:
            prev_tgt: [B, U], previous target
        Return:
            lprobs: [B, U, H], lstm output
        '''
        prev_tgt = prev_tgt.unsqueeze(1)
        logits = self.lstm_module.forward(prev_tgt)
        lprobs = logits.log_softmax(dim=2)
        return lprobs

    def forward_step(self, prev_tgt, global_state_in=None):
        '''
        forward step.

        Args:
          prev_tgt[torch.Tensor]: [Batch]
          global_state_in[torch.Tensor]: [Batch, state_size]
        '''
        lprobs, states = self.lstm_module.forward_step(prev_tgt, global_state_in)
        lprobs = lprobs.unsqueeze(1)
        states = concat_global_states(*states)
        return lprobs, states

    def sample_inputs(self):
        prev_tgt = self._generate_input_data(
            0, dynamic_axis=[128], low=0, high=self._tgt_vocab_size
        )
        if self._convert_stream_flag:
            global_state_in = self._generate_input_data(1)
            return prev_tgt, global_state_in
        return prev_tgt

    def sample_inputs_from_dataloader(self):
        '''
        sample some real data from dataloader.
        '''
        prev_char = []
        num_batch = 4
        for _ in range(num_batch):
            batch_data = self._data_loader.next()
            prev_char.append(batch_data['prev_char'].detach().cpu().numpy())
        datas = {'prev_tgt': prev_char}
        return datas


@register_solution("BaseNNLMModel")
class BaseNNLMModel(BaseSolution):
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
        super().__init__()
        self.args = args
        self.nnlm_model = eval(args.nnlm_model_type)(args)

        self.criterion_module = eval(args.criterion_type)(args)
        if args.get('xavier_init', False):
            xavier_init(self)

    def forward(self, batch_data):
        '''
        Forward for RNNLM base module
        '''

        lm_src = batch_data['src']
        lm_out = self.nnlm_model.forward(lm_src)
        src_mask = batch_data['src_mask']
        target = batch_data['char']
        target_mask = batch_data['char_mask']
        # criterion for loss computation
        forward_out = self.criterion_module(lm_out, src_mask, target, target_mask)

        return forward_out

    def get_log_prob(self, batch_data):
        '''
        get log prob
        '''
        forward_out = self.forward(batch_data)
        log_prob = forward_out['lprobs']
        return log_prob

    def forward_step(self, prev_tgt, states):
        '''Froward by a token step'''
        return self.nnlm_model.forward_step(prev_tgt, states)

    @staticmethod
    def max_decoder_positions():
        '''
        # used for decoding
        '''
        return 10000

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        infers = [LSTMLMExporter(self.nnlm_model, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
