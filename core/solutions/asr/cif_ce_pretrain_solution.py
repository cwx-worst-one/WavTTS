''' BaseRnntModel '''
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.solutions.base_solution import register_solution
from core.utils.dict import FalconDict
from .base_cif_model import BaseCifModel


@register_solution("BaseCIFPretrainModel")
class BaseCIFPretrainModel(BaseCifModel):
    '''
    Base encoder model for CIF encoder.
    - Encoder
    '''

    def __init__(self, args):
        '''
        init function for CIF pretrain model.
        '''
        super().__init__(args=args, is_pretrain=True)
        self.args = FalconDict({**args, '__default_falcon_value': None})
        self.encoder_pretrain_proj = nn.Linear(
            self.args.hidden_size, self.args.vocab_size, bias=False
        )

    def forward(self, batch_data):
        '''
        Forward for CIF encoder base module
        '''
        ############ Encoder #############
        inputs = batch_data['src'].unsqueeze(-1)
        inputs_mask = batch_data['src_mask']
        hybrid_ce_targets = batch_data['ce_label']
        bsz = inputs_mask.shape[0]
        hybrid_ce_targets = hybrid_ce_targets.view(bsz, -1, self.args.downsampling_size)[:, :, 0]
        targets = batch_data['char']
        encoder_outputs, not_padding, a = self.encoder(inputs, inputs_mask)
        sum_a = a.sum(-1)
        encoder_pretrain_logits = self.encoder_pretrain_proj(encoder_outputs)
        forward_out = self.criterion(
            logits=None,
            targets=targets,
            src_mask=inputs_mask,
            not_padding_on_encoder=not_padding,
            sum_a=sum_a,
            hybrid_ce_logits=encoder_pretrain_logits,
            hybrid_ce_targets=hybrid_ce_targets,
            hybrid_ce_targets_mask=not_padding,
        )
        return forward_out
