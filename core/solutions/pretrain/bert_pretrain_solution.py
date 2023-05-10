"""bert pretrain"""

from core.models.pretrained.bert_model import *
from core.models.pretrained.bert_pipeline_model import *
from core.solutions.base_solution import BaseSolution, register_solution


@register_solution("BaseBertPretrainModel")
class BaseBertPretrainModel(BaseSolution):
    """bert pretrain model"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args
        self.bert_model = eval(args.bert_type)(args)

    def forward(self, batch_data):
        """forward"""
        input_ids = batch_data['src']
        attention_mask = batch_data.get('src_mask', None)
        token_type_ids = batch_data.get('token_type_ids', None)
        position_ids = batch_data.get('position_ids', None)
        head_mask = batch_data.get('head_mask', None)
        inputs_embeds = batch_data.get('inputs_embeds', None)
        labels = batch_data.get('char', None)
        next_sentence_label = batch_data.get('next_sentence_label', None)
        output_attentions = self.args.get('output_attentions', False)
        output_hidden_states = self.args.get('output_hidden_states', False)
        return_dict = self.args.get('use_return_dict', True)
        forward_out = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            next_sentence_label=next_sentence_label,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return forward_out
