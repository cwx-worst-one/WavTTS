''' BaseEmotionRecognitionModel'''
# pylint: disable=abstract-method,invalid-name
# pylint: disable=unused-argument
# pylint: disable=unused-import
# pylint: disable=protected-access
# pylint: disable=import-error
import os
import os.path as osp
from collections import OrderedDict
import copy
import re
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (
    AutoConfig,
    Wav2Vec2ForCTC,
    HubertForCTC,
    HubertForSequenceClassification,
    Wav2Vec2ForSequenceClassification,
)
from transformers import BertModel as BertModel_huggingface
from transformers import ElectraModel as ElectraModel_huggingface
import numpy as np
from core.models.pretrained.bert_model import BertModel
from core.models.pretrained.electra_model import ElectraModel
from core.models.pretrained.wav2vec2_model import *
from core.models.aed.classifier import *
from core.solutions.base_solution import BaseSolution, register_solution

from core.models.layers.pooling import *
from core.criterions import *
from core.solutions.inference import BaseInfer, INFERS
from core.utils import FalconDict
from core.utils import hdfs_get, get_local_rank, dist_barrier


class BaseEmotionRecognitionExporter(BaseInfer):
    '''export emotion model to onnx'''

    NAME = 'emotion_recognition'
    INPUTS = [
        FalconDict(name='waveform', type=torch.int16, shape=['B', 'T']),
        FalconDict(name='src_mask', type=torch.int32, shape=['B', 1]),
    ]
    OUTPUTS = [
        FalconDict(name='original', type=torch.float32, shape=['B', 1]),
        FalconDict(name='emotion', type=torch.float32, shape=['B', 11]),
    ]
    # TEST_ATOL = 1e-3

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.args = model.args
        self.model = model

    @torch.no_grad()
    def forward(self, waveform, src_mask):
        '''
        Forward for emotion base module
        '''
        batch_data = OrderedDict()
        batch_data['waveform'] = waveform.float()
        max_len = waveform.shape[1]
        batch_data['src_mask'] = (
            torch.arange(max_len).cuda().expand(src_mask.shape[0], max_len) < src_mask
        ).float()

        outputs = self.model.forward(batch_data)
        return outputs['original'].reshape(-1, 1), outputs['emotion']

    def sample_inputs(self):

        waveform1 = self._generate_input_data(0, dynamic_axis=[5, 160000], low=-16000, high=16000)
        src_mask1 = self._generate_input_data(1, dynamic_axis=[5, 1], low=1, high=16000)
        tmp = np.load('tmp.npy', allow_pickle=True)
        waveform = tmp.item()['waveform']
        src_mask = tmp.item()['src_mask']
        waveform = waveform.type_as(waveform1)
        src_mask = src_mask.type_as(src_mask1)

        return waveform, src_mask


class MultimodalEmotionRecognitionExporter(BaseInfer):
    '''export emotion model to onnx'''

    name = 'emotion_recognition'
    INPUTS = [
        FalconDict(name='waveform', type=torch.int16, shape=['B', 'T']),
        FalconDict(name='src_mask', type=torch.int32, shape=['B', 1]),
        FalconDict(name='input_id', type=torch.int32, shape=['B', 'T']),
        FalconDict(name='input_lens', type=torch.int32, shape=['B', 1]),
    ]

    OUTPUTS = [FalconDict(name='emotion', type=torch.float32, shape=['B', -1])]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.args = model.args
        self.model = model
        self.onnx_emotion_type = self.args.get('onnx_emotion_type', 'negative')
        ctg_dict = {'negative': 3, 'impatient': 2, 'confront': 3, 'arousal_QA': 2, 'arousal': 2}
        self.OUTPUTS[0]['shape'][1] = ctg_dict[self.onnx_emotion_type]

    @torch.no_grad()
    def forward(self, waveform, src_mask, input_id, input_lens):
        '''
        Forward for emotion base module
        '''
        batch_data = OrderedDict()
        batch_data['waveform'] = waveform.float()
        max_len = waveform.shape[1]
        batch_data['src_mask'] = (
            torch.arange(max_len).cuda().expand(src_mask.shape[0], max_len) < src_mask
        ).float()

        batch_data['text'] = input_id
        max_len = input_id.shape[1]
        batch_data['text_mask'] = (
            torch.arange(max_len).cuda().expand(input_lens.shape[0], max_len) < input_lens
        ).float()

        outputs = self.model.forward(batch_data)
        output = outputs['m_emotion']

        preds = torch.argmax(output, dim=-1)

        # export onnx for customer negative
        if self.onnx_emotion_type == 'negative':
            preds = torch.div(preds, 2).type('torch.LongTensor')
            ctg = int((self.args.emotion_num + 1) / 2)

        # export onnx for service impatient
        if self.onnx_emotion_type == 'impatient':
            preds1 = torch.gt(preds, 0).type('torch.LongTensor')
            preds2 = torch.lt(preds, 3).type('torch.LongTensor')
            preds = preds1 * preds2
            ctg = 2

        # export onnx for service confront
        if self.onnx_emotion_type == 'confront':
            preds1 = torch.div(torch.add(preds, 1), 2).type('torch.LongTensor')
            preds2 = torch.lt(preds, 5).type('torch.LongTensor')
            preds = preds1 * preds2
            ctg = 3

        # export onnx for service arousal
        if self.onnx_emotion_type == 'arousal_QA':
            preds1 = torch.gt(preds, 0).type('torch.LongTensor')
            preds2 = torch.lt(preds, 4).type('torch.LongTensor')
            preds3 = torch.gt(input_lens[:, 0], 22).type('torch.LongTensor')
            preds = preds1 * preds2 * preds3
            ctg = 2

        # export onnx for service arousal_practice
        if self.onnx_emotion_type == 'arousal':
            preds1 = torch.gt(preds, 0).type('torch.LongTensor')
            preds2 = torch.lt(preds, 11).type('torch.LongTensor')
            preds3 = ((output * torch.tensor(range(13)).cuda()).sum(axis=-1) / 10).cpu()
            preds4 = preds1 * preds2 * preds3
            preds = preds1 * preds2
            ctg = 2

        preds = torch.reshape(preds, (-1, 1)).to('cuda')

        y_one_hot = torch.zeros(preds.shape[0], ctg).to('cuda').scatter_(1, preds, 1)
        y_one_hot = torch.reshape(y_one_hot, (-1, ctg)).type('torch.FloatTensor')
        if self.onnx_emotion_type == 'arousal':
            y_one_hot[:, 1] = y_one_hot[:, 1] * preds4

        return y_one_hot

    def sample_inputs(self):

        waveform1 = self._generate_input_data(0, dynamic_axis=[5, 160000], low=-16000, high=16000)
        src_mask1 = self._generate_input_data(1, dynamic_axis=[5, 1], low=1, high=16000)
        text1 = self._generate_input_data(2, dynamic_axis=[5, 16], low=1, high=2000)
        text_mask1 = self._generate_input_data(3, dynamic_axis=[5, 1], low=1, high=100)
        return waveform1, src_mask1, text1, text_mask1


@register_solution("MultimodalEmotionRecognitionModel")
class MultimodalEmotionRecognitionModel(BaseSolution):
    '''emotion recognition for different output'''

    def __init__(self, args):
        '''init of emotion recognition model'''
        # pylint:disable=too-many-branches
        super().__init__()
        self.args = args
        self.apply_mask = args.apply_mask
        predict_sets = args.get('predict_sets', 'emotion_phone')
        self.predict_sets = predict_sets.strip().split('|')
        self.fns = {}

        if (
            'a_emotion' in self.predict_sets
            or 'm_emotion' in self.predict_sets
            or 'mix_emotion' in self.predict_sets
        ):
            self.w2v_model = eval(args.encoder_type)(args)
            a_emotion_dim = args.get('a_emotion_dim', 1024)
            self.a_classifier = EmotionSingleOutHead(
                args,
                args.emotion_num,
                final_dropout=args.get('a_drop_out', 0.0),
                encoder_embed_dim=a_emotion_dim,
            )
            if self.args.get('freeze_acoustic_encoder', False):
                for name, param in self.w2v_model.named_parameters():
                    param.requires_grad = False

            if self.args.get('activate_part_acoustic_encoder', False):
                for name, param in self.w2v_model.named_parameters():
                    if name.startswith('encoder.layers.23') or name.startswith(
                        'encoder.layer_norm'
                    ):
                        param.requires_grad = True
            self.fns['a_emotion'] = [self.utterance_classification, self.a_classifier, 'emotion']

        if (
            't_emotion' in self.predict_sets
            or 'm_emotion' in self.predict_sets
            or 'mix_emotion' in self.predict_sets
        ):

            t_emotion_dim = args.get('t_emotion_dim', 768)

            if self.args.get('bert_vocab_size', None) is not None:
                if t_emotion_dim == 768:
                    self.bert_model = BertModel(args)
                else:
                    self.bert_model = ElectraModel(
                        args, add_pooling_layer=self.args.get('add_bert_pooling_layer', False)
                    )
            else:
                if args.init_bert_model.startswith('hdfs'):
                    local_checkpoint = args.init_bert_model.split('/')[-1]
                    local_config = args.init_bert_config.split('/')[-1]
                    local_rank = get_local_rank()
                    if local_rank == 0:
                        try:
                            os.system('rm -rf {}'.format(local_checkpoint))
                            os.system('rm -rf {}'.format(local_config))
                            hdfs_get(args.init_bert_model, local_checkpoint)
                            hdfs_get(args.init_bert_config, local_config)
                        except Exception:
                            pass
                    args.init_bert_model = local_checkpoint
                    args.init_bert_config = local_config
                    dist_barrier()

                config = AutoConfig.from_pretrained(
                    args.init_bert_config,
                )
                if t_emotion_dim == 768:
                    self.bert_model = BertModel_huggingface.from_pretrained(
                        args.init_bert_model, config=config
                    )
                else:
                    self.bert_model = ElectraModel_huggingface.from_pretrained(
                        args.init_bert_model, config=config
                    )

            self.t_classifier = EmotionSeqOutHead(
                args,
                args.emotion_num,
                final_dropout=args.get('t_drop_out', 0.0),
                encoder_embed_dim=t_emotion_dim,
            )

            self.t_linear = EmotionSeqOutHead(
                args,
                t_emotion_dim,
                final_dropout=args.get('t_emotion_drop_out', 0.0),
                encoder_embed_dim=t_emotion_dim,
            )

            if self.args.get('freeze_text_encoder', False):
                for name, param in self.bert_model.named_parameters():
                    param.requires_grad = False

            if self.args.get('activate_part_text_encoder', False):
                for name, param in self.bert_model.named_parameters():
                    if name.startswith('encoder.layer.11') or name.startswith('pooler.dense'):
                        param.requires_grad = True
            self.fns['t_emotion'] = [self.multimodal_classification, self.t_classifier, 'emotion']

        if 'm_emotion' in self.predict_sets:
            m_emotion_dim = args.get('m_emotion_dim', a_emotion_dim + t_emotion_dim)

            self.m_classifier = EmotionSeqOutHead(
                args,
                args.emotion_num,
                final_dropout=args.get('m_drop_out', 0.0),
                encoder_embed_dim=m_emotion_dim,
            )
            if self.args.get('attention_merge', False):
                self.ta_encoder = EmotionMuiltmodalSeqOutHead(
                    args,
                    t_emotion_dim,
                    final_dropout=args.get('m_drop_out', 0.0),
                    a_emotion_dim=t_emotion_dim,
                    t_emotion_dim=a_emotion_dim,
                )
                self.at_encoder = EmotionMuiltmodalSeqOutHead(
                    args,
                    a_emotion_dim,
                    final_dropout=args.get('m_drop_out', 0.0),
                    a_emotion_dim=a_emotion_dim,
                    t_emotion_dim=t_emotion_dim,
                )
            self.fns['m_emotion'] = [self.multimodal_classification, self.m_classifier, 'emotion']

        if 'fbank' in self.predict_sets:
            self.mfcc_encoder = EmotionSingleLSTMHead(
                args,
                128,
                final_dropout=args.get('mfcc_drop_out', 0.0),
                encoder_embed_dim=args.get('fbank_dim', 80),
            )
            self.mfcc_classifier = EmotionSeqOutHead(
                args,
                args.emotion_num,
                final_dropout=args.get('mfcc_drop_out', 0.0),
                encoder_embed_dim=128,
            )
            self.fns['fbank'] = [self.multimodal_classification, self.mfcc_classifier, 'emotion']

        self.seqclassifier_loss_fct = CTC(args)
        self.regressor_loss_fct = MSELoss()
        self.classifier_loss_fct = CrossEntropyLoss()

    def tf_chkpt_converter(self, tf_var_dict):
        """ Load tf checkpoints in a pytorch model, revised from https://github.com/huggingface/\
            transformers/blob/main/src/transformers/models/electra/modeling_electra.py """
        # pylint:disable=too-many-branches
        model = self.bert_model
        for name in tf_var_dict:
            array = tf_var_dict[name]
            name = name.replace('.', '/')
            name = name.replace('//ATTRIBUTES/VARIABLE_VALUE', '')
            name = name.replace('S', '')
            name = name.replace('bert/', '')

            original_name: str = name
            try:
                name = name.replace("dense_1", "dense_prediction")
                name = name.replace("generator_predictions/output_bias", "generator_lm_head/bias")

                if name.find('intent_labels_dense') != -1:
                    pointer = self.t_classifier.pred_fc
                    name = name.replace('intent_labels_dense/', '')
                else:
                    pointer = model

                name = name.split("/")

                # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to \
                # calculated m and v, which are not required for using pretrained model
                if any(n in ["global_step", "temperature"] for n in name):
                    continue
                m_name = ''
                for m_name in name:
                    if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                        scope_names = re.split(r"_(\d+)", m_name)
                    else:
                        scope_names = [m_name]
                    if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                        pointer = getattr(pointer, "weight")
                    elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                        pointer = getattr(pointer, "bias")
                    elif scope_names[0] == "output_weights":
                        pointer = getattr(pointer, "weight")
                    elif scope_names[0] == "squad":
                        pointer = getattr(pointer, "classifier")
                    else:
                        pointer = getattr(pointer, scope_names[0])
                    if len(scope_names) >= 2:
                        num = int(scope_names[1])
                        pointer = pointer[num]
                if m_name.endswith("_embeddings"):
                    pointer = getattr(pointer, "weight")
                elif m_name == "kernel":
                    array = np.transpose(array)
                try:
                    assert (
                        pointer.shape == array.shape
                    ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
                except AssertionError:
                    continue
                print(f"Initialize PyTorch weight {name}", original_name)
                pointer.data = torch.from_numpy(array)
            except AttributeError as e:
                print(f"Skipping {original_name}", name, e)
                continue
        # return model

    def sequence_classification(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        classifier,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for sequence classification'''
        encoder_output = classifier(encoder_outputs['encoder_output'])
        forward_out[logit_tag] = nn.functional.softmax(encoder_output, dim=-1)
        if label_tag in batch_data:
            net_output = {
                "encoder_out": encoder_output,  # B x T x C
                "padding_mask": encoder_outputs['padding_mask'],  # B x T
                "labels": batch_data[label_tag],
            }
            loss = self.seqclassifier_loss_fct(net_output)
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] = forward_out['backward_loss'] + loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def multimodal_classification(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        classifier,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for sequence classification'''
        encoder_output = classifier(encoder_outputs['encoder_output'])
        forward_out[logit_tag] = nn.functional.softmax(encoder_output, dim=-1)
        if label_tag in batch_data:
            loss = self.classifier_loss_fct(
                encoder_output,
                torch.Tensor(batch_data[label_tag]).type(torch.long).to('cuda'),
            )
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] = forward_out['backward_loss'] + loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def utterance_lstm_classification(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        classifier,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for single utterance classification'''
        encoder_output = classifier(encoder_outputs['encoder_output'])
        forward_out[logit_tag] = nn.functional.softmax(encoder_output, dim=-1)
        if label_tag in batch_data:
            loss = self.classifier_loss_fct(
                encoder_output,
                torch.Tensor(batch_data[label_tag]).type(torch.long).to('cuda'),
            )
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] = forward_out['backward_loss'] + loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def utterance_classification(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        classifier,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for single utterance classification'''
        encoder_output = classifier(
            encoder_outputs['encoder_output'],
            encoder_outputs['padding_mask'],
            encoder_outputs['hidden_states'],
        )

        forward_out[logit_tag] = nn.functional.softmax(encoder_output, dim=-1)
        if label_tag in batch_data:
            loss = self.classifier_loss_fct(
                encoder_output,
                torch.Tensor(batch_data[label_tag]).type(torch.long).to('cuda'),
            )
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] = forward_out['backward_loss'] + loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def utterance_regression(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        regressor,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for utterance regression'''
        encoder_output = regressor(
            encoder_outputs['encoder_output'],
            encoder_outputs['padding_mask'],
            encoder_outputs['hidden_states'],
        )
        forward_out[logit_tag] = encoder_output.view(-1)
        if label_tag in batch_data:
            loss = self.regressor_loss_fct(
                encoder_output.view(-1), torch.Tensor(batch_data[label_tag]).to('cuda').view(-1)
            )
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] = forward_out['backward_loss'] + loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    @staticmethod
    def get_pool_data(data, data_mask=None, reverse=True):
        '''get pool result'''
        if data_mask is None:
            data_pool = data.mean(dim=1)
        else:
            if reverse:
                data_mask = 1 - data_mask.long()
            data = data * data_mask.view(data_mask.shape[0], data_mask.shape[1], 1)
            data_pool = data.sum(dim=1) / data_mask.sum(dim=1).view(-1, 1)
        return data, data_pool

    def forward(self, batch_data):
        """forward"""
        # pylint:disable=too-many-branches

        encoder_outputs = {}
        if self.args.get('acoustic_mode', None) in ['fbank_only', 'both']:
            fbank_encoder_pool = self.mfcc_encoder(batch_data['src'])
            encoder_outputs['fbank'] = {
                'encoder_output': fbank_encoder_pool,
                'padding_mask': None,
                'extracted_features': None,
                'hidden_states': None,
            }
        if self.args.get('acoustic_mode', None) in ['a_emotion_only', 'both']:
            if 'waveform' not in batch_data:
                batch_data['waveform'] = batch_data['input_values']
            if 'src_mask' not in batch_data:
                batch_data['src_mask'] = batch_data['attention_mask']  # 320000

            w2v_args = {
                "batch_data": batch_data,
                "mask": self.apply_mask and self.training,
                "output_feature_extractor": self.args.get('output_feature_extractor', False),
                "output_hidden_states": self.args.get('output_hidden_states', False),
            }

            if self.args.get('freeze_acoustic_encoder', False):
                with torch.no_grad():
                    acoustic_encoder_outputs = self.w2v_model.forward_freeze_internal(**w2v_args)
            else:
                acoustic_encoder_outputs = self.w2v_model.forward_freeze_internal(**w2v_args)

            encoder_outputs['a_emotion'] = {
                'encoder_output': acoustic_encoder_outputs['encoder_output'],
                'padding_mask': acoustic_encoder_outputs['padding_mask'],
                'extracted_features': None,
                'hidden_states': None,
            }
            acoustic_encoder, acoustic_encoder_pool = self.get_pool_data(
                acoustic_encoder_outputs['encoder_output'], acoustic_encoder_outputs['padding_mask']
            )

        if 't_emotion' in self.predict_sets or 'm_emotion' in self.predict_sets:
            bert_args = {
                'input_ids': batch_data['text'],
                'attention_mask': batch_data['text_mask'],
            }
            outputs = self.bert_model(**bert_args)

            if 'pooler_output' in outputs:
                text_encoder_pool = outputs['pooler_output']
            elif self.args.get('text_pool_out', False):
                _, text_encoder_pool = self.get_pool_data(
                    outputs['last_hidden_state'], batch_data['text_mask'], False
                )
            else:
                text_encoder_pool = outputs['last_hidden_state'][:, 0, :]
            encoder_outputs['t_emotion'] = {
                'encoder_output': text_encoder_pool,
                'padding_mask': 1 - batch_data['text_mask'],
                'extracted_features': None,
                'hidden_states': None,
            }

        if 'm_emotion' in self.predict_sets:
            if self.args.get('acoustic_mode', None) == 'fbank_only':
                encoder_outputs['m_emotion'] = {
                    'encoder_output': torch.cat((text_encoder_pool, fbank_encoder_pool), 1),
                }
            if self.args.get('attention_merge', False):

                text_attention_acoustic = self.at_encoder(
                    acoustic_encoder, outputs['last_hidden_state'], batch_data['text_mask']  # 10
                )

                acoustic_attention_text = self.ta_encoder(
                    outputs['last_hidden_state'],
                    acoustic_encoder,
                    1 - acoustic_encoder_outputs['padding_mask'].long(),  #
                )
                _, text_encoder_pool1 = self.get_pool_data(
                    acoustic_attention_text, batch_data['text_mask'], False
                )
                _, acoustic_encoder_pool1 = self.get_pool_data(
                    text_attention_acoustic, acoustic_encoder_outputs['padding_mask']
                )

                text_encoder_pool = torch.add(text_encoder_pool, text_encoder_pool1)
                acoustic_encoder_pool = torch.add(acoustic_encoder_pool, acoustic_encoder_pool1)

            if self.args.get('acoustic_mode', None) == 'a_emotion_only':
                encoder_outputs['m_emotion'] = {
                    'encoder_output': torch.cat((text_encoder_pool, acoustic_encoder_pool), 1),
                }

        forward_out = OrderedDict()

        for predict_tag in self.predict_sets:
            if predict_tag == 'mix_emotion':
                continue
            label_tag = 'emotion'
            if predict_tag != 'fbank':
                label_tag = predict_tag.split('_')[-1]
            forward_out = self.fns[predict_tag][0](
                batch_data,
                encoder_outputs[predict_tag],
                forward_out,
                self.fns[predict_tag][1],
                predict_tag,
                label_tag,
            )

        if 'mix_emotion' in self.predict_sets:
            for tag in ['t_emotion', 'a_emotion', 'fbank', 'm_emotion']:
                if tag in forward_out:
                    if 'mix_emotion' in forward_out:
                        forward_out['mix_emotion'] = forward_out['mix_emotion'] + forward_out[tag]
                    else:
                        forward_out['mix_emotion'] = forward_out[tag]
        if 'src_mask' in batch_data:
            forward_out['frame_size'] = batch_data['src_mask'].float().sum()
        elif 'text_mask' in batch_data:
            forward_out['frame_size'] = batch_data['text_mask'].float().sum()

        return forward_out

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        infers = [MultimodalEmotionRecognitionExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def get_onnx(self, out_dir):
        '''register infer object for export and beamsearch.'''
        self.onnx_exporter = MultimodalEmotionRecognitionExporter(self, **self.args)
        self.onnx_type = self.args.get('onnx_type', 'origin')
        self.onnx_exporter.load_model(
            osp.join(
                self.onnx_exporter._onnx_dir,
                '{}_{}.onnx'.format(self.onnx_exporter.NAME, self.onnx_type),
            )
        )

    def test_export_onnx_batch(self, inputs):
        '''test onnx'''
        default_values = {
            'waveform': inputs['waveform'].cpu().detach().numpy()
            if 'waveform' in inputs
            else inputs['input_values'].cpu().detach().numpy(),
            'src_mask': inputs['src_mask'].sum(-1).long().cpu().detach().numpy().reshape(-1, 1)
            if 'src_mask' in inputs
            else inputs['attention_mask'].sum(-1).long().cpu().detach().numpy().reshape(-1, 1),
            'input_id': inputs['text'].cpu().detach().numpy(),
            'input_lens': inputs['text_mask'].sum(-1).long().cpu().detach().numpy().reshape(-1, 1),
        }
        default_values = {
            'waveform': default_values['waveform'].astype(np.int16),
            'src_mask': default_values['src_mask'].astype(np.int32),
            'input_id': default_values['input_id'].astype(np.int32),
            'input_lens': default_values['input_lens'].astype(np.int32),
        }
        outputs2 = {}
        output_name = self.onnx_exporter.get_output()
        results = self.onnx_exporter.sess.run(self.onnx_exporter.get_output(), default_values)
        for i, value in enumerate(output_name):
            outputs2[value] = results[i]
        return outputs2


@register_solution("BaseEmotionRecognitionModel")
class BaseEmotionRecognitionModel(BaseSolution):
    '''emotion recognition for different output'''

    def __init__(self, args):
        '''init of emotion recognition model'''
        super().__init__()
        self.args = args
        self.apply_mask = args.apply_mask

        predict_sets = args.get('predict_sets', 'emotion_phone')
        self.predict_sets = predict_sets.strip().split('|')
        if not self.args.get('huggingface_model', False):
            self.w2v_model = eval(args.encoder_type)(args)
        else:
            if args.init_huggingface_model.startswith('hdfs'):
                local_checkpoint = args.init_huggingface_model.split('/')[-1]
                local_config = args.init_huggingface_config.split('/')[-1]
                local_rank = get_local_rank()
                if local_rank == 0:
                    try:
                        os.system('rm -rf {}'.format(local_checkpoint))
                        os.system('rm -rf {}'.format(local_config))
                        hdfs_get(args.init_huggingface_model, local_checkpoint)
                        hdfs_get(args.init_huggingface_config, local_config)
                    except Exception:
                        pass
                args.init_huggingface_model = local_checkpoint
                args.init_huggingface_config = local_config
                dist_barrier()

            if 'emotion_phone' in self.predict_sets:
                num_labels = args.tgt_size
            else:
                num_labels = args.emotion_num
            config = AutoConfig.from_pretrained(
                args.init_huggingface_config,
                ignore_mismatched_sizes=True,
                num_labels=num_labels,
                gradient_checkpointing=True,
                ctc_loss_reduction="mean",
                pad_token_id=0,
                vocab_size=args.tgt_size,
                use_weighted_layer_sum=args.get('use_weighted_layer_sum', False),
            )
            self.w2v_model = eval(args.pretrain_model).from_pretrained(
                args.init_huggingface_model, ignore_mismatched_sizes=True, config=config
            )
            self.w2v_model.freeze_feature_extractor()

        self.seqclassifier = EmotionSeqOutHead(args, args.tgt_size)
        self.seqclassifier_p = EmotionSeqOutHead(args, args.dim_tgt_size)
        self.seqclassifier_a = EmotionSeqOutHead(args, args.dim_tgt_size)
        self.seqclassifier_d = EmotionSeqOutHead(args, args.dim_tgt_size)

        self.classifier = EmotionSingleOutHead(args, args.emotion_num)
        self.classifier_p = EmotionSingleOutHead(args, args.dim_emotion_num)
        self.classifier_a = EmotionSingleOutHead(args, args.dim_emotion_num)
        self.classifier_d = EmotionSingleOutHead(args, args.dim_emotion_num)

        self.regressor = EmotionSingleOutHead(args, args.emotion_num, regression=True)
        self.regressor_p = EmotionSingleOutHead(args, args.dim_emotion_num, regression=True)
        self.regressor_a = EmotionSingleOutHead(args, args.dim_emotion_num, regression=True)
        self.regressor_d = EmotionSingleOutHead(args, args.dim_emotion_num, regression=True)

        self.seqclassifier_loss_fct = CTC(args)
        self.regressor_loss_fct = MSELoss()
        self.classifier_loss_fct = CrossEntropyLoss()

    def sequence_classification(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        classifier,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for sequence classification'''
        encoder_output = classifier(encoder_outputs['encoder_output'])
        forward_out[logit_tag] = nn.functional.softmax(encoder_output, dim=-1)
        if label_tag in batch_data:
            net_output = {
                "encoder_out": encoder_output,  # B x T x C
                "padding_mask": encoder_outputs['padding_mask'],  # B x T
                "labels": batch_data[label_tag],
            }
            loss = self.seqclassifier_loss_fct(net_output)
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] += loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def utterance_classification(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        classifier,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for single utterance classification'''
        encoder_output = classifier(
            encoder_outputs['encoder_output'],
            encoder_outputs['padding_mask'],
            encoder_outputs['hidden_states'],
        )

        forward_out[logit_tag] = nn.functional.softmax(encoder_output, dim=-1)
        if label_tag in batch_data:
            loss = self.classifier_loss_fct(
                encoder_output, torch.Tensor(batch_data[label_tag]).type(torch.long).to('cuda')
            )
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] += loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def utterance_regression(
        self,
        batch_data,
        encoder_outputs,
        forward_out,
        regressor,
        logit_tag='logits',
        label_tag='labels',
    ):
        '''emotion recognition model for utterance regression'''
        encoder_output = regressor(
            encoder_outputs['encoder_output'],
            encoder_outputs['padding_mask'],
            encoder_outputs['hidden_states'],
        )
        forward_out[logit_tag] = encoder_output.view(-1)
        if label_tag in batch_data:
            loss = self.regressor_loss_fct(
                encoder_output.view(-1), torch.Tensor(batch_data[label_tag]).to('cuda').view(-1)
            )
            if 'backward_loss' in forward_out:
                forward_out['backward_loss'] += loss
            else:
                forward_out['backward_loss'] = loss
                forward_out['tgt_size'] = len(batch_data[label_tag])
            forward_out['loss'] = forward_out['backward_loss']
        return forward_out

    def forward(self, batch_data):
        """forward"""
        if self.args.get('huggingface_model', False):
            if 'input_values' not in batch_data:
                batch_data['input_values'] = batch_data['waveform']
            if 'attention_mask' not in batch_data:
                batch_data['attention_mask'] = batch_data['src_mask']  # 320000

            label_tag = self.predict_sets[0]
            labels = (
                torch.tensor(batch_data[label_tag], dtype=torch.long).to('cuda')
                if label_tag in batch_data
                else None
            )
            attention_mask = (
                batch_data['attention_mask'] if 'attention_mask' in batch_data else None
            )
            encoder_outputs = self.w2v_model(
                batch_data['input_values'],
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            forward_out = OrderedDict()
            forward_out[label_tag] = encoder_outputs["logits"]
            if labels is not None:
                forward_out['backward_loss'] = encoder_outputs["loss"]
                forward_out['loss'] = forward_out['backward_loss']
                forward_out['tgt_size'] = len(batch_data[label_tag])
            return forward_out

        if 'waveform' not in batch_data:
            batch_data['waveform'] = batch_data['input_values']
        if 'src_mask' not in batch_data:
            batch_data['src_mask'] = batch_data['attention_mask']  # 320000

        w2v_args = {
            "batch_data": batch_data,
            "mask": self.apply_mask and self.training,
            "output_feature_extractor": self.args.get('output_feature_extractor', False),
            "output_hidden_states": self.args.get('output_hidden_states', False),
        }
        encoder_outputs = self.w2v_model.forward_freeze_internal(**w2v_args)
        forward_out = OrderedDict()

        fns = {
            'emotion_phone': [self.sequence_classification, self.seqclassifier],
            'emotion_phone_p': [self.sequence_classification, self.seqclassifier_p],
            'emotion_phone_a': [self.sequence_classification, self.seqclassifier_a],
            'emotion_phone_d': [self.sequence_classification, self.seqclassifier_d],
            'emotion': [self.utterance_classification, self.classifier],
            'emotion_p': [self.utterance_classification, self.classifier_p],
            'emotion_a': [self.utterance_classification, self.classifier_a],
            'emotion_d': [self.utterance_classification, self.classifier_d],
            'original': [self.utterance_regression, self.regressor],
            'original_p': [self.utterance_regression, self.regressor_p],
            'original_a': [self.utterance_regression, self.regressor_a],
            'original_d': [self.utterance_regression, self.regressor_d],
        }

        for predict_tag in self.predict_sets:
            forward_out = fns[predict_tag][0](
                batch_data,
                encoder_outputs,
                forward_out,
                fns[predict_tag][1],
                predict_tag,
                predict_tag,
            )
        if 'src_mask' in batch_data:
            forward_out['frame_size'] = batch_data['src_mask'].float().sum()

        return forward_out

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        infers = [BaseEmotionRecognitionExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def get_onnx(self, out_dir):
        '''register infer object for export and beamsearch.'''
        self.onnx_exporter = BaseEmotionRecognitionExporter(self, **self.args)
        self.onnx_type = self.args.get('onnx_type', 'origin')
        self.onnx_exporter.load_model(
            osp.join(
                self.onnx_exporter._onnx_dir,
                '{}_{}.onnx'.format(self.onnx_exporter.NAME, self.onnx_type),
            )
        )

    def test_export_onnx_batch(self, inputs):
        '''test onnx'''
        default_values1 = {}
        default_values1['waveform'] = inputs['input_values']
        default_values1['src_mask'] = inputs['attention_mask'].sum(-1).reshape(-1, 1)
        default_values = {
            'waveform': inputs['waveform'].cpu().detach().numpy()
            if 'waveform' in inputs
            else inputs['input_values'].cpu().detach().numpy(),
            'src_mask': inputs['src_mask'].sum(-1).long().cpu().detach().numpy().reshape(-1, 1)
            if 'src_mask' in inputs
            else inputs['attention_mask'].sum(-1).long().cpu().detach().numpy().reshape(-1, 1),
        }
        output_data = self.onnx_exporter(**default_values1)
        print(output_data)

        default_values = {
            'waveform': default_values['waveform'].astype(np.int16),
            'src_mask': default_values['src_mask'].astype(np.int32),
        }
        outputs2 = {}
        output_name = self.onnx_exporter.get_output()

        results = self.onnx_exporter.sess.run(self.onnx_exporter.get_output(), default_values)
        print(results)
        for i, value in enumerate(output_name):
            outputs2[value] = results[i]
        return outputs2
