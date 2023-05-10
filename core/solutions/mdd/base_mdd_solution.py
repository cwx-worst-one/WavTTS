''' BaseSidModel and BaseSidSolution '''

from collections import OrderedDict
import sys
import torch
from torch import nn

import torch.nn.functional as F

from core.models.mdd.acoustic_encoder import *
from core.criterions import *
from core.solutions.inference import BaseInfer, INFERS
from core.solutions.base_solution import BaseSolution, register_solution
from core.utils import FalconDict

EPSILON = 1e-5


@register_solution("SiameseEmbeddNet")
class SiameseEmbeddNet(BaseSolution):
    """
    Siamese network for embedding extraction
    """

    def __init__(self, args):
        super().__init__()

        self.ref_encoder = eval(args.encoder_type)(args)
        self.usr_encoder = eval(args.encoder_type)(args)

        self.out = nn.Linear(args.hidden_dim, args.out_dim)
        self.criterion_module = eval(args.criterion_type)(args)

        self.args = args

    def forward(self, batch_data):  # x, x_utt, y, y_utt, x_mask=None, y_mask=None):
        """
        Args:
            input:
                x (Batch x Time x Channel): Features extracted from reference sample
                x_utt (Batch x Channel): utterance level features extracted from reference sample
                y (Batch x Time x Channel): Features extracted from user's sample
                y_utt (Batch x Channel): utterance level features extracted from user's sample
                x_mask (Batch x Time x Channel): padding mask for reference sample
                y_mask (Batch x Time x Channel): padding mask for user's sample
            output:
                sim_score (Batch x 1): output cosine similarity between ref embedding and
                    user embedding followed by an MSE loss for optimization
                class_score (Batch x Class): output posterior for score classes
                    followed by a CrossEntropy loss for optimization
        """
        x = batch_data['ref_feat']
        x_utt = batch_data['ref_utt']
        x_mask = batch_data['ref_mask']
        x_score = batch_data['ref_score']

        y = batch_data['usr_feat']
        y_utt = batch_data['usr_utt']
        y_mask = batch_data['usr_mask']
        y_score = batch_data['usr_score']
        y_class = batch_data['usr_score_class']

        x_mask = (1 - x_mask).int().bool()
        y_mask = (1 - y_mask).int().bool()
        ref_embedd = self.ref_encoder(x, x_utt, x_mask)
        usr_embedd = self.usr_encoder(y, y_utt, y_mask)

        if self.criterion_type == 'MSE_utt':
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim_score = cos(usr_embedd, ref_embedd)

            ref_score = y_score / x_score.float()
            forward_out = self.criterion_module(sim_score, ref_score)

        elif self.criterion_type == 'Xentropy_utt':
            dis = torch.abs(ref_embedd - usr_embedd)
            out_class = self.out(dis)
            forward_out = self.criterion_module(out_class, y_class)
        return forward_out

        # dis = torch.abs(ref_embedd - usr_embedd)
        # class_score = self.out(dis)

        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # sim_score = cos(ref_embedd, usr_embedd)

        # return sim_score, class_score


@register_solution("ScoreClassificationNet")
class ScoreClassificationNet(BaseSolution):
    """
    score classification network:
    """

    def __init__(self, args):
        super().__init__()

        self.acoustic_encoder = eval(args.encoder_type)(args)
        self.linear_out = nn.Linear(args.hidden_dim, args.out_dim)
        self.criterion_module = eval(args.criterion_type)(args)

    def forward(self, batch_data):  # x, x_utt, x_mask=None):
        """
        Args:
            input:
                x (Batch x Time x Channel): input features for rhythm prediction
                x_utt (Batch x Channel): utterance level features for rhythm prediction
                x_mask (Batch x Time x Channel): padding mask for input features
            output:
                out_score (Batch x classes): output posterior for score classes
                    By default it consists of 5 score slots: 0, 1-3, 4-6, 7-9 and 10
        """
        x = batch_data['usr']
        x_utt = batch_data['usr_utt']
        x_mask = batch_data['usr_mask']

        ref_score = batch_data['usr_score']

        x_mask = (1 - x_mask).int().bool()
        embedd = self.acoustic_encoder(x, x_utt, x_mask)
        out_score = self.linear_out(embedd)

        forward_out = self.criterion_module(out_score, ref_score)
        return forward_out


class StressClassificationExporter(BaseInfer):
    '''export rnnt predictor to onnx'''

    NAME = 'stress_classification'
    INPUTS = [FalconDict(name='x', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='out_socre', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, args):
        super().__init__(args)
        self.acoustic_encoder = eval(args.encoder_type)(args)
        self.linear_out = nn.Linear(args.hidden_dim, args.out_dim)
        self._inputs[0].shape[2] = args.input_dim
        self._outputs[0].shape[2] = args.out_dim

    def forward(self, x, x_mask=None):
        '''
        Forward for StressClassification module
        '''
        embedd = self.acoustic_encoder(x, x_mask)
        out_score = self.linear_out(embedd)
        return out_score

    def sample_inputs(self):
        x = self._generate_input_data(0, method='ones', dynamic_axis=[1, 100, 260])
        return x


@register_solution("StressClassificationNet")
class StressClassificationNet(BaseSolution):
    """
    Stress classification network:
    """

    def __init__(self, args):
        super().__init__()

        self.acoustic_encoder = eval(args.encoder_type)(args)
        self.linear_out = nn.Linear(args.hidden_dim, args.out_dim)
        self.criterion_module = eval(args.criterion_type)(args)
        self.args = args

    def forward(self, batch_data):  # x, x_utt, x_mask=None):
        """
        Args:
            input:
                x (Batch x Time x Channel): input features for rhythm prediction
                x_mask (Batch x Time x Channel): padding mask for input features
            output:
                out_score (Batch x classes): output posterior for stress classes
                    By default it consists of 3 score classes:
                                                                0 unstressed word
                                                                1 stressed word
                                                                2 silence
        """
        x = batch_data['src']
        x_mask = batch_data['src_mask']

        tgt_score = batch_data['ce_label']
        tgt_mask = batch_data['ce_label_mask']

        x_mask = (1 - x_mask).int().bool()
        embedd = self.acoustic_encoder(x, x_mask)
        out_score = self.linear_out(embedd)

        if self.args.speech_loss:
            silence_mask = tgt_score.ge(1)  # only speech part
            target_mask = tgt_mask.float() * silence_mask.float()
        else:
            target_mask = tgt_mask

        forward_out = self.criterion_module(out_score, x_mask, tgt_score, target_mask)
        return forward_out

    def inference(self, batch_data):  # x, x_utt, x_mask=None):
        """
        Args:
            input:
                x (Batch x Time x Channel): input features for rhythm prediction
                x_mask (Batch x Time x Channel): padding mask for input features
            output:
                out_score (Batch x classes): output posterior for stress classes
                    By default it consists of 3 score classes:
                                                                0 unstressed word
                                                                1 stressed word
                                                                2 silence
        """
        x = batch_data['src']
        x_mask = batch_data['src_mask']

        x_mask = (1 - x_mask).int().bool()

        embedd = self.acoustic_encoder(x, x_mask)
        out_score = self.linear_out(embedd)

        # out_score = F.softmax(out_score.float(), dim=-1)

        return out_score

    def evaluation(self, batch_data):  # x, x_utt, x_mask=None):
        """
        Args:
            input:
                x (Batch x Time x Channel): input features for rhythm prediction
                x_mask (Batch x Time x Channel): padding mask for input features
            output:
                out_score (Batch x classes): output posterior for stress classes
                    By default it consists of 3 score classes:
                                                                0 unstressed word
                                                                1 stressed word
                                                                2 silence
        """
        x = batch_data['src']
        x_mask = batch_data['src_mask']
        # print(x.shape)

        feat_in = x[:, :, 0 : self.args.input_dim]

        tgt_score = batch_data['ce_label']
        tgt_mask = batch_data['ce_label_mask']

        x_mask = (1 - x_mask).int().bool()
        embedd = self.acoustic_encoder(feat_in, x_mask)
        out_score = self.linear_out(embedd)

        if self.args.evaluation == 'frame':
            forward_out = self.frame_accuracy(out_score, tgt_score, tgt_mask)
        elif self.args.evaluation == 'phone':
            duration_vect = x[:, :, -2]
            forward_out = self.phone_syllable_accuracy(
                duration_vect, out_score, tgt_score, tgt_mask
            )
        elif self.args.evaluation == 'syllable':
            duration_vect = x[:, :, -1]
            forward_out = self.phone_syllable_accuracy(
                duration_vect, out_score, tgt_score, tgt_mask
            )

        return forward_out

    def frame_accuracy(self, out_score, tgt_score, tgt_mask):
        """
        calculate frame level accuracy
        """

        # frame-level
        if self.args.eval_seg == 'overall':
            target_mask = tgt_mask
        elif self.args.eval_seg == 'speech':
            silence_mask = tgt_score.ge(1)  # only speech part
            target_mask = tgt_mask.float() * silence_mask.float()
        elif self.args.eval_seg == 'silence':
            speech_mask = tgt_score.eq(0)  # only silence part
            target_mask = tgt_mask.float() * speech_mask.float()
        else:
            print('please select propoer flag')
            sys.exit()

        lprobs = F.softmax(out_score.float(), dim=-1)
        hyp_tgt = lprobs.max(dim=-1)[1]
        hyp_acc = ((hyp_tgt == tgt_score).float() * (target_mask.float())).sum() / (
            target_mask.float().sum() + EPSILON
        )

        frame_size = target_mask.float().sum()
        tgt_size = target_mask.float().sum()  # tokens

        forward_out = OrderedDict()
        forward_out['utt_num'] = out_score.shape[0]
        forward_out['acc'] = hyp_acc
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size

        return forward_out

    def phone_syllable_accuracy(self, duration_vect, out_score, tgt_score, tgt_mask):
        """
        calculate syllable level accuracy
        """
        # pylint:disable=too-many-branches

        b_size = out_score.shape[0]
        idx = 0
        correct_idx = 0.0
        for i in range(b_size):
            data_len = tgt_mask[i, :].sum()
            dur_feat = duration_vect[i, :]
            str_p = 0
            cur_dur = 0
            while str_p < data_len:
                cur_dur = int(dur_feat[str_p].item())

                if cur_dur == 0:
                    break

                cur_x = out_score[i, str_p : str_p + cur_dur, :]
                cur_y = tgt_score[i, str_p : str_p + cur_dur]

                # print('str_p: %d, end_p: %d' % (str_p, str_p+cur_dur))

                str_p += cur_dur

                lprobs = F.softmax(cur_x.float(), dim=-1)
                lprobs = torch.mean(lprobs, dim=0)
                hyp_y = lprobs.max(dim=-1)[1]

                if hyp_y == cur_y[0]:
                    if self.args.eval_seg == 'overall':
                        if cur_y[0] >= 0:
                            correct_idx += 1
                    elif self.args.eval_seg == 'speech':
                        if cur_y[0] > 0:
                            correct_idx += 1
                    elif self.args.eval_seg == 'silence':
                        if cur_y[0] == 0:
                            correct_idx += 1

                if self.args.eval_seg == 'overall':
                    if cur_y[0] >= 0:
                        idx += 1
                elif self.args.eval_seg == 'speech':
                    if cur_y[0] > 0:
                        idx += 1
                elif self.args.eval_seg == 'silence':
                    if cur_y[0] == 0:
                        idx += 1

        # print('batch: %d, total word: %d, correct word: %d' % (b_size, idx, correct_idx))
        frame_size = idx
        tgt_size = idx  # tokens
        forward_out = OrderedDict()
        forward_out['utt_num'] = b_size
        forward_out['acc'] = correct_idx / (idx + EPSILON)
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size

        return forward_out

    def register_infers(self):
        infers = [StressClassificationExporter(self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
