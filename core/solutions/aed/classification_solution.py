'''Aed classification solution'''
import torch
from core.solutions.base_solution import BaseSolution, register_solution
from core.models.layers.vgg import *
from core.models.aed.aed_transformer import *
from core.criterions.criterion import *
from core.solutions.inference import BaseInfer, INFERS
from core.utils import FalconDict


class AedClassificationExporter(BaseInfer):
    '''AedClassificationExporter'''

    NAME = 'AedClassification'
    INPUTS = [
        FalconDict(name='features', type=torch.float32, shape=['B', 'T', 'H']),
    ]
    OUTPUTS = [
        FalconDict(name='predicts', type=torch.float32, shape=['B', 'T', 'C']),
    ]

    def __init__(self, classifier_module, **kwargs):
        '''init'''
        super().__init__(kwargs)
        self.classifier_module = classifier_module
        self.fbank_dim = self._cfg.get('fbank_dim')

    @torch.no_grad()
    def forward(self, features):
        '''forward'''
        feat_shape = features.shape
        model_inp = features.reshape(feat_shape[0] * feat_shape[1], 1, -1, self.fbank_dim)
        _, predicts = self.classifier_module(model_inp)
        return predicts.reshape(feat_shape[0], feat_shape[1], -1)

    def sample_inputs(self):
        '''sample_inputs'''
        return self._generate_input_data(0, dynamic_axis=[1, 256, 100 * self.fbank_dim])


@register_solution("AedClassificationModel")
class AedClassificationModel(BaseSolution):
    '''AED Classification Model'''

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.args = args
        self.classifier = eval(args.classifier)(args)
        self.criterion = eval(args.criterion_type)(args)

    def weights_regularization_losses(self):
        '''l2 regularization losses for weights of conv2d and linear'''
        weight_decay = self.args.get('weight_decay', 0.0)
        if weight_decay <= 0.0:
            return 0.0

        losses = 0.0
        count = 0
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                losses += torch.sum(torch.square(module.weight))
                count += 1
        return weight_decay * losses / count

    def forward(self, batch_data):
        '''forward AED Classification Model'''
        features = batch_data['features']
        labels = batch_data.get('labels', None)
        onehot_labels = batch_data.get('onehot_labels', None)

        logits, predicts = self.classifier(features.unsqueeze(1))
        solution_out = {
            'frame_size': features.shape[0] * features.shape[1],
            'tgt_size': features.shape[0],
            'predicts': predicts,
        }
        if labels is None and onehot_labels is None:
            return solution_out
        # get top k accuracy
        max_labels, _ = torch.max(onehot_labels, dim=-1, keepdim=True)
        label_thresh = torch.full_like(max_labels, self.args.get('label_thresh', 1.0))
        label_thresh = torch.minimum(max_labels, label_thresh)
        label_flags = torch.ge(onehot_labels, label_thresh)
        label_flags = label_flags.reshape(-1, 1, self.args.num_classes)
        for k in [1, self.args.topk]:
            _, top_indices = torch.topk(predicts, k, dim=-1)
            top_labels = F.one_hot(top_indices, self.args.num_classes)
            top_labels = top_labels.reshape(-1, k, self.args.num_classes)
            hits = torch.logical_and(label_flags, top_labels)
            hits = torch.any(hits.reshape(-1, k * self.args.num_classes), dim=-1)
            topk = hits.float().sum()
            if k == 1:
                solution_out['top1'] = topk
            else:
                solution_out['topk'] = topk
        # get losses
        ce_loss = self.criterion(logits, labels, onehot_labels)
        l2_loss = self.weights_regularization_losses()
        loss = ce_loss + l2_loss
        solution_out.update(
            {
                'ce_loss': ce_loss,
                'l2_loss': l2_loss,
                'loss': loss,
                'backward_loss': loss,
            }
        )
        return solution_out

    def register_infers(self):
        '''register infer object for export'''
        infers = [AedClassificationExporter(self.classifier, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    @staticmethod
    def generate_fake_batch_data():
        '''generate dummy input for slim model trace'''
        feature = torch.ones([2, 99, 64], dtype=torch.float32)
        batch_data = {'features': feature}
        return batch_data
