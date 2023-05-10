''' BaseSidModel and BaseSidSolution '''
import torch
from core.models.sid.frame_level_net import *
from core.models.sid.segment_level_net import *
from core.models.sid.pooling_net import *
from core.criterions import *
from core.utils import FalconDict
from core.solutions.base_solution import BaseSolution, register_solution
from core.solutions.inference import (
    BaseInfer,
    INFERS,
)


class SidExporter(BaseInfer):
    '''
    Exporter for SID model.
    '''

    NAME = 'sid'
    INPUTS = [
        FalconDict(name='feature', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = []

    def __init__(self, model, transpose, **kwargs):
        '''init'''
        super().__init__(kwargs)
        self.model = model
        self.transpose = transpose
        for node in self.model.embedding_nodes:
            self._outputs.append(FalconDict(name=node, type=torch.float32, shape=['B', -1]))
        fbank_dim = kwargs.get('fbank_dim')
        if self.transpose:
            self._inputs[0].shape[2] = fbank_dim
        else:
            self._inputs[0].shape = ['B', fbank_dim, 'T']
        self._use_mask = kwargs.get('use_mask', False)
        if self._use_mask:
            self._inputs.append(FalconDict(name='mask', type=torch.float32, shape=['B', 'T']))
        self.eval()

    def forward(self, feature, mask=None):
        '''The basic forward implementation'''
        if self.transpose:
            feature = feature.transpose(1, 2)
        frame_level_feature, backbone_mask = self.model.acoustic_backbone_module.forward(
            feature, mask
        )
        segment_level_feature, embedding = self.model.pooling_module(
            frame_level_feature, backbone_mask
        )
        embedding_utt = self.model.segment_network_module.forward(
            segment_level_feature, None, step=0, export_onnx=True
        )
        embedding.update(embedding_utt)
        return [embedding[node] for node in self.model.embedding_nodes]

    def sample_inputs(self):
        if self.transpose:
            feature = self._generate_input_data(0, dynamic_axis=[1, 100, -1])
        else:
            feature = self._generate_input_data(0, dynamic_axis=[1, -1, 100])
        if self._use_mask:
            mask = self._generate_input_data(1, method='zeros', dynamic_axis=[1, 100])
            return feature, mask
        return feature


class SidFrmExporter(SidExporter):
    '''Exporter for the frame-level network.'''

    NAME = 'sid_frm'
    INPUTS = [
        FalconDict(name='feature', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = [FalconDict(name='frm_embed', type=torch.float32, shape=['B', -1, 'T'])]

    def __init__(self, model, transpose, **kwargs):
        super().__init__(model, transpose, **kwargs)
        self._outputs = self.OUTPUTS

    def forward(self, feature, mask=None):
        '''forward'''
        if self.transpose:
            feature = feature.transpose(1, 2)
        frame_level_feature, _ = self.model.acoustic_backbone_module.forward(feature, mask)
        if len(frame_level_feature.shape) == 4:
            bs, chan, feat_dim, length = frame_level_feature.shape
            frame_level_feature = frame_level_feature.view(bs, chan * feat_dim, length)
        return frame_level_feature


class SidUttExporter(SidExporter):
    '''Exporter for the utterance-level network.'''

    NAME = 'sid_utt'
    INPUTS = [
        FalconDict(name='frm_embed', type=torch.float32, shape=['B', -1, 'T']),
    ]
    OUTPUTS = []

    def __init__(self, model, **kwargs):
        super().__init__(model, transpose=None, **kwargs)
        self._inputs[0].shape[1] = self.model.acoustic_backbone_module.output_dim

    def forward(self, frm_embed, mask=None):
        '''forward'''
        segment_level_feature, embedding = self.model.pooling_module(frm_embed, mask)
        embedding_utt = self.model.segment_network_module.forward(
            segment_level_feature, None, step=0, export_onnx=True
        )
        embedding.update(embedding_utt)
        return [embedding[node] for node in self.model.embedding_nodes]

    def sample_inputs(self):
        feature = self._generate_input_data(0, dynamic_axis=[1, -1, 100])
        if self._use_mask:
            mask = self._generate_input_data(1, method='zeros', dynamic_axis=[1, 100])
            return feature, mask
        return feature


@register_solution("BaseSidModel")
class BaseSidModel(BaseSolution):
    '''
    Base model for SID/LID.
    - frame-level network
    - pooling
    - segment-level network + different logits
    - criterion
    '''

    def __init__(self, args):
        '''
        The input feature is expected to be [B, L, D], in which L is the length.
        '''
        super().__init__()
        self.args = args
        self._embedding_nodes = None

        # The output of the backbone is the feature-level features (aka. speaker features).
        # The outputs of different layers in the frame-level network are not recorded
        # since they may be too large if we have many hidden layers.
        self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)

        # The output of the pooling module is the pooled feature (aka. utterace-level feature).
        # The internal outputs of the pooling layer are saved in a dict and we can
        # use them later (e.g. the stddev of the stat pooling).
        self.pooling_module = eval(args.pooling_layer_type)(
            args, self.acoustic_backbone_module.output_dim
        )

        # The output of the segment-level network is the output of the last layer.
        # The outputs of different layers are saved in a dict as well. These outputs
        # can be used as the speaker embedding during the inference.
        self.segment_network_module = eval(args.segment_network_type)(
            args, self.pooling_module.output_dim
        )

        if not self.args.is_inference:
            self.criterion_module = eval(args.criterion_type)(args)

    def _forward_impl(self, feature, mask=None, label=None, step=0):
        '''The basic forward implementation'''
        # Most network architectures need [B, D, L], so we transpose the input features
        # in the first place.
        feature = feature.transpose(1, 2)

        # frame_level_feature is the extracted 'speaker feature'.
        # Aggregation or other processes has been done if required.
        frame_level_feature, backbone_mask = self.acoustic_backbone_module.forward(feature, mask)

        # Here, embedding is a dict that we can choose the node that we are interested.
        segment_level_feature, embedding = self.pooling_module(frame_level_feature, backbone_mask)

        # the output of the hidden layer in the segment-level network is the speaker
        # embedding
        logits, embedding_utt, meta = self.segment_network_module.forward(
            segment_level_feature, label, step=step
        )
        embedding.update(embedding_utt)
        return logits, embedding, meta

    def forward(
        self, batch_data, feature_key='feature', mask_key='mask', label_key='label', step=0
    ):
        '''
        Forward for SID base model.
        Args:
            batch_data: the batched data. It requires the following contents:
                feature: the input feature with shape [B, L, D]
                feature_mask: the input mask (indicating which frames are valid)
                              with shape [B, L]
                label: the label for each utterance. shape [B]
            step: the number of traininig step.
        '''
        feature = batch_data[feature_key]
        feature_mask = batch_data.get(mask_key, None)
        label = batch_data.get(label_key, None)
        logits, _, meta = self._forward_impl(feature, feature_mask, label=label, step=step)

        if not self.args.is_inference:
            # The criterion always needs three inputs:
            # logits: [B, L, N], labels: [B, L] and masks: [B, L]
            # where L=1, since we are an utterance-based task rather than frame-based
            target_mask = feature_mask.sum(1, keepdim=True) > 0
            if isinstance(self.criterion_module, Xentropy):
                forward_out = self.criterion_module(
                    logits.unsqueeze(1), feature_mask, label.unsqueeze(1), target_mask
                )
            else:
                raise ValueError("Unsupported criterion.")

        # Get other metric information
        forward_out.update(meta)
        return forward_out

    @torch.no_grad()
    def inference(self, feature, mask=None):
        '''
        Extract the embedding (for verification) or posterior (for identification)
        Unlike training or validation, the inference does not need the label.
        We do not use batch_data here, since this is more convenient for model export.

        Args:
            feature: the input feature with shape [B, T, D]
            mask: the mask with shape [B, T]
        '''
        is_training = self.training
        if is_training:
            self.eval()
        if self._embedding_nodes is None:
            raise ValueError("The embedding nodes should be intialized before inference.")
        _, embedding, _ = self._forward_impl(feature, mask)
        if is_training:
            self.train()
        return [embedding[node] for node in self._embedding_nodes]

    @property
    def embedding_nodes(self):
        '''return current embedding nodes'''
        return self._embedding_nodes

    @embedding_nodes.setter
    def embedding_nodes(self, nodes):
        '''set the embedding nodes'''
        if not isinstance(nodes, list):
            raise ValueError("The model only accept a list as the nodes.")
        self._embedding_nodes = nodes

    def register_infers(self):
        transpose = self.args.get('transpose', False)
        export_nodes = self.args.get('export_nodes', 'utt_output')
        self.embedding_nodes = export_nodes.split(';')
        infers = [
            SidExporter(self, transpose, **self.args),
            SidFrmExporter(self, transpose, **self.args),
            SidUttExporter(self, **self.args),
        ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    @staticmethod
    def generate_fake_batch_data():
        '''generate dummy input for slim model trace'''
        feature = torch.ones([16, 320, 64], dtype=torch.float32)
        mask = torch.ones([16, 320], dtype=torch.float32)
        label = torch.ones([16], dtype=torch.int64)
        batch_data = {'feature': feature, 'label': label, 'mask': mask}
        return batch_data
