''' WavLMSidModel '''
import torch
from core.models.sid.frame_level_net import *
from core.models.sid.segment_level_net import *
from core.models.sid.pooling_net import *
from core.models.pretrained.wavlm_model import *
from core.criterions import *
from core.solutions.base_solution import register_solution
from core.solutions.sid.base_sid_solution import BaseSidModel


@register_solution("WavLMSidModel")
class WavLMSidModel(BaseSidModel):
    '''
    Sid Model using WavLM
    '''

    def __init__(self, args):
        '''
        The input feature is expected to be waveform with shape [B, L]
        '''
        super().__init__(args)

        self.feature_extract = PretrainedWavLM(args)
        self.total_feat_num = args.encoder_layers + 1
        self.feat_num = args.get("wavlm_feat_num", self.total_feat_num)
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(args.encoder_embed_dim)
        self._register_load_state_dict_pre_hook(self._model_load_hook)

    def _forward_impl(self, feature, mask=None, label=None, step=0):
        '''The basic forward implementation'''
        if self.args.freeze_upstream:
            with torch.no_grad():
                res_dic = self.feature_extract.forward(
                    batch_data={"waveform": feature, "src_mask": mask},
                    mask=self.args.apply_mask and self.training,
                )
        else:
            res_dic = self.feature_extract.forward(
                batch_data={"waveform": feature, "src_mask": mask},
                mask=self.args.apply_mask and self.training,
            )
        # Use the first (N-1) elements as the input of each encoder layer.
        # The last element is replaced by the final output of the encoder.
        for index, res in enumerate(res_dic['layer_results'][1:-1]):
            res_dic['layer_results'][index + 1] = res[0] if isinstance(res, tuple) else res
        res_dic['layer_results'][-1] = res_dic['x']
        x = torch.stack(res_dic['layer_results'], dim=0)
        norm_weights = (
            F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6
        feature = self.instance_norm(x)
        if res_dic['padding_mask'] is not None:
            backbone_mask = 1 - res_dic['padding_mask'].float()
        else:
            backbone_mask = None

        frame_level_feature, backbone_mask = self.acoustic_backbone_module.forward(
            feature, backbone_mask
        )
        segment_level_feature, embedding = self.pooling_module(frame_level_feature, backbone_mask)
        logits, embedding_utt, meta = self.segment_network_module.forward(
            segment_level_feature, label, step=step
        )
        embedding.update(embedding_utt)
        return logits, embedding, meta

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
        """Transform unispeech-chkpt to dolphin-chkpt"""
        assert isinstance(self.feature_extract, PretrainedWavLM)
        old_state_dict = state_dict.copy()
        state_dict.clear()

        for name, param in old_state_dict.items():
            if name == 'mask_emb':
                # the feature mask
                new_name = 'feature_extract.mask_emb'
            elif name.endswith("encoder.layers.0.self_attn.relative_attention_bias.weight"):
                # the positional embedding
                new_name = 'feature_extract.pos_enc.weight'
            elif (
                name.startswith("layer_norm.")
                or name.startswith("feature_extractor.")
                or name.startswith("post_extract_proj.")
                or name.startswith("encoder.")
            ):
                # loading from original wavlm
                new_name = 'feature_extract.{}'.format(name)
            elif name.startswith("feature_extract.model."):
                # loading from finetuned wavlm
                new_name = name.replace("feature_extract.model.", 'feature_extract.')
            else:
                new_name = name
            state_dict[new_name] = param
