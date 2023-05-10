''' test panther symbol in onnx. '''
import sys
import torch
import onnx

from core.utils import FalconDict
from core.models.layers.conformer import ConformerLayer
from core.solutions.inference import BaseInfer


class _ConformerExporter(BaseInfer):
    '''Conformer exporter test.'''

    NAME = 'test_conformer'
    INPUTS = [
        FalconDict(name='x', type=torch.float32, shape=[1, 128, 512]),
        FalconDict(name='pos_emb', type=torch.float32, shape=[1, 255, 512]),
    ]
    OUTPUTS = [
        FalconDict(name='y', type=torch.float32, shape=[1, 128, 512]),
    ]

    def __init__(self, m, **kwargs):
        '''init.'''
        super().__init__(kwargs)
        self.model = m

    def forward(self, x, pos_emb, **_kwargs):
        '''forward.'''
        (y, _), _ = self.model([(x, pos_emb), None])
        return y

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        x = self._generate_input_data(0)
        pos_emb = self._generate_input_data(1)
        return x, pos_emb


def test_conformer():
    torch.cuda.empty_cache()
    args = FalconDict(
        backbone_memory_size=512,
        conformer_normalize_before=1,
        conformer_attention_heads=8,
        conformer_mask_topology='[[160, 160]]*14',
        conformer_linear_units=2048,
        conformer_num_blocks=14,
        conformer_dropout_rate=0.1,
        conformer_positional_dropout_rate=0.1,
        conformer_attention_dropout_rate=0.0,
        conformer_positionwise_layer_type='linear',
        conformer_activation_fn='gelu',
        conformer_positionwise_conv_kernel_size=1,
        conformer_macaron_style=1,
        conformer_pos_enc_layer_type='fix_rel_pos',
        conformer_selfattention_layer_type='rel_selfattn',
        conformer_use_cnn_module=1,
        conformer_cnn_module='ConvolutionModule',
        conformer_cnn_module_kernel='15,15',
        conformer_layernorm_interval=0,
        conformer_weight_scale=1.0,
        conformer_half_pooling=0,
        onnx_dir='.',
        onnx_graph_optimization=True,
    )
    conformer_mask_topology = [160, 160]

    module = ConformerLayer(args, conformer_mask_topology).cuda()
    exporter = _ConformerExporter(module, **args)
    exporter.export()


if __name__ == '__main__':
    model = onnx.load(sys.argv[1])
    # onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))  # pylint: disable=no-member
