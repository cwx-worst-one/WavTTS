'''
extension modules for dolhin.
'''
try:
    from panther.custom_ops.torch import (
        add_noise,
        kaldi_add_rir,
        time_mask,
        freq_mask,
    )
    from panther.custom_ops.torch import (
        clip_grads_,
        edit_distance,
        ShallowJointFunction,
        ctc_loss,
        dynamic_cmvn,
        beam_search_unique_roads,
        rnnt_loss,
        rnnt_force_alignment,
        fused_ffn,
        cif_calculator,
        fused_conformer_convolution,
        fused_mem_mask_rel_attn,
        fused_prev_char,
        fused_bmuf,
        gather_fn,
        clear_cuda_error,
        fused_layernorm,
        fused_transformer,
        fused_multihead_attn,
        fused_adaptive_softmax,
        LSTMFunction,
        fused_dfsmn,
        ext_ctc_force_alignment,
        PantherLogFbank,
        multi_heads_chunk_wise_attention,
        TpTransformerFunc,
        conv1d,
        conv2d,
        fused_conv_feature_extraction,
    )
    from panther.torch.amp import (
        init as amp_init,
        master_params,
        get_amp_level,
        AmpHandler,
        AmpEnable,
    )
    from panther.profiler import FlopsProfiler
    from panther.torch.grad_acc import GradientsAccumulator
    from panther.torch import checkpoint_wrapper
    from panther.torch import auto_parallel as dist_parallel
    from panther.torch.largescale import (
        PipelineModule,
        LayerSpec,
        TiedLayerSpec,
        mpu,
        initialize_topology,
    )
    import panther.custom_ops.torch as pnn
except Exception as e:
    raise Exception("Panther not be installed correctly!") from e

# TODO(zhengyijie): The zero_grad function call will be uniformly modified
# instead of rewriting the zero_grad function.
# This function could be removed in the torch2.0 version.
# But now, we need to keep it for compatibility with torch1.8.1 version.
# Because the zero_grad function in apex optimizer is not uniform.
def zero_grad_(self, **_kwargs):
    '''
    do zero grads inplace.
    '''
    for group in self.param_groups:
        for p in group['params']:
            p.grad = None


__all__ = [
    'clip_grads_',
    'ext_ctc_force_alignment',
    'edit_distance',
    'zero_grad_',
    'ShallowJointFunction',
    'LSTMFunction',
    'fused_dfsmn',
    'ctc_loss',
    'fused_bmuf',
    'gather_fn',
    'get_amp_level',
    'master_params',
    'amp_init',
    'clear_cuda_error',
    'ctc_loss',
    'rnnt_loss',
    'rnnt_force_alignment',
    'beam_search_unique_roads',
    'dynamic_cmvn',
    'fused_prev_char',
    'AmpHandler',
    'AmpEnable',
    'checkpoint_wrapper',
    'PipelineModule',
    'LayerSpec',
    'TiedLayerSpec',
    'dist_parallel',
    'mpu',
    'initialize_topology',
    'fused_layernorm',
    'fused_transformer',
    'FlopsProfiler',
    'GradientsAccumulator',
    'add_noise',
    'kaldi_add_rir',
    'time_mask',
    'freq_mask',
    'fused_ffn',
    'multi_heads_chunk_wise_attention',
    'cif_calculator',
    'fused_conformer_convolution',
    'fused_mem_mask_rel_attn',
    'fused_multihead_attn',
    'TpTransformerFunc',
    'fused_conv_feature_extraction',
    'pnn',
]
