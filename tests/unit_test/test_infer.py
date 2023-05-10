'''
test infer.
more test should add to export onnx.
'''
import os
import sys
import copy
import numpy as np
import torch

try:
    import panther
except Exception:
    panther = None  # pylint: disable=invalid-name
from core.utils import Config
from core.solutions import setup_solution
from core.solutions.inference.infer import Factory, INFERS


def _print_panther_session_info():
    '''print panther session infer, easy for debug.'''
    if panther is None:
        return
    model_filepath = sys.argv[1]
    providers = ['CUDAExecutionProvider']
    infer_option = panther.SessionOptions()
    infer_option.intra_op_num_threads = 1
    infer_option.graph_optimization_level = panther.GraphOptimizationLevel.PTH_ENABLE_ALL
    print(model_filepath, providers, infer_option)
    sess = panther.InferenceSession(model_filepath, providers=providers, sess_options=infer_option)
    inputs = sess.get_inputs()
    print(type(inputs))
    print('all inputs', [(v.name, v.type, v.shape) for v in inputs])
    outputs = sess.get_outputs()
    print('all outputs', [(v.name, v.type, v.shape) for v in outputs])
    print(type(inputs[0].type))


def _penguin_run_encoder(encoder_inputs, backend=None):
    '''run encoder with most likely penguin code.'''
    config = None
    encoder_inference = Factory.get_inference('encoder', config)
    encoder_input_name = encoder_inference.get_input()
    encoder_output_name = encoder_inference.get_output()
    encoder_out = encoder_inference.run(encoder_inputs, backend=backend)

    assert len(encoder_inputs) >= len(encoder_input_name)
    assert len(encoder_out) >= len(encoder_output_name)
    return encoder_out


def _penguin_run_jointer(jointer_inputs, backend=None):
    '''run jointer with most likely penguin code.'''
    config = None
    jointer_inference = Factory.get_inference('jointer', config)
    jointer_input_name = jointer_inference.get_input()
    jointer_output_name = jointer_inference.get_output()
    jointer_out = jointer_inference.run(jointer_inputs, backend=backend)

    assert len(jointer_inputs) >= len(jointer_input_name)
    assert len(jointer_out) >= len(jointer_output_name)
    return jointer_out


def _penguin_run_predictor(predictor_inputs, backend=None):
    '''run predictor with most likely penguin code.'''
    config = None
    predictor_inference = Factory.get_inference('predictor', config)
    predictor_input_name = predictor_inference.get_input()
    predictor_output_name = predictor_inference.get_output()
    predictor_out = predictor_inference.run(predictor_inputs, backend=backend)

    assert len(predictor_inputs) >= len(predictor_input_name)
    assert len(predictor_out) >= len(predictor_output_name)
    return predictor_out


def _penguin_run_nnlm(nnlm_inputs, backend=None):
    '''run nnlm with most likely penguin code.'''
    config = None
    nnlm_inference = Factory.get_inference('lstm_lm', config)
    nnlm_input_name = nnlm_inference.get_input()
    nnlm_output_name = nnlm_inference.get_output()
    nnlm_out = nnlm_inference.run(nnlm_inputs, backend=backend)

    assert len(nnlm_inputs) >= len(nnlm_input_name)
    assert len(nnlm_out) >= len(nnlm_output_name)
    return nnlm_out


def test_rnnt_transformer_infer():
    '''test jointer infer.'''
    # empty torch cache for Panther BFC GPU mem cache
    torch.cuda.empty_cache()
    # setup solution, and get onnx file
    cfg = Config.fromfile('configs/asr/tel_rnnt_transfomer_80kh.py')
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    solution_cfg.setdefault('fbank_dim', 80)
    tgt_vocab_size = (
        solution_cfg.adaptive_head_size
        + solution_cfg.adaptive_tail_size * solution_cfg.adaptive_tail_groups
    )
    solution_cfg.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution_cfg.setdefault('encoder_convert_stream', False)  # Transformer is not stream

    train_cfg = cfg.train
    onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)
    solution.cuda()
    solution.register_infers()

    # get onnx file by export
    Factory.get_inference('jointer').export()

    # build input data
    bsz, jointer_hidden_size = 1, solution_cfg.get('jointer_hidden_size', -1)
    use_jointer_temperature = solution_cfg.get('use_jointer_temperature', False)
    encoder_out = torch.rand([bsz, jointer_hidden_size]).cuda()
    predictor_out = torch.rand([bsz, jointer_hidden_size]).cuda()
    if use_jointer_temperature:
        temperature = torch.ones([1]).cuda()
        jointer_inputs = (encoder_out, predictor_out, temperature)
    else:
        jointer_inputs = (encoder_out, predictor_out)

    # run penguin code with panther and torch backend
    panther_out = _penguin_run_jointer(jointer_inputs)
    torch_out = _penguin_run_jointer(jointer_inputs, backend='torch')

    # compare result
    for name in panther_out.keys():
        result1 = panther_out[name]
        result2 = torch_out[name].detach().cpu().numpy()
        # we also can use panther_out.output
        assert result1.shape == result2.shape, name
        assert np.allclose(result1, result2, atol=1e-5), name

    INFERS.clear()


def test_rnnt_lstm_infer():
    '''test encoder infer.'''
    torch.manual_seed(20220211)
    torch.cuda.manual_seed(145058)

    # setup solution, and get onnx file
    config = 'configs/asr/child_zh_time_reduce_online_prosody_baseline.py'
    cfg = Config.fromfile(config)
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    solution_cfg.setdefault('fbank_dim', 80)
    solution_cfg['onnx_stack_frame'] = 320
    solution_cfg['onnx_with_mask'] = False
    solution_cfg['backbone_mask'] = False
    tgt_vocab_size = (
        solution_cfg.adaptive_head_size
        + solution_cfg.adaptive_tail_size * solution_cfg.adaptive_tail_groups
    )
    solution_cfg.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution_cfg.setdefault('encoder_convert_stream', True)
    solution_cfg.setdefault('predictor_convert_stream', True)

    train_cfg = cfg.train
    onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)
    solution.cuda()
    solution.register_infers()

    # get onnx file by export
    Factory.get_inference('encoder').export()
    Factory.get_inference('predictor').export()
    # Factory.get_inference('jointer').export()

    # build input data for encoder
    bsz, seg_frames, onnx_stack_frame = 1, 20, solution_cfg.get('onnx_stack_frame', -1)
    unfold_fbank = torch.rand([bsz, seg_frames, onnx_stack_frame]).cuda()
    # in torch,
    lstm_layers = 8
    lstm_hidden = solution_cfg.backbone_hidden_size
    global_state_in = torch.rand([bsz, lstm_hidden * 2 * lstm_layers]).cuda()
    x_sign = torch.zeros([1], dtype=torch.int32).cuda()
    encoder_inputs = (unfold_fbank, global_state_in, x_sign)
    prev_char = torch.randint(low=0, high=tgt_vocab_size, size=[bsz]).cuda()
    predictor_state_size = (
        solution_cfg.predictor_lstm_hidden_size * 2 * solution_cfg.predictor_lstm_layer_num
    )
    predictor_state_in = torch.rand([bsz, predictor_state_size]).cuda()
    predictor_inputs = (prev_char, predictor_state_in)

    torch_out = _penguin_run_predictor(predictor_inputs, backend='torch')
    panther_out = _penguin_run_predictor(predictor_inputs)

    # compare result
    for name in panther_out.keys():
        result1 = panther_out[name]
        result2 = torch_out[name].detach().cpu().numpy()
        # we also can use panther_out.output
        assert result1.shape == result2.shape, name
        assert np.allclose(result1, result2, atol=1e-5), name

    # run penguin code with panther and torch backend
    torch_out = _penguin_run_encoder(encoder_inputs, backend='torch')
    panther_out = _penguin_run_encoder(encoder_inputs)

    # compare result
    for name in panther_out.keys():
        result1 = panther_out[name]
        result2 = torch_out[name].detach().cpu().numpy()
        # we also can use panther_out.output
        assert result1.shape == result2.shape, name
        assert np.allclose(result1, result2, atol=1e-5), name

    INFERS.clear()


def test_nnlm_infer():
    '''test nnlm infer.'''
    torch.manual_seed(20220211)
    torch.cuda.manual_seed(145058)

    # setup solution, and get onnx file
    config = 'configs/lm/lm_tomato_vs_lstm.py'
    cfg = Config.fromfile(config)
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    tgt_vocab_size = 12963
    solution_cfg.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution_cfg.setdefault('nnlm_convert_stream', True)

    train_cfg = cfg.train
    onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)
    solution.cuda()
    solution.register_infers()

    # get onnx file by export
    Factory.get_inference('lstm_lm').export()

    # build input data for encoder
    nnlm_state_size = solution_cfg.lstm_cell_size * 2 * solution_cfg.lstm_num_layers
    bsz = 1
    prev_char = torch.randint(low=0, high=tgt_vocab_size, size=[bsz]).cuda()
    nnlm_state_in = torch.rand([bsz, nnlm_state_size]).cuda()
    nnlm_inputs = (prev_char, nnlm_state_in)

    panther_out = _penguin_run_nnlm(nnlm_inputs)
    torch_out = _penguin_run_nnlm(nnlm_inputs, backend='torch')

    # compare result
    for name in panther_out.keys():
        result1 = panther_out[name]
        result2 = torch_out[name].detach().cpu().numpy()
        # we also can use panther_out.output
        assert result1.shape == result2.shape, name
        assert np.allclose(result1, result2, atol=1e-3), name

    INFERS.clear()


def test_rnnt_dfsmn_infer():
    '''test encoder infer.'''
    torch.manual_seed(20220211)
    torch.cuda.manual_seed(145058)

    # setup solution, and get onnx file
    config = 'configs/asr/dfsmn_rnnt.py'
    cfg = Config.fromfile(config)
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    solution_cfg.setdefault('fbank_dim', 80)
    solution_cfg['onnx_stack_frame'] = 80
    solution_cfg['onnx_with_mask'] = False
    solution_cfg['backbone_mask'] = False
    tgt_vocab_size = (
        solution_cfg.adaptive_head_size
        + solution_cfg.adaptive_tail_size * solution_cfg.adaptive_tail_groups
    )
    solution_cfg.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution_cfg.setdefault('encoder_convert_stream', True)
    solution_cfg.setdefault('predictor_convert_stream', True)

    train_cfg = cfg.train
    onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)

    solution.cuda()
    solution.register_infers()

    # get onnx file by export
    Factory.get_inference('encoder').export()

    # build input data for encoder
    bsz, total_len, onnx_stack_frame = 4, 500, solution_cfg.get('onnx_stack_frame', -1)
    fbank = torch.rand([bsz, total_len, onnx_stack_frame]).cuda()

    # build encoder
    encoder_infer = Factory.get_inference('encoder', None)
    global_state_size = (
        encoder_infer.acoustic_front_end_module.state_size
        + encoder_infer.acoustic_backbone_module.state_size
    )

    ############## nonstream test ##############
    # torch backend, forward
    torch_output_nonstream = encoder_infer.forward(fbank)

    # panther backend
    x_sign = torch.ones([1], dtype=torch.int32).cuda() * 3  # nonstream mode
    global_state_in = torch.zeros([bsz, global_state_size]).cuda()
    encoder_inputs = (fbank, global_state_in, x_sign)
    panther_out = _penguin_run_encoder(encoder_inputs)
    panther_output_nonstream = torch.Tensor(panther_out['output']).cuda()

    assert torch.allclose(panther_output_nonstream, torch_output_nonstream, atol=1e-5)
    # torch backend, forward step x_sign 3
    torch_output_nonstream, _ = encoder_infer.forward_step(
        fbank, fbank_mask=None, global_state_in=global_state_in, x_sign=x_sign
    )
    # check nonstream result
    assert torch.allclose(panther_output_nonstream, torch_output_nonstream, atol=1e-5)

    ############## stream test    ##############
    global_state_in = torch.zeros([bsz, global_state_size]).cuda()

    stream_first_len = 128
    stream_mid_len = 20
    panther_output_stream_list = []
    torch_output_stream_list = []
    # first
    x_sign = torch.ones([1], dtype=torch.int32).cuda()
    panther_encoder_inputs = (fbank[:, :stream_first_len, :], global_state_in.clone(), x_sign)
    panther_out = _penguin_run_encoder(panther_encoder_inputs)
    torch_encoder_inputs = (fbank[:, :stream_first_len, :], global_state_in.clone(), x_sign)
    torch_out = _penguin_run_encoder(torch_encoder_inputs, backend='torch')
    panther_output_stream_list.append(panther_out['output'])
    torch_output_stream_list.append(torch_out['output'])
    # mid
    for cur_t in range(stream_first_len, total_len - stream_mid_len, stream_mid_len):
        x_sign = torch.zeros([1], dtype=torch.int32).cuda()
        panther_encoder_inputs = (
            fbank[:, cur_t : cur_t + stream_mid_len, :],
            panther_out['global_state_out'],
            x_sign,
        )
        panther_out = _penguin_run_encoder(panther_encoder_inputs)
        torch_encoder_inputs = (
            fbank[:, cur_t : cur_t + stream_mid_len, :],
            torch_out['global_state_out'],
            x_sign,
        )
        torch_out = _penguin_run_encoder(torch_encoder_inputs, backend='torch')
        panther_output_stream_list.append(panther_out['output'])
        torch_output_stream_list.append(torch_out['output'])
    # last
    x_sign = torch.ones([1], dtype=torch.int32).cuda() * 2
    panther_encoder_inputs = (
        fbank[:, cur_t + stream_mid_len :, :],
        panther_out['global_state_out'],
        x_sign,
    )
    panther_out = _penguin_run_encoder(panther_encoder_inputs)
    torch_encoder_inputs = (
        fbank[:, cur_t + stream_mid_len :, :],
        torch_out['global_state_out'],
        x_sign,
    )
    torch_out = _penguin_run_encoder(torch_encoder_inputs, backend='torch')
    panther_output_stream_list.append(panther_out['output'])
    torch_output_stream_list.append(torch_out['output'])
    # panther_output_stream = torch.cat(panther_output_stream_list, dim=1)
    _panther_output_stream = torch.Tensor(np.concatenate(panther_output_stream_list, axis=1)).cuda()
    torch_output_stream = torch.cat(torch_output_stream_list, dim=1)

    # TODO(liancaijiang.1212): fix stream dfsmn panther output
    # assert torch.allclose(panther_output_stream, torch_output_stream, atol=1e-5)
    assert torch.allclose(panther_output_nonstream, torch_output_stream, atol=1e-5)
    INFERS.clear()


if __name__ == '__main__':
    # _print_panther_session_info()
    test_rnnt_lstm_infer()
    test_rnnt_dfsmn_infer()
