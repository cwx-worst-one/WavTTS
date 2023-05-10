'''export onnx'''
import copy
import os
import os.path as osp
import pickle
from packaging import version

import numpy as np
import torch
from torch import nn

try:
    # pylint: disable=import-error, no-name-in-module
    import onnx
    import panther

    if version.parse(panther.__version__) > version.parse('1.6.4'):
        import panther.stream_converter2 as cv
    elif version.parse(panther.__version__) == version.parse('1.6.4'):
        import panther.stream_converter as cv
    elif version.parse(panther.__version__) > version.parse('0.4.2'):
        import panther.stream_converter.converter as cv
    else:
        import panther.slim.converter as cv
    from panther.model_converter import onnx2panther
except Exception:
    # Panther is installed as default.
    # An exception will be raised if panther is used but not available.
    pass
try:
    from byteslim.quant.post_quant import QuantizeConfig, create_post_training_quantizer
except ImportError:
    # To use byteslim, include byteslim scm at task building stage.
    # For more detail, see dolphin tutorial.
    # An exception will be raised if byteslim is used.
    pass
try:
    from panther_utilities.torch_symbolic import module_map, replace_module_with_panther_op
except ImportError:
    module_map, replace_module_with_panther_op = None, None


from core.utils import get_rank, get_local_rank, logging, mkdir_or_exist, FalconDict


def _flatten_list(data):
    '''Recursively flat a list.'''
    if isinstance(data, (np.ndarray, torch.Tensor)):
        return [data]
    if isinstance(data, (list, tuple)):
        out = []
        for d in data:
            out.extend(_flatten_list(d))
        return out
    return []


def concat_global_states(*states, dim=1, reshape=False, bsz=0):
    '''Recursively concat all state tensor.'''
    states = _flatten_list(states)
    if reshape:
        states = [v.reshape(bsz, -1) for v in states]
    return torch.cat(states, dim=dim)


def convert_to_np(data):
    '''Recursively convert all tensor to numpy for ONNX run.'''
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        return data
    if isinstance(data, (list, tuple)):
        return [convert_to_np(item) for item in data]
    if isinstance(data, dict):
        return {k: convert_to_np(v) for k, v in data.items()}
    return data


def convert_to_tensor(data, use_gpu=True):
    '''Recursively convert all numpy array to tensor for PyTorch run.'''
    if isinstance(data, torch.Tensor):
        return data.cuda() if use_gpu else data.cpu()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        return data.cuda() if use_gpu else data.cpu()
    if isinstance(data, (list, tuple)):
        return [convert_to_tensor(item, use_gpu) for item in data]
    if isinstance(data, dict):
        return {k: convert_to_tensor(v, use_gpu) for k, v in data.items()}
    return data


def to_panther_dtype_str(dtype):
    '''convert torch data type to panther data type str.'''
    # pylint: disable=too-many-return-statements
    if dtype == torch.float32:
        return "tensor(float)"
    if dtype == torch.uint8:
        return "tensor(uint8)"
    if dtype == torch.int8:
        return "tensor(int8)"
    if dtype == torch.int16:
        return "tensor(int16)"
    if dtype == torch.int32:
        return "tensor(int32)"
    if dtype == torch.int64:
        return "tensor(int64)"
    if dtype == torch.bool:
        return "tensor(bool)"
    if dtype == torch.float16:
        return "tensor(float16)"
    if dtype == torch.float64:
        return "tensor(double)"
    if dtype == torch.bfloat16:
        return "tensor(bfloat16)"
    return dtype


def is_tensor_meta_align(torch_tensors, panther_nodes):
    '''
    check tensor info if aligned with panther session info.
    '''
    # panther may add some state node when streaming model.
    assert len(torch_tensors) <= len(panther_nodes)

    for torch_tensor in torch_tensors:
        panther_node = None
        for node in panther_nodes:
            if node.name == torch_tensor.name:
                panther_node = node
                break

        if panther_node is None:
            return False
        if len(panther_node.shape) != len(torch_tensor.shape):
            return False
        for a, b in zip(torch_tensor.shape, panther_node.shape):
            if a != b and a != -1 and not isinstance(a, str):
                return False
        if panther_node.type != to_panther_dtype_str(torch_tensor.type):
            return False
    return True


def _meta_str(nodes):
    '''convert node info to string.'''
    return str([(node.name, node.type, node.shape) for node in nodes])


def parse_shape(torch_shape, pth_shape):
    '''parse shape to list'''
    shape = []
    for i, d in enumerate(torch_shape):
        if isinstance(d, str):
            if d.upper() == 'B' or d.upper() == 'T':
                shape.append(d.upper())
            elif isinstance(pth_shape[i], int):
                shape.append(pth_shape[i])
            else:
                shape.append(1)
        else:
            if isinstance(pth_shape[i], int):
                shape.append(pth_shape[i])
            else:
                shape.append(max(1, d))
    return shape


class BaseInfer(nn.Module):
    '''
    Base Model class for inference.
    1. support different backend, torch or panther.
    2. integrate export and test into this Base class.
    '''

    # pylint: disable=invalid-name

    NAME = ''
    INPUTS = []
    OUTPUTS = []
    ORIGIN_TEST_ATOL = 1e-4
    ORIGIN_TEST_RTOL = 1e-4
    ORIGIN_TEST_MSE = 1e-4
    OPTIMIZED_TEST_ATOL = 1e-3
    OPTIMIZED_TEST_RTOL = 1e-3
    OPTIMIZED_TEST_MSE = 1e-3

    def __init__(self, cfg):
        '''
        init
        '''
        super().__init__()

        self._cfg = copy.deepcopy(cfg)
        self._backend = cfg.get('backend', 'torch')
        self._onnx_dir = cfg.get('onnx_dir', './')
        self._onnx_file = osp.join(self._onnx_dir, '{}.onnx'.format(self.NAME))
        self._onnx_files = []
        self._providers = (
            ['CUDAExecutionProvider']
            if cfg.get('panther_use_gpu', True)
            else ['CPUExecutionProvider']
        )
        self._inputs = self.INPUTS.copy()
        self._outputs = self.OUTPUTS.copy()
        self.sess = None
        self.need_jit_script = cfg.get("need_jit_script", False)
        self.numpy_out = False

        # init some configs
        self._optimize_graph_flag = cfg.get('onnx_graph_optimization', True)
        self._convert_stream_flag = False
        self._export_stream_flag = False
        self._quantize_model_flag = cfg.get('onnx_quantization', False)
        self._optimize_level = int(cfg.get('onnx_optimize_level', 1))
        self._conv_w_list = None
        # arguments for stream converter
        # input shape of original model, like "B,T,512;B"
        # where 'B' indicates batch, 'T' indicates sequence length
        self._stream_trial_shape = cfg.get('stream_trial_shape', None)
        # valid input value range of the model, like "1;1",
        # it means the first input value is 1, the seconde value is 1
        self._stream_trial_value = cfg.get('stream_trial_value', None)
        # valid input sequence length sample of the model,
        # format is like "4,8,12", means 4,8,12 are the valid sequence length
        self._stream_trial_t = cfg.get('stream_trial_t', None)

        self._skip_test = False
        self.with_slim = cfg.get('slim_config', False) or cfg.get('slim_init_config', False)

        self.module_map = module_map

    def forward(self, *_args, **_kwargs):
        '''
        forward func for non streamed torch backend.
        It's for onnx export.

        Args:
            one or multi Tensor. other object type is not supported.
        Return:
            list of tuple of tensor. other object type is not supported.
        '''
        raise NotImplementedError

    def forward_step(self, *_args, **_kwargs):
        '''
        forward step func for streamed torch backend.
        It's for streamed run.

        Args:
            one or multi Tensor. other object type is not supported.
        Return:
            list of tuple of tensor. other object type is not supported.
        '''
        if self._convert_stream_flag:
            raise NotImplementedError

    def load_model(self, onnx_file, providers=None, **kwargs):
        '''
        load onnx model.
        compatible with penguin, and enable other keywords.
        '''
        infer_option = panther.SessionOptions()
        infer_option.intra_op_num_threads = 1
        infer_option.graph_optimization_level = panther.GraphOptimizationLevel.PTH_ENABLE_ALL
        for k, v in kwargs.items():
            if hasattr(infer_option, k):
                setattr(infer_option, k, v)
        self.sess = panther.InferenceSession(
            onnx_file,
            providers=providers or self._providers,
            sess_options=infer_option,
            device_id=get_rank(),
        )
        # align inputs for streaming model
        panther_input = [arg.name for arg in self.sess.get_inputs()]
        stream_input = ['global_state_in', 'x_sign']
        self_input = self._inputs
        self._inputs = []
        for model_input in self_input:
            if model_input['name'] not in stream_input:
                self._inputs.append(model_input)
            elif model_input['name'] in panther_input:
                self._inputs.append(model_input)

        if not is_tensor_meta_align(self._inputs, self.sess.get_inputs()):
            raise RuntimeError(
                self.NAME,
                'load_model: inputs meta not align:\ntorch:',
                _meta_str(self._inputs),
                '\npanther:',
                _meta_str(self.sess.get_inputs()),
            )
        if not is_tensor_meta_align(self._outputs, self.sess.get_outputs()):
            raise RuntimeError(
                self.NAME,
                'load_model: outputs meta not align:\ntorch:',
                _meta_str(self._outputs),
                '\npanther:',
                _meta_str(self.sess.get_outputs()),
            )

    def get_input(self):
        '''
        get input names.
        compatible with penguin
        '''
        # TODO: state tensor added to onnx model when convert stream.
        if self.sess is None:
            return [v.name for v in self._inputs]
        return [node.name for node in self.get_input_all()]

    def get_input_all(self):
        '''
        get input meta info.
        compatible with penguin
        '''
        if self.sess is None:
            return [
                FalconDict(name=node.name, type=node.type, shape=node.shape)
                for node in self._inputs
            ]
        inputs = self.sess.get_inputs()
        return inputs

    def get_output_all(self):
        '''
        get output meta info.
        compatible with penguin
        '''
        if self.sess is None:
            return [
                FalconDict(name=node.name, type=node.type, shape=node.shape)
                for node in self._outputs
            ]
        outputs = self.sess.get_outputs()
        return outputs

    def get_output(self):
        '''
        get output names.
        compatible with penguin
        '''
        if self.sess is None:
            return [v.name for v in self._outputs]
        return [node.name for node in self.get_output_all()]

    @torch.no_grad()
    def run(self, input_data, backend=None):
        '''
        run model to get outputs.

        Args:
            input_data[FalconDict]: input data
        '''
        backend = backend or self._backend
        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]
        if self.sess is None and backend == 'panther':
            self.load_model(self._onnx_file)
        input_data = dict(zip(self.get_input(), input_data))
        output_names = self.get_output()

        if backend == 'torch':
            # prepare env for inference
            self.eval()
            cudnn_tf32_flag = torch.backends.cudnn.allow_tf32
            cublas_tf32_flag = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            input_data = convert_to_tensor(input_data)
            if self._convert_stream_flag:
                # pylint: disable=assignment-from-no-return
                output_data = self.forward_step(**input_data)
            else:
                output_data = self(**input_data)

            torch.backends.cudnn.allow_tf32 = cudnn_tf32_flag
            torch.backends.cuda.matmul.allow_tf32 = cublas_tf32_flag
        elif backend == 'panther':
            input_data = convert_to_np(input_data)
            # NOTE: x_sign and global_state_in should be set before calling run
            output_data = self.sess.run(output_names, input_feed=input_data)
        else:
            raise RuntimeError(self.NAME, 'Infer run: unsupport backend', backend)
        if not isinstance(output_data, (list, tuple)):
            output_data = [output_data]
        if self.numpy_out:
            output_data = convert_to_np(output_data)
        return FalconDict(zip(output_names, output_data))

    def sample_inputs(self):
        '''
        sample some inputs as example.
        may be used for jit trace or test.
        '''
        raise NotImplementedError

    def sample_inputs_from_dataloader(self):
        '''
        sample from dataloader for post training quantization
        each exporter need to rewrite this function
        output format: dict
            key: onnx input names
            value: list of inputs (cpu numpy format) for the key
        if return is None, skip quantization
        '''
        # For pylint
        return None if self._quantize_model_flag else None

    def get_module_map(self):
        '''
        get module_map, which is a dict mapping module class to panther op
        for default, return module_map imported from panther_utilities
        if needed, modify module_map to specify how to map module class to panther op
        '''
        return self.module_map

    # @torch.no_grad()
    def export(self, *_args, **kwargs):
        '''
        export onnx for online.
        every step need test.
        1. export origin onnx
        2. do graph optimization
        3. do convert stream
        4. do quantize
        '''
        with torch.no_grad():
            self.prepare_export()
            if self._cfg.get('backend', None) == 'torch':
                return
            if get_local_rank() != 0:
                return
            if self._cfg.get('skip_onnx_export', False):
                mkdir_or_exist(self._onnx_dir)
                return

            if self._cfg.get('use_symbolic_export', False):
                replace_module_with_panther_op(self, self.get_module_map())

            self._data_loader = kwargs.get('data_loader', None)
            # remove stream state, Now we have not convert stream
            origin_inputs = self._inputs
            origin_outputs = self._outputs
            origin_stream_flag = self._convert_stream_flag
            if self._convert_stream_flag:
                self._inputs = [
                    v for v in origin_inputs if v.name not in ('global_state_in', 'x_sign')
                ]
                self._outputs = [v for v in origin_outputs if v.name not in ('global_state_out',)]
                self._convert_stream_flag = False

            mkdir_or_exist(self._onnx_dir)
            self._export_onnx()
            self._optimize_graph()

            # recover origin state
            if origin_stream_flag:
                self._set_stream_trial_shape(self._onnx_files[-1])
                self._inputs = origin_inputs
                self._outputs = origin_outputs
                self._convert_stream_flag = origin_stream_flag
            del origin_inputs, origin_outputs, origin_stream_flag

            self._convert_stream()

        # Need grad when quantizing
        self._quantize_stream_onnx_model()

        # copy last onnx file as final output
        previous_file = self._onnx_files[-1]
        os.system('cp -f {} {}'.format(previous_file, self._onnx_file))
        if not osp.exists(self._onnx_file):
            raise RuntimeError('rank', get_rank(), self.NAME, 'failed to export', self._onnx_file)

    def prepare_export(self):
        '''prepare for onnx export.'''

    def _export_onnx(self):
        '''export origin onnx and test.'''
        onnx_file = osp.join(self._onnx_dir, '{}_{}.onnx'.format(self.NAME, 'origin'))
        inputs = self.sample_inputs()
        self._do_export_onnx(onnx_file, inputs)
        self._onnx_files.append(onnx_file)
        if not self.with_slim:
            # TODO: add more test
            self.load_model(onnx_file)
            self.test_onnx(inputs, msg='origin')
            self.sess = None

    def _do_export_onnx(self, onnx_file, inputs, **kwargs):
        '''do export onnx.'''
        dynamic_axes = {
            node.name: {i: val for i, val in enumerate(node.shape) if isinstance(val, str)}
            for node in self._inputs + self._outputs
        }
        input_names = [v.name for v in self._inputs]
        output_names = [v.name for v in self._outputs]
        self.eval()
        os.system('rm -rf {}'.format(onnx_file))
        if self.need_jit_script:
            example_outputs = self(*inputs)
            model = torch.jit.script(self)
        else:
            model = self
            example_outputs = None
        torch.onnx.export(
            model,
            inputs,
            onnx_file,
            verbose=True,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            enable_onnx_checker=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            example_outputs=example_outputs,
            **kwargs,
        )
        if not osp.exists(onnx_file):
            raise RuntimeError('rank', get_rank(), self.NAME, 'failed to export', onnx_file)

    def _optimize_graph(self):
        '''graph optimization and test.'''
        if not self._optimize_graph_flag:
            return

        previous_file = self._onnx_files[-1]
        current_file = osp.join(self._onnx_dir, '{}_{}.onnx'.format(self.NAME, 'optimized'))
        os.system('rm -rf {}'.format(current_file))
        try:
            # Some OP may not supported on CPU, default try CUDA.
            onnx2panther.from_onnx_model(
                previous_file,
                optimize_level=self._optimize_level,
                output_onnx_path=current_file,
                provider='CPU',
                with_slim=self.with_slim,
            )
        except Exception:
            onnx2panther.from_onnx_model(
                previous_file,
                optimize_level=self._optimize_level,
                output_onnx_path=current_file,
                provider='CUDA',
                with_slim=self.with_slim,
            )
        if not osp.exists(current_file):
            raise RuntimeError(
                'rank', get_rank(), self.NAME, 'failed on optimizing graph', current_file
            )
        self._onnx_files.append(current_file)
        # TODO: add more test
        self.load_model(current_file)
        inputs = self.sample_inputs()
        self.test_onnx(inputs, msg='optimized')
        self.sess = None

    def _set_stream_trial_shape(self, onnx_file):
        """read input shape info and merge it with self._inputs"""
        origin_sess = self.sess
        self.load_model(onnx_file)
        input_info = self.sess.get_inputs()
        self.sess = origin_sess

        if not self._stream_trial_shape:
            self._stream_trial_shape = []
            for pth_input in input_info:
                for torch_input in self._inputs:
                    if pth_input.name == torch_input.name:
                        self._stream_trial_shape.append(
                            parse_shape(torch_input.shape, pth_input.shape)
                        )
                        break

    def _convert_stream(self):
        '''convert stream and test.'''
        if not self._convert_stream_flag:
            return

        if version.parse(panther.__version__) <= version.parse('1.6.3'):
            previous_file = self._onnx_files[-1]
            current_file = osp.join(self._onnx_dir, '{}_{}.onnx'.format(self.NAME, 'stream'))
            os.system('rm -rf {}'.format(current_file))
            model = onnx.load(previous_file)
            config = cv.StreamConverterConfig(conv_w_list=self._conv_w_list)
            stream_converter = cv.StreamConverter(model, config)
            stream_converter.convert()
            stream_converter.save(current_file)
            if not osp.exists(current_file):
                raise RuntimeError(
                    'rank', get_rank(), self.NAME, 'failed on convert stream', current_file
                )
            self._onnx_files.append(current_file)
        else:
            previous_file = self._onnx_files[-1]
            current_file = osp.join(self._onnx_dir, '{}_{}.onnx'.format(self.NAME, 'stream'))
            os.system('rm -rf {}'.format(current_file))
            if not self._stream_trial_shape:
                raise RuntimeError(
                    'Require input shape of model',
                    previous_file,
                    'to convert it to stream.\n',
                    'input shape is like [["B", "T", 512]] '
                    'where "T" indicates the sequence dimension.',
                )
            input_shape = self._stream_trial_shape
            input_value = self._stream_trial_value
            if self._stream_trial_t:
                trial_t = (
                    self._stream_trial_t
                    if isinstance(self._stream_trial_t, list)
                    else list(map(int, self._stream_trial_t.split(',')))
                )
            else:
                trial_t = None
            cv.convert_to_stream(
                previous_file, current_file, input_shape, input_value=input_value, trial_t=trial_t
            )

            if not osp.exists(current_file):
                raise RuntimeError(
                    'rank', get_rank(), self.NAME, 'failed on convert stream', current_file
                )
            self._onnx_files.append(current_file)

        # TODO support stream onnx model test

    def _quantize_stream_onnx_model(self):
        '''do quantize on onnx model and test.'''
        if not self._quantize_model_flag or self._data_loader is None:
            return
        self._dynamic_quant_flag = False

        # prepare calibration data
        datas = self.sample_inputs_from_dataloader()
        if datas is None and not self._dynamic_quant_flag:
            return

        # load original model
        previous_file = self._onnx_files[-1]
        model_fp32 = onnx.load(previous_file)

        if self._dynamic_quant_flag:
            # use dynamic quantization
            config = QuantizeConfig()
            quantizer = create_post_training_quantizer(model_fp32, 'DynamicQuantizer', config)
            quantizer.quantize()
        else:
            if 'onnx_quant_ops' not in self._cfg:
                return
            types_to_quantize = [s.strip() for s in self._cfg['onnx_quant_ops'].split(',')]
            # quantization config
            config = QuantizeConfig(
                bits=8,
                per_channel=True,
                types_to_quantize=types_to_quantize,
            )

            # post-training quantize the model
            quantizer = create_post_training_quantizer(model_fp32, 'PostStaticQuantizer', config)
            # Fallback the first QLSTM of encoder to old version
            # For CER robustness, use --solution.loose_quant 1 to skip the first LSTMP of encoder
            quantizer.quantize(
                datas,
                fallback=self.NAME == 'encoder',
                provider='cuda' if self._cfg.get('panther_use_gpu', True) else 'cpu',
                loose=self._cfg.get('loose_quant', False) and self.NAME == 'encoder',
            )
        current_file = osp.join(self._onnx_dir, '{}_{}.onnx'.format(self.NAME, 'quantized'))
        quantizer.save(current_file)
        self._onnx_files.append(current_file)
        self._skip_test = True

    def test_onnx(self, inputs, msg=''):
        '''do test.'''
        atol = self.OPTIMIZED_TEST_ATOL if msg == 'optimized' else self.ORIGIN_TEST_ATOL
        rtol = self.OPTIMIZED_TEST_RTOL if msg == 'optimized' else self.ORIGIN_TEST_RTOL
        msetol = self.OPTIMIZED_TEST_MSE if msg == 'optimized' else self.ORIGIN_TEST_MSE
        if self._skip_test:
            return True
        torch_out = self.run(inputs, backend='torch')
        torch_out = convert_to_np(torch_out)
        onnx_out = self.run(inputs, backend='panther')

        rank = get_rank()
        all_pass = True
        for name in torch_out.keys():
            result1 = torch_out[name]
            result2 = onnx_out[name]
            result = np.allclose(result2, result1, atol=atol, rtol=rtol)
            err = np.absolute(result1 - result2)
            mse = (np.square(result1 - result2)).mean()
            absolute_err = np.where(err > atol, atol, err)
            relative_err = np.absolute((err - absolute_err) / (result1 + 1e-6))
            ind = np.unravel_index(np.argmax(relative_err, axis=None), err.shape)
            rtl = relative_err[ind]
            if rtl > 0:
                atl = atol
            else:
                ind = np.unravel_index(np.argmax(absolute_err, axis=None), err.shape)
                atl = absolute_err[ind]
            if not result or mse > msetol:
                all_pass = False
                logging.error(
                    'rank %d: %s %s(%s): %s test failed, atol %.5f/%.5f, rtol %.5f/%.5f., '
                    'mse %.5f/%.5f, inputs and outputs is saved in %s/test_data/%s/%s/',
                    rank,
                    self.__class__.__name__,
                    self.NAME,
                    msg,
                    name,
                    atl,
                    atol,
                    rtl,
                    rtol,
                    mse,
                    msetol,
                    self._onnx_dir,
                    self.NAME,
                    msg,
                )
                logging.error(
                    'The max difference: {}{} (torch)={}, (onnx)={}'.format(
                        name, list(ind), result1[ind], result2[ind]
                    )
                )
            else:
                logging.info(
                    'rank %d: %s %s(%s): %s test passed, atol %.5f/%.5f, rtol %.5f/%.5f.,'
                    ' mse %.5f/%.5f',
                    rank,
                    self.__class__.__name__,
                    self.NAME,
                    msg,
                    name,
                    atl,
                    atol,
                    rtl,
                    rtol,
                    mse,
                    msetol,
                )
        if not all_pass:
            self._save_inputs_outputs(msg, inputs, torch_out, onnx_out)
        return all_pass

    def _generate_input_data(
        self, input_idx, method='rand', dynamic_axis=1, device='cuda', low=0, high=-1
    ):
        '''
        helper function for sample input data.
        generate data by input infos.
        '''
        shape = self._inputs[input_idx].shape
        dtype = self._inputs[input_idx].type
        if isinstance(dynamic_axis, (list, tuple)):
            shape = [da if isinstance(dim, str) else dim for dim, da in zip(shape, dynamic_axis)]
        else:
            shape = [dynamic_axis if isinstance(dim, str) else dim for dim in shape]
        if any(dim == -1 for dim in shape):
            raise RuntimeError(self.NAME, 'shape not set', shape)
        assert method in ('rand', 'ones', 'zeros')
        if method == 'rand':
            if dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16):
                data = torch.rand(shape)
            else:
                data = torch.randint(low=low, high=high, size=shape)
        elif method == 'ones':
            data = torch.ones(shape)
        else:
            data = torch.zeros(shape)
        return data.to(dtype).to(device)

    def _save_inputs_outputs(self, msg, inputs, torch_out, onnx_out):
        '''save inputs and outputs.'''
        file_path = f'{self._onnx_dir}/test_data/{self.NAME}/{msg}/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(f'{file_path}inputs.pkl', 'wb') as f:
            pickle.dump(inputs, f)
        with open(f'{file_path}torch_out.pkl', 'wb') as f:
            pickle.dump(torch_out, f)
        with open(f'{file_path}onnx_out.pkl', 'wb') as f:
            pickle.dump(dict(onnx_out), f)
        if not self._cfg.get('skip_export_test_error', False):
            raise RuntimeError('infer test failed')


INFERS = dict()


class Factory:
    '''
    Infer Object factory.
    Compatible with Penguin penguin/core/factory.py.
    '''

    @staticmethod
    def get_inference(model_name, *_args):
        '''
        get infer object.

        compatible with penguin API.
        function API in Penguin.
            >>> def get_inference(model_name, config)

        the confg argument is only to get use_gpu option.
        Now this option is set by panther_use_gpu in Infer args.
        '''
        return INFERS[model_name]

    # NOTE: get_inference and get_new_processor may need to add
