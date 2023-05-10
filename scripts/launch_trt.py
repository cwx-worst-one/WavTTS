'''launch tensorrt model'''
import os
import argparse
import librosa
import random
import pycuda.autoinit
import os.path as osp
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger()

class MemoryManager(object):
    '''manage host and device memory'''
    def __init__(self, binding, shape, dtype):
        '''__init__'''
        self._host = cuda.pagelocked_empty(trt.volume(shape), dtype)
        self._device = cuda.mem_alloc(self._host.nbytes)
        self._binding = binding
        self._shape = shape
        self._dtype = dtype

    def __str__(self):
        '''__str__'''
        return 'binding: {}, shape: {}, dtype: {}, host: {}, device: {}'.format(
            self._binding, self._shape, self._dtype, self._host, self._device
        )

    def buffer(self):
        '''buffer'''
        return int(self._device)

    def htod(self, stream):
        '''memcpy from host to device'''
        cuda.memcpy_htod_async(self._device, self._host, stream)

    def dtoh(self, stream):
        '''memcpy from device to host'''
        cuda.memcpy_dtoh_async(self._host, self._device, stream)

    def set_data(self, np_data):
        '''copy numpy data to device'''
        np.copyto(self._host, np_data.ravel())

    def get_data(self):
        '''copy device data to np array'''
        out = np.array(self._host, dtype=self._dtype)
        return np.reshape(out, self._shape)

    @property
    def shape(self):
        '''self._shape getter'''
        return self._shape

    @property
    def dtype(self):
        '''self._dtype getter'''
        return self._dtype


class TensorRTEngine(object):
    '''manage tensorrt engine object'''
    def __init__(self, engine, batch_size, batch_index=0, inputs=None):
        '''__init__
        Arguments:
            engine: cuda engine object, created or deserialize by trt.Runtime.
            batch_size: int, batch_size of the inputs data.
            batch_index: int, the dimension index of the batch dim.
            inputs: list of MemoryManager objects that bind to the previous TensorRTEngine
                    or None
        '''
        self._engine = engine
        self._context = engine.create_execution_context()

        self._inputs = []
        self._outputs = []
        self._bindings = []
        idx = 0
        # get inputs and outputs bindings of the engine
        for binding in engine:
            mode = engine.get_tensor_mode(binding)
            assert mode in [trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT]
            shape = list(engine.get_tensor_shape(binding))
            shape[batch_index] = batch_size
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            self._context.set_binding_shape(engine[binding], shape)

            if mode == trt.TensorIOMode.INPUT and inputs:
                mem = inputs[idx]
                idx += 1
                assert mem.shape == shape and mem.dtype == dtype
            else:
                mem = MemoryManager(binding, shape, dtype)

            self._bindings.append(mem.buffer())
            if mode == trt.TensorIOMode.INPUT:
                self._inputs.append(mem)
            else:
                self._outputs.append(mem)

    def __str__(self):
        '''__str__'''
        message = "TensorRTEngine object:\nInputs:"
        for i in self._inputs:
            message += '\n\t'
            message += str(i)
        message += '\nOutputs:'
        for o in self._outputs:
            message += '\n\t'
            message += str(o)
        return message

    def __call__(self, stream):
        '''__call__'''
        self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=stream.handle
        )

    @property
    def inputs(self):
        '''self._inputs getter'''
        return self._inputs

    @property
    def outputs(self):
        '''self._outputs getter'''
        return self._outputs


class TensorRTRunner(object):
    '''TensorRTRunner'''
    def __init__(self, engines, batch_size, batch_index):
        '''__init__'''
        self._engines = []
        inputs = None
        for e in engines:
            engine = TensorRTEngine(e, batch_size, batch_index, inputs)
            inputs = engine.outputs
            self._engines.append(engine)

    def __str__(self):
        '''__str__'''
        message = '>' * 50
        for engine in self._engines:
            message += '\n'
            message += str(engine)
            message += '\n'
            message += '>' * 50
        return message

    def __call__(self, stream, inputs):
        '''__call__'''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        engine_inputs = self._engines[0].inputs
        for mem, data in zip(engine_inputs, inputs):
            mem.set_data(data)
            mem.htod(stream)

        for engine in self._engines:
            engine(stream)

        engine_outputs = self._engines[-1].outputs
        for out in engine_outputs:
            out.dtoh(stream)
        stream.synchronize()

        outputs = [out.get_data() for out in engine_outputs]
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @property
    def inputs(self):
        '''inputs getter of first engine'''
        return self._engines[0].inputs

    @property
    def outputs(self):
        '''outputs getter of last engine'''
        return self._engines[-1].outputs


def local_filepath(model_root, model_file, local_root):
    '''local file path'''
    filepath = osp.join(model_root, model_file)
    if filepath.startswith('hdfs://'):
        filename = filepath.split('/')[-1]
        local_file = osp.join(local_root, filename)
        if osp.exists(local_file):
            os.remove(local_file)
        ret = os.system('hdfs dfs -get {} {}'.format(filepath, local_root))
        assert ret == 0, 'failed to get {} from hdfs'.format(filepath)
        filepath = local_file
    return filepath


# TODO(mashengtao) support more common data format
def load_aed_wav(data_patterns, local_root, inputs, batch_index):
    '''load wav data'''
    assert len(inputs) == 1
    assert len(inputs[0].shape) == 2
    input_shape = inputs[0].shape
    batch_size = input_shape[batch_index]
    wav_length = input_shape[1 - batch_index]

    local_data_root = osp.join(local_root, 'data')
    if not osp.isdir(local_data_root):
        os.makedirs(local_data_root, mode=777, exist_ok=True)

    data_files = []
    for pattern in data_patterns:
        if pattern.startswith("hdfs://"):
            text_wrapper = os.popen(f"hdfs dfs -ls {pattern}")
        else:
            text_wrapper = os.popen(f"ls {pattern}")

        for line in text_wrapper:
            data_files.append(line.strip().split(" ")[-1])

    data = []
    for data_file in data_files:
        filepath = local_filepath('', data_file, local_data_root)
        wav, _ = librosa.load(filepath, sr=16000)
        wav *= 32768
        # resize to wav_length
        wav_len = len(wav)
        if wav_len < wav_length:
            pad_len = wav_length - wav_len
            pad_l = random.randint(0, pad_len)
            pad_r = pad_len - pad_l
            wav_data = np.concatenate([np.zeros(pad_l), wav, np.zeros(pad_r)])
        else:
            pos = random.randint(0, wav_len - wav_length)
            wav_data = wav[pos : (pos + wav_length)]
        wav_data.astype(inputs[0].dtype)
        data.append(wav_data)
    # batch
    batch_data = []
    for pos in range(batch_size, len(data) + 1, batch_size):
        batch_data.append(np.stack(data[(pos - batch_size) : pos]))
    return batch_data


def launch():
    '''launch entry'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_root',
        type=str,
        required=True,
        help='common root dir of model_files'
    )
    parser.add_argument(
        '--model_files',
        type=str,
        required=True,
        help='tensorrt model files, seperated by comma',
    )
    parser.add_argument('--data_patterns', type=str, required=True)
    parser.add_argument('--local_root', type=str, required=False, default='local_trt')
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--batch_index', type=int, required=False, default=0)
    args = parser.parse_args()
    local_root = args.local_root
    model_files = args.model_files.split(',')
    data_patterns = args.data_patterns.split(',')

    if not osp.isdir(local_root):
        os.makedirs(local_root, mode=777, exist_ok=True)

    stream = cuda.Stream()

    engines = []
    with trt.Runtime(TRT_LOGGER) as runtime:
        for model_file in model_files:
            filepath = local_filepath(args.model_root, model_file, local_root)
            with open(filepath, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
                engines.append(engine)
        runner = TensorRTRunner(engines, args.batch_size, args.batch_index)
        batch_data = load_aed_wav(
            data_patterns, local_root, runner.inputs, args.batch_index
        )
        print(runner)

        for batch in batch_data:
            print(runner(stream, batch))


if __name__ == '__main__':
    launch()
