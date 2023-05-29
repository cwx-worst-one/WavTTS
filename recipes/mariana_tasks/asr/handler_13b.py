import logging
logging.basicConfig(level=logging.INFO)

from typing import Dict, List, Union
import torch

from fex.matx.ops import MatxBertTokenizer
from fuxi.tasks.alice.tools.recover_text import recover_text
from quicksilver import proto_util

import lego_ops
lego_ops.load_ft_torch()

MAX_LENGTH = 1280

# TODO: move to bytedlaplace
def ndarray_encode(array):
    return proto_util.make_tensor_proto(array).SerializeToString()


def ndarray_decode(data):
    if isinstance(data, str):
        data = data.encode()
    tensor_proto = proto_util.tensor_pb2.TensorProto()
    tensor_proto.ParseFromString(data)
    return proto_util.make_ndarray(tensor_proto)


class EndpointHandler:
    def __init__(self, ckpt_path="./model/model_state.th", config_path="./model/model_conf.yaml", generate_config_path="./model/generate.yaml"):
        self.generator = torch.jit.load("gpt2_generator.pt").eval()
        self.tokenizer = self.make_tokenizer()
        self.eos_token, = self.tokenizer.tokens2idx([b'[EOS]'])

        # do some warmup
        for _ in range(4):
            warmup_request = {
                "prompt": ["你好".encode()],
                "top_k" : [10],
                "top_p" : [0.7],
                "temperature" : [1.0],
            }
            warmup_response = self(warmup_request)
            print("warmup output", warmup_response['output'][0].decode())

    def make_tokenizer(self):
        vocab_path = "./model/vocab.txt"
        tokenizer = MatxBertTokenizer(vocab_path,
                                      do_cut=True,
                                      cut_mode='libcut/search',
                                      lower_case=True,
                                      max_tokens_per_input=MAX_LENGTH)
        return tokenizer

    def preprocess_text(self, text_list, with_sep):
        final_ids = []
        sep_id = self.tokenizer.tokens2idx(["[SEP]"])
        eos_id = self.tokenizer.tokens2idx(["[EOS]"])
        for idx, text in enumerate(text_list[:-1]):
            tokens = self.tokenizer(text)
            ids = self.tokenizer.tokens2idx(tokens)
            final_ids += ids
            if idx % 2 == 0:
                final_ids += sep_id
            else:
                final_ids += eos_id

        final_prompt = text_list[-1]
        tokens = self.tokenizer(final_prompt)
        ids = self.tokenizer.tokens2idx(tokens)
        if with_sep:
            ids += sep_id
        final_ids += ids

        final_ids = final_ids[-512:]
        return torch.tensor(final_ids).unsqueeze(0).to(torch.long).cuda()

    @torch.no_grad()
    def __call__(
        self, request: Dict[str, Union[List[bytes], List[int], List[float]]]
    ) -> Dict[str, Union[List[bytes], List[int], List[float]]]:
        text = [t.decode() if type(t)==bytes else t for t in request["prompt"]]
        bs = len(text)
        logging.info(f"PROMPT: {text}")

        top_k = 1
        if "top_k" in request:
            top_k = request["top_k"][0]
        top_p = 1.0
        if "top_p" in request:
            top_p = request["top_p"][0]
        temperature = 1.0
        if "temperature" in request:
            temperature = request["temperature"][0]

        with_sep = True
        if "segment_id" in request:
            with_sep = request["segment_id"][0] == 0

        if "input_ids" in request and request["input_ids"][0]:
            input_ids = torch.from_numpy(ndarray_decode(request["input_ids"][0]))[-512:].unsqueeze(0).cuda()
        else:
            input_ids = self.preprocess_text(text, with_sep)

        _, prompt_len = input_ids.shape

        max_new_tokens = MAX_LENGTH
        if "max_tokens" in request:
            max_new_tokens = request["max_tokens"][0]

        self.generator.set_generator_strategy(
            max_new_tokens=max_new_tokens,
            top_k=top_k, top_p=top_p, temperature=temperature
        )
        out = self.generator(input_ids)
        out2 = out[:,prompt_len:]

        if out.shape[1] >= MAX_LENGTH:
            is_end = [1] * bs
        else:
            is_end = (out2 == self.eos_token).long().max(dim=1)[0].tolist() * bs

        if with_sep:
            completion = recover_text(' '.join(self.tokenizer.idx2tokens(out[0][prompt_len:])))
        else:
            full_tokens = self.tokenizer.idx2tokens(out[0])
            full_text = recover_text(' '.join(full_tokens))
            prev_text = recover_text(' '.join(full_tokens[:prompt_len]))
            completion = full_text[len(prev_text):]

        logging.info(f"RESPONSE: {completion}")

        completion = [completion.encode()] + [b''] * (bs - 1)
        output_ids = [ndarray_encode(out[0].cpu().numpy())] + [b''] * (bs - 1)
        return dict(output=completion, is_end=is_end, output_ids=output_ids)


if __name__ == "__main__":
    # multi-round
    endpoint_handler = EndpointHandler()
    inputs = dict(
        prompt=["天空是什么颜色的？".encode(), "蓝色。".encode(), "为什么是蓝色的？".encode()],
    )
    rsp = endpoint_handler(inputs)
    print(inputs, [t.decode() for t in rsp['output']])

    # using input_ids
    inputs = dict(
        prompt=["天空是什么颜色的？".encode(), "蓝色。".encode(), "为什么是蓝色的？".encode()],
        max_tokens=[4],
    )
    rsp = endpoint_handler(inputs)
    print(inputs, [t.decode() for t in rsp['output']])
    output_ids = rsp["output_ids"]

    inputs = dict(
        prompt=["DONT USE THIS!".encode()],
        input_ids=output_ids,
    )
    rsp = endpoint_handler(inputs)
    print(inputs, [t.decode() for t in rsp['output']])
