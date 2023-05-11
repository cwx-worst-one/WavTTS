import pytest
import torch

from tests.helpers.runif import RunIf
from tests.helpers.testing_utils import torch_device
from tests.unittests.models.utils import ids_tensor
from samantha.models.gpt2 import GPT2, GPT2Config, GPT2Model


class GPT2ModelTester:
    def __init__(
        self,
        batch_size=14,
        seq_len: int = 7,
        vocab_size: int = 99,
        n_head: int = 2,
        n_embd: int = 32,
        n_layer: int = 2,
        n_inner: int = 37,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_bias: bool = False,
        layer_norm_epsilon: float = 0.00001,
        initializer_range: float = 0.02,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_bias = layer_norm_bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

    def get_inputs(self):
        return ids_tensor(
            (self.batch_size, self.seq_len), self.vocab_size, device=torch_device
        )

    def get_config(self):
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.seq_len,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            n_inner=self.n_inner,
            resid_pdrop=self.resid_pdrop,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            layer_norm_bias=self.layer_norm_bias,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
        )

    def create_and_test_model(self):
        model = GPT2Model(self.get_config())
        model = model.to(torch_device)
        model.eval()
        return model

    def create_and_test_model_with_lm_head(self):
        model = GPT2(self.get_config())
        model = model.to(torch_device)
        model.eval()
        return model


@pytest.fixture()
def model_tester() -> GPT2ModelTester:
    yield GPT2ModelTester()


def test_gpt2_model(model_tester: GPT2ModelTester) -> None:
    inputs = model_tester.get_inputs()
    model = model_tester.create_and_test_model()
    out = model(inputs)
    assert out.shape == (
        model_tester.batch_size,
        model_tester.seq_len,
        model_tester.n_embd,
    )


def test_gpt2_model_with_lm_head(model_tester: GPT2ModelTester) -> None:
    inputs = model_tester.get_inputs()
    model = model_tester.create_and_test_model_with_lm_head()
    out = model(inputs)
    assert out.shape == (
        model_tester.batch_size,
        model_tester.seq_len,
        model_tester.vocab_size,
    )


@RunIf(min_torch="2.0")
def test_gpt2_model_compile(model_tester: GPT2ModelTester) -> None:
    inputs = model_tester.get_inputs()

    model = model_tester.create_and_test_model()
    model = torch.compile(model)
    out = model(inputs)
    assert out.shape == (
        model_tester.batch_size,
        model_tester.seq_len,
        model_tester.n_embd,
    )
