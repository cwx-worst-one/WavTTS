import pytest

from recipes.soundstorm2.lightning.soundstorm import SoundStorm, SoundStormConfig
from recipes.soundstorm2.lightning.soundstream import SoundStreamSpeech24k
from tests.helpers.runif import RunIf
from tests.helpers.testing_utils import torch_device
from tests.unittests.models.utils import ids_tensor


class SoundStormModelTester:
    def __init__(
        self,
        batch_size=14,
        seq_len: int = 500,
        n_head: int = 2,
        n_embd: int = 32,
        n_layer: int = 2,
        n_quantizers: int = 12,
        attention_kwargs={"enable_flash": True, "enable_mem_efficient": False},
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_quantizers = n_quantizers
        self.attention_kwargs = attention_kwargs

        self.audio_model = SoundStreamSpeech24k()
        self.config = SoundStormConfig(
            sample_rate=24000,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
        )

    def get_inputs(self):
        audio_tokens = ids_tensor(
            (self.batch_size, self.n_quantizers, self.seq_len),
            self.model.audio_model.codebook_size,
            device=torch_device,
        )
        selected_qs = list(range(0, self.n_quantizers)) * self.batch_size
        return audio_tokens, selected_qs

    def create_and_test_model(self):
        self.model = SoundStorm(
            config=self.config,
            audio_model=self.audio_model,
            optimizer_cls=None,
            scheduler_cls=None,
        )
        self.model = self.model.to(torch_device)
        self.model.apply(self.model._init_weights)
        self.model.eval()
        return self.model


@pytest.fixture()
def soundstorm() -> SoundStormModelTester:
    yield SoundStormModelTester()


def test_model(soundstorm: SoundStormModelTester) -> None:
    model = soundstorm.create_and_test_model()
    audio_tokens, selected_qs = soundstorm.get_inputs()
    out = model(audio_tokens, selected_qs)
    assert out.shape == (
        soundstorm.batch_size,
        soundstorm.num_quantizers,
        soundstorm.seq_len,
        soundstorm.model.out_dim,
    )
