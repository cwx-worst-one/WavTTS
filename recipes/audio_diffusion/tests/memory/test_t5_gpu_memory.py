import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from samantha.utils.hparams import DotDict
from tests.helpers.runif import RunIf


@RunIf(min_cuda_gpus=1)
def test_memory_t5():

    hparams_file = (
        "./recipes/audio_diffusion/conf/140223-singsong/singsong_coarse2coarse.yaml"
    )

    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)

    pl_module = cfg.pl_module

    model = pl_module.acoustic_token_model.to("cuda")

    batch_size = 1
    n_quantizers = 6
    quantizer_seq_len = 800
    target_acoustic_tokens = torch.randint(
        0, 1024, (batch_size, n_quantizers, quantizer_seq_len), device="cuda"
    )
    cond_embeddings = torch.randint(
        0, 1024, (batch_size, n_quantizers, quantizer_seq_len), device="cuda"
    )

    # from pytorch_memlab import MemReporter
    # reporter = MemReporter()

    def prof(tca, ce):
        loss, accuracy = model.loss(tca, ce)
        return loss

    # optimizer = cfg.optimizer_class(model.parameters())
    for e in tqdm(range(1)):
        _ = prof(target_acoustic_tokens.clone(), cond_embeddings.clone())
        # loss.backward()
