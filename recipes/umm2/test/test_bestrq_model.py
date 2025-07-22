import sys
sys.path.append("/Users/bytedance/Documents/workspace/samantha")
sys.path.append("/Users/bytedance/Documents/workspace/samantha/apps/mariana")
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from samantha.utils.hparams import DotDict



def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def prepare_input(config, batch_size, time_steps):
    """Helper function to prepare input tensors."""
    n_mels = config.n_mels
    # Input shape (B, n_mels, Time, 1)
    x = torch.randn(batch_size, time_steps , n_mels)
    # Input x_lengths shape (B,)
    x_lengths = torch.randint(time_steps // 2, time_steps + 1, (batch_size,), dtype=torch.long)
    #x_lengths = _lengths_to_padding_mask(x_lengths)

    idx = torch.randint(0, len(x_lengths), (1,))
    x_lengths[idx] = time_steps
    return x, x_lengths

def load_config_from_file(config_fp: str):
    with open(config_fp, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)
    return DotDict(hparams)


def test_bestrq_model():
    config_fp = "/Users/bytedance/Documents/workspace/samantha/recipes/umm2/conf/conformer_unified_data/dataloader/umm_conformer_stage1_unified_VQ_baseline_re_lite.yaml"
    config = load_config_from_file(config_fp)
    model = config.model

    print(model)
    


if __name__ == "__main__":
    test_bestrq_model()



