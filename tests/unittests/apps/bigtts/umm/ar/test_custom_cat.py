import pytest
import torch

from apps.bigtts.umm.ar.utility.common import _custom_cat


def ref_cat(
    lyrics_tokens: torch.Tensor,
    sos_ids: torch.Tensor,
    target_ids: torch.Tensor,
    input_lens: torch.Tensor,
    target_lens: torch.Tensor,
    bsz: int,
    t: int,
):
    h = torch.zeros([bsz, t]).long()
    for i in range(bsz):
        h[i, : input_lens[i] + 1 + 1 + target_lens[i] + 1] = torch.cat(
            (
                lyrics_tokens[i, : input_lens[i]],
                # torch.zeros([prompt_lens[i]]).to(sos_ids.device), # query placeholder
                sos_ids[i, :],
                torch.zeros([1]).to(sos_ids.device),  # placeholder
                target_ids[i, : target_lens[i] + 1],
            )
        )
    return h


@pytest.mark.parametrize("target_max_T", [32, 64])
@pytest.mark.parametrize("input_max_T", [32, 64])
@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("D", [128, 256])
def test_custom_cat(target_max_T, input_max_T, batch_size, D):
    T = target_max_T + input_max_T + 2
    lyrics_tokens = torch.randint(low=0, high=D, size=(batch_size, input_max_T))
    sos_ids = torch.ones([batch_size, 1]).long() * (D + 1)  # D + 1 as sos id
    target_ids = torch.randint(low=0, high=D, size=(batch_size, target_max_T))
    input_lens = torch.randint(low=0, high=input_max_T, size=(batch_size,))
    target_lens = torch.randint(low=0, high=target_max_T, size=(batch_size,))

    ref_output = ref_cat(
        lyrics_tokens, sos_ids, target_ids, input_lens, target_lens, batch_size, T
    )
    custom_output = _custom_cat(
        lyrics_tokens, sos_ids, target_ids, input_lens, target_lens, batch_size, T
    )
    torch.testing.assert_allclose(ref_output, custom_output, atol=0, rtol=0)
