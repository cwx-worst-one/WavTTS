import os
import torch
from collections import OrderedDict
from recipes.musiclm.utils.dist import local_zero_first
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.audio_quality_classifier.models.discriminator import MultiScaleSTFTDiscriminator

def init_audio_quality_classifier(checkpoint_path, local_rank, cache_dir):
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)

        model = MultiScaleSTFTDiscriminator(filters=48)

        state_dict = torch.load(local_path)["state_dict"]
        new_dict = OrderedDict()
        for key in state_dict:
            new_key = key.replace('model.', '')
            new_dict[new_key] = state_dict[key]

        model.load_state_dict(new_dict, strict=True)
        model.to(device)
        model.eval()

        return { "classifier": model }


@torch.no_grad()
def aq_classifier_inference(requires, samples, params):
    model = requires["classifier"]

    assert len(samples.shape) == 3, f"Expected input shape (batch, channel, length)"
    # samples shape  (b c l) 
    logits, _ = model(samples.float())
    # print(len(logits))
    # assert 1==2
    output = []
    for logit in logits:
        output.append(logit.mean(dim=tuple(range(1, logit.ndim))))
    output = torch.stack(output).T.mean(dim=-1)

    return output


