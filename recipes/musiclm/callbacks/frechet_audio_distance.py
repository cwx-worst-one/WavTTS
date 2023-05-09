from functools import partial

import pytorch_lightning as pl
import torch
from frechet_audio_distance import FrechetAudioDistance
from frechet_audio_distance.compat.vggish_tf.transform import (
    TFCompatibleVGGishTransform,
)

# from frechet_audio_distance.models.vggish.convert import convert_tf_vggish_to_torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchaudio.functional import resample


class FrechetAudioDistanceCallback(pl.Callback):
    @rank_zero_only
    def __init__(
        self,
        dataset,
        dataset_sample_rate: int,
        temperature: float,
        total_seconds_to_compare: int = 30,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_sample_rate = dataset_sample_rate
        self.temperature = temperature
        self.total_seconds_to_compare = total_seconds_to_compare

        # TODO: Conversion hogs up GPU memory
        # model = convert_tf_vggish_to_torch("https://tfhub.dev/google/vggish/1")
        # torch.save(model.cpu(), "vggish.pt")
        # exit()

        fp = (
            "/mnt/bn/audio-diffusion/pretrained_models/frechet_audio_distance/vggish.pt"
        )
        model = torch.load(fp, map_location="cpu")

        self.fad = FrechetAudioDistance(model)
        self.transform = TFCompatibleVGGishTransform()
        self.resample = partial(
            resample, orig_freq=dataset_sample_rate, new_freq=self.transform.sample_rate
        )

    @rank_zero_only
    def on_validation_epoch_start(self, _, module):
        mels_ground_truth = []
        mels_sampled_audio = []
        total_seconds = 0
        for data in self.dataset:
            if total_seconds >= self.total_seconds_to_compare:
                break

            audios = data[0]
            for audio in audios:
                # TODO: Batched sampling, but it's quite slow..
                sampled_token_ids = module.sample_with_audio_conditioning(
                    audio.unsqueeze(dim=0), temperature=self.temperature
                )

                with torch.no_grad():
                    module.audio_model.eval()
                    modeled_quantizers = module.input_quantizers
                    modeled_token_labels = module.audio_model(
                        audio.unsqueeze(dim=0).to(module.device)
                    )
                    modeled_token_labels[:, modeled_quantizers] = sampled_token_ids
                    decoded_modeled_preds = module.audio_model.decode(
                        modeled_token_labels
                    )
                    decoded_modeled_preds = decoded_modeled_preds.cpu()

                mels_ground_truth.append(self.transform(self.resample(audio)))
                mels_sampled_audio.append(
                    self.transform(self.resample(decoded_modeled_preds.squeeze(dim=0)))
                )
                total_seconds += audio.shape[-1] / self.dataset_sample_rate

        assert len(mels_ground_truth) == len(mels_sampled_audio)
        mels_ground_truth = torch.cat(mels_ground_truth)
        mels_sampled_audio = torch.cat(mels_sampled_audio)

        score = self.fad(mels_ground_truth, mels_sampled_audio)
        module.log("frechet_audio_distance/valid", score)
