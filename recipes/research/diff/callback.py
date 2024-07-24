import os
from typing import Union

import torch
import torchaudio
from pytorch_lightning import Callback, Trainer
from tqdm import tqdm

from recipes.research.diff import DiffInstrumental
from samantha.data.utils import read_audio
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


def chunks(l, n):
    for i in range(0, n):
        yield l[i::n]


class AudioSampleCallback(Callback):

    def __init__(self, cfg_weight: float, schedule_tau: float):
        super().__init__()
        self.cfg_weight = cfg_weight
        self.schedule_tau = schedule_tau

        # self.fps = [
        #     "/mnt/bn/audio-diffusion/assets/music/01-02 Can You Hear The Music.mp3",
        #     "/mnt/bn/audio-diffusion/assets/music/24. Fellowship.mp3",
        #     "/mnt/bn/audio-diffusion/assets/music/instrumental/42 Rey's Theme.mp3",
        #     "/mnt/bn/audio-diffusion/assets/music/instrumental/26 - Back to the Future - End Credits.mp3",
        #     "/mnt/bn/audio-diffusion/assets/music/instrumental/02 - Chevaliers De Sangreal (From The Da Vinci Code Original Motion Picture Soundtrack).mp3",
        # ]

        self.fps = [
            "/mnt/bn/janne-research-xl/assets/music/01 - The Times They Are A-Changin'.mp3",
            "/mnt/bn/janne-research-xl/assets/music/10 No Surprises.mp3",
            "/mnt/bn/janne-research-xl/assets/music/french/11 - L'amour.mp3",
            # "/mnt/bn/janne-research-xl/assets/music/amy_winehouse/13 - Rehab.mp3",
            "/mnt/bn/janne-research-xl/assets/music/amy_winehouse/06 - Back To Black.mp3",
            "/mnt/bn/janne-research-xl/assets/music/Paul Kalkbrenner - Sky and Sand (Official Music Video).mp3",
            "/mnt/bn/janne-research-xl/assets/music/11 - Billie Eilish - What Was I Made For.mp3",
            # "/mnt/bn/janne-research-xl/assets/music/16 - Billie Eilish - No Time To Die.mp3",
            # "/mnt/bn/janne-research-xl/assets/music/Fred again.., Lil Yachty & Overmono - stayinit.mp3",
            "/mnt/bn/janne-research-xl/assets/music/01-02 Can You Hear The Music.mp3",
            # "/mnt/bn/janne-research-xl/assets/music/24. Fellowship.mp3",
            "/mnt/bn/janne-research-xl/assets/music/instrumental/42 Rey's Theme.mp3",
            "/mnt/bn/janne-research-xl/assets/music/instrumental/26 - Back to the Future - End Credits.mp3",
            "/mnt/bn/janne-research-xl/assets/music/instrumental/Hans Zimmer - Time.mp3",
            "/mnt/bn/janne-research-xl/assets/music/instrumental/02 - Chevaliers De Sangreal (From The Da Vinci Code Original Motion Picture Soundtrack).mp3",
        ]

    def on_validation_end(
        self, trainer: Trainer, pl_module: Union[DiffInstrumental]
    ) -> None:
        if trainer.global_step == 0:
            return

        commit_step = f"step={trainer.global_step}"
        log_dir = trainer.log_dir if trainer.log_dir is not None else ".tmp"
        out_dir = os.path.join(log_dir, commit_step)
        os.makedirs(out_dir, exist_ok=True)

        rank_fps = list(chunks(self.fps, n=trainer.world_size))
        if len(rank_fps) < trainer.global_rank:
            return

        rank_fps = rank_fps[trainer.global_rank]

        for fp in tqdm(rank_fps, desc=f"Sampling to {out_dir}..."):
            audio, sample_rate = read_audio(
                fp, pl_module.config.sample_rate, normalize_loudness=True
            )
            audio = audio.to(pl_module.device)

            t = 50
            with torch.no_grad():
                pred_noise = pl_module.sample_from_audio(
                    audio,
                    sample_rate,
                    t,
                    cfg_weight=self.cfg_weight,
                    schedule_tau=self.schedule_tau,
                )
            pred_audio = pl_module.decode_audio(pred_noise)

            fn = os.path.basename(fp)
            torchaudio.save(
                os.path.join(out_dir, f"{fn}_real.flac"),
                audio[0].cpu(),
                pl_module.config.sample_rate,
            )
            torchaudio.save(
                os.path.join(
                    out_dir,
                    f"{fn}_t={t}_cfg={self.cfg_weight}_tau={self.schedule_tau}_gen.flac",
                ),
                pred_audio[0].cpu(),
                pl_module.config.sample_rate,
            )


class ValidationSampleCallback(Callback):

    def __init__(self, n_batches: int):
        super().__init__()
        self.n_batches = n_batches

    def on_validation_end(
        self, trainer: Trainer, pl_module: Union[DiffInstrumental]
    ) -> None:
        if trainer.global_step == 0:
            return

        commit_step = f"step={trainer.global_step}"
        log_dir = trainer.log_dir if trainer.log_dir is not None else ".tmp"
        out_dir = os.path.join(log_dir, commit_step)
        os.makedirs(out_dir, exist_ok=True)

        t = 50
        schedule_tau = 0.4
        validation_loader = trainer.val_dataloaders

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(validation_loader, desc=f"Sampling to {out_dir}...")
            ):
                if batch_idx == self.n_batches:
                    break

                for idx in range(len(batch.audio)):
                    audio = batch.audio[idx : idx + 1].to(pl_module.device)
                    descriptions = [batch.index[idx]["keywords"]]
                    genres = [batch.index[idx]["final_genre_1"]]

                    for cfg_weight in [1.0, 1.3, 1.5, 2.0, 3.0, 4.0, 7.0]:

                        # pred_noise = pl_module.ddim_sample(audio, pl_module.config.sample_rate, descriptions, genres, t, cfg_weight, schedule_tau)
                        pred_noise = pl_module.sample_from_audio(
                            audio,
                            pl_module.config.sample_rate,
                            t=t,
                            cfg_weight=cfg_weight,
                            schedule_tau=schedule_tau,
                        )
                        pred_audio = pl_module.decode_audio(pred_noise.permute(0, 2, 1))

                        fn = f"{batch_idx}-{idx}"
                        torchaudio.save(
                            os.path.join(out_dir, f"{fn}_real.flac"),
                            audio[0].cpu(),
                            pl_module.config.sample_rate,
                        )
                        torchaudio.save(
                            os.path.join(
                                out_dir,
                                f"{fn}_t={t}_cfg={cfg_weight}_tau={schedule_tau}_gen.flac",
                            ),
                            pred_audio[0].cpu(),
                            pl_module.config.sample_rate,
                        )

                        with open(os.path.join(out_dir, f"{fn}_text.txt"), "wb") as f:
                            f.write("\n".join(descriptions).encode("utf-8"))

