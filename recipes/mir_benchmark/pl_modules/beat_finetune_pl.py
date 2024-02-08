import torch
from recipes.mir_benchmark.pl_modules.beat_pl import LitBeat
from einops import rearrange


class LitFinetuneBeat(LitBeat):
    def __init__(
        self,
        model,
        lr=0.001,
        scheduler_patience=20,
        scheduler_decay_factor=0.5,
        hop_length=240,
        beat_window_length=2,
        sample_rate=24000,
        sample_len=6,
        label_hop=0.02,
        n_beats=3,
        n_tempo=300,
        hop_factor=2,
        do_tempo_loss=False,
        f_measure_threshold=0.07,
        use_tempo_prior=False,
        model_batch_size=32,
        pretrained_path=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
            hop_length=hop_length,
            beat_window_length=beat_window_length,
            n_beats=n_beats,
            n_tempo=n_tempo,
            hop_factor=hop_factor,
            sample_rate=sample_rate,
            sample_len=sample_len,
            label_hop=label_hop,
            do_tempo_loss=do_tempo_loss,
            f_measure_threshold=f_measure_threshold,
            use_tempo_prior=use_tempo_prior,
            model_batch_size=model_batch_size,
            pretrained_path=pretrained_path,
        )
        self._beat_sample_len = 6
        self._audio_sample_len = int(self._sample_rate * self._beat_sample_len)
        self._label_sample_len = int(1 / self._beat_label_hop * self._beat_sample_len)

    def _beat_training_step(self, batch):
        inputs = {}
        audio = batch[0].unfold(1, self._audio_sample_len, self._audio_sample_len)
        b, c, t = audio.shape
        audio = audio.reshape(-1, self._audio_sample_len)

        inputs["audio"] = audio
        inputs["aug_hop_size"] = self._hop_length

        # model prediction
        oup = self.model(inputs)[0]
        beat_pred, _ = oup['beat']
        _, t, f = beat_pred.shape
        beat_pred = beat_pred.reshape(b, c, t, f).reshape(b, -1, f)

        beat_tar = batch[1]

        loss = self._train_beat(
            beat_tar,
            self._hop_length,
            inputs["aug_hop_size"],
            beat_pred,
            self._beat_window_length,
            self._n_beats,
            self._n_tempo,
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._beat_training_step(batch)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        return super().predict_step(batch, batch_idx)
