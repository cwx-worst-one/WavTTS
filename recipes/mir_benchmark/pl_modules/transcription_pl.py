import numpy as np
import torch
from mir_eval import transcription, util
from torch.nn.functional import binary_cross_entropy_with_logits

from recipes.beat.pl_modules.pl_module import BaseLightningModule
from recipes.mir_benchmark.utils.transcription_processor import SLAKH_INSTRUMENTS
from recipes.transcription.utils.eval_utils import merge_and_sort, undo_make_frames_avg
from recipes.vocal2midi.utils.evaluation import (
    make_frames,
    output_dict_to_midi_events,
    pad_audio,
)


class LitTranscription(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        hop_length,
        sample_rate,
        sample_len,
        label_hop,
        sample_hop_sec=1,
        max_batch=100,
        do_cqt_augmentation=False,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._max_batch = max_batch
        self._hop_length = hop_length
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self._label_hop = label_hop
        self._sample_hop_sec = sample_hop_sec
        self._do_cqt_augmentation = do_cqt_augmentation

    def training_step(self, batch, batch_idx):
        audio = batch[0].sum(dim=1)
        note_tar = batch[1]
        onset_tar = batch[2]

        # model prediction
        note_pre, onset_pre = self.model(audio)

        loss = binary_cross_entropy_with_logits(note_pre, note_tar)
        loss += binary_cross_entropy_with_logits(onset_pre, onset_tar)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_duration_sample = int(self._sample_len * self._sample_rate)
        audio_hop = int(self._sample_rate * self._sample_hop_sec)
        label_hop = int(self._sample_hop_sec / self._label_hop)

        origin_waveforms, gt_note_seq = batch

        # waveform input
        pad_waveforms = pad_audio(origin_waveforms, audio_duration_sample, audio_hop)
        audio_batch = make_frames(
            pad_waveforms, audio_duration_sample, audio_hop
        ).float()
        notes, onsets = [], []
        for b in torch.split(audio_batch, self._max_batch):
            inputs = {}
            inputs["audio"] = b.squeeze(1)
            inputs["aug_hop_size"] = self._hop_length
            note_pre, onset_pre = self.model(inputs)
            notes.append(note_pre.detach().cpu())
            onsets.append(onset_pre.detach().cpu())

        # merge pitch segments
        notes = undo_make_frames_avg(notes, label_hop)
        onsets = undo_make_frames_avg(onsets, label_hop)

        # Remove the padding parts
        original_audio_len_sec = origin_waveforms.shape[1] / self._sample_rate
        notes = notes[: int(np.round(original_audio_len_sec / self._label_hop))]
        onsets = onsets[: int(np.round(original_audio_len_sec / self._label_hop))]

        ref_intervals, ref_pitches = [], []
        est_intervals, est_pitches = [], []
        for i, inst in enumerate(SLAKH_INSTRUMENTS):
            if inst == "Drums" or inst not in gt_note_seq:
                continue
            note_list = output_dict_to_midi_events(notes[i], onsets[i], 0.25, 0.25)
            ref_interval, ref_pitch = [], []
            for n in gt_note_seq[inst]:
                ref_interval.append([n["start"], n["end"]])
                ref_pitch.append(n["pitch"])
            ref_intervals.append(ref_interval)
            ref_pitches.append(ref_pitch)

            est_interval, est_pitch = [], []
            for n in note_list:
                est_interval.append([n["start"], n["end"]])
                est_pitch.append(n["pitch"])
            est_intervals.append(est_interval)
            est_pitches.append(est_pitch)

        ref_intervals, ref_pitches = merge_and_sort(ref_intervals, ref_pitches)
        est_intervals, est_pitches = merge_and_sort(est_intervals, est_pitches)

        # midi 2 hz
        ref_pitches = util.midi_to_hz(ref_pitches)
        est_pitches = util.midi_to_hz(est_pitches)

        # eval
        scores = {}
        if len(est_intervals) == 0:
            scores["onset_pitch_offset_f1"] = 0
            scores["onset_pitch_f1"] = 0
            scores["onset_f1"] = 0
        else:
            raw_data = transcription.evaluate(
                ref_intervals,
                ref_pitches,
                est_intervals,
                est_pitches,
                onset_tolerance=0.05,
                pitch_tolerance=50,
            )

            scores["onset_pitch_offset_f1"] = raw_data["F-measure"]
            scores["onset_pitch_f1"] = raw_data["F-measure_no_offset"]
            scores["onset_f1"] = raw_data["Onset_F-measure"]

        self.log_dict(scores, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(scores)
