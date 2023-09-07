import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import repeat
from mir_eval import transcription, util
from torch import nn

from recipes.mir_benchmark.pl_modules.beat_pl import BaseLightningModule
from recipes.vocal2midi.utils.evaluation import *


class Vocal2MidiModule(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        loss_function,
        model_batch_size,
        scheduler_patience,
        scheduler_decay_factor,
        optimizer_class=torch.optim.Adam,
        target_sample_rate=16000,
        audio_duration=6,
        label_time_resolution=0.02,
        label_hop_sec=3,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.model_batch_size = model_batch_size

        # self.save_hyperparameters()
        # logging.info("Building pytorch lightning model - done")

        self.target_sample_rate = target_sample_rate
        self.audio_duration = audio_duration
        self.label_time_resolution = label_time_resolution
        self.label_hop_sec = label_hop_sec

    def forward(self, waveforms):
        # in lightning, forward defines the prediction/inference actions
        note, onset = self.model({"audio": waveforms, "output_name": "vocal"})

        return note, onset

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        ret = self._shared_step(batch)
        self.log("train_loss", ret["loss"], prog_bar=True, on_step=True, sync_dist=True)
        return ret

    def validation_step(self, batch, batch_idx):
        scores, partial_scores = self._validation_step(batch)
        self.log_dict(scores, prog_bar=True, batch_size=1, sync_dist=True)
        self.log_dict(partial_scores, prog_bar=True, batch_size=1, sync_dist=True)
        return scores

    def test_step(self, batch, batch_idx):
        scores = self._validation_step(batch)
        self.log_dict(scores, prog_bar=True, batch_size=1, sync_dist=True)
        return scores

    def _get_note_list(self, _notes, _onsets, label_hop, original_audio_len_sec):
        # merge pitch segments
        notes = merge_segments(_notes, label_hop)
        onsets = merge_segments(_onsets, label_hop)

        # Remove the padding parts
        notes = notes[
            : int(np.round(original_audio_len_sec / self.label_time_resolution))
        ]
        onsets = onsets[
            : int(np.round(original_audio_len_sec / self.label_time_resolution))
        ]

        _notes = np.zeros((notes.shape[0], 128))
        _onsets = np.zeros((onsets.shape[0], 128))
        _notes[..., 38 : 38 + 60] = notes[..., 1:]
        _onsets[..., 38 : 38 + 60] = onsets

        midi_event = output_dict_to_midi_events(_notes, _onsets, 0.45, 0.35)

        note_list, onset_list = [], []
        for event in midi_event:
            note_list.append(event)
            onset_list.append(
                {"start": event["start"], "end": None, "pitch": event["pitch"]}
            )
        return note_list, onset_list

    def predict_step(self, batch, batch_idx):
        audio_duration_sample = int(self.audio_duration * self.target_sample_rate)
        audio_hop = int(self.target_sample_rate * self.label_hop_sec)
        label_hop = int(self.label_hop_sec / self.label_time_resolution)

        origin_waveforms, _ = batch
        original_audio_len_sec = origin_waveforms.shape[1] / self.target_sample_rate

        # waveform input
        pad_waveforms = pad_audio(origin_waveforms, audio_duration_sample, audio_hop)
        audio_batch = make_frames(pad_waveforms, audio_duration_sample, audio_hop)

        vocal_notes, vocal_onsets, bass_notes, bass_onsets = [], [], [], []
        for b in torch.split(audio_batch, self.model_batch_size):
            (
                p_vocal,
                o_vocal,
                p_bass,
                o_bass,
                p_drums,
                o_drums,
                p_guitar,
                o_guitar,
                p_piano,
                o_pian,
            ) = self(b)
            vocal_notes.append(p_vocal.detach().cpu())
            vocal_onsets.append(o_vocal.detach().cpu())
            bass_notes.append(p_bass.detach().cpu())
            bass_onsets.append(o_bass.detach().cpu())

        output = {}
        output["bass"] = {}
        output["bass"]["pitch"], output["bass"]["onset"] = self._get_note_list(
            bass_notes, bass_onsets, label_hop, original_audio_len_sec
        )
        output["vocal"] = {}
        output["vocal"]["pitch"], output["vocal"]["onset"] = self._get_note_list(
            vocal_notes, vocal_onsets, label_hop, original_audio_len_sec
        )

        return output

    def _shared_step(self, batch):
        origin_waveforms, pitch_labels, onset_labels = batch

        # random perumte the batch for model_batch_size < dataloader_batch_size
        pitch_labels = torch.cat(
            (pitch_labels[:, :, :1], pitch_labels[:, :, 39 : 39 + 60]), -1
        )
        onset_labels = onset_labels[:, :, 39 : 39 + 60]

        note, onset = self(origin_waveforms)
        min_length = min(note.shape[1], pitch_labels.shape[1])

        # Loss
        pitch_loss = torch.mean(
            self.loss_function(note[:, :min_length], pitch_labels[:, :min_length])
        )
        onset_loss = torch.mean(
            self.loss_function(onset[:, :min_length], onset_labels[:, :min_length])
        )

        loss = 1.0 * pitch_loss + 1.0 * onset_loss

        return {"loss": loss, "logits": note, "targets": pitch_labels.int()}

    def _validation_step(self, batch):
        audio_duration_sample = int(self.audio_duration * self.target_sample_rate)
        audio_hop = int(self.target_sample_rate * self.label_hop_sec)
        label_hop = int(self.label_hop_sec / self.label_time_resolution)

        origin_waveforms, gt_note_seq, dataset_name = batch
        dataset_name = dataset_name[0].split("_")[0] + "_"

        # waveform input
        pad_waveforms = pad_audio(origin_waveforms, audio_duration_sample, audio_hop)
        audio_batch = make_frames(pad_waveforms, audio_duration_sample, audio_hop)

        notes, onsets = [], []
        for b in torch.split(audio_batch, self.model_batch_size):
            note, onset = self(b.squeeze(1))
            note = note[:, : int(self.audio_duration / self.label_time_resolution), :]
            onset = onset[:, : int(self.audio_duration / self.label_time_resolution), :]
            notes.append(torch.sigmoid(note).detach().cpu())
            onsets.append(torch.sigmoid(onset).detach().cpu())

        # merge pitch segments
        notes = merge_segments(notes, label_hop)
        onsets = merge_segments(onsets, label_hop)

        # Remove the padding parts
        original_audio_len_sec = origin_waveforms.shape[1] / self.target_sample_rate
        notes = notes[
            : int(np.round(original_audio_len_sec / self.label_time_resolution))
        ]
        onsets = onsets[
            : int(np.round(original_audio_len_sec / self.label_time_resolution))
        ]

        _notes = np.zeros((notes.shape[0], 128))
        _onsets = np.zeros((onsets.shape[0], 128))
        _notes[..., 38 : 38 + 60] = notes[..., 1:]
        _onsets[..., 38 : 38 + 60] = onsets

        note_list = output_dict_to_midi_events(_notes, _onsets, 0.45, 0.35)
        # 0.45, 0.35 -> vocal
        ref_intervals, ref_pitches = [], []
        for n in gt_note_seq[0]:
            ref_intervals.append([n["start"], n["end"]])
            ref_pitches.append(n["pitch"])
        ref_intervals, ref_pitches = np.array(ref_intervals), np.array(ref_pitches)

        est_intervals, est_pitches = [], []
        for n in note_list:
            est_intervals.append([n["start"], n["end"]])
            est_pitches.append(n["pitch"])
        est_intervals, est_pitches = np.array(est_intervals), np.array(est_pitches)

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
        partial_scores = {dataset_name + k: v for k, v in scores.items()}
        return scores, partial_scores
