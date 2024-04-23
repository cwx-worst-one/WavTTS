from collections import defaultdict

import numpy as np
import torch
import subprocess
import scipy
from torch.nn.functional import binary_cross_entropy_with_logits

from recipes.beat.pl_modules.pl_module import BaseLightningModule
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs, merge_multiple_temporal_probs_inference
from recipes.chord.utils.mireval_chord import get_chord_score, idx2chord, idx2triad


class LitChord(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        hop_length,
        chord_pool,
        sample_rate,
        sample_len,
        resnet_pools,
        hop_factor=6,
        do_cqt_augmentation=False,
        pretrain_path=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._hop_length = hop_length
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self._label_chord_hop = (
            hop_length
            * np.prod([p for p in chord_pool])
            * np.prod([p[1] for p in resnet_pools])
        ) / sample_rate
        self._hop_factor = hop_factor
        self._do_cqt_augmentation = do_cqt_augmentation

        if pretrain_path is not None:
            if pretrain_path.startswith("hdfs://"):
                name = pretrain_path.split("/")[-1]
                subprocess.run(f"hdfs dfs -get {pretrain_path}", shell=True)
            else:
                name = pretrain_path
            checkpoint = torch.load(name)
        
            for key in list(checkpoint["state_dict"]):
                if "teacher" in key:
                    checkpoint["state_dict"].pop(key)
                else:
                    checkpoint["state_dict"][key[6:]] = checkpoint["state_dict"].pop(key)

            model_dict = self.model.state_dict()
            model_dict.update(checkpoint["state_dict"]) 
            self.model.load_state_dict(model_dict)


    def training_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}
        data = {}

        inputs["audio"] = batch[0]
        inputs["aug_hop_size"] = self._hop_length

        data["chord_root"] = batch[1]
        data["chord_triad"] = batch[2]
        data["chord_note"] = batch[3]
        data["chord_ignore"] = batch[4]

        if self._do_cqt_augmentation:
            self.model.stages[0], note_idx = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1
            )

            data["chord_root"][..., 1:] = torch.roll(
                data["chord_root"][..., 1:], note_idx * -1, -1
            )
            data["chord_note"][..., :12] = torch.roll(
                data["chord_note"][..., :12], note_idx * -1, -1
            )
            data["chord_note"][..., 12:24] = torch.roll(
                data["chord_note"][..., 12:24], note_idx * -1, -1
            )

        # model prediction
        outputs["chord_root"], outputs["chord_triad"] = self.model(inputs)

        loss = self._train_chord(outputs, data)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}
        data = {}

        if self._do_cqt_augmentation:
            self.model.stages[0], _ = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1, note="C1"
            )

        sample = batch[0]
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        sample = sample.unfold(
            1, self._train_num_samples, int(self._train_num_samples / self._hop_factor)
        )

        inputs["audio"] = sample.squeeze(0).float()
        inputs["aug_hop_size"] = self._hop_length

        data["chord_root"] = batch[1]
        data["chord_triad"] = batch[2]
        data["chord_note"] = batch[3]
        data["chord_ignore"] = batch[4]

        outputs["chord_root"], outputs["chord_triad"] = self.model(inputs)

        processed_data = self._postprocess_chord(
            data,
            outputs,
            int(self._sample_len / self._label_chord_hop),
            int(self._sample_len / self._hop_factor / self._label_chord_hop),
        )

        scores = get_chord_score(
            processed_data["pred"], processed_data["truth"], self._label_chord_hop
        )

        # Filter None
        scores = {k: v for k, v in scores.items() if v is not None}
        self.log_dict(scores, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(scores)

    def predict_step(self, batch, batch_idx):
        if type(batch) == dict:
            sample = batch["target_audio"]
        else:
            sample = batch[0]
        bsz = sample.shape[0]
        # Remove channel, assume mono
        if sample.ndim == 3:
            sample = sample.squeeze(1)

        duration = sample.shape[-1] / self._sample_rate

        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        # (bsz, num_segments, seg_len)
        sample = sample.unfold(
            -1, self._train_num_samples, int(self._train_num_samples / self._hop_factor)
        )
        num_segments = sample.shape[1]
        # (bsz * num_segments, seg_len)
        sample = sample.reshape(bsz * num_segments, -1).float()

        inputs = {}
        inputs["audio"] = sample
        inputs["aug_hop_size"] = self._hop_length

        # (bsz * num_segments, num_frames, output_dim)
        batch_roots, batch_triads = self.model(inputs)
        # (bsz, num_segments, num_frames, output_dim)
        batch_roots = batch_roots.reshape(bsz, num_segments, batch_roots.shape[1], -1)
        batch_triads = batch_triads.reshape(bsz, num_segments, batch_triads.shape[1], -1)

        batch_labels = []
        for root, triad in zip(batch_roots, batch_triads):
            root = merge_multiple_temporal_probs_inference(
                torch.softmax(root, -1), 
                root.shape[-1], 
                int(np.ceil(duration / self._label_chord_hop)), 
                int(self._sample_len / self._label_chord_hop), 
                int(self._sample_len / self._label_chord_hop / self._hop_factor)
            ).cpu().numpy()

            triad = merge_multiple_temporal_probs_inference(
                torch.softmax(triad, -1), 
                triad.shape[-1], 
                int(np.ceil(duration / self._label_chord_hop)), 
                int(self._sample_len / self._label_chord_hop), 
                int(self._sample_len / self._label_chord_hop / self._hop_factor)
            ).cpu().numpy() 

            root_prob = np.max(root, axis=-1)               # the highest vluae is considered as probability
            # root_prob[np.argmax(root, axis=-1)==0] = 0    # index=0 is "N" label, which should set probility to 0
            triad_prob = np.max(triad, axis=-1)             # the highest vluae is considered as probability
            # triad_prob[np.argmax(triad, axis=-1)==0] = 0  # index=0 is "N" label, which should set probility to 0
            chord_prob = (root_prob + triad_prob) / 2
            chord_prob[np.isnan(chord_prob)] = 0

            root = np.argmax(root, -1)
            triad = np.argmax(triad, -1)
            root = scipy.signal.medfilt(root, kernel_size=9)
            triad = scipy.signal.medfilt(triad, kernel_size=9)

            labels = []
            i_prev = 0
            for i in range(len(root)):
                if idx2chord[root[i]] == "N" or idx2triad[triad[i]] == "N":
                    label = "N"
                else:
                    triads = idx2triad[triad[i]]
                    label = f"{idx2chord[root[i]]}:{triads}"

                if i*self._label_chord_hop >= duration:
                    break

                if len(labels) == 0:
                    labels.append([i*self._label_chord_hop, (i+1)*self._label_chord_hop, label, 0])
                elif label == labels[-1][2]:
                    labels[-1][1] = (i+1)*self._label_chord_hop
                else:
                    prob = np.mean(chord_prob[i_prev: i])
                    i_prev = i
                    labels[-1][-1] = prob
                    labels.append([i*self._label_chord_hop, (i+1)*self._label_chord_hop, label, 0])
                if i == len(root)-1:
                    prob = np.mean(chord_prob[i_prev:])
                    labels[-1][-1] = prob = prob
            batch_labels.append(labels)

        return batch_labels

    def _train_chord(self, prediction, target, crf=None):
        chord_loss = 0

        eval_name = ["chord_root", "chord_triad"]
        for name in eval_name:
            chord_pre, chord = prediction[name], target[name]
            chord_index = chord.sum(1).sum(1) > 0
            if len(chord_pre[chord_index, :]) > 0:
                min_length = min(chord_pre.shape[1], chord.shape[1])
                chord_pre = chord_pre[:, :min_length]
                chord = chord[:, :min_length]
                ignore = target["chord_ignore"][:, :min_length]

                weight = (1 - ignore).unsqueeze(-1).repeat(1, 1, chord.shape[-1])
                if name == "chord_root":
                    note_weight = torch.ones(chord.shape).to(chord.device)
                    note_weight[..., 1:] += target["chord_note"]
                    weight *= note_weight

                loss = binary_cross_entropy_with_logits(chord_pre, chord, weight)
                chord_loss += loss

        return chord_loss

    def _postprocess_chord(self, target, prediction, sample_len, sample_hop):
        eval_name = ["chord_root", "chord_triad"]
        processed_data = defaultdict(dict)
        for name in eval_name:
            chord_pre, chord = prediction[name], target[name]
            chord = chord.squeeze()
            chord_pre = merge_multiple_temporal_probs(
                chord_pre, chord, sample_len, sample_hop
            )
            chord_pre = torch.softmax(chord_pre, -1)
            min_length = min(chord_pre.shape[0], chord.shape[0])
            chord_labels = chord[:min_length]
            chord_preds = chord_pre[:min_length]
            processed_data["truth"][name] = chord_labels.cpu().numpy()
            processed_data["pred"][name] = chord_preds.cpu().numpy()

        processed_data["truth"]["chord_ignore"] = (
            target["chord_ignore"].squeeze(0).cpu().numpy()
        )

        return processed_data
