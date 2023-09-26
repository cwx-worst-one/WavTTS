import pytorch_lightning as pl
from typing import Any
from recipes.bigmusic.utils.format_utils import concat_metadata_list, update_json
from pathlib import Path
import numpy as np
from recipes.mir_benchmark.tagging_inference.genre import GenreTagging
from recipes.mir_benchmark.tagging_inference.instrument import InstrumentTagging
from recipes.mir_benchmark.tagging_inference.vocal import VocalTagging

class MIRTagMetricsCallback(pl.Callback):
    def __init__(self, tags="genre,instrument,vocal"):
        super().__init__()
        tagging_models = {}
        if "genre" in tags:
            tagging_models['genre'] = GenreTagging()
        if "instrument" in tags:
            tagging_models['instrument'] = InstrumentTagging()
        if "vocal" in tags:
            tagging_models['vocal'] = VocalTagging()
        self.tagging_models = tagging_models
        self.batch_accuracies = { tag: [] for tag in tagging_models.keys() }

    @staticmethod
    def gt_model_predict(model, target_audio, generated_audio, device):
        model.to(device) # to fix bug in pytorch lightning where "setup" does not have correct device yet
        if target_audio.shape == 2:
            target_audio = target_audio.unsqueeze(1)
        if generated_audio.shape == 2:
            generated_audio = generated_audio.unsqueeze(1)
        gt_tags, gt_preds = model.predict(target_audio.to(device))
        gen_tags, gen_preds = model.predict(generated_audio.to(device))
        accuracies = MIRTagMetricsCallback.multilabel_accuracy(gt_tags, gen_tags)
        return accuracies

    @staticmethod
    def semantic_model_predict(model, target_tags, generated_audio, device):
        model.to(device) # to fix bug in pytorch lightning where "setup" does not have correct device yet
        target_tags = [tags.split(',') if isinstance(tags, str) else tags for tags in target_tags]
        if generated_audio.shape == 2:
            generated_audio = generated_audio.unsqueeze(1)
        gen_tags, gen_preds = model.predict(generated_audio.to(device))
        accuracies = MIRTagMetricsCallback.multilabel_accuracy(target_tags, gen_tags)
        return accuracies

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # for now, calculate using target audio and generate audio
        device = pl_module.device
        generated_audio = outputs['generated_audio_tensor']
        batch_size = generated_audio.shape[0]
        if 'target_audio' in batch: # Ground Truth use case
            target_audio = batch['target_audio']
            tag_metadata = {}
            for tag, model in self.tagging_models.items():
                accuracies = MIRTagMetricsCallback.gt_model_predict(model, target_audio, generated_audio, device)
                accuracy_metadata = [ { tag: round(x, 3) } for x in accuracies ]
                tag_metadata = concat_metadata_list(tag_metadata, accuracy_metadata)
            self.batch_accuracies[tag].extend(accuracies)
            outputs['metadata'] = concat_metadata_list(outputs.get('metadata'), tag_metadata)
        elif 'style_text' in batch: # style_text use case
            pass # TODO (AS) finish tagging for MIR


    @staticmethod
    def multilabel_accuracy(pred_labels, target_labels):
        accuracies = []
        for pred, target in zip(pred_labels, target_labels):
            pred_set = set(pred)
            target_set = set(target)
            target_set - pred_set
            correct = len(pred_set & target_set)
            if len(target_set) == 0:
                accuracy = 1
            else:
                accuracy = correct / len(target_set)
            accuracies.append(accuracy)
        return accuracies
    
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tag_metadata = { tag: np.mean(tag_accuracies) for tag, tag_accuracies in self.batch_accuracies.items() }
        output_dir = pl_module.extra_params.output_dir
        metrics_fp = Path(output_dir)/'metrics.json'
        update_json(metrics_fp, { 'MIR tag acc': tag_metadata })
