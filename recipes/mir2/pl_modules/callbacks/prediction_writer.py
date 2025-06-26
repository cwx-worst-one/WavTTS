from pytorch_lightning.callbacks import BasePredictionWriter
import numpy as np
import os
import pandas as pd
import json

from recipes.mir2.utils.midi_utils import process_transcription_result, note2midi, chord_to_midi

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ChordWriter(BasePredictionWriter):
    r"""Writes beat prediction to disk.
    """

    def __init__(self, output_dir=None) -> None:
        super().__init__()
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if self.output_dir is not None:
            output_path = os.path.join(self.output_dir, batch[1][0]+'.txt')
            with open(output_path, 'w') as f:
                for line in prediction:
                    f.write(f'{str(line[0])}\t{str(line[1])}\t{str(line[2])}\n')


class StructureWriter(BasePredictionWriter):
    r"""Writes beat prediction to disk.
    """

    def __init__(self, output_dir=None) -> None:
        super().__init__()
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, batch[1][0]+'.txt'), 'w') as f:
                for line in prediction:
                    f.write(f"{str(line['interval'][0])}\t{str(line['interval'][1])}\t{str(line['funct_name'])}\n")

class TaggingWriter(BasePredictionWriter):
    r"""Writes beat prediction to disk.
    """

    def __init__(self, output_dir=None) -> None:
        super().__init__()
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if self.output_dir is not None:
            output_json = os.path.join(self.output_dir, batch[1][0]+'.json')
            if pl_module.final_pooling:
                json.dump(prediction, open(output_json, 'w'), indent=2,  cls=json_serialize)
            else:
                json.dump(prediction, open(output_json, 'w'), cls=json_serialize)


class MultiTaskWriter(BasePredictionWriter):
    r"""Writes beat prediction to disk.
    """

    def __init__(self, tasks, output_dir=None) -> None:
        super().__init__()
        self.tasks = tasks
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def call_json2midi(self, json_data, out_fn):
        try:
            process_transcription_result(json_data, out_fn)
        except:
            pass

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if self.output_dir is None:
            return

        save_path = os.path.join(self.output_dir, batch[1][0])
        os.makedirs(save_path, exist_ok=True)

        for task in self.tasks:
            if task not in prediction:
                continue

            if task in ['trans_5stem', 'trans_12stem']: 
                self.call_json2midi(prediction[task], os.path.join(save_path, f"{task}.mid"))
            elif task == 'vocal2midi': 
                note2midi(prediction[task]['notes'], prediction[task]['end_time'], os.path.join(save_path, f"{task}.mid"))
            elif task == 'trans_vocal':
                note2midi(prediction[task]['notes']['vocal'], prediction[task]['end_time'], os.path.join(save_path, f"{task}.mid"))
            elif task in ['beat', 'vocalbeat']: 
                bb = pd.DataFrame(prediction[task])
                if len(prediction[task]) > 0:
                    bb = bb.astype({0:float, 1:int})
                    bb.to_csv(os.path.join(save_path, "beat.txt"), sep='\t', index=False, header=False)
                #open(os.path.join(save_path, "tempo.txt"), 'w').write(str(prediction[task]['tempo']))
            elif task == 'chord':
                pd.DataFrame(prediction[task]).to_csv(os.path.join(save_path, "chord.txt"), sep='\t', index=False, header=False)
                chord_result = {'intervals': [[t[0], t[1]] for t in prediction[task]], 'chords': [t[2] for t in prediction[task]]}
                chord_to_midi(chord_result, os.path.join(save_path, "chord.mid"))
            elif task == 'structure':
                #segs = [(t[0][0], t[1]) for t in prediction[task]] + [(prediction[task][-1][0][1], 'end')]
                segs = [(line['interval'][0], line['funct_name']) for line in prediction[task]] + [(prediction[task][-1]['interval'][1], 'end')]
                pd.DataFrame(segs).to_csv(os.path.join(save_path, "structure.txt"), sep='\t', index=False, header=False)
            elif task == 'key':
                pd.DataFrame(prediction[task]).to_csv(os.path.join(save_path, "key.txt"), sep='\t', index=False, header=False)
           