from sacrebleu import corpus_bleu
import json 
import numpy as np 
import os 
from collections import defaultdict
from pathlib import Path
import threading
import sys 
import torch
import subprocess
from recipes.musiclm.inference.utils import load_wav
from logging import getLogger
import io 
import soundfile as sf 
import pytorch_lightning as pl
from recipes.bigmusic.utils.format_utils import update_json

logger = getLogger(__name__)
BIGMUSIC_SAMI_MODELS_FOUND=False
BASE_BATCH_SIZE = 1  # 1 for V100-32G; 4 for A100-80G

try: 
    import sami_models
    BIGMUSIC_SAMI_MODELS_FOUND = True 
except ImportError as e:
    logger.error(f"`bigmusic_sami_models` library was not found in PYTHONPATH {e}")


class Vocal2MidiCallback(pl.Callback):
    def __init__(self, sr = 24000):
        super().__init__()
        self.sr = sr 

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

        run_vocal2midi_metrics(generated_output_fps=generated_output_fps, sr=self.sr)


class Vocal2MidiPredictor:
    def __init__(self, batch_size=16, tasks=['vocal2midi']):
        # sys.path.insert(0, "/mnt/bn/lf-yiqinglu/code/bigmusic_sami_models")
        from bigmusic_sami_models_path import MODEL_WEIGHT_PATH
        ckpt_prefix = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/sami_models'
        self.logger = getLogger(__name__)

        local_weight_paths = {}
        for k, path in MODEL_WEIGHT_PATH.items():
            if 'vocal2midi' in k:
                local_weight_paths[k] = get_hdfs_file(ckpt_prefix, path, self.logger)

        # from sami_models.mir_models.perceiver_5stem.mir_perceiver_5stem_model import \
        #     MIRPerceiver5StemModel
        # from sami_models.mir_models.perceiver_12stem.mir_perceiver_12stem_model import \
        #     MIRPerceiverTFModel
        # from sami_models.mir_models.perceiver_vocal.mir_perceiver_vocal_model import \
        #     MIRPerceiverVocalModel
        # from sami_models.mir_models.sheetdoctor.mir_sheetdoctor_model import \
        #     MirSheetdoctorModel
        # from sami_models.mir_models.structure.mir_structure_model import \
        #     StructureSpecTntModel
        from sami_models.mir_models.vocal2midi.mir_vocal2midi_model import \
            MirVocalMelodyExtractionModel

        self.batch_size = batch_size

        self.tasks = tasks
        num_gpus = torch.cuda.device_count()
        self.device = torch.device('cuda') if num_gpus > 0 else torch.device('cpu')
        # self.device = torch.cuda.current_device()
        if isinstance(self.device, list):
            self.device = self.device[0]

        self.models = {}
        for task in tasks:
            if task in ['beat', 'chord', 'key', 'vocalbeat', 'vocalkey']:
                # batch_size depends on input audio length
                if task == 'chord':
                    self.models[task] = MirSheetdoctorModel(
                        model_path=local_weight_paths[task],
                        beat_model=self.models['beat'],
                        task=task,
                        map_location=self.device)
                else:
                    self.models[task] = MirSheetdoctorModel(
                        model_path=local_weight_paths[task],
                        task=task,
                        map_location=self.device)
            elif task == 'structure':
                self.models[task] = StructureSpecTntModel(
                    config_path=local_weight_paths['structure_config'],
                    model_path=local_weight_paths['structure_downbeat'],
                    batch_size=BASE_BATCH_SIZE * 16,
                    map_location=self.device)
            elif task == 'vocal2midi':
                self.models[task] = MirVocalMelodyExtractionModel(
                    model_path=local_weight_paths['vocal2midi'],
                    batch_size=BASE_BATCH_SIZE * 2,
                    map_location=self.device)
            elif task == 'trans_5stem':
                self.models[task] = MIRPerceiver5StemModel(
                    model_path=local_weight_paths['perceiver_5stem'],
                    batch_size=BASE_BATCH_SIZE * 2,
                    map_location=self.device)
            elif task == 'trans_vocal':
                self.models[task] = MIRPerceiverVocalModel(
                    model_path=local_weight_paths['perceiver_vocal'],
                    batch_size=BASE_BATCH_SIZE * 2,
                    map_location=self.device)
            elif task == 'trans_12stem':
                self.models[task] = MIRPerceiverTFModel(
                    model_path=local_weight_paths['perceiver_12stem'],
                    batch_size=BASE_BATCH_SIZE * 2,
                    map_location=self.device)

        self.lock = threading.Lock()

    @torch.no_grad()
    def predict_one(self, wav_bytes):

        with self.lock:
            output = {}
            for task, model in self.models.items():
                out_dict = model.predict(wav_bytes)
                try:
                    json_temp = json.dumps(out_dict, cls=json_serialize)
                    output[task] = json.loads(json_temp)
                except Exception:
                    print(f"(Non JSON serializable) Failed to convert {task} to json")

            return output


class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def get_hdfs_file(hdfs_prefix, file_path, logger, local_prefix='/tmp'):
    local_path = os.path.join(local_prefix, os.path.dirname(file_path))
    os.makedirs(local_path, exist_ok=True)
    local_filename = os.path.join(local_prefix, file_path)
    if not os.path.exists(local_filename):
        hdfs_path = os.path.join(hdfs_prefix, file_path)
        subprocess.call(["hdfs", "dfs", "-get", hdfs_path, local_path])
        logger.info(f"Downloaded {hdfs_path} into {local_path}")
    return local_filename




def load_json(path, preprocess=True, no_last_sil=True):
    if preprocess:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = path

    notes = data.get('notes')
    duration = data.get('end_time')
    note = []
    for line in notes:
        note += [[line['start'], line['end'], line['pitch']]]
        
    start = 0
    res = []
    for item in note:
        if item[0] != start:
            res += [[start, item[0], 'sp']]
        res += [item]
        start = item[1]

    if not no_last_sil:
        if res[-1][1] != duration:
            res += [[res[-1][1], duration, 'sp']]

    return res 


def change_dur_to_seq(res, unit=0.25):
    seq = []
    true_seq = []
    for i in res:
        pit = i[2]
        dur = i[1] - i[0]
        dur_s = round(dur / unit) * unit
        seq += [pit, dur_s]
        true_seq += [pit, dur]

    return seq, true_seq


def upsample_to_umm_rate(seq, umm_token_rate=25):
    dur = seq[-1][1]
    arr = np.zeros(int(round(umm_token_rate * dur)))

    for item in seq:
        if item[-1] == 'sp':
            continue
        start, end = round(item[0] * 25), round(item[1] * 25)
        arr[start: end] = int(item[-1])

    if len(arr) < 10:
        return [0], ''
    # print(len(arr))
    # strip the zeros at head and tail 
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] != 0:
            break

    pitch = arr[:i + 1]
    res = []
    for i in pitch:
        if i == 0:
            # res += ['sp']
            res += ['0']
        else:
            res += [str(int(i))]
    
    return pitch, ' '.join(res)



def bleu_v0(pred_json_path, gt_json_path, need_process=True, unit=0.25):
    seq1 = load_json(pred_json_path, need_process)
    seq2 = load_json(gt_json_path, need_process)

    if len(seq2) == 0:
        return 0, '', ''

    tgt, true_tgt = change_dur_to_seq(seq2, unit=unit)
    pred, true_pred = change_dur_to_seq(seq1, unit=unit)

    pred = ' '.join([str(i) for i in pred])
    tgt = ' '.join([str(i) for i in tgt])
    # bleu 4 default
    bleu = corpus_bleu([pred], [[tgt]], tokenize='none')
    
    return bleu, tgt, pred


def bleu_v1(pred_json_path, gt_json_path, need_process=True):
    seq1 = load_json(pred_json_path, need_process)
    seq2 = load_json(gt_json_path, need_process)
    if len(seq2) == 0:
        return 0, '', ''

    pred_pitch, pred_trans_pitch = upsample_to_umm_rate(seq1)
    tgt_pitch, tgt_trans_pitch = upsample_to_umm_rate(seq2)
    bleu = corpus_bleu([pred_trans_pitch], [[tgt_trans_pitch]], tokenize='none')
    return bleu, tgt_trans_pitch, pred_trans_pitch


def run_vocal2midi_metrics(generated_output_fps, sr=24000):
    predictor = Vocal2MidiPredictor()
    UNIT = 0.25
    category2wer = defaultdict(list)
    count = 0

    for idx, generated_output_fp in enumerate(generated_output_fps):
        gt_output_fp = generated_output_fp.parent / generated_output_fp.stem.replace('generated', 'target_audio.wav')
        wav = load_wav(str(generated_output_fp))
        gt_wav = load_wav(str(gt_output_fp))
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        audio_buffer = io.BytesIO()
        sf.write(file=audio_buffer, data=wav, samplerate=sr, format='WAV')    
        audio_buffer.seek(0)
        result_json = predictor.predict_one(audio_buffer.getvalue())
        gen_json = result_json['vocal2midi']

        audio_buffer.seek(0)
        audio_buffer.truncate()
        sf.write(file=audio_buffer, data=gt_wav, samplerate=sr, format='WAV')    
        audio_buffer.seek(0)
        gt_json = predictor.predict_one(audio_buffer.getvalue())['vocal2midi']

        bleu, tgt_trans_pitch, pred_trans_pitch = bleu_v0(gen_json, gt_json, need_process=False, unit=UNIT)
        if bleu == 0 and tgt_trans_pitch == pred_trans_pitch == '':
            count += 1 
            continue
           
        score = bleu.score / bleu.bp if bleu.bp < 1 else bleu.score

        midi_metadata = {
            'bleu4' : score,
            'bleu_check_method': '0',
            'predict_transcript': pred_trans_pitch,
            'tgt_transcript': tgt_trans_pitch,
        }


        update_json(metadata_fp, { 'vocal2midi_metadata': midi_metadata })

        if isinstance(generated_output_fp, str):
            category_dir = os.path.dirname(os.path.dirname(generated_output_fp))

        else:
            category_dir = generated_output_fp.parent.resolve()

        if score <= 101:
            category2wer[str(category_dir)].append([score]) # append to base directory to calculate total wer
        else:
            count += 1

        # update total metrics
    for dir_path, wers in category2wer.items():
        metrics_fp = Path(dir_path)/'vocal2midi_metrics.json'
        avg_score = np.array(wers).mean()
        print(np.array(wers).shape)
        wer_metadata = {
            'bleu4': round(avg_score, 3),
        }
        update_json(metrics_fp, {'avg_bleu4': wer_metadata, "abandon_count" : count})
        print(f"output_dir={str(category_dir)}, avgBLEU4={wer_metadata}")



def run_vocal2midi_metrics_with_results(generated_output_fps,  method=0):
    category2wer = defaultdict(list)
    count = 0
    for idx, generated_output_fp in enumerate(generated_output_fps):
        gt_fp = generated_output_fp.parent.parent / 'target_results' / generated_output_fp.name
        # test
        metadata_fp = os.path.join(generated_output_fp.parent.parent, generated_output_fp.stem.replace('vocal2midi', '')+'.wav.metadata.json')

        if not method:
            bleu, tgt_trans_pitch, pred_trans_pitch = bleu_v0(generated_output_fp, gt_fp)
        else:
            bleu, tgt_trans_pitch, pred_trans_pitch = bleu_v1(generated_output_fp, gt_fp)

        if bleu == 0 and tgt_trans_pitch == pred_trans_pitch == '':
            count += 1 
            continue
           
        score = bleu.score / bleu.bp if bleu.bp < 1 else bleu.score

        midi_metadata = {
            'bleu4' : score,
            'bleu_check_method': method,
            'predict_transcript': pred_trans_pitch,
            'tgt_transcript': tgt_trans_pitch,
        }


        update_json(metadata_fp, { 'vocal2midi_metadata': midi_metadata })

        if isinstance(generated_output_fp, str):
            category_dir = os.path.dirname(os.path.dirname(generated_output_fp))
            # category_dir = os.path.dirname(generated_output_fp)

        else:
            category_dir = generated_output_fp.parent.parent.resolve()

        if score <= 101:
            category2wer[str(category_dir)].append([score]) # append to base directory to calculate total wer
        else:
            count += 1

        # update total metrics
    for dir_path, wers in category2wer.items():
        metrics_fp = Path(dir_path)/'vocal2midi_metrics.json'
        avg_score = np.array(wers).mean()
        print(np.array(wers).shape)
        wer_metadata = {
            'bleu4': round(avg_score, 3),
        }
        update_json(metrics_fp, {'avg_bleu4': wer_metadata, "abandon_count" : count})
        print(f"output_dir={str(category_dir)}, avgBLEU4={wer_metadata}")


if '__main__' == __name__:
    import argparse
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    run_vocal2midi_metrics(list(Path(args.input_dir).glob('**/*generated.wav')))


