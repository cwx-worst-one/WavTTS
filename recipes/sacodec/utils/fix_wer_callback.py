import pytorch_lightning as pl
from recipes.bigmusic.utils.format_utils import (
    update_json,
)
from pathlib import Path
import json

is_vocal = {'j_01_the_times_they_are_a_changin': True,
 'j_02_can_you_hear_the_music': False,
 'j_03_chevaliers': False,
 'j_04_lamour': False,
 'j_05_hans_zimmer_time': False,
 'j_06_back_to_black': True,
 'j_07_stayinit': False,
 'j_08_reys_theme': False,
 'j_09_fellowship': False,
 'j_10_no_surprises': True,
 'j_11_what_was_i_made_for': True,
 'j_12_back_to_the_future': False,
 'j_13_rehab': True,
 'j_14_sky_and_sand': True,
 'j_16_no_time_to_die': True,
 'sd_1051200': True,
 'sd_1100002': False,
 'sd_1151403': False,
 'sd_1169701': True,
 'sd_12301': False,
 'sd_133600': False,
 'sd_1357005': True,
 'sd_1366203': True,
 'sd_1408401': False,
 'sd_18500': True,
 'sd_246601': False,
 'sd_39105': False,
 'zh_1599310': True,
 'zh_1599313': True,
 'zh_1599363': True,
 'zh_1599368': True,
 'zh_1599432': True,
 'zh_1599470': True,
 'zh_1601784': True,
 'zh_1601910': True,
 'zh_1601917': True,
 'zh_1615002': True,
 'zh_1693299': True,
 'zh_1693467': False,
 'zh_1693554': False,
 'zh_1693586': False,
 'zh_1693653': False,
 'zh_1699660': False,
 'zh_1699688': True,
 'zh_1699708': True,
 'zh_1699839': False,
 'zh_1699933': False,
 'zh_1907349': False,
 'zh_1907668': False,
 'zh_1907674': True,
 'zh_1907757': False,
 'zh_1907887': True,
 'zh_如果生活不爱你-女声': True,
 'zh_如果生活不爱你-男声': True}

class FixWERCallback(pl.Callback):
    def __init__(self, asr_model_path='en_punc', transliteration=False, parallel=1, no_gt_lyrics=False):
        super().__init__()
        self.asr_model_path = asr_model_path
        self.transliteration = transliteration
        self.parallel = parallel
        self.no_gt_lyrics = no_gt_lyrics

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None) -> None:
        # process current chunk
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        run_fix_wer(output_dir)


def run_fix_wer(output_dir):
    output_dir = Path(output_dir)
    try:
        metadata_fps = list(output_dir.glob("**/*.metadata.json"))
        wers = {}
        for metadata_fp in metadata_fps:
            name = metadata_fp.name.split('.')[0]
            if not is_vocal[name]: continue
            with open(metadata_fp, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            wers[name] = metadata['wer']['wer']
        wer_fixed = sum(wers.values()) / len(wers)
        wer_fixed_metadata = { 'wer_fixed': { 'wer_fixed': wer_fixed } }
        update_json(output_dir/'metrics.json', wer_fixed_metadata)
        print(f"output_dir={output_dir}, fad_mulan_distance={wer_fixed_metadata}")
    except Exception as e:
        print("error with file", output_dir, e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    run_fix_wer(args.input_dir)


# python3 recipes/sacodec/utils/fix_wer_callback.py --input_dir /mnt/bn/ashaw-lq/eval/results_sacodec_diff/0707_v7_mos_stats_cfg16/MOS_422/
