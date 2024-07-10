from pathlib import Path
from collections import defaultdict
from recipes.bigmusic.utils.format_utils import update_json
import json
from recipes.musiclm.utils.dist import local_zero_last, is_local_zero
import pytorch_lightning as pl

def convert_values_to_lists(nested_dict):
    def convert_dict(d):
        converted_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                converted_dict[key] = convert_dict(value)
            else:
                if not isinstance(value, list):
                    converted_dict[key] = [value]
                else:
                    converted_dict[key] = value
        return converted_dict

    return convert_dict(nested_dict)

def merge_nested_dicts(dict_list):
    def merge_dicts(d1, d2):
        for key, value in d2.items():
            if key in d1:
                if isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dicts(d1[key], value)
                else:
                    d1[key].append(value)  # or handle as per requirement, e.g., append to a list if values are lists
            elif isinstance(value, dict):
                d1[key] = convert_values_to_lists(value)
            else:
                d1[key] = [value]

    merged_dict = {}
    for dictionary in dict_list:
        merge_dicts(merged_dict, dictionary)
    
    return merged_dict


def average_nested_dict_values(nested_dict):
    def average_values(value_list):
        return sum(value_list) / len(value_list) if value_list else 0

    def average_dict(d):
        averaged_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                averaged_dict[key] = average_dict(value)
            else:
                try:
                    averaged_dict[key] = average_values(value)
                except:
                    # ignore non number cases
                    pass
        return averaged_dict

    return average_dict(nested_dict)

def run_average_metrics(output_dir):
    output_dir = Path(output_dir)
    metadata_fps = list(output_dir.glob('**/*.metadata.json'))
    if len(metadata_fps) == 0:
        return
    category2wer = defaultdict(list)
    for idx, metadata_fp in enumerate(metadata_fps):
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # update total metrics
        category_dir = metadata_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2wer[str(category_dir)].append(metadata)
        category2wer[str(output_dir)].append(metadata) # append to base directory to calculate total wer
        
    for dir_path, values in category2wer.items():
        metrics_fp = Path(dir_path)/'metrics.json'
        merged_values = merge_nested_dicts(values)
        update_json(metrics_fp, average_nested_dict_values(merged_values))

# Generalized version of "save_outputs.AverageMetricsCallback"
class AverageMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        with local_zero_last():
            if is_local_zero():
                try:
                    run_average_metrics(output_dir)
                except Exception as e:
                    print('Could not run average metrics:', e)

