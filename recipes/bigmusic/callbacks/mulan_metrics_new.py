# Created by @ Yatong Bai
# Computes MuLan text similarity and per-song FAD
# Required pip packages: pyloudnorm, confusables, ToJyutping

import json, os
import math
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
import pytorch_lightning as pl

from recipes.musiclm.inference.utils import load_wav, tensor_resample, level_norm_lufs
from recipes.bigmusic.datasets.utils.zh_meta import infer_freeform_text_from_style_text
from recipes.bigmusic.utils.mulan_utils import MulanUtil


def compute_nested_means(
    json_file_fps: List[str],
    categories: List[str] = ["Style_MCS", "Freeform_MCS", "PSFAD", "PSFAD_PoolExt"]
) -> Dict:
    """
    Computes the mean of nested numerical fields across all JSON files in a directory.
    Ignores any NaN values found in the data.

    Args:
        json_file_fps (List[str]): A list of paths to JSON files to average.
        categories (List[str], optional): A list of categories to average.
            Defaults to ["Style_MCS", "Freeform_MCS", "PSFAD", "PSFAD_PoolExt"].
    Returns:
        Dict: A nested dictionary with the computed means and the total file count.
    """

    # Dictionary to accumulate values for each key in the nested structure
    # The structure will be {category: {key: [value1, value2, ...]}}
    accumulated_data = {category: {} for category in categories}
    # Dictionary to store the final mean values
    mean_data = {category: {} for category in categories}

    file_count = 0
    # Loop through all files in the specified directory
    for file_path in json_file_fps:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            nan_found_in_file = False
            # Iterate through the main categories and their nested keys
            for category in categories:
                if category in data and isinstance(data[category], dict):
                    for key, value in data[category].items():
                        # Check if the value is a valid number
                        if isinstance(value, (int, float)):
                            # Check if the value is NaN
                            if math.isnan(value):
                                nan_found_in_file = True
                                logging.warning(
                                    f"Warning: Found NaN value in file '{file_path}' for key '{category}.{key}'. "
                                    "This value will be ignored."
                                )
                            else:
                                if key not in accumulated_data[category]:
                                    accumulated_data[category][key] = []
                                accumulated_data[category][key].append(value)

            # Only increment file_count if the file was processed without errors
            if not nan_found_in_file:
                file_count += 1

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error processing file {os.path.basename(file_path)}: {e}")

    # Check if any files were processed to avoid division by zero
    if file_count == 0:
        print("No JSON files found or processed in the directory.")
        return {}

    # Compute the mean for each key
    for category, keys in accumulated_data.items():
        for key, values in keys.items():
            if values and len(values) > 0:
                mean_data[category][key] = sum(values) / len(values)

    # Add the file count directly to the mean_data dictionary
    mean_data["File_Count"] = file_count

    # Return the single dictionary
    return mean_data


class MuLanMetricsCallback(pl.Callback):
    def __init__(
        self,
        mulan_name: str = 'mulan89',
        ref_stats_dir: str = 'MuLan_Ref_Stats',
        use_loud_norm: bool = True,  # If True, do loudness normalization to -23dB LUFS
        mulan_shift_seconds: float = 5.
    ) -> None:

        super().__init__()

        self.mulan = MulanUtil(mulan_name)
        self.use_loud_norm = use_loud_norm
        self.mulan_shift_seconds = mulan_shift_seconds
        logging.info(f"use_loud_norm: {use_loud_norm}, mulan_sift_seconds: {mulan_shift_seconds}")

        pool_ext_ref_stats_path = f"{ref_stats_dir}/PGC_4pts_genre_pooled_MuLan_stats_dict.pt"
        no_pool_ref_stats_path = f"{ref_stats_dir}/PGC_4pts_genre_MuLan_stats_dict.pt"
        self.no_pool_fad_ref_stats = torch.load(no_pool_ref_stats_path, map_location="cuda")
        self.pool_ext_fad_ref_stats = torch.load(pool_ext_ref_stats_path, map_location="cuda")

    def save_json_and_pt(
        self,
        generated_output_fps: List[str],
        style_texts: List[str],
        freeform_texts: List[str],
        mulan_scores_dict: Dict
    ) -> List[str]:
        """ Save JSON (for scores) and PT (for embeddings) files for each data_id. """

        def check_dup_and_rename(fn):
            # Avoid overwriting when multiple data share the same data_id and round_num
            postfix = fn.split(".")[-1]
            if os.path.exists(fn):
                fn = fn.replace(f".{postfix}", f"_2.{postfix}")
            cntr = 2
            while os.path.exists(fn):
                fn = fn.replace(f"_{cntr}.{postfix}", f"_{cntr+1}.{postfix}")
                cntr += 1
            return fn

        metrics_fps = []
        for idx, (gen_out_fp, style_text, freeform_text) in enumerate(
            zip(generated_output_fps, style_texts, freeform_texts)
        ):
            cur_style_sim_dict = {k: v[idx] for k, v in mulan_scores_dict["Style_MCS"].items()}
            cur_freeform_sim_dict = {k: v[idx] for k, v in mulan_scores_dict["Freeform_MCS"].items()}
            cur_psfad_dict = {k: v[idx] for k, v in mulan_scores_dict["PSFAD"].items()}
            cur_psfad_pool_ext_dict = {k: v[idx] for k, v in mulan_scores_dict["PSFAD_PoolExt"].items()}
            cur_embds_dict = {k: v[idx] for k, v in mulan_scores_dict["Embeddings"].items()}

            cur_json_dict = {
                "style_text": style_text,
                "freeform_text": freeform_text,
                # Save "Raw_Mean", "Raw_Median", and "Pooled" fields of mulan_scores
                "Style_MCS": {k: v.tolist() for k, v in cur_style_sim_dict.items() if k != "Raw"},
                "Freeform_MCS": {k: v.tolist() for k, v in cur_freeform_sim_dict.items() if k != "Raw"},
                "PSFAD": {k: v.tolist() for k, v in cur_psfad_dict.items()},
                "PSFAD_PoolExt": {k: v.tolist() for k, v in cur_psfad_pool_ext_dict.items()},
            }
            # Write embeddings and results to disk
            metrics_fp = str(gen_out_fp).replace('generated.wav', 'mulan_metrics.json')
            metrics_fp = check_dup_and_rename(metrics_fp)
            metrics_fps.append(metrics_fp)
            with open(metrics_fp, "w") as f:
                json.dump(cur_json_dict, f, indent=4)

            cur_pickle_dict = {
                "style_text": style_text,   
                "freeform_text": freeform_text,
                "Style_MCS": cur_style_sim_dict,
                "Freeform_MCS": cur_freeform_sim_dict,
                "PSFAD": cur_psfad_dict,
                "PSFAD_PoolExt": cur_psfad_pool_ext_dict,
                "Embeddings": cur_embds_dict
            }
            pickle_fp = str(gen_out_fp).replace('generated.wav', 'mulan_embeddings.pt')
            pickle_fp = check_dup_and_rename(pickle_fp)
            torch.save(cur_pickle_dict, pickle_fp)

            del cur_pickle_dict, cur_json_dict, cur_embds_dict, cur_psfad_dict, cur_style_sim_dict, cur_freeform_sim_dict

        return metrics_fps

    def get_scores_for_batch(
        self, wavs_np: List[np.ndarray], style_texts: List[str], freeform_texts: List[str]
    ) -> Dict:

        target_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() \
            else torch.device("cpu")

        wavs = [torch.FloatTensor(wnp).to(target_device) for wnp in wavs_np]

        # Compute MuLan similarity and per-song FAD
        mulan_scores_dict = self.mulan.get_scores_from_tensor(
            wav=wavs, style_text=style_texts, freeform_text=freeform_texts,
            ref_stats=self.no_pool_fad_ref_stats,
            ref_stats_pool_ext=self.pool_ext_fad_ref_stats,
            return_embd=True, shift_seconds=self.mulan_shift_seconds
        )
        return mulan_scores_dict

    def on_predict_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch_size: int = 8, 
        output_dir: Optional[str] = None
    ) -> None:
        """
        Computes MuLan metrics for the generated outputs and saves them to disk.
        """

        if trainer is None or trainer.is_global_zero:
            # Gather all generated output file paths
            if output_dir is None:
                output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            logging.info(f"output_dir: {output_dir}")

            wavs_np, style_texts, freeform_texts, gen_out_fps, json_file_fps = [], [], [], [], []
            for idx, generated_output_fp in enumerate(tqdm(generated_output_fps)):
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')

                # Convert to mono by averaging across channels
                wav_np = load_wav(str(generated_output_fp), sr=44100, mono=True)
                # # Convert to mono by only keeping the first channel
                # wav_np = load_wav(str(generated_output_fp), sr=44100, mono=False)[0, :]

                wav_np = tensor_resample(wav=torch.from_numpy(wav_np), orig_freq=44100, new_freq=24000).numpy()
                if self.use_loud_norm:
                    wav_np = level_norm_lufs(wav_np, sr=24000, target_lufs=-23.)
                wavs_np.append(wav_np)

                with open(metadata_fp, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                #breakpoint()
                freeform_text = metadata.get("prompt", None)
                style_tags = metadata.get("style_text", None)
                style_text = infer_freeform_text_from_style_text(style_tags, freeform_dropout=0., shuffle=False)

                style_texts.append(style_text.replace(" , ", ""))
                freeform_texts.append(freeform_text)
                gen_out_fps.append(generated_output_fp)

                if (idx + 1) % batch_size == 0 or idx + 1 == len(generated_output_fps):
                    # Compute MuLan similarity and per-song FAD for this batch
                    mulan_scores_dict = self.get_scores_for_batch(wavs_np, style_texts, freeform_texts)

                    # Post-process results and save to disk
                    json_file_fp = self.save_json_and_pt(gen_out_fps, style_texts, freeform_texts, mulan_scores_dict)
                    json_file_fps.extend(json_file_fp)

                    # Clear lists for the next batch
                    wavs_np, style_texts, freeform_texts, gen_out_fps = [], [], [], []

        if trainer is None or trainer.is_global_zero:
            # Compute the mean of nested numerical fields across all JSON files in a directory.
            logging.info(f"Processed {len(json_file_fps)} files.")

            mean_data = compute_nested_means(json_file_fps)

            logging.info("============================================================================")
            logging.info(f"MuLan metrics:")
            logging.info(
                f"Average similarity with style text: {mean_data['Style_MCS']['Raw_Mean']} (higher is better)"
            )
            logging.info(
                f"Average similarity with freeform text: {mean_data['Freeform_MCS']['Raw_Mean']} (higher is better)"
            )
            logging.info(
                f"Average per-song FAD: {mean_data['PSFAD']['All_4pts_MuLan_LN']} (lower is better)"
            )
            logging.info(
                f"Average pool-extended per-song FAD: {mean_data['PSFAD_PoolExt']['All_4pts_MuLan_LN']} (lower is better)"
            )
            logging.info("============================================================================")

            # Save the mean data to disk
            mean_data_fp = os.path.join(output_dir, "mulan_metrics_mean.json")
            with open(mean_data_fp, "w") as f:
                json.dump(mean_data, f, indent=4)


def my_callback_offline(output_dir: str) -> None:
    callback = MuLanMetricsCallback(use_loud_norm=True)
    callback.on_predict_end(trainer=None, pl_module=None, output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    my_callback_offline(args.output_dir)
