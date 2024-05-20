import os

import numpy as np
import torch
import torchaudio

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.modules.pitch_predictor_task import PitchPredictorTask


### Functions for running the trained model and extracting a standalone .pt file. ###
def run_trained_model():
    """Function for loading a ckpt and checking predictions with a figure and saved audio."""
    from transformers import BertTokenizer

    from recipes.datasets.mcc.mix_mkii import MixMSSDataModule

    # Configure the Pitch Predictor Module
    # CKPT_PATH = "hdfs://haruna/home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/pitch_predictor_trained_on_vocal/checkpoints/step=0200000.ckpt"
    CKPT_PATH = "hdfs://haruna/home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/pitch_predictor_trained_on_full_mix/checkpoints/step=0640000.ckpt"
    pl_pitch_module = PitchPredictorTask.load_from_checkpoint(CKPT_PATH)
    pl_pitch_module.setup("predict")

    # Configure the small run
    NUM_DATALOADER_ITERATIONS = 6
    NUM_TESTING_SAMPLES = 1
    SAMPLE_RATE = 24000
    OUTPUT_DIR = "./.test_pitch_model"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configure the data loader
    tokenizer = None  # BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    pl_datamodule = MixMSSDataModule(
        data_ids=[1038],
        data_weights=[1],
        sample_rate=SAMPLE_RATE,
        batch_size=SAMPLE_RATE * 10 * 30,
        shuffle_buffer_size=100,
        num_workers=1,  # prevents a data segmentation bug when loading data outside of PyTorch.Trainer()
        tokenizer=None,
    )
    train_loader = pl_datamodule.train_dataloader()
    train_loader = iter(train_loader)

    for i in range(NUM_DATALOADER_ITERATIONS):
        # Run model
        batch = next(train_loader)
        output_dict = pl_pitch_module.forward(batch)
        f_dict = {k: v.cpu().detach().numpy() for k, v in output_dict.items()}

        for j in range(NUM_TESTING_SAMPLES):
            # Extract data for a single sample
            f0_pred_i, vuv_pred_i = f_dict["f0_pred"][i, :], f_dict["vuv_pred"][i, :]
            f0_gt_i, vuv_gt_i = f_dict["f0_gt"][i, :], f_dict["vuv_gt"][i, :]
            mels_i = f_dict["x_mels"][i, :]
            # Save figure of raw outputs
            vuv_pred_i = pitch_utils.post_process_vuv_logits_to_binary_state(
                vuv_pred_i
            )  # convert output logits into binary voice/unvoiced state
            fig_combo = pitch_utils._create_fig_mel_with_f0_pred_and_gt(
                mels_i, f0_pred_i, f0_gt_i, vuv_pred_i, vuv_gt_i
            )
            pitch_utils._save_figure(
                fig_combo, os.path.join(OUTPUT_DIR, f"{i}-{j}_raw")
            )
            # Save figure of post processed predictions
            f0_pred_processed_i = pitch_utils.mask_f0_by_vuv(
                f0_pred_i,
                pitch_utils.post_process_vuv_logits_to_binary_state(vuv_pred_i),
            )
            fig_combo2 = pitch_utils._create_fig_mel_with_f0_pred_and_gt(
                mels_i, f0_pred_processed_i, f0_gt_i
            )
            pitch_utils._save_figure(
                fig_combo2, os.path.join(OUTPUT_DIR, f"{i}-{j}_post_processed")
            )
            # Save audio
            audio_i = np.expand_dims(
                f_dict["audio_in"][i], axis=0
            )  # [channels=1, num_samples]
            filename = os.path.join(OUTPUT_DIR, f"audio_{i}.wav")
            torchaudio.save(filename, torch.Tensor(audio_i), sample_rate=SAMPLE_RATE)


def extract_trained_model():
    """Extracts standalone .pt file from .ckpt and checks the output of loaded model is correct."""
    from recipes.umm.models.voc_modules.pitch_predictor.model import PitchPredictor

    # Configure the Pitch Predictor Module
    # BASE_PATH = "hdfs://haruna/home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/pitch_predictor_trained_on_vocal"
    # CKPT_PATH = os.path.join(BASE_PATH, "checkpoints/step=0200000.ckpt")

    BASE_PATH = "hdfs://haruna/home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/pitch_predictor_trained_on_full_mix"
    CKPT_PATH = os.path.join(BASE_PATH, "checkpoints/step=0640000.ckpt")

    pl_pitch_module = PitchPredictorTask.load_from_checkpoint(CKPT_PATH)
    pl_pitch_module.setup("predict")

    # Extract the pitch model
    pitch_model = pl_pitch_module.model

    # Configure save path and name
    FILE_NAME = "perceptual_pitch_model_1_26_2024.pt"
    OUTPUT_DIR = "./.saved_models"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, FILE_NAME)
    torch.save(pitch_model.state_dict(), file_path)

    # Verify model can be loaded
    loaded_model = PitchPredictor()
    loaded_model.load_state_dict(torch.load(file_path))

    batch = {
        k: pitch_utils.gen_test_sin() for k in ["audio", "audio_vocal", "audio_inst"]
    }
    output_dict = pl_pitch_module.forward(batch)

    # Compare predicted f0 from loaded .ckpt pl_module and loaded .pt model
    expected_f0 = output_dict["f0_pred"]  # compare with pl_module model
    f0 = loaded_model.forward(output_dict["x_mels"])[
        :, :, 0
    ]  # use x_mels computed from pl_module
    trim_len = pitch_utils.compute_min_lengths(expected_f0, f0)  # [1, n_frames]
    assert torch.allclose(expected_f0[:, :trim_len], f0[:, :trim_len])

    # Ensure hidden state is accesible
    h_out = loaded_model.get_hidden_state()
    assert list(h_out.size()) == [1, 201, 128]  # [batch_size, n_frames, hidden_state]


if __name__ == "__main__":
    # run_trained_model()
    extract_trained_model()
