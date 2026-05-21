from pathlib import Path


def test_import_package():
    import f5_tts

    assert f5_tts.__doc__ == "WavTTS package."


def test_main_config_loads_waveform_training_baseline():
    from omegaconf import OmegaConf

    config_path = Path(
        "src/f5_tts/configs/"
        "WavTTS_scale_9_16k.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.model.wav_input is True
    assert cfg.model.cfm.prediction == "x_pred"
    assert cfg.model.waveform.wav_frame_len == 160


def test_waveform_cfm_toy_model_initializes():
    from f5_tts.model import CFM, DiT

    transformer = DiT(
        dim=16,
        depth=1,
        heads=2,
        dim_head=8,
        wav_frame_len=8,
        text_num_embeds=16,
        text_dim=8,
        ff_mult=2,
    )
    model = CFM(
        transformer=transformer,
        waveform_kwargs={
            "target_sample_rate": 16000,
            "hop_length": 8,
            "win_length": 32,
            "n_fft": 32,
            "wav_frame_len": 8,
        },
        vocab_char_map={" ": 0},
        prediction="x_pred",
        loss_space="v",
    )

    assert model.wav_input_only is True
    assert model.wav_frame_len == 8
    assert model.num_channels == 8
    assert model.transformer.wav_input_only is True


if __name__ == "__main__":
    test_import_package()
    test_main_config_loads_waveform_training_baseline()
    test_waveform_cfm_toy_model_initializes()
