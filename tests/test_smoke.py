from pathlib import Path


def test_import_package():
    import f5_tts
    import wavtts

    assert f5_tts.__doc__ == "WavTTS package."
    assert wavtts.__doc__ == "WavTTS package."


def test_main_config_loads_waveform_training_baseline():
    from omegaconf import OmegaConf

    config_path = Path(
        "src/wavtts/configs/"
        "WavTTS_scale_9_16k.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.model.cfm.prediction == "x_pred"
    assert cfg.model.waveform.wav_frame_len == 160


def test_waveform_cfm_toy_model_initializes():
    from wavtts.model import CFM, DiT

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
            "wav_frame_len": 8,
        },
        vocab_char_map={" ": 0},
        prediction="x_pred",
        loss_space="v",
    )

    assert model.wav_frame_len == 8
    assert model.num_channels == 8


if __name__ == "__main__":
    test_import_package()
    test_main_config_loads_waveform_training_baseline()
    test_waveform_cfm_toy_model_initializes()
