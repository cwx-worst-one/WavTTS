from recipes.bigmusic.datasets.symbolic_music.fl_p2t52a_codec import FLP2T52ACodec


def test_fl_p2t52a_codec_decode():
    from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes

    config = FLP2T52ACodec.Config(
        lyrics_seq_len=2000,
        leadsheet_seq_len=5000,
    )
    dm = FLP2T52ACodec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=1,
        num_workers=0,
        shuffle_buffer=0,
    )
    codec = FLP2T52ACodec(config)

    for i, batch in enumerate(dm.val_dataloader()):
        from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
        target_length = batch["target_lengths"][0] - 1
        leadsheet_tokens = batch["target_ids"][0][:target_length].numpy()
        midi_obj = codec.decode(leadsheet_tokens)
        from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip

    # output_dir = os.path.join(
    #     os.environ["DUMP_DIR"], "ai_music/tmp/"
    # )

    # codec = FLP2T5Codec(config)
    # midi_obj = codec.decode(arr)
    # midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
    # midi_output_path = os.path.join(output_dir, f"20240314.20240309.l2l.base.0.mid")
    # with open(midi_output_path, "wb") as f:
    #     f.write(midi_bytes)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip