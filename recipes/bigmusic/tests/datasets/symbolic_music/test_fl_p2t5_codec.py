import os

import numpy as np

from recipes.bigmusic.datasets.symbolic_music.fl_p2t5_codec import FLP2T5Codec


def test_fl_p2t5_codec_lyrics_and_leadsheet_seq_len_stat():
    config = FLP2T5Codec.Config(
        lyrics_seq_len=2000,
        leadsheet_seq_len=5000,
    )
    dm = FLP2T5Codec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=24,
        num_workers=0,
        shuffle_buffer=0,
    )

    lyrics_seq_lens = []
    leadsheet_seq_lens = []

    cum_num_samples = 0
    for i, batch in enumerate(dm.val_dataloader()):
        cum_num_samples += len(batch["__key__"])
        print(i, cum_num_samples)
        if i > 30: break
        # target_length = batch["target_lengths"][0] - 1
        # leadsheet_tokens = batch["target_ids"][0][:target_length].numpy()
        lyrics_seq_lens.extend(batch["original_lyrics_seq_len"])
        leadsheet_seq_lens.extend(batch["original_leadsheet_seq_len"])

    print(f"num samples {len(lyrics_seq_lens)}")
    print(f"lyrics ave {np.mean(lyrics_seq_lens)}, min {np.min(lyrics_seq_lens)}, max {np.max(lyrics_seq_lens)}")
    print(f"leadsheet ave {np.mean(leadsheet_seq_lens)}, min {np.min(leadsheet_seq_lens)}, max {np.max(leadsheet_seq_lens)}")
    # num samples 744
    # lyrics ave 1098.1102150537633, min 119, max 3801
    # leadsheet ave 7660.552419354839, min 1080, max 31293

    ## Entire val set (data_id 1839)
    # num samples 5141
    # lyrics ave 1107.8583933086948, min 119, max 4830
    # leadsheet ave 7605.998832911885, min 655, max 37385
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_fl_p2t5_codec_decode():
    from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes

    config = FLP2T5Codec.Config(
        lyrics_seq_len=2000,
        leadsheet_seq_len=5000,
    )
    dm = FLP2T5Codec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=1,
        num_workers=0,
        shuffle_buffer=0,
    )
    codec = FLP2T5Codec(config)

    for i, batch in enumerate(dm.val_dataloader()):
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