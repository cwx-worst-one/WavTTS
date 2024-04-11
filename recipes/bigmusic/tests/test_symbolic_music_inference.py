import os

import numpy as np
import pandas as pd
import torch

from recipes.bigmusic.datasets.symbolic_music.base import get_indexer
from recipes.bigmusic.datasets.symbolic_music.fixed_length_phoneme_and_vocal2midi_codec import FixedLengthPhonemeAndVocal2MidiCodec
from recipes.bigmusic.inference.token_generator import TokenGenerator, GeneratorState
from recipes.bigmusic.inference.base_observer import ResultRecordObserver
from recipes.bigmusic.inference.controllers import FixedLengthLyricsToLeadsheetController
from recipes.bigmusic.inference.utils import get_sami_tokenizer, download_checkpoint_from_hdfs
from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes


def test_symbolic_music_l2l_sanity():
    exp_id = "20240318.l2l.valcopy.testtb"
    ckpt_filename = "step=094000-tr_loss=0.0005-val_loss_0=0.0004.ckpt"
    hdfs_ckpt_path = f"/home/byte_speech_sv/haonanchen/logs/{exp_id}/checkpoints/{ckpt_filename}"
    local_ckpt_dir = os.path.join(os.environ["DUMP_DIR"], f"ai_music/20240312.symbolic.dump/logs/{exp_id}/checkpoints")
    os.makedirs(local_ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(local_ckpt_dir, ckpt_filename)
    download_checkpoint_from_hdfs(hdfs_ckpt_path, ckpt_path)

    from recipes.bigmusic.lightning.symbolic_music_modules import Lyrics2SymbolicMusicModule
    pl_module = Lyrics2SymbolicMusicModule.load_from_checkpoint(ckpt_path)
    result_obs = ResultRecordObserver()
    token_gen = TokenGenerator(pl_module, GeneratorState())
    token_gen.register_observer(result_obs)

    ## Fetch data from datamodule
    config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=2000,
        leadsheet_seq_len=5000,
    )
    dm = FixedLengthPhonemeAndVocal2MidiCodec.get_datamodule(
        config=config,
        train_id=1839,
        val_id=1839,
        batch_size=1,
        num_workers=0,
    )

    controller = FixedLengthLyricsToLeadsheetController(
        token_gen, config
    )

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        for batch in dm.val_dataloader():
            tokens = batch["model_inputs"]["input_ids"][0].tolist()
            controller.generate(tokens)
            midi_obj = controller.codec.decode(result_obs.all_ids[:-1])
            output_dir = os.path.join(
                os.environ["DUMP_DIR"], "ai_music/tmp/"
            )
            midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
            midi_output_path = os.path.join(
                output_dir,
                f"20240320.{exp_id}.mid",
            )
            with open(midi_output_path, "wb") as f:
                f.write(midi_bytes)


def test_symbolic_music_l2l_inference():
    exp_id = "20240309.l2l.base"
    # ckpt_filename = "step=014000-tr_loss=0.6262-val_loss_0=0.6314.ckpt"
    ckpt_filename = "step=041000-tr_loss=0.5820-val_loss_0=0.5866.ckpt"

    # exp_id = "20240325.l2l.nosilence.ms86m"
    # ckpt_filename = "step=017000-tr_loss=0.5943-val_loss_0=0.5745.ckpt"
    ckpt_path = f"/home/byte_speech_sv/haonanchen/logs/{exp_id}/checkpoints/{ckpt_filename}"
    test_lyrics_file = os.path.join(
        os.environ["DUMP_DIR"],
        "ai_music/20240223.svs.data/20240223.en_lyrics_sft_10_baseline.csv"
    )
    df_info = pd.read_csv(test_lyrics_file)

    output_dir = os.path.join(os.environ["DUMP_DIR"], "ai_music/20240312.symbolic.dump")
    local_ckpt_path = os.path.join(output_dir, ckpt_filename)
    if not os.path.exists(local_ckpt_path):
        os.system(f'hdfs dfs -get {ckpt_path} {local_ckpt_path}')
    assert os.path.exists(local_ckpt_path), "Model download failed"
    lyrics = df_info.lyrics.iloc[0]
    tokenizer = get_sami_tokenizer()
    tokens = np.concatenate([np.ravel(tokenizer(l)["input_ids"]) for l in lyrics.split("\n")])

    from recipes.bigmusic.lightning.symbolic_music_modules import Lyrics2SymbolicMusicModule
    pl_module = Lyrics2SymbolicMusicModule.load_from_checkpoint(local_ckpt_path)
    result_obs = ResultRecordObserver()
    token_gen = TokenGenerator(pl_module, GeneratorState())
    token_gen.register_observer(result_obs)

    data_config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=2000,
        leadsheet_seq_len=5000,
    )

    controller = FixedLengthLyricsToLeadsheetController(
        token_gen, data_config
    )

    def generate_single_song(tokens, midi_output_path):
        controller.generate(tokens)
    
        ## The last token is pad or eos
        midi_obj = controller.codec.decode(result_obs.all_ids[:-1])

        output_dir = os.path.join(
            os.environ["DUMP_DIR"], "ai_music/tmp/"
        )
        midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
        midi_output_path = os.path.join(output_dir, f"20240314.20240309.l2l.base.0.mid")
        with open(midi_output_path, "wb") as f:
            f.write(midi_bytes)

    output_dir = os.path.join(
        os.environ["DUMP_DIR"], "ai_music/20240312.symbolic.dump/"
    )
    # print(result_obs.all_ids[:-1])
    # events = []
    # phone_tokens = []
    # for t in result_obs.all_ids:
    #     event = controller.codec.indexer.inverse[t]
    #     events.append(event)
    #     if event.startswith("phone"):
    #         phone_tokens.append(t)
    # gt_phone_tokens = np.array([t for t in tokens if t > 2 and t < 264 and t != 85])
    # print("utterance phone:", gt_phone_tokens)
    # print("leadsheet phone:", np.array(phone_tokens))

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        controller.generate(tokens)
    
        ## The last token is pad or eos
        midi_obj = controller.codec.decode(result_obs.all_ids[:-1])

        from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes
        output_dir = os.path.join(
            os.environ["DUMP_DIR"], "ai_music/tmp/"
        )
        midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
        midi_output_path = os.path.join(output_dir, f"20240326.{exp_id}.0.mid")
        with open(midi_output_path, "wb") as f:
            f.write(midi_bytes)

        print(result_obs.all_ids[:-1])
        events = []
        phone_tokens = []
        for t in result_obs.all_ids:
            event = controller.codec.indexer.inverse[t]
            events.append(event)
            if event.startswith("phone"):
                phone_tokens.append(t)
        gt_phone_tokens = np.array([t for t in tokens if t > 2 and t < 264 and t != 85])
        print("utterance phone:", gt_phone_tokens)
        print("leadsheet phone:", np.array(phone_tokens))
        from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_symbolic_music_melodyonly_inference():
    exp_id = "20240309.melodyonly.base"
    # ckpt_filename = "step=014000-tr_loss=0.6262-val_loss_0=0.6314.ckpt"
    ckpt_filename = ""
    ckpt_path = f"/home/byte_speech_sv/haonanchen/logs/{exp_id}/checkpoints/checkpoints/{ckpt_filename}"