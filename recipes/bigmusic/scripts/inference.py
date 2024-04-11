import os
import sys

import torch
import numpy as np
import pandas as pd

from recipes.bigmusic.datasets.symbolic_music.base_codec import IndexerConfig
from recipes.bigmusic.inference.base_observer import ResultRecordObserver
from recipes.bigmusic.inference.token_generator import TokenGenerator, GeneratorState
from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes
from recipes.bigmusic.datasets.symbolic_music import FixedLengthPhonemeAndVocal2MidiCodec


def _get_l2l_inf_controller(
    exp_id: str,
    ckpt_filename: str,
    data_config: FixedLengthPhonemeAndVocal2MidiCodec.Config,
):
    from recipes.bigmusic.inference.controllers import FixedLengthLyricsToLeadsheetController
    from recipes.bigmusic.lightning.symbolic_music_modules import Lyrics2SymbolicMusicModule

    # ckpt_filename = "step=041000-tr_loss=0.5820-val_loss_0=0.5866.ckpt"
    ckpt_path = f"/home/byte_speech_sv/haonanchen/logs/{exp_id}/checkpoints/checkpoints/{ckpt_filename}"
    output_dir = os.path.join(os.environ["DUMP_DIR"], f"ai_music/20240312.symbolic.dump/logs/{exp_id}")
    os.makedirs(output_dir, exist_ok=True)
    local_ckpt_path = os.path.join(output_dir, ckpt_filename)
    if not os.path.exists(local_ckpt_path):
        os.system(f'hdfs dfs -get {ckpt_path} {local_ckpt_path}')
    assert os.path.exists(local_ckpt_path), "Model download failed"
    pl_module = Lyrics2SymbolicMusicModule.load_from_checkpoint(local_ckpt_path)
    result_obs = ResultRecordObserver()
    token_gen = TokenGenerator(pl_module, GeneratorState())
    token_gen.register_observer(result_obs)

    controller = FixedLengthLyricsToLeadsheetController(
        token_gen, data_config
    )
    return controller, result_obs


def l2l():
    from recipes.bigmusic.inference.utils import get_sami_tokenizer
    from recipes.bigmusic.datasets.symbolic_music import FixedLengthPhonemeAndVocal2MidiCodec

    inf_id = sys.argv[2]

    num_samples = 10
    output_root_dir = os.path.join(
        os.environ["DUMP_DIR"], "ai_music/20240312.symbolic.dump/"
    )
    lyrics_seq_len = 2000
    leadsheet_seq_len = 5000

    if inf_id == "20240314.20240309.l2l.base.lyrics0":
        exp_id = "20240309.l2l.base"
        ckpt_filename = "step=014000-tr_loss=0.6262-val_loss_0=0.6314.ckpt"
        num_samples = 10

        test_lyrics_file = os.path.join(
            os.environ["DUMP_DIR"],
            "ai_music/20240223.svs.data/20240223.en_lyrics_sft_10_baseline.csv"
        )
        df_info = pd.read_csv(test_lyrics_file)
        lyrics = df_info.lyrics.iloc[0]
        tokenizer = get_sami_tokenizer()
        tokens = np.concatenate([np.ravel(tokenizer(l)["input_ids"]) for l in lyrics.split("\n")])

    data_config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=lyrics_seq_len,
        leadsheet_seq_len=leadsheet_seq_len,
    )

    controller, result_obs = _get_l2l_inf_controller(exp_id, ckpt_filename, data_config)

    def generate_single_song(tokens, midi_output_path):
        controller.generate(tokens)
        ## The last token is pad or eos
        midi_obj = controller.codec.decode(result_obs.all_ids[:-1])
        midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
        with open(midi_output_path, "wb") as f:
            f.write(midi_bytes)

    output_dir = os.path.join(output_root_dir, inf_id)
    os.makedirs(output_dir, exist_ok=True)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        for i in range(num_samples):
            midi_output_path = os.path.join(output_dir, f"{inf_id}.{i}.mid")
            generate_single_song(tokens, midi_output_path)


def l2l_copy_train():
    """check if model can successfully copy training data
    """
    lyrics_seq_len = 2000
    leadsheet_seq_len = 5000
    exp_id = "20240309.l2l.base"
    ckpt_filename = "step=041000-tr_loss=0.5820-val_loss_0=0.5866.ckpt"
    data_config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=lyrics_seq_len,
        leadsheet_seq_len=leadsheet_seq_len,
    )
    codec = FixedLengthPhonemeAndVocal2MidiCodec(data_config)

    controller, result_obs = _get_l2l_inf_controller(exp_id, ckpt_filename, data_config)

    dm = FixedLengthPhonemeAndVocal2MidiCodec.get_datamodule(
        data_config,
        1838,
        1839,
        batch_size=1,
        num_workers=0,
    )
    tdl = dm.train_dataloader()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        for batch in tdl:
            tokens = batch["model_inputs"]["input_ids"][0, :lyrics_seq_len]
            controller.generate(tokens)
            gt_target = batch["model_inputs"]["input_ids"][0, lyrics_seq_len:]
            result_obs.all_ids
            from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def slm():
    raise NotImplementedError("TODO")
    inf_id = sys.argv[2]

    config = None

    context_events = []
    observers = []
    temperature = 1.0

    if inf_id == "20240322.20240320.20240320.licensed.part.llama110.20480.uncond.seq20480":
        from recipes.bigmusic.datasets.symbolic_music.trans5stems_codec import Trans5StemsCodec

        exp_id = "20240320.20240320.licensed.part.llama110.20480"
        # checkpoint_filename = "epoch=4-step=49996.ckpt"
        checkpoint_filename = "epoch=2-step=20080.ckpt"
        num_samples = 10
        seq_len = 20480
        vocab_size = 321
        config = IndexerConfig(
            include_phoneme=False,
            include_utterance_phoneme_tokens=False,
            include_bar_event = True,
            include_stem_indicator = True,
            include_drum_events = True,
            include_section_indicator_each_bar = True,
            include_bpm_levels = True,
            include_prompt = True,
        )
        codec = Trans5StemsCodec(config)

        indexer = get_remi_indexer(config)
        context_events = [
            "eom",
            "bar",
            "sec_2",
            # "bpm_level_5",
            # "stem_0",
        ]
        fsm = FSM(ZeroOrMore(AnyState(indexer)))
        logs_subdir = "logs/leadsheet_lyrics"


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        locals()[sys.argv[1]]()
    else:
        [print(k) for k, v in locals().items() if callable(v)]
