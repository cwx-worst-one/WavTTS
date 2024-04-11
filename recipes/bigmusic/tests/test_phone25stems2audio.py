import tqdm
from recipes.bigmusic.datasets.symbolic_music.fixed_length_phoneme_and_trans5stems_audio_codec import FixedLengthTransPhone25Stems2AudioCodec

config = FixedLengthTransPhone25Stems2AudioCodec.Config(
    lyrics_seq_len=600,
    leadsheet_seq_len=2000,
    audio_max_duration=60,
    semantic_frame_rate=25,
    sample_rate=24000,
    audio_key='wav',
    conditions="lyrics_tokens",
    include_utterance_phoneme_tokens=False,
    stems="vocal,piano,guitar,bass,drums",
)

datamodule = FixedLengthTransPhone25Stems2AudioCodec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1881,
        batch_size=1,
        num_workers=1)

dataset = datamodule.train_dataloader().__iter__()
target_tokens_length = []
remi_leadsheet_token_len = []
for i in tqdm.tqdm(range(100)):
    batch = next(dataset)
    print(batch)
    # target_tokens_length.extend(batch["original_target_tokens_length"].cpu().tolist())
    remi_leadsheet_token_len.extend(batch["original_remi_leadsheet_tokens_length"].cpu().tolist())

    import matplotlib.pyplot as plt
    plt.cla()
    plt.hist(remi_leadsheet_token_len, bins=1000)
    plt.savefig("remi_leadsheet_token_len.1839.fixed_length_trans5stem_and_lyric2audio_codec.png")