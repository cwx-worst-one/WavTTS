import matplotlib.pyplot as plt
import torchaudio
from tqdm import tqdm

from recipes.mir.datasets.mix import DataModule

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
save_audio = False


def test_mix_datamodule():
    sample_rate = 24000
    max_length = 3072
    pl_datamodule = DataModule(
        sample_rate=sample_rate,
        batch_size=3,
        shuffle_buffer_size=10,
        num_workers=2,
        # tempo, tag, structure, key, gender, dialect, caption, asr-music, asr-speech
        weights=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        region="CN",
        dynamic_batch=False,
        max_length=max_length,
    )
    tokenizer = pl_datamodule.tokenizer
    train_loaders = pl_datamodule.val_dataloader()
    for loader_idx, train_loader in enumerate(train_loaders):
        train_loader = iter(train_loader)
        for i, batch in enumerate(train_loader):
            mono_map = batch["mono_map"]
            homo_map = batch["homo_map"]
            mono_audio = batch["mono_audio"]
            homo_audio = batch["homo_audio"]
            mono_text = batch["mono_text"]
            homo_text = batch["homo_text"]
            for j, (task_type, task_id, it_i, ia_i, tt_i, ta_i) in enumerate(mono_map):
                print(f"[{loader_idx} - {i} - {j}] [{task_type} - {task_id}]")
                print(f"  [Input Text] {None if it_i is None else tokenizer.decode(mono_text[it_i], skip_special_tokens=True, clean_up_tokenization_spaces=True)}")
                if ia_i is not None:
                    print(f"  [Input Audio] {mono_audio[ia_i].size()}")
                    if save_audio:
                        torchaudio.save(
                            f"./test_out/batch-{i}-mono-{j}-input.mp3",
                            mono_audio[ia_i],
                            sample_rate,
                            format="mp3",
                        )
                else:
                    print(f"  [Input Audio] None")
                print(f"  [Target Text] {None if tt_i is None else tokenizer.decode(mono_text[tt_i], skip_special_tokens=True, clean_up_tokenization_spaces=True)}")
                if ta_i is not None:
                    print(f"  [Target Audio] {mono_audio[ta_i].size()}")
                    if save_audio:
                        torchaudio.save(
                            f"./test_out/batch-{i}-mono-{j}-target.mp3",
                            mono_audio[ta_i],
                            sample_rate,
                            format="mp3",
                        )
                else:
                    print(f"  [Target Audio] None")
                print(f"=============================================")

            for j, (task_type, task_id, it_i, ia_i, tt_i, ta_i) in enumerate(homo_map):
                print(f"[{loader_idx} - {i} - {j}] [{task_type} - {task_id}]")
                print(f"  [Input Text] {None if it_i is None else tokenizer.decode(homo_text[it_i], skip_special_tokens=True, clean_up_tokenization_spaces=True)}")
                print(f"  [Input Audio] {homo_audio[ia_i].size()}")
                if save_audio:
                    torchaudio.save(
                        f"./test_out/batch-{i}-homo-{j}-input.mp3",
                        homo_audio[ia_i],
                        sample_rate,
                        format="mp3",
                    )
                print(f"  [Target Text] {None if tt_i is None else tokenizer.decode(homo_text[tt_i], skip_special_tokens=True, clean_up_tokenization_spaces=True)}")
                print(f"  [Target Audio] {homo_audio[ta_i].size()}")
                if save_audio:
                    torchaudio.save(
                        f"./test_out/batch-{i}-homo-{j}-target.mp3",
                        homo_audio[ta_i],
                        sample_rate,
                        format="mp3",
                    )
                print(f"=============================================")

test_mix_datamodule()
