import torch
import torchaudio

assert torch.cuda.is_available()

from jiwer import cer, wer
from tqdm import tqdm

from recipes.datasets.mcc.mix_mkii import DataModule
from recipes.umm_062.requires.model_initializer import init_stage3, init_unified_decoder


def slugify(value, allow_unicode=False):
    import re
    import unicodedata

    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


@torch.no_grad()
def evaluate(it, verbose=True, max_i=1000):
    for i, batch in tqdm(enumerate(it)):
        if len(batch["mono_map"]) > 0:
            batch["mono_audio"] = batch["mono_audio"].to(device)
            batch["mono_text"] = batch["mono_text"].to(device)
        if len(batch["homo_map"]) > 0:
            batch["homo_audio"] = batch["homo_audio"].to(device)
            batch["homo_text"] = batch["homo_text"].to(device)
        token_seq, token_length, target_length = decoder.prepare_feature(batch)
        assert token_seq.size(0) == 1
        model_input = []
        for j, (tok_len, tar_len) in enumerate(zip(token_length, target_length)):
            model_input.append(token_seq[j, : tok_len - tar_len])
        model_input = torch.stack(model_input, dim=0)
        model_output = decoder.predict(model_input, temperature=0.4)
        tokenizer_input = []
        for token in model_output[0]:
            if token < decoder.extra_params.vocab_size_text:
                tokenizer_input.append(token)
        tokenizer_input = torch.stack(tokenizer_input, dim=0)
        predicted_text = slugify(
            tokenizer.decode(tokenizer_input, skip_special_tokens=True).strip()
        )
        label_text = slugify(
            tokenizer.decode(batch["mono_text"][1], skip_special_tokens=True).strip()
        )
        if verbose:
            print(f"Index: {i}, Label: {label_text}, Predicted: {predicted_text}")
            torchaudio.save(
                f"{out_dir}/{i}-{label_text}-{predicted_text}.mp3",
                batch["mono_audio"][0].cpu(),
                sample_rate,
                format="mp3",
            )
        if i >= max_i:
            break
    return


if __name__ == "__main__":
    out_dir = "test_mix"
    sample_rate = 24000
    hop_length = 240
    shuffle_buffer_size = 20
    num_workers = 2
    min_duration = 2
    max_duration = 30
    batch_size = 20

    print("Loading UnifiedMirModel...")
    decoder = init_unified_decoder(
        hpath="hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_decoder_bert-base-uncased-30522_vq32768/checkpoints/step=130000.ckpt",
        local_rank=0,
        cache_dir=".module_cache/umm/unified_decoder",
    )["unified_decoder"]
    print("UnifiedMirModel loaded.")
    print("Loading Stage3...")
    stage3 = init_stage3(
        hpath="hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt",
        local_rank=0,
        cache_dir=".module_cache/umm/",
    )
    decoder.requires.update(stage3)
    print("Stage3 loaded.")
    device = decoder.device
    # ".module_cache/umm/unified_decoder/step=130000.ckpt"
    pl_datamodule = DataModule(
        sample_rate=sample_rate,
        batch_size=1,
        shuffle_buffer_size=1,
        num_workers=1,
        region="CN",
        weights=[1, 0, 0, 0],
        dynamic_batch=False,
        max_length=decoder.extra_params.max_length,
        tokenizer=decoder.extra_params.tokenizer,
    )
    tokenizer = pl_datamodule.tokenizer
    dataloader = iter(pl_datamodule.train_dataloader())

    max_i = 9
    verbose = True
    print("Start evaluation...")
    evaluate(dataloader, verbose, max_i)
