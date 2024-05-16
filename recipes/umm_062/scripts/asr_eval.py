import torch

assert torch.cuda.is_available()

from jiwer import cer
from tqdm import tqdm

from recipes.datasets.mcc.mix import DouyinMusicDataModule, MixZhWebDataModule
from recipes.datasets.mcc.sami_tokenizer import SamiTokenizer, all_phones, offset
from recipes.umm_062.modules.lit_module import ASR
from recipes.umm_062.requires.model_initializer import init_sami_tts_api


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=266, padding=0):
        super().__init__()
        self.blank = blank
        self.padding = padding

    def forward(self, emission_batch):
        res = []
        for emission in emission_batch:
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [
                i.item() for i in indices if (i != self.blank and i != self.padding)
            ]
            res.append(indices)
        return res


@torch.no_grad()
def evaluate(it, verbose=True, max_i=1000):
    char_s = 65
    mean_cer = []
    for i, batch in tqdm(enumerate(it)):
        input_batch = {"audio": batch["audio"].to("cuda"), "text": batch["text"]}
        model_input = umm_model.prepare_feature(input_batch)
        model_output = umm_model.model(model_input)

        actual_transcript = batch["text"]
        actual_phonemes = []
        for phonemes in model_input["text_ids"]:
            actual_phonemes.append(
                [
                    i.item()
                    for i in phonemes
                    if (i != greedy_decoder.blank and i != greedy_decoder.padding)
                ]
            )
        greedy_phonemes = greedy_decoder(model_output["ctc_out"])
        for j, (at, ap, gp) in enumerate(
            zip(actual_transcript, actual_phonemes, greedy_phonemes)
        ):
            greedy_cer = cer(
                "".join([chr(char_s + i) for i in ap]),
                "".join([chr(char_s + i) for i in gp]),
            )
            if verbose:
                res_log.write("=============================\n")
                res_log.write(f"Actual transcript: {at}")
                res_log.write("\n")
                ap_string = "\t".join(all_phones[i - offset] for i in ap)
                res_log.write(f"Actual phonemes: {ap_string}")
                res_log.write("\n")
                gp_string = "\t".join(all_phones[i - offset] for i in gp)
                res_log.write(f"Greedy phonemes: {gp_string}")
                res_log.write("\n")
                res_log.write(f"CER: {greedy_cer}")
                res_log.write("\n")
            mean_cer.append(greedy_cer)
        if max_i is not None and i >= max_i:
            break
    mean_cer = sum(mean_cer) / len(mean_cer)
    res_log.write(f"Mean CER: {mean_cer}\n")
    return mean_cer


if __name__ == "__main__":
    res_log = open("asr_eval_out_hotgalaxy.txt", "w")
    init_sami_tts_api("42.0", "tts_chinese_frontend_model")
    sample_rate = 24000
    hop_length = 240
    shuffle_buffer_size = 10
    num_workers = 1
    min_duration = 2
    max_duration = 30
    batch_size = 20

    pl_datamodule = MixZhWebDataModule(
        # languages=["zh"],
        sample_rate=sample_rate,
        batch_size=batch_size * max_duration * sample_rate,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        weights=[1],
        region="CN",
        min_duration=min_duration,
        max_duration=max_duration,
        normalize_audio=False,
        tokenizer=None,
    )

    ckpt_path = ".module_cache/umm/umm_asr_zh_douyin/step=020000.ckpt"
    umm_model = ASR.load_from_checkpoint(ckpt_path).to("cuda").eval()
    umm_model.tokenizer = SamiTokenizer()

    greedy_decoder = GreedyCTCDecoder()

    dataloader = iter(pl_datamodule.train_dataloader())

    max_i = 600  # 100hrs
    verbose = True
    eval_out = evaluate(dataloader, verbose, max_i)
    res_log.close()
