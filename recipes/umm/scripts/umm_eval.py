import torch
import torchaudio
assert torch.cuda.is_available()

from tqdm import tqdm
from transformers import BertTokenizer

from recipes.datasets.mcc.mix import MixWebDataModule
from recipes.umm.modules.lit_module import Stage3

from jiwer import wer, cer


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission_batch):
        """Given a sequence emission over labels, get the best path
        Args:
          emission_batch (Tensor): Logit tensors. Shape `[batch, num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        res = []
        for emission in emission_batch:
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            joined = " ".join([self.labels[i] for i in indices])
            res.append(joined.replace("|", " ").strip())
        return res

@torch.no_grad()
def evaluate(it, verbose=True, max_i=1000):
    mean_wer = []
    mean_cer = []
    mean_edit_distance = []
    mean_mel_loss = []
    mean_chroma_loss = []
    for i, batch in tqdm(enumerate(it)):
        input_batch = {"audio": batch["audio"].to("cuda"), "text": batch["text"]}
        model_input = umm_model.prepare_feature(input_batch)
        model_output = umm_model.model(model_input)

        loss_dict = umm_model.criterion(
            recon_feature=model_output["recon_feature"],
            feature=model_input["feature"],
            logits=model_output["logits"],
            text_ids=model_input["text_ids"],
            recon_chroma=model_output["recon_chroma"],
            chroma=model_input["chroma"],
        )
        mean_mel_loss.append(loss_dict["stft_loss"])
        mean_chroma_loss.append(loss_dict["chroma_stft_loss"])

        actual_transcript = batch["text"]
        greedy_transcript = greedy_decoder(model_output["logits"])
        for j, (a, g) in enumerate(zip(actual_transcript, greedy_transcript)):
            greedy_wer = wer(a.lower(), g.lower())
            greedy_cer = cer(a.lower(), g.lower())
            greedy_edit_distance = torchaudio.functional.edit_distance(a.lower(), g.lower()) / len(a)
            if verbose:
                print("=============================")
                print(f"Actual transcript: {a}")
                print(f"Greedy transcript: {g}")
                print(f"WER: {greedy_wer}")
            mean_wer.append(greedy_wer)
            mean_cer.append(greedy_cer)
            mean_edit_distance.append(greedy_edit_distance)
        if max_i is not None and i >= max_i:
            break
    mean_mel_loss = sum(mean_mel_loss) / len(mean_mel_loss)
    mean_chroma_loss = sum(mean_chroma_loss) / len(mean_chroma_loss)
    mean_wer = sum(mean_wer) / len(mean_wer)
    mean_cer = sum(mean_cer) / len(mean_cer)
    mean_edit_distance = sum(mean_edit_distance) / len(mean_edit_distance)
    print(f"Mean Mel loss: {mean_mel_loss}")
    print(f"Mean Chroma loss: {mean_chroma_loss}")
    print(f"Mean WER: {mean_wer}")
    print(f"Mean CER: {mean_cer}")
    print(f"Mean Edit Distance: {mean_edit_distance}")
    return mean_mel_loss, mean_chroma_loss, mean_wer, mean_cer, mean_edit_distance


if __name__ == "__main__":
    sample_rate = 24000
    hop_length = 240
    shuffle_buffer_size = 20
    num_workers = 2
    min_duration = 2
    max_duration = 30
    batch_size = 20

    pl_datamodule = MixWebDataModule(
        sample_rate=sample_rate,
        batch_size=batch_size * max_duration * sample_rate,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        region="CN",
        weights=[1, 1, 1],
        max_num_crops=None,
        min_duration=min_duration,
        max_duration=max_duration,
        normalize_audio=False,
    )

    ckpt_path = ".module_cache/umm/umm_chroma_25hz_vq32768x32_warmup30000_baseline/checkpoints/step=030000.ckpt"
    # ckpt_path = ".module_cache/umm/umm_chroma_25hz_vq32768x32_warmup30000_mcc1m/checkpoints/step=030000.ckpt"
    umm_model = Stage3.load_from_checkpoint(ckpt_path).to("cuda").eval()
    umm_model.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    vocab = list(umm_model.tokenizer.get_vocab().keys())
    greedy_decoder = GreedyCTCDecoder(vocab)

    val_loaders = pl_datamodule.val_dataloader()
    val_karaoke, val_libritts = val_loaders[0], val_loaders[1]
    val_iter_karaoke, val_iter_libritts = iter(val_karaoke), iter(val_libritts)

    max_i = None
    verbose = False
    eval_out_libritts = evaluate(val_iter_libritts, verbose, max_i)
    eval_out_karaoke = evaluate(val_iter_karaoke, verbose, max_i)
    print(eval_out_karaoke)
    print(eval_out_libritts)
