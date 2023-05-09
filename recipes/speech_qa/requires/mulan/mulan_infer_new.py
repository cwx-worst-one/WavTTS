import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from collections import defaultdict
from transformers import AutoModel, AutoProcessor, BertTokenizer


def tokenize_text(tokenizer, mode="eval"):
    def _tokenize_text(data):
        text = data["text"]

        # Skip empty text
        if text == "":
            return None

        # Tokenize the text
        encodings = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        data["input_ids"] = encodings["input_ids"]
        data["token_type_ids"] = encodings["token_type_ids"]
        data["attention_mask"] = encodings["attention_mask"]

        # Randomly knock out tokens for training
        if mode == "train":
            # scheme 1:
            original_attention_masks = data["attention_mask"]
            mask_of_mask = torch.rand(original_attention_masks.shape)
            sampled_mask = (
                mask_of_mask > 0.05
            ) * original_attention_masks  # TODO random knock out 5%
            data["attention_mask"] = sampled_mask

        return data

    return _tokenize_text


class MusicEncoder(nn.Module):
    def __init__(self, pretrained_model, sample_rate: int = 24000):
        super(MusicEncoder, self).__init__()
        processor = AutoProcessor.from_pretrained(pretrained_model)
        music_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_extract = {  # Use a dict to avoid auto convert fp16
            "mel": torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                # 400 (25ms) in AST with 16K.
                # 42.6ms if using 1024 in 24K.
                # 46.43ms in music tagging (1024, 22050Hz)
                n_fft=1024,
                # 10ms in AST with 16K .
                # 21.3ms if using 512 in 24K.
                # 23.2ms in music tagging task (512, 22050Hz)
                hop_length=240,
                f_min=0,
                f_max=sample_rate // 2,
                n_mels=128,
                window_fn=torch.hann_window,
                power=2.0,
                center=True,
                pad_mode="reflect",
            ),
            "db": torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
            "mel_spec_mean": torch.nn.Parameter(
                torch.tensor(-2.0645375), requires_grad=False
            ),
            "mel_spec_std": torch.nn.Parameter(
                torch.tensor(0.94333875), requires_grad=False
            ),
        }
        self.processor = processor
        self.music_model = music_model
        self.music_model.gradient_checkpointing_enable()
        self.music_linear = nn.Linear(768, 128)
        # spec_aug:
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=192)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=48)

    def manually_to_device(self, device):
        for k, v in self.feat_extract.items():
            self.feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False, normalize=True):
        x = audio
        x = x - torch.mean(x, dim=1, keepdim=True)  # remove DC offset
        mel_spec = self.feat_extract["mel"](x.float())
        mel_spec = self.feat_extract["db"](mel_spec)
        mel_spec = (mel_spec - self.feat_extract["mel_spec_mean"]) / (
            self.feat_extract["mel_spec_std"] * 2
        )  # normalize with AST setting
        if spec_aug:
            mel_spec = self.fm(self.tm(mel_spec))
        mel_spec = mel_spec.to(x.dtype)
        mel_spec = F.pad(mel_spec, ((0, 1024 - mel_spec.shape[-1])))
        mel_spec = mel_spec.permute(0, 2, 1)
        outputs = self.music_model(input_values=mel_spec)
        last_hidden_state = outputs["last_hidden_state"]
        music_output = self.music_linear(last_hidden_state[:, 0, :])
        if normalize:
            music_embed = F.normalize(music_output, p=2, dim=1)
            return music_embed
        else:
            return music_output


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super(TextEncoder, self).__init__()
        self.text_model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.text_model.gradient_checkpointing_enable()
        self.text_linear = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]
        text_output = self.text_linear(last_hidden_state[:, 0, :])
        text_embed = F.normalize(text_output, p=2, dim=1)
        return text_embed


def get_text_encoder(text_encoder="bert"):
    if text_encoder == "bert":
        return TextEncoder("bert-base-uncased")
    else:
        raise NotImplementedError


def get_music_encoder(music_encoder="ast"):
    sample_rate = 24000
    if music_encoder == "ast":
        return MusicEncoder("MIT/ast-finetuned-audioset-10-10-0.4593", sample_rate)
    else:
        raise NotImplementedError


class LitMuLanModule(pl.LightningModule):
    def __init__(
        self, music_encoder, text_encoder, spec_aug, lr, weight_decay, temperature
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        self.music_encoder = get_music_encoder(music_encoder)
        self.text_encoder = get_text_encoder(text_encoder)
        self.spec_aug = spec_aug
        self.lr = lr
        self.weight_decay = weight_decay
        if temperature == "learnable":
            self.temperature = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
            )
        else:
            self.temperature = torch.log(torch.tensor(1 / temperature))

        # Validation outputs
        self.val_outputs = dict()
        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_text(self, mixed_text):
        tokenized_text = self.tokenizer(
            mixed_text.lower(),
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        token_type_ids = tokenized_text["token_type_ids"]
        return input_ids, attention_mask, token_type_ids

    def _shared_step(self, batch, spec_aug=False):
        music_embed = self.music_encoder(batch["audio"], spec_aug)
        input_ids, attention_mask, token_type_ids = self.tokenize_text(batch["texts"])
        text_embed = self.text_encoder(input_ids, attention_mask, token_type_ids)
        return {"text_vec": text_embed, "music_vec": music_embed}

    def encode_text(self, text):
        input_ids, attention_mask, token_type_ids = self.tokenize_text(text)
        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        text_embed = self.text_encoder(input_ids, attention_mask, token_type_ids)
        return text_embed


def create_mulan_model(ckpt_path, device):
    engine = LitMuLanModule.load_from_checkpoint(ckpt_path)
    for k, v in engine.music_encoder.feat_extract.items():
        engine.music_encoder.feat_extract[k] = v.to(device)
    engine.to(device)
    engine.eval()
    for k,v in engine.named_parameters():
        v.requires_grad = False

    return engine


@torch.no_grad()
def mulan_inference(
    model, text=None, music=None, device="cpu", avg=True, shift_seconds=5, normalize=True
):
    assert (text is not None) ^ (
        music is not None
    ), "text inputs and music input can only select one"

    if text is not None:
        emb = model.encode_text(text)

    if music is not None:
        # print(f"mulan_inference: wav shape is {music.shape}")
        music_encoder = model.music_encoder
        music = music.unfold(1, 24000 * 10, 24000 * shift_seconds)  # [b, n, t]
        # print(f"mulan_inference: unfolded wav shape is {music.shape}")
        b, n, t = music.shape
        music = music.reshape(b * n, t)
        # print(f"mulan_inference: reshaped wav shape is {music.shape}")
        emb = music_encoder(music, normalize=normalize)
        # print(f"mulan_inference: embe shape is {emb.shape}, b={b}, n={n}, t={t}")
        emb = emb.reshape(b, n, -1)
        if avg:
            # print("averaging embeds")
            emb = emb.mean(dim=1)
    return emb


def mulan_rvq_indexs(z, centers):
    # z: [b, d]
    # center: [n, d]
    indexs = []
    ds = []
    for center in centers:
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(center**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z, rearrange(center, "n d -> d n"))
        )
        min_index = torch.argmin(d, dim=1)  # [b, ]

        # RVQ
        z = z - torch.nn.functional.embedding(min_index, center)

        indexs.append(min_index)
        ds.append(d.min())
    indexs = torch.stack(indexs, dim=1)  # [b, 12]
    return indexs, ds
