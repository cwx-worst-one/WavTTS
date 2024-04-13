import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from apps.bigmusic.umm.ar.utils.utils import sample

from samantha.utils.hparams import DotDict

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_flash_attn=False):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.use_flash_attn = use_flash_attn

    def forward(self, x, rotary_emb=None, masked_n_unmasked=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if rotary_emb is not None:
            if masked_n_unmasked is not None:
                # expand qk to original dimension before rotate
                # temp solution
                original_len = (
                    masked_n_unmasked[0].shape[1] + masked_n_unmasked[1].shape[1]
                )
                b, h, _, d = q.shape
                expand_q = torch.zeros(
                    (b, h, original_len, d), device=q.device, dtype=q.dtype
                )
                expand_k = torch.zeros(
                    (b, h, original_len, d), device=k.device, dtype=k.dtype
                )

                batch_range = torch.arange(b, device=q.device)[:, None]
                expand_q.permute(0, 2, 1, 3)[
                    batch_range, masked_n_unmasked[1]
                ] = q.permute(0, 2, 1, 3)
                expand_k.permute(0, 2, 1, 3)[
                    batch_range, masked_n_unmasked[1]
                ] = k.permute(0, 2, 1, 3)

                q = (
                    rotary_emb.rotate_queries_or_keys(expand_q)
                    .permute(0, 2, 1, 3)[batch_range, masked_n_unmasked[1]]
                    .permute(0, 2, 1, 3)
                )
                k = (
                    rotary_emb.rotate_queries_or_keys(expand_k)
                    .permute(0, 2, 1, 3)[batch_range, masked_n_unmasked[1]]
                    .permute(0, 2, 1, 3)
                )

            else:
                q = rotary_emb.rotate_queries_or_keys(q)
                k = rotary_emb.rotate_queries_or_keys(k)

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        checkpointing=True,
        use_flash_attn=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_0": RMSNorm(dim),
                        "attn": Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_flash_attn=use_flash_attn,
                        ),
                        "norm_1": RMSNorm(dim),
                        "ff": FeedForward(dim, mlp_dim, dropout=dropout),
                    }
                )
            )
        self.checkpointing = checkpointing

    def _forward_checkpointing(self, x, rotary_emb=None, masked_n_unmasked=None):
        for layer in self.layers:
            norm_x = checkpoint(layer["norm_0"], x, use_reentrant=False)

            if rotary_emb is not None:
                if masked_n_unmasked is not None:
                    x = (
                        checkpoint(
                            layer["attn"],
                            norm_x,
                            rotary_emb,
                            masked_n_unmasked,
                            use_reentrant=False,
                        )
                        + x
                    )
                else:
                    x = (
                        checkpoint(
                            layer["attn"], norm_x, rotary_emb, use_reentrant=False
                        )
                        + x
                    )
            else:
                x = checkpoint(layer["attn"], norm_x, use_reentrant=False) + x

            x = (
                checkpoint(
                    layer["ff"],
                    checkpoint(layer["norm_1"], x, use_reentrant=False),
                    use_reentrant=False,
                )
                + x
            )
        return x

    def _forward(self, x, rotary_emb=None, masked_n_unmasked=None):
        for layer in self.layers:
            if rotary_emb is not None:
                if masked_n_unmasked is not None:
                    x = (
                        layer["attn"](layer["norm_0"](x), rotary_emb, masked_n_unmasked)
                        + x
                    )
                else:
                    x = layer["attn"](layer["norm_0"](x), rotary_emb) + x
            else:
                x = layer["attn"](layer["norm_0"](x)) + x
            x = layer["ff"](layer["norm_1"](x)) + x
        return x

    def forward(self, x, rotary_emb=None, masked_n_unmasked=None):
        if self.checkpointing:
            return self._forward_checkpointing(x, rotary_emb, masked_n_unmasked)
        else:
            return self._forward(x, rotary_emb, masked_n_unmasked)


class LogMel(nn.Module):
    def __init__(self, sample_rate=24000):
        super().__init__()

        self.feat_extract = {
            "mel": torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
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

    def forward(self, audio):
        with torch.autocast(device_type="cuda", enabled=False):
            audio = audio - torch.mean(audio, dim=-1, keepdim=True)  # remove DC offset
            mel_spec = self.feat_extract["mel"](audio.float())[
                ..., :-1
            ]  # remove last frame
            mel_spec = self.feat_extract["db"](mel_spec)
            mel_spec = (mel_spec - self.feat_extract["mel_spec_mean"]) / (
                self.feat_extract["mel_spec_std"] * 2
            )  # normalize
        mel_spec = mel_spec.to(audio.dtype)
        return mel_spec


class MuT(nn.Module):
    def __init__(
        self,
        *,
        spec_shape,
        patch_shape,
        num_classes,
        sample_rate,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        checkpointing=True,
        use_flash_attn=False,
        output_type="emb",
    ):
        super().__init__()

        self.logmel_frontend = {"logmel": LogMel(sample_rate=sample_rate)}

        spec_height, spec_width = spec_shape
        patch_height, patch_width = patch_shape
        assert (
            spec_height % patch_height == 0 and spec_width % patch_width == 0
        ), "Spectrogram dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width
        self.patch_dim = patch_dim
        self.dim = dim
        self.to_patch_embedding = nn.ModuleDict(
            {
                "to_patch": Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                "norm_0": RMSNorm(patch_dim),
                "linear": nn.Linear(patch_dim, dim),
                "norm_1": RMSNorm(dim),
            }
        )

        assert output_type in {
            "cls",
            "mean",
            "seq",
        }, "pool type must be either cls (cls token), mean (mean pooling) or seq (output sequence)"  # noqa

        self.output_type = output_type
        if output_type in ["cls"]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.rotary_emb = RotaryEmbedding(dim=int(dim / heads))

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            checkpointing,
            use_flash_attn=use_flash_attn,
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=192)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=48)

    def forward(self, audio, masked_n_unmasked=None, spec_aug=False):
        # to mel spectrogram
        mel_spec = self.logmel_frontend["logmel"](audio)

        if spec_aug:
            mel_spec = self.fm(self.tm(mel_spec))

        # to patch embedding
        for _, layer in self.to_patch_embedding.items():
            mel_spec = layer(mel_spec)
        patch_emb = nn.Identity()(mel_spec)
        b, n, _ = patch_emb.shape

        # add cls token
        if self.output_type in ["cls"]:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
            x = torch.cat((cls_tokens, patch_emb), dim=1)
        else:
            x = nn.Identity()(patch_emb)
        x = self.dropout(x)

        x = self.transformer(x, self.rotary_emb, masked_n_unmasked)

        if self.output_type == "mean":
            x = x.mean(dim=1)
        elif self.output_type == "cls":
            x = x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)


class PretrainedMuTWrapper(nn.Module):
    def __init__(
        self,
        output_layer,  # output dim from mut is 1280
        checkpointing=True,
        use_flash_attn=False,
        output_type="cls",
        num_layers=32,
        pretained_path: str = "mutmae-step=177600-loss_1=5-sf.pth",
    ):
        super(PretrainedMuTWrapper, self).__init__()
        mut = MuT(
            spec_shape=(128, 1000),
            patch_shape=(128, 2),
            num_classes=1000,
            sample_rate=24000,
            dim=1280,
            depth=num_layers,
            heads=16,
            dim_head=80,
            channels=1,
            mlp_dim=5120,
            checkpointing=checkpointing,
            use_flash_attn=use_flash_attn,
            output_type=output_type,
        )
        # state_dict = torch.load(pretained_path, map_location='cpu')
        # mut.load_state_dict(state_dict, strict=False)
        mut.mlp_head = output_layer
        self.mut = mut

    def manually_to_device(self, device):
        for k, v in self.mut.logmel_frontend["logmel"].feat_extract.items():
            self.mut.logmel_frontend["logmel"].feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False):
        out = self.mut(audio, spec_aug=spec_aug)

        return out


class MuTinyWrapper(nn.Module):
    def __init__(self, emb_dim: int = 128, output_type="seq"):
        super(MuTinyWrapper, self).__init__()

        self.emb_dim = emb_dim
        mlp_head = nn.Sequential(RMSNorm(1280), nn.AvgPool2d((2, 1)), nn.Linear(1280, emb_dim))
        mut = PretrainedMuTWrapper(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=177600-loss_1=5-sf.pth",
            num_layers=16,
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", emb_dim: int = 128, output_type="cls"):
        super(TextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.output_type = output_type
        self.text_model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.text_model.gradient_checkpointing_enable()
        self.text_linear = nn.Linear(1024, emb_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]
        if self.output_type == "cls":
            text_output = last_hidden_state[:, 0, :]
        else:
            text_output = last_hidden_state
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed


def get_text_encoder(text_encoder="bert", emb_dim=128):
    if text_encoder == "bert":
        return TextEncoder("bert-large-uncased", emb_dim, output_type="seq")
    else:
        raise NotImplementedError


def get_music_encoder(music_encoder="mut-tiny", emb_dim=128):
    if music_encoder == "mut-tiny":
        return MuTinyWrapper(emb_dim)
    else:
        raise NotImplementedError


class FiLMGenModel(pl.LightningModule):
    def __init__(
        self,
        music_encoder,
        text_encoder,
        decoder_cls,
        criterion_cls,
        required_modules,
        size_params,
        lr,
        gen_lr,
        weight_decay,
        gen_batch_size: int = 4,
        temperature: float = 0.1,
        mulan_loss_weight: float = 0.5,
        use_flatclr: bool = False,
        gather_batches: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        self.size_params = DotDict(size_params)

        self.music_encoder = get_music_encoder(
            music_encoder, self.size_params.emb_dim
        )
        self.text_encoder = get_text_encoder(
            text_encoder, self.size_params.emb_dim
        )
        self.decoder = decoder_cls()
        self.criterion = criterion_cls()
        self.requires = {}

        self.temperature = torch.log(torch.tensor(1 / self.hparams.temperature))

        # Validation outputs
        self.val_outputs = dict()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    def tokenize_text(self, texts):
        tokenized_text = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=250,
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        token_type_ids = tokenized_text["token_type_ids"]
        return input_ids, attention_mask, token_type_ids

    def encode_text(self, texts):
        input_ids, attention_mask, token_type_ids = self.tokenize_text(texts)
        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        text_embed = self.text_encoder(input_ids, attention_mask, token_type_ids)
        return text_embed
    
    def decode_coarse(self, text_embeds, temp=0.9, sample_mode="gumbel"):
        device = text_embeds.device
        b = text_embeds.size(0)
        num_coarse = 4
        soundstream_codebook_size = 1024
        soundstream_frame_rate = 50
        coarse_duration = 10
        duration = 10
        sample_len = 2000
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=torch.long, device=device)
            + num_coarse * soundstream_codebook_size
        )
        input_ids = sos_ids
        pbar = tqdm(range(sample_len))
        pbar.set_description("FilmGen")
        coarse_samples = None
        past_key_values = None
        for i in pbar:
            model_output = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=text_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = model_output["past_key_values"]
            logits = model_output["logits"]
            layer_idx = i % num_coarse
            predict_logits = logits[:, -1:, layer_idx * 1024 : (layer_idx + 1) * 1024]
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            samples = samples + layer_idx * 1024
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return coarse_samples


def create_mulan_model(ckpt_path, device):
    litmodel = FiLMGenModel.load_from_checkpoint(ckpt_path)

    # text tower
    litmodel.text_encoder.eval()
    litmodel.text_encoder.to(device)
    # decoder
    litmodel.decoder.eval()
    litmodel.decoder.to(device)

    return litmodel


@torch.no_grad()
def mulan_inference(
    model, texts=None, device="cpu", avg=True, shift_seconds=1
):
    assert (texts is not None), "text inputs cannot be None"

    emb = model.encode_text(texts)
    coarse_samples = model.decode_coarse(emb)

    return coarse_samples

if __name__ == "__main__":
    ckpt_path = "/mnt/bn/audio-diffusion/filmgen_exp/film_72_gen_8_diff_lr/checkpoints/mulan-step=029000-median_rank_0=70-kaggle.ckpt"
    model = create_mulan_model(ckpt_path, device="cuda")
    text = ["piano", "guitar music"]
    coarse = mulan_inference(model, texts=text)
