import itertools
from collections import defaultdict

import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig, ClapModel

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


class TextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model: str = "bert-base-uncased",
        emb_dim: int = 128,
        text_emb_dim: int = 1024,
    ):
        super(TextEncoder, self).__init__()
        if pretrained_model in {"bert-base-uncased", "bert-large-uncased"}:
            config = AutoConfig.from_pretrained(pretrained_model)
            self.text_model = AutoModel.from_config(config, add_pooling_layer=False)
            self.text_model.gradient_checkpointing_enable()
        else:
            if pretrained_model == "laion/larger_clap_general":
                try:
                    self.text_model = ClapModel.from_pretrained('.module_cache/huggingface/larger_clap_general')
                except Exception as e:
                    print(f"Failed to load from local cache, download from huggingface instead: {e}")
                    self.text_model = ClapModel.from_pretrained(pretrained_model)
            else:
                self.text_model = ClapModel.from_pretrained(pretrained_model)
        self.text_linear = nn.Linear(text_emb_dim, emb_dim)
        self.pretrained_model = pretrained_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.pretrained_model == "laion/larger_clap_general":
            outputs = self.text_model.get_text_features(
                input_ids, attention_mask=attention_mask
            )
            text_output = self.text_linear(outputs)
        else:
            outputs = self.text_model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            last_hidden_state = outputs["last_hidden_state"]
            text_output = self.text_linear(last_hidden_state[:, 0, :])
        text_embed = F.normalize(text_output, p=2, dim=1)
        return text_embed


def get_text_encoder(text_encoder="bert", emb_dim=128):
    if text_encoder == "bert":
        return TextEncoder("bert-large-uncased", emb_dim)
    elif text_encoder == "clap":
        return TextEncoder("laion/larger_clap_general", emb_dim, text_emb_dim=512)
    else:
        raise NotImplementedError


class MusicEncoder(nn.Module):
    def __init__(self, pretrained_model, emb_dim: int = 128, sample_rate: int = 24000):
        super(MusicEncoder, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        processor = AutoProcessor.from_config(config)
        music_model =  AutoModel.from_config(config)
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
        self.music_linear = nn.Linear(768, emb_dim)
        # spec_aug:
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=192)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=48)

    def manually_to_device(self, device):
        for k, v in self.feat_extract.items():
            self.feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False):
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
        music_embed = F.normalize(music_output, p=2, dim=1)
        return music_embed


class PretrainedMuTSSTKWrapper(nn.Module):
    def __init__(
        self,
        output_layer,  # output dim from mut is 1280
        pretrained_path,
        checkpointing=True,
        use_flash_attn=False,
        output_type="cls",
        num_layers: int = 32,
        patch_shape=(128, 4),
    ):
        super(PretrainedMuTSSTKWrapper, self).__init__()
        mut = MuT(
            spec_shape=(128, 1000),
            patch_shape=patch_shape,
            num_classes=1000,
            sample_rate=24000,
            dim=1280,
            depth=32, #32
            heads=16,
            dim_head=80,
            channels=1,
            mlp_dim=5120,
            checkpointing=True,
            use_flash_attn=False,
            output_type=output_type,
        )
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            mut.load_state_dict(state_dict, strict=False)
        mut.mlp_head = output_layer
        self.mut = mut

    def manually_to_device(self, device):
        for k, v in self.mut.logmel_frontend["logmel"].feat_extract.items():
            self.mut.logmel_frontend["logmel"].feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False):
        out = self.mut(audio, spec_aug=spec_aug)

        return out

class MuTSSTKMAEWrapper(nn.Module):
    def __init__(self, emb_dim: int = 128, output_type="cls", seq_len=500, version="v1"):
        super(MuTSSTKMAEWrapper, self).__init__()

        self.emb_dim = emb_dim
        if seq_len == 500:
            mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        else:
            raise NotImplementedError(f"Seq len {seq_len} is not supported yet.")

        if version == "v1":
            pretrained_path = "/mnt/bn/audio-diffusion/xuchen/mulan/models/mutmae-step=563200-loss_0=7-kaggle.pth"
            patch_shape = (128, 4)
        else:
            pretrained_path = "/mnt/bn/audio-diffusion/weituo/sstk_mulan/assets/mutmae-step=177600-loss_1=5-sf.pth"
            patch_shape = (128, 2)
        mut = PretrainedMuTSSTKWrapper(
            output_layer=mlp_head,
            pretrained_path=pretrained_path,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            patch_shape=patch_shape,
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
   

def get_music_encoder(music_encoder="sstk", emb_dim=128, version="v1"):
    return MuTSSTKMAEWrapper(emb_dim, version=version)


class LitMuLanModule(pl.LightningModule):
    def __init__(
        self,
        music_encoder,
        text_encoder,
        emb_dim,
        spec_aug,
        lr,
        weight_decay,
        temperature,
        version="v1",
        load_text_tower: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        print(f"music_encoder: {music_encoder}, text_encoder: {text_encoder}")
        self.music_encoder = get_music_encoder(music_encoder, emb_dim, version=version)
        if load_text_tower:
            self.text_encoder = get_text_encoder(text_encoder, emb_dim)
            if text_encoder == 'clap':
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(".module_cache/huggingface/larger_clap_general")
                except Exception as e:
                    self.tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(".module_cache/huggingface/bert-large-uncased")
                except Exception as e:
                    self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

        else:
            print("Skipping load text tower")
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

    def on_fit_start(self):
        self.music_encoder.mut.manually_to_device(self.device)

    def on_predict_start(self):
        # Temp solution, hardcode ckpt here
        state_dict = torch.load(
            "hdfs://haruna/home/byte_speech_sv/user/weituo/"
            "mulan_ckpt/music_encoder_mulan_149.ckpt"
        )
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("music_model", "music_encoder")] = v
        self.music_encoder.load_state_dict(state_dict)
        self.music_encoder.manually_to_device(self.device)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            return DeepSpeedCPUAdam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            optimizer = FusedAdam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            base_params = []
            norm_params = []
            no_decay = [
                "temperature",
                "bn",
                "bias",
                "LayerNorm",
                "embeddings",
                "layernorm",
                "norm.bias",
                "norm.weight",
                "rotary",
            ]
            for name, param in self.named_parameters():  # self.parameters()
                _found = False
                for k in no_decay:
                    if k in name:
                        norm_params.append(param)
                        _found = True
                        break
                if not _found:
                    base_params.append(param)

            optimizer = torch.optim.AdamW(
                [{"params": base_params}, {"params": norm_params, "weight_decay": 0.0}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        return optimizer

    def _multiview_loss(self, text_vec, music_vec, music_id):
        # Size of text_vec: (batch_size * world_size, embed_dim),
        # eg: (8*4, 128)=(32,128)
        # Size of music_vec: (batch_size * world_size, embed_dim),
        # eg: (8*4, 128)
        # Slice text embed based on self.global_rank
        # Then calculate the dot product between text and music
        # Size of dot_product: (batch_size, batch_size * world_size), eg: (8, 32)
        batch_size = text_vec.shape[0] // self.trainer.world_size
        start = self.global_rank * batch_size
        end = start + batch_size
        dot_product = torch.mm(text_vec[start:end, :], music_vec.t())

        # Gather all the dot products from all the GPUs
        # Size of dot_product: (batch_size * world_size, batch_size * world_size),
        # eg: (32, 32)
        dot_product = rearrange(
            self.all_gather(dot_product, sync_grads=True), "w b d -> (w b) d"
        )
        dot_product = dot_product * self.temperature.exp()
        dot_product = dot_product - dot_product.max()
        # Calculate exp(dot_product/temperature)
        loss_mat = torch.exp(dot_product)

        # Collect music id
        info_dict = defaultdict(list)
        for i, item in enumerate(music_id.tolist()):
            info_dict[item].append(i)

        false_negative = list()
        for kk in info_dict:
            false_negative.extend(itertools.product(info_dict[kk], repeat=2))

        select_negative = torch.ones_like(loss_mat)

        for item in false_negative:
            select_negative[item[0], item[1]] = 0

        negative_mat = loss_mat * select_negative
        negative_mat[~select_negative.bool()] = negative_mat[
            ~select_negative.bool()
        ].detach()

        loss = (loss_mat.diagonal()) / (
            loss_mat.diagonal()
            + negative_mat.sum(dim=1)
            + negative_mat.sum(dim=0)
            + torch.finfo(loss_mat.dtype).tiny
        )

        loss = torch.mean(-torch.log(loss + torch.finfo(loss.dtype).tiny))
        self.log("tr_loss", loss, prog_bar=True)
        return loss

    def _hit_score_rank(self, source_embed, target_embed):
        assert source_embed.shape[1] == target_embed.shape[1]
        sample_size = source_embed.shape[0]
        mat = torch.matmul(source_embed, target_embed.T)
        smat, indx = mat.sort(dim=1, descending=True)
        score, ranklst = list(), list()
        for i in range(sample_size):
            rank = (indx[i] == i).nonzero(as_tuple=True)[0]
            ranklst.append(rank)
            score.append((sample_size - rank) / sample_size)

        median_rank = sorted(ranklst)[len(ranklst) // 2]

        return {"hit_score": sum(score) / len(score), "median_rank": median_rank}

    def training_step(self, batch, batch_idx):
        # Combine multiple dataloader batches into one batch

        text_vec, music_vec = self._shared_step(batch, spec_aug=self.spec_aug).values()
        text_vec = rearrange(
            self.all_gather(text_vec, sync_grads=True), "w b d -> (w b) d"
        )
        music_vec = rearrange(
            self.all_gather(music_vec, sync_grads=True), "w b d -> (w b) d"
        )
        music_id = rearrange(
            self.all_gather(batch["music_id"], sync_grads=True), "w b -> (w b) "
        )

        loss = self._multiview_loss(text_vec, music_vec, music_id)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        result = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(result)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            text_vecs = []
            music_vecs = []
            for output in outputs:
                text_vecs += output["text_vec"]
                music_vecs += output["music_vec"]

            text_vecs = torch.stack(text_vecs)
            music_vecs = torch.stack(music_vecs)
            text_vecs = rearrange(self.all_gather(text_vecs), "w b d -> (w b) d")
            music_vecs = rearrange(self.all_gather(music_vecs), "w b d -> (w b) d")

            rank_zero_info(
                f"text_vecs.shape: {text_vecs.shape}, "
                f"music_vecs.shape: {music_vecs.shape}"
            )

            score = self._hit_score_rank(text_vecs, music_vecs)
            self.log_dict(
                {
                    f"median_rank_{dataloader_idx}": score["median_rank"],
                    f"hit_score_{dataloader_idx}": score["hit_score"],
                },
                prog_bar=True,
            )
            self.val_outputs[dataloader_idx] = []

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        music_embed = self.music_encoder(batch["audio"])
        return {"music_vec": music_embed, "music_id": batch["music_id"]}

    def _shared_step(self, batch, spec_aug=False):
        music_embed = self.music_encoder(batch["audio"].unsqueeze(1), spec_aug=spec_aug)
        text_embed = self.text_encoder(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        return {"text_vec": text_embed, "music_vec": music_embed}

    def tokenize_text(self, text):
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=400,
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        token_type_ids = tokenized_text.get("token_type_ids", None)
        return input_ids, attention_mask, token_type_ids

    def encode_text(self, text, return_hidden_state):
        input_ids, attention_mask, token_type_ids = self.tokenize_text(text)
        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if return_hidden_state:
            result_dict = self.text_encoder.text_model.text_model(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
            return result_dict.hidden_states[-2]
        text_embed = self.text_encoder(input_ids, attention_mask, token_type_ids)
        return text_embed


def create_mulan_model(ckpt_path, device, version="v1", load_text_tower=True):
    strict = load_text_tower
    litmodel = LitMuLanModule.load_from_checkpoint(
        ckpt_path,
        version=version,
        map_location='cpu',
        strict=strict,
        load_text_tower=load_text_tower
    )

    # audio tower
    litmodel.music_encoder.eval()
    litmodel.music_encoder.to(device)
    litmodel.music_encoder.mut.manually_to_device(device)
    if load_text_tower:
        # text tower
        litmodel.text_encoder.eval()
        litmodel.text_encoder.to(device)

    # litmodel
    litmodel.to(device)

    # litmodel
    litmodel.to(device)

    return litmodel


@torch.no_grad()
def mulan_inference(
    model,
    text=None,
    music=None,
    device="cpu",
    avg=True,
    shift_seconds=5,
    normalize_text=False,
    return_hidden_state=False,
    return_sequence=0,
):
    assert (text is not None) ^ (
        music is not None
    ), "text inputs and music input can only select one"

    if text is not None:
        if normalize_text:
            if isinstance(text, str):
                text = [text]
            text = [" ".join(x.split(',')) for x in text]
        emb = model.encode_text(text, return_hidden_state)

    if music is not None:
        # music needs to be in 2D: [b, t]
        if len(music.shape) == 3:
            music = music.squeeze(1)
        # print(f"mulan_inference: wav shape is {music.shape}")
        music_encoder = model.music_encoder
        music = music.unfold(1, 24000 * 10, 24000 * shift_seconds)  # [b, n, t]
        # print(f"mulan_inference: unfolded wav shape is {music.shape}")
        b, n, t = music.shape
        music = music.reshape(b * n, t)
        # print(f"mulan_inference: reshaped wav shape is {music.shape}")

        if return_sequence:
            saved_output_type = music_encoder.mut.mut.output_type
            music_encoder.mut.mut.output_type = "seq"
            emb = music_encoder(music.unsqueeze(1))
            music_encoder.mut.mut.output_type = saved_output_type
            if return_sequence == 2:
                # do not unfold sequence. for previous fad_mulan compatibility
                return emb
            t, e = emb.shape[-2:] # fold back into original shape
            emb = emb.reshape(b, n * t, e)
            return emb

        emb = music_encoder(music.unsqueeze(1))
        # print(f"mulan_inference: embe shape is {emb.shape}, b={b}, n={n}, t={t}")
        emb = emb.reshape(b, n, -1)
        if avg:
            # print("averaging embeds")
            emb = F.normalize(emb.mean(dim=1), p=2, dim=1)
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
