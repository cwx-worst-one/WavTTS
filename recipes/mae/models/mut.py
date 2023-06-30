import torch
import torch.nn.functional as F
import torchaudio
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.utils.checkpoint import checkpoint

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
        self.num_layers = depth
        self.hidden_size = dim
        self.intermediate_size = mlp_dim

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
        }, "pool type must be either cls (cls token), mean (mean pooling) or seq (output sequence)"
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
        pretained_path: str = "mutmae-step=177600-loss_1=5-sf.pth",
    ):
        super(PretrainedMuTWrapper, self).__init__()
        mut = MuT(
            spec_shape=(128, 1000),
            patch_shape=(128, 2),
            num_classes=1000,
            sample_rate=24000,
            dim=1280,
            depth=32,
            heads=16,
            dim_head=80,
            channels=1,
            mlp_dim=5120,
            checkpointing=checkpointing,
            use_flash_attn=use_flash_attn,
            output_type=output_type,
        )
        state_dict = torch.load(pretained_path, map_location="cpu")
        mut.load_state_dict(state_dict, strict=False)
        mut.mlp_head = output_layer
        self.mut = mut

    def manually_to_device(self, device):
        for k, v in self.mut.logmel_frontend["logmel"].feat_extract.items():
            self.mut.logmel_frontend["logmel"].feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False):
        out = self.mut(audio, spec_aug=spec_aug)

        return out


if __name__ == "__main__":
    # model = MuT(
    #     spec_shape=(128, 1000),
    #     patch_shape=(128, 2),
    #     num_classes=1000,
    #     sample_rate=24000,
    #     dim=1024,
    #     depth=6,
    #     heads=8,
    #     dim_head=128,
    #     channels=1,
    #     mlp_dim=2048
    # )
    # dummy_input = torch.randn(3, 1, 240000)
    # print(model(dummy_input).shape)
    torch.manual_seed(42)
    output_layer = nn.Sequential(RMSNorm(1280), nn.Linear(1280, 20))
    mut = PretrainedMuTWrapper(
        output_layer, checkpointing=True, use_flash_attn=False, output_type="seq"
    )
    device = torch.device("cuda:0")
    mut.to(device)
    mut.eval()
    mut.manually_to_device(device)
    dummy_input = torch.randn(3, 1, 240000)
    output = mut(dummy_input.to(device))
    print(output[0][0])
    print(output.shape)
