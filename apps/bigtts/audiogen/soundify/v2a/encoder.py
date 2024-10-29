import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from apps.bigtts.audiogen.open_clip import create_model_from_pretrained
from apps.bigtts.audiogen.soundify.v2a.action import ActionEncoder

from apps.bigtts.audiogen.soundify.modules.based_ctiga_llama import LLaMa
from apps.bigtts.audiogen.soundify.modules.based_ctiga_llama import ModelArgs as LLamaArgs


@dataclass
class ModelArgs:
    in_channels: int = 512
    out_channels: int = 512
    encoder_dim: int = 1536
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    causal: bool = False
    llama_provider: str = 'ctiga'
    use_unet_style_skip_connect: bool = True
    use_qk_norm: str = 'head'
    bias: bool = False


class VideoProjector(nn.Module):

    def __init__(self, feat_dim=768, hidden_dim=512, out_dim=512, mlp_depth=3, bias=False):
        super(VideoProjector, self).__init__()

        modules = [nn.Linear(feat_dim, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

        self.mlp = nn.Sequential(*modules)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, video_feats):

        video_embed = self.mlp(video_feats)
        video_embed = self.linear(video_embed)

        return video_embed


class VideoEncoder(nn.Module):

    def __init__(self):
        super(VideoEncoder, self).__init__()

        hp = ModelArgs()

        llama_config = LLamaArgs(dim=hp.encoder_dim,
                                 n_layers=hp.encoder_n_layers,
                                 n_heads=hp.encoder_n_heads,
                                 causal=hp.causal,
                                 use_unet_style_skip_connect=hp.use_unet_style_skip_connect,
                                 use_qk_norm=hp.use_qk_norm)

        self.preprocess = transforms.Compose([
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

        ########################################################################
        # 24 layers, 427.62M
        self.image_backbone = create_model_from_pretrained('hf-hub:apple/DFN2B-CLIP-ViT-B-16',
                                                           cache_dir=".deploy_cache")
        del self.image_backbone.transformer
        self.image_backbone.eval()
        for name, param in self.image_backbone.named_parameters():
            param.requires_grad = False
        ########################################################################
        self.action_encoder = ActionEncoder(embed_dim=hp.in_channels)
        ########################################################################
        self.video_fuse = nn.Linear(hp.in_channels, hp.encoder_dim)
        self.video_llama = LLaMa(llama_config, hp.llama_provider)
        self.video_projector = VideoProjector(feat_dim=hp.encoder_dim,
                                              hidden_dim=hp.encoder_dim,
                                              out_dim=hp.out_channels,
                                              mlp_depth=3,
                                              bias=False)

    def forward(self, video):

        bts, t, c, h, w = video.shape  # [b, t, c, h, w]
        video = video.flatten(0, 1)  # [b*t, c, h, w]
        video = self.preprocess(video)
        ##################################################################
        image_embed = self.image_backbone.encode_image(video)
        image_embed = image_embed.reshape(bts, t, -1)  # [b, t, d]
        ##################################################################
        video = video.reshape(bts, t, c, h, w)
        action_embed = self.action_encoder(video)
        ##################################################################
        video_embed = self.video_fuse(image_embed + action_embed)
        video_embed = self.video_llama(video_embed, seqlen=image_embed.shape[1])
        video_embed = self.video_projector(video_embed)

        return video_embed  # [b, t, d]

    @torch.no_grad()
    def infer(self, video):

        bts, t, c, h, w = video.shape  # [b, t, c, h, w]
        video = video.flatten(0, 1)  # [b*t, c, h, w]
        video = self.preprocess(video)
        ##################################################################
        image_embed = self.image_backbone.encode_image(video)
        image_embed = image_embed.reshape(bts, t, -1)  # [b, t, d]
        ##################################################################
        video = video.reshape(bts, t, c, h, w)
        action_embed = self.action_encoder(video)
        ##################################################################
        video_embed = self.video_fuse(image_embed + action_embed)
        video_embed = self.video_llama(video_embed, seqlen=image_embed.shape[1])
        video_embed = self.video_projector(video_embed)

        return video_embed  # [b, t, d]

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":

    device = "cuda"

    videos = torch.randn([2, 8, 3, 224, 224]).to(device)

    with torch.autocast(device_type="cuda", enabled=True):

        video_encoder = VideoEncoder().to(device)
        video_embed = video_encoder(videos)

    print(video_embed.shape)

    total_num = sum(p.numel() for p in video_encoder.parameters())
    trainable_num = sum(p.numel() for p in video_encoder.parameters() if p.requires_grad)
    print(f'Total {total_num/1024/1024} M, Trainable {trainable_num/1024/1024} M')
