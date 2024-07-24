import torch

from recipes.research.audio_codec import AudioCodec, AudioCodecConfig

if __name__ == "__main__":
    state_dict = torch.load("/mnt/bn/janne-research-xl/models/s-vocoder-2.pt")

    config = AudioCodecConfig()
    audio_codec = AudioCodec(config)

    generator = audio_codec.generator
    generator.load_state_dict(state_dict)

    audio_codec = audio_codec.to_torchscript()
    audio_codec.save(f"uac-7c355ea-step=2000000.ckpt-latent=64.pt")
