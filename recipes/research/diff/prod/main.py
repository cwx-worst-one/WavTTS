from recipes.research.audio_codec.zoo import AudioCodec_f81b3fa_64l
from recipes.research.diff.prod import load_diffusion_model, sample

if __name__ == "__main__":
    commit_hash = "55ff351"
    model = load_diffusion_model(commit_hash, cache=True)
    # torch.save(model, "model.pt")

    pred_audio, sample_rate = sample(
        model,
        "test",
        seconds_start=0,
        seconds_total=60,
        t=50,
        cfg_weight=2.5,
        schedule_tau=1.0,
        batch_size=4,
        solver="dpmpp-3m-sde",
    )

    print(pred_audio.shape)
