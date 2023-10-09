import torch
import torchaudio

assert torch.cuda.is_available()

from tqdm import tqdm

from recipes.datasets.mcc.mix import MixWebDataModule, collate_audio
from recipes.umm.modules.lit_module import Stage3


def forward(model, wav):
    wav = model.pad_audio(wav.float())
    feature = model.preprocessing(wav)["mel"]
    audio_feature = model.audio_encoder(feature)
    hidden_states = model.encoder_input_dropout(audio_feature)
    position_embeddings = model.embed_positions(hidden_states)
    for i, layer in enumerate(model.encoder_layers):
        if i == model.config.vq_layer_idx:
            print(f"[Hidden 1024] {hidden_states.mean().item():.3f} {hidden_states.std().item():.3f}")
            hidden_states = model.vq_proj_in(hidden_states)
            print(f"[Hidden   32] {hidden_states.mean().item():.3f} {hidden_states.std().item():.3f}")
            vq_embs, vq_ids, vq_loss = model.vq(hidden_states)
            print(f"[VQ emb   32] {vq_embs.mean().item():.3f} {vq_embs.std().item():.3f}")
            return vq_ids
        hidden_states = layer(
            hidden_states, position_embeddings=position_embeddings
        )

@torch.no_grad()
def evaluate(it, verbose=True, max_i=1000):
    for i, batch in tqdm(enumerate(it)):
        wav = batch["audio"].to("cuda")
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        print("\n")
        forward(umm_model, wav)
        print("----------------")
        forward(umm_model_2, wav)
        print("\n================\n")
        if max_i is not None and i >= max_i:
            break
    return 


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
        max_num_crops=1,
        min_duration=min_duration,
        max_duration=max_duration,
        normalize_audio=False,
        collate_fn=collate_audio,
    )

    ckpt_path = ".module_cache/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12_from_w_loss_ctc_2/checkpoints/step=010000.ckpt"
    umm_model = Stage3.load_from_checkpoint(ckpt_path).model.to("cuda").eval()
    umm_model.config.interfere_audio = False
    ckpt_path = ".module_cache/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12_from_and_to_w_loss_ctc_2/checkpoints/step=010000.ckpt"
    umm_model_2 = Stage3.load_from_checkpoint(ckpt_path).model.to("cuda").eval()
    umm_model_2.config.interfere_audio = False

    dataloader = iter(pl_datamodule.train_dataloader())

    max_i = 10
    verbose = False
    eval_out_libritts = evaluate(dataloader, verbose, max_i)
