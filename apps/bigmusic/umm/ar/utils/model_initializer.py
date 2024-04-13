import torch

from apps.bigmusic.umm.ar.lightning.acoustic_modules import CoarseModule, FineModule


def init_coarse(hpath, local_rank, cache_dir):
    device = torch.device(f"cuda:{local_rank}")
    coarse = CoarseModule.load_from_checkpoint(hpath).eval().to(device)
    return {"coarse": coarse}


def init_fine(hpath, local_rank, cache_dir):
    device = torch.device(f"cuda:{local_rank}")
    fine = FineModule.load_from_checkpoint(hpath).eval().to(device)
    return {"fine": fine}


def run_2ar(requires, semantic_samples, extra_params, semantic_embeds=None):
    # coarse_module = self.coarse_module
    # fine_module = self.fine_module
    coarse_module = requires["coarse"]
    fine_module = requires["fine"]
    ss_decoder = requires["ss_dec"]
    # semantic_samples = tokens.
    coarse_samples = coarse_module.predict(
        semantic_samples=semantic_samples,
        hp=extra_params,
        semantic_embeds=semantic_embeds,
    )
    fine_samples = fine_module.predict(coarse_samples, extra_params)
    bs = coarse_samples.size(0)

    num_coarse = coarse_module.extra_params.num_coarse
    num_fine = fine_module.extra_params.num_fine
    soundstream_codebook_size = coarse_module.extra_params.soundstream_codebook_size

    coarse_samples = coarse_samples.view([bs, -1, num_coarse])
    fine_samples = fine_samples.view([bs, -1, num_fine])
    vqgan_inputs = (
        torch.cat([coarse_samples, fine_samples], dim=2)
        - torch.arange(num_coarse + num_fine, device=coarse_samples.device)
        * soundstream_codebook_size
    )  # [b, t, n_codebook]
    vqgan_inputs = vqgan_inputs.transpose(
        1, 2
    )  # [b, t, n_codebook] -> [b, n_codebook, t]
    wavs = ss_decoder(vqgan_inputs).squeeze(1)
    return wavs
