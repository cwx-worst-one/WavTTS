import torch
from scipy.io.wavfile import write
from recipes.text2semantic.utils.model_init import init_sound_stream_decoder, init_sound_stream_encoder
from recipes.text2semantic.lit_modules import ValleCoarse, ValleFine
from recipes.bark.lit_modules.v1.valle_coarse import ValleCoarseModule

def prepare_models(args, device):
    ar_model = ValleCoarse.load_from_checkpoint(
        args.ar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    ar_model.eval()

    nar_model = ValleFine.load_from_checkpoint(
        args.nar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    nar_model.eval()

    vqgan_model_encoder = init_sound_stream_encoder(
        args.codec_ckpt_path, device.split(':')[1], cache_dir=".cache_dir")["ss_enc"]
    vqgan_model_encoder.eval()

    vqgan_model_decoder = init_sound_stream_decoder(
        args.codec_ckpt_path, device.split(':')[1], cache_dir=".cache_dir")["ss_dec"]
    vqgan_model_decoder.eval()
    vqgan = {"encoder": vqgan_model_encoder, "decoder": vqgan_model_decoder}

    return ar_model, nar_model, vqgan

def prepare_models_llama(args, device):
    ar_model = ValleCoarseModule.load_from_checkpoint(
        args.ar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    ar_model.eval()

    nar_model = ValleFine.load_from_checkpoint(
        args.nar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    nar_model.eval()

    vqgan_model_encoder = init_sound_stream_encoder(
        args.codec_ckpt_path, device.split(':')[1], cache_dir=".cache_dir")["ss_enc"]
    vqgan_model_encoder.eval()

    vqgan_model_decoder = init_sound_stream_decoder(
        args.codec_ckpt_path, device.split(':')[1], cache_dir=".cache_dir")["ss_dec"]
    vqgan_model_decoder.eval()
    vqgan = {"encoder": vqgan_model_encoder, "decoder": vqgan_model_decoder}

    return ar_model, nar_model, vqgan

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device

def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch