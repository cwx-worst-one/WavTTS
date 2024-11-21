import os
import sys
import torch
import samantha.utils.hdfs_helper as hh
from collections import OrderedDict


def get_ckpt_path(ckpt_path, ckpt_cache=".deploy_cache", force_update=True):

    if not force_update:
        fn = os.path.basename(ckpt_path)
        local_path = os.path.join(ckpt_cache, fn)
        if os.path.exists(local_path):
            return local_path

    if hh.ishdfs(ckpt_path):
        os.makedirs(ckpt_cache, exist_ok=True)
        fn = os.path.basename(ckpt_path)
        local_path = os.path.join(ckpt_cache, fn)
        if os.path.exists(local_path):
            os.remove(local_path)
        success = hh.get(ckpt_path, local_path)
        if not success:
            raise ConnectionError(f"failed to retrieve {ckpt_path} to {local_path}")
        return local_path
    else:
        return ckpt_path


def export_video(ckpt, save_path):

    from apps.bigtts.audiogen.soundify.v2a.encoder import VideoEncoder

    new_ckpt = OrderedDict()

    encoder = VideoEncoder()
    key = "model.video_encoder"

    for name, param in ckpt.items():
        print(name)
        if key in name:
            print(name[(len(key) + 1)::], "->", name)
            new_ckpt[name[(len(key) + 1)::]] = param

    msg = encoder.load_state_dict(new_ckpt)

    print(f"encoder {msg}")
    print(save_path)


def export_video(ckpt, save_path):

    from apps.bigtts.audiogen.soundify.v2a.encoder import VideoEncoder

    new_ckpt = OrderedDict()

    model = VideoEncoder()
    key = "model.video_encoder."

    for name, param in ckpt.items():
        print(name)
        if key in name:
            print(name[len(key)::], "->", name)
            new_ckpt[name[len(key)::]] = param

    msg = model.load_state_dict(new_ckpt)

    torch.save(new_ckpt, save_path)

    print(f"encoder {msg}")
    print(save_path)


def export_diffusion(ckpt, save_path):

    from apps.bigtts.audiogen.soundify.v2a.diffusion import Diffusion

    new_ckpt = OrderedDict()

    model = Diffusion()

    key1 = "model.video_encoder."
    key2 = "model.vocoder."

    key = "model."

    for name, param in ckpt.items():
        print(name)
        if (key in name) and (key1 not in name) and (key2 not in name):
            print(name[len(key)::], "->", name)
            new_ckpt[name[len(key)::]] = param

    msg = model.load_state_dict(new_ckpt)

    torch.save(new_ckpt, save_path)

    print(f"model {msg}")
    print(save_path)


def export_vocoder(ckpt, save_path):

    from apps.bigtts.audiogen.soundify.v2a.vocoder import Vocoder

    new_ckpt = OrderedDict()

    model = Vocoder()

    key1 = "model.vocoder.decoder."
    key2 = "model.vocoder.post_quant_conv."
    key = "model.vocoder."

    for name, param in ckpt.items():
        if key1 in name or key2 in name:
            print(name, "->", name[len(key)::])
            new_ckpt[name[len(key)::]] = param

    msg = model.load_state_dict(new_ckpt)

    torch.save(new_ckpt, save_path)

    print(f"model {msg}")
    print(save_path)


if __name__ == "__main__":

    file_name = "v2a_15w_32k_ft12_74000"

    local_path = f"/mnt/bn/zxb-lq/workspace/samantha/.deploy_cache/{file_name}.ckpt"

    encoder_path = os.path.join(".deploy_cache", f"{file_name}_encoder.ckpt")
    diffusion_path = os.path.join(".deploy_cache", f"{file_name}_diffusion.ckpt")
    vocoder_path = os.path.join(".deploy_cache", f"{file_name}_vocoder.ckpt")

    print("loading ckpt ......")
    ckpt = torch.load(local_path, map_location="cpu")
    print("loading done")

    export_video(ckpt=ckpt, save_path=encoder_path)
    export_diffusion(ckpt=ckpt, save_path=diffusion_path)
    export_vocoder(ckpt=ckpt, save_path=vocoder_path)
