def init_umm(ckpt_path, device, umm_version):
    if umm_version == "0.6.2":
        from recipes.umm_062.requires.model_initializer import init_stage3

        token_model = init_stage3(ckpt_path, device.index, "./")["Stage3"].eval()
        return token_model
    else:
        raise ValueError(f"invalid {umm_version=}")
