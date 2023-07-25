import argparse

from .dprtnet import DPRTNet
from .v_diffusion import ARVSampler, Diffusion


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        diffusion_steps=0,
        num_chunks=20,
        chunk_length=125, # 20*125=2500 for training
        sampling_length=2500,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        input_dim=256,
        feature_dim=1024,
        num_blocks=8,
        segment_size=64,
        segment_stride=32,
        context_dim=512,
        dropout=0.,
        intra_seq2seq='lstm',
        inter_seq2seq='lstm',
        predict_xstart=False,
        end2end=False,
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(**kwargs):
    model = create_model(**kwargs)
    locals().update(kwargs)
    if kwargs["diffusion_steps"] <= 0:
        diffusion = create_v_diffusion(
            steps=kwargs["diffusion_steps"],
            num_chunks=kwargs["num_chunks"],
            chunk_length=kwargs["chunk_length"],
            training=(kwargs["diffusion_steps"] <= 0)
        )
    else:
        diffusion = ARVSampler(
            model,
            in_channels=kwargs["input_dim"],
            length=kwargs["num_chunks"]*kwargs["chunk_length"],
            num_splits=kwargs["num_chunks"]
        )
    return model, diffusion


def create_model(
    *,
    input_dim=256,
    feature_dim=1024,
    num_blocks=8,
    num_chunks=1,
    segment_size=64,
    segment_stride=32,
    context_dim=512,
    dropout=0,
    intra_seq2seq='lstm',
    inter_seq2seq='lstm',
    predict_xstart=False,
    end2end=False,
    diffusion_steps=1000,
    **kwargs
):

    return DPRTNet(
        input_dim=input_dim,
        feature_dim=feature_dim,
        num_blocks=num_blocks,
        num_chunks=num_chunks,
        segment_size=segment_size,
        segment_stride=segment_stride,
        diffusion_steps=diffusion_steps,
        dropout=dropout,
        intra_seq2seq=intra_seq2seq,
        inter_seq2seq=inter_seq2seq,
    )


def create_v_diffusion(
    steps=0,
    num_chunks=20,
    chunk_length=125, # 20*125=2500 for training
    training=True
):
    return Diffusion(
        steps=steps,
        num_chunks=num_chunks,
        chunk_length=chunk_length,
        training=training,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

