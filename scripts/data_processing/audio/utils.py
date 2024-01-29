import io
import logging
import os
import pickle
from multiprocessing import Pool

import torch

from samantha.dataio.parquet import ParquetWriter
from samantha.utils.distributed import rank_zero_first
from samantha.utils.hdfs_helper import get, hdfs_ls, isdir, ishdfs
from scripts.data_processing.audio import (
    byt5,
    ser,
    spk_embed_utils,
    ss_utils,
    umm_token,
    umm_tokenizer,
    wvae_mel_token,
    wvae_mel_utils,
    wvae_utils,
    noisy_token,
)

logger = logging.getLogger(__name__)


class Consumer:
    def __init__(
        self, output_url, filesystem, feature_type, target_sample_rate, domain, **kwargs
    ):
        self.writer = ParquetWriter(output_url, verbose=True, filesystem=filesystem)
        self.feature_type = feature_type
        self.feature_name = feature_name_mapping(feature_type)
        self.target_sample_rate = target_sample_rate
        self.domain = domain
        self.kwargs = kwargs
        self.resampler = {}

    def write(self, uttid, feature, dataset_name):

        item = {"uttid": uttid, "dataset_name": str(dataset_name)}
        if isinstance(self.feature_name, list):
            assert len(self.feature_name) == len(feature)
            item.update(
                {
                    name: pickle.dumps(feat)
                    for name, feat in zip(self.feature_name, feature)
                }
            )
        elif isinstance(self.feature_name, str):
            item.update({self.feature_name: pickle.dumps(feature)})

        self.writer.write(item)

    def close(self):
        self.writer.close()

    def consume(self, uttids, model, batch, device, dataset_name):
        for uttid, feature in zip(
            uttids,
            process_batch(self.feature_type)(
                model, batch, device, self.target_sample_rate
            ),
        ):
            if feature is None:
                continue
            self.write(uttid=uttid, feature=feature, dataset_name=dataset_name)

    def preprocess(self, item, device):
        if self.domain == "data":
            return preprocess_audio(self.feature_type)(
                audio_bin=io.BytesIO(item["audio"]),
                sample_rate=self.target_sample_rate,
                resampler=self.resampler,
                device=device,
                **self.kwargs,
            )
        else:
            return preprocess_index(self.feature_type)(
                index_item=item, device=device, **self.kwargs
            )


def preprocess_index(feature_type):
    if feature_type == "byt5":
        return byt5.preprocess_index
    else:
        raise ValueError(f"{feature_type=} is not support for preprocess_index.")


def preprocess_audio(feature_type):
    if feature_type == "soundstream":
        return ss_utils.preprocess_audio
    elif feature_type == "wavevae":
        return wvae_utils.preprocess_audio
    elif feature_type == "wavevae_mel":
        return wvae_mel_utils.preprocess_audio
    elif feature_type == "wavevae_mel_token":
        return wvae_mel_token.preprocess_audio
    elif feature_type == "speaker_embed":
        return spk_embed_utils.preprocess_audio
    elif feature_type == "umm_tokenizer":
        return umm_tokenizer.preprocess_audio
    elif feature_type == "umm_token":
        return umm_token.preprocess_audio
    elif feature_type == "ser":
        return ser.preprocess_audio
    elif feature_type == "noisy_token":
        return noisy_token.preprocess_audio
    else:
        raise ValueError(f"{feature_type=} is not support for preprocess_audio.")


def process_batch(feature_type):
    if feature_type == "soundstream":
        return ss_utils.process_batch
    elif feature_type == "wavevae":
        return wvae_utils.process_batch
    elif feature_type == "wavevae_mel":
        return wvae_mel_utils.process_batch
    elif feature_type == "wavevae_mel_token":
        return wvae_mel_token.process_batch
    elif feature_type == "speaker_embed":
        return spk_embed_utils.process_batch
    elif feature_type == "umm_tokenizer":
        return umm_tokenizer.process_batch
    elif feature_type == "umm_token":
        return umm_token.process_batch
    elif feature_type == "ser":
        return ser.process_batch
    elif feature_type == "byt5":
        return byt5.process_batch
    elif feature_type == "noisy_token":
        return noisy_token.process_batch
    else:
        raise ValueError(f"{feature_type=} is not support for process_batch.")


def model_path_patten(feature_type, feature_version):
    if feature_type == "soundstream":
        return ss_utils.model_path_patten(feature_version)
    elif feature_type == "wavevae":
        return wvae_utils.model_path_patten(feature_version)
    elif feature_type == "wavevae_mel":
        return wvae_mel_utils.model_path_patten(feature_version)
    elif feature_type in ["speaker_embed", "umm_token", "ser", "byt5", "noisy_token"]:
        return None
    elif feature_type == "wavevae_mel_token":
        return wvae_mel_token.model_path_patten(feature_version)
    elif feature_type == "umm_tokenizer":
        return umm_tokenizer.model_path_patten(feature_version)
    else:
        raise ValueError(f"{feature_type=} is not support for model_path_patten.")


def feature_name_mapping(feature_type):
    if feature_type == "soundstream":
        return "ss"
    if feature_type in ["wavevae", "wavevae_mel"]:
        return "bns"
    if feature_type == "wavevae_mel_token":
        return "zvq_token"
    if feature_type == "speaker_embed":
        return "spk_emb"
    if feature_type in ["umm_tokenizer", "umm_token"]:
        return "umm_token"
    if feature_type in ["ser"]:
        return ["emo_tag", "emo_deg", "emo_emb"]
    if feature_type in ["byt5"]:
        return "byt5"
    if feature_type in ["noisy_token"]:
        return ["noisy_wav", "noisy_bns", "noisy_umm_token", "noisy_meta"]


def _load_torch_script_model(model_path, device):
    if model_path is None:
        return None
    local_model_path = os.path.basename(model_path)
    if os.path.exists(local_model_path):
        os.remove(local_model_path)
    get(model_path, local_model_path)
    model = torch.jit.load(local_model_path).eval().to(device)
    return model


def load_model(feature_type):
    if feature_type in [
        "soundstream",
        "wavevae",
        "wavevae_mel",
        "umm_tokenizer",
        "wavevae_mel_token",
    ]:
        return _load_torch_script_model
    elif feature_type in ["speaker_embed"]:
        return spk_embed_utils.load_model
    elif feature_type in ["umm_token"]:
        return umm_token.load_model
    elif feature_type in ["ser"]:
        return ser.load_model
    elif feature_type in ["byt5"]:
        return byt5.load_model
    elif feature_type in ["noisy_token"]:
        return noisy_token.load_model


def download_model(ckpt_path):
    if ckpt_path is None:
        return None
    if not ishdfs(ckpt_path):
        return ckpt_path
    local_path = os.path.basename(ckpt_path)
    with rank_zero_first(is_global=False):
        if not os.path.exists(local_path):
            multi_get(ckpt_path, local_path)
    return local_path


def multi_get(ckpt_path, local_path):
    pool = Pool(20)

    def inner_fn(ckpt_path, local_path):
        if not isdir(ckpt_path):
            pool.apply_async(
                func=get,
                args=(ckpt_path, local_path),
                error_callback=lambda exn: logger.error(
                    "error on download model", exc_info=exn
                ),
            )
            return
        os.makedirs(local_path, exist_ok=True)
        for item in hdfs_ls(ckpt_path):
            local_path_inner = os.path.join(local_path, os.path.basename(item))
            inner_fn(item, local_path_inner)

    inner_fn(ckpt_path=ckpt_path, local_path=local_path)
    pool.close()
    pool.join()


def dummy_process_after_downloading(*_, **__):
    return {}


def process_after_downloading(feature_type):
    if feature_type in ["noisy_token"]:
        return noisy_token.process_after_downloading
    return dummy_process_after_downloading
