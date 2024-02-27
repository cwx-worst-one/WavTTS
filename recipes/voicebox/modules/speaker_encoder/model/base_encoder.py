import numpy as np
import torch
import torchaudio
import os
from coqpit import Coqpit
from torch import nn
import fsspec
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Union



def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict



def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)

class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class BaseEncoder(nn.Module):
    """Base `encoder` class. Every new `encoder` model must inherit this.

    It defines common `encoder` specific functions.
    """

    # pylint: disable=W0102
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            # TorchSTFT(
            #     n_fft=audio_config["fft_size"],
            #     hop_length=audio_config["hop_length"],
            #     win_length=audio_config["win_length"],
            #     sample_rate=audio_config["sample_rate"],
            #     window="hamming_window",
            #     mel_fmin=0.0,
            #     mel_fmax=None,
            #     use_htk=True,
            #     do_amp_to_db=False,
            #     n_mels=audio_config["num_mels"],
            #     power=2.0,
            #     use_mel=True,
            #     mel_norm=None,
            # )
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    @torch.no_grad()
    def inference(self, x, l2_norm=True):
        return self.forward(x, l2_norm)

    @torch.no_grad()
    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        # map to the waveform size
        if self.use_torch_spec:
            num_frames = num_frames * self.audio_config["hop_length"]

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings

    # def get_criterion(self, c: Coqpit, num_classes=None):
    #     if c.loss == "ge2e":
    #         criterion = GE2ELoss(loss_method="softmax")
    #     elif c.loss == "angleproto":
    #         criterion = AngleProtoLoss()
    #     elif c.loss == "softmaxproto":
    #         criterion = SoftmaxAngleProtoLoss(c.model_params["proj_dim"], num_classes)
    #     else:
    #         raise Exception("The %s  not is a loss supported" % c.loss)
    #     return criterion

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str,
        eval: bool = False,
        use_cuda: bool = False,
        criterion=None,
        cache=False,
    ):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        try:
            self.load_state_dict(state["model"])
            print(" > Model fully restored. ")
        except (KeyError, RuntimeError) as error:
            # If eval raise the error
            if eval:
                raise error

            print(" > Partial model initialization.")
            model_dict = self.state_dict()
            model_dict = set_init_dict(model_dict, state["model"], c)
            self.load_state_dict(model_dict)
            del model_dict

        # load the criterion for restore_path
        # if criterion is not None and "criterion" in state:
        #     try:
        #         criterion.load_state_dict(state["criterion"])
        #     except (KeyError, RuntimeError) as error:
        #         print(" > Criterion load ignored because of:", error)

        # # instance and load the criterion for the encoder classifier in inference time
        # if (
        #     eval
        #     and criterion is None
        #     and "criterion" in state
        #     and getattr(config, "map_classid_to_classname", None) is not None
        # ):
        #     criterion = self.get_criterion(config, len(config.map_classid_to_classname))
        #     criterion.load_state_dict(state["criterion"])

        if use_cuda:
            self.cuda()
        #     if criterion is not None:
        #         criterion = criterion.cuda()

        # if eval:
        #     self.eval()
        #     assert not self.training

        # if not eval:
        #     return criterion, state["step"]
        # return criterion