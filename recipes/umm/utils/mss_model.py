import collections
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, OrderedDict, Union
from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_only
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SerializeOptions:
    params_path: str = None
    dynamic_axes: Dict = None
    tolerance: float = None
    sample_input_shape: Dict = None
    graph_path: str = None
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    extra_opt: Dict = field(default_factory=dict)


class BaseStage(nn.Module):
    r"""Base class for all stages. Provide a generic interface for data computing.

    :class:`~sami_ai.core.BaseStage` defines functions such as pre-processor,
    post-processor, encoder, decoder. It follows yaml-style format. Users can
    define what the function takes and provides after processing. A sample
    definition is as below:

    .. code-block:: yaml


        stages:
          - !new:recipes.sample_project.modules.model.CNNModelStage
             takes: [img_tensor]
             provides: [label]
             num_classes: 10
             serialize_opts:
                io_meta:
                    image_tensor: [b, 3, 224, 224]
                    label: [b, 10]
                sample_input_shape:
                    image_tensor: [10, 3, 224, 224]
                graph_path: cnn_model_stage.onnx
                tolerance: 1e-5

    For more info, please refer to: `Redesign SAMI Model Repo & Engine`_

    Args:
        takes (List[str]): the names that this stage will take.
        provides (List[str]): the names that this stage will output.
        serialize_opts(Dict): some necessary options needed that graph
                              export needs.

    .. _Redesign SAMI Model Repo & Engine:
       https://bytedance.feishu.cn/docx/doxcnaAWhP8inq8vGMGJri3Y9Rf

    """

    SAMI_STAGE = "BaseStage"

    def __init__(self, takes, provides, serialize_opts=None, *args, **kwargs):
        super().__init__()
        self.takes = takes
        self.provides = provides
        self._serialize_opts = self._prepare_opts(serialize_opts)
        self._dump_params = {"takes": takes, "provides": provides}

    def forward(self, data: Dict) -> Dict:
        r"""Perform forward based on takes and provides

        Args:
            data (dict): A dictionary contains all keys in ``takes``.

        Returns:
            A dictionary contains all keys in ``provides``.

        """

        outputs = self._compute(*(data[k] for k in self.takes))
        return dict(zip(self.provides, outputs))

    @rank_zero_only
    def export(self, **kwargs):
        r"""Perform stage export"""

        graph_path = self._export_onnx_graph(**kwargs)
        params_path = self._export_params()

        export_paths = []
        if graph_path:
            export_paths.append(graph_path)
        if params_path:
            export_paths.append(params_path)

        return export_paths

    def _compute(self, *args):
        # do some awesome things
        raise NotImplementedError(
            "Subclass must provide implementation of this functon"
        )

    @torch.no_grad()
    def _export_onnx_graph(self, **kwargs):
        r"""Perform onnx graph export under given serialize options"""
        if self._serialize_opts.graph_path is None:
            logger.error("Serialization about graph are not provided.")
            return None
        if not self._mkdir(self._serialize_opts.graph_path):
            return None

        extra_opts = self._serialize_opts.extra_opt
        device = extra_opts.pop("device", torch.device("cpu"))
        data = self._generate_sample_input(device=device)
        opset_version = extra_opts.pop("opset", 9)

        # we don't need None tensor in graph and params.
        for name, tensor in data.items():
            if tensor is None:
                self._dump_params["takes"].remove(name)

        torch.onnx.export(
            model=self,
            args=({"data": data},),
            f=self._serialize_opts.graph_path,
            input_names=self._serialize_opts.input_names,
            output_names=self._serialize_opts.output_names,
            dynamic_axes=self._serialize_opts.dynamic_axes,
            opset_version=opset_version,
            **extra_opts,
        )

        self._check_consistency(device=device)
        return self._serialize_opts.graph_path

    @torch.no_grad()
    def _check_consistency(self, device):
        sample_input = self._generate_sample_input(device=device)
        sample_input_numpy = {
            name: tensor.cpu().numpy()
            for name, tensor in sample_input.items()
            if tensor is not None
        }
        sess = ort.InferenceSession(self._serialize_opts.graph_path)
        python_output = self(sample_input)
        ort_output = sess.run(
            self._serialize_opts.output_names, input_feed=sample_input_numpy
        )
        for i, name in enumerate(self._serialize_opts.output_names):
            try:
                np.testing.assert_allclose(
                    python_output[name].cpu().numpy(),
                    ort_output[i],
                    atol=self._serialize_opts.tolerance,
                )
            except AssertionError as e:
                logger.error(
                    f"Check graph consistency error on output {name} with error {e}"
                )

    def _generate_sample_input(self, device=torch.device("cpu")) -> Dict:
        data = {}
        for name, shape in self._serialize_opts.sample_input_shape.items():
            if shape is None:
                data[name] = None
            else:
                data[name] = torch.rand(shape, device=device)
        return data

    def _prepare_opts(self, serialize_opt: Dict) -> Optional[SerializeOptions]:

        if serialize_opt is None:
            return SerializeOptions()

        opts_copy = serialize_opt.copy()
        params_path = opts_copy.pop("params_path", None)
        if not opts_copy:
            return SerializeOptions(params_path=params_path)

        if "io_meta" not in opts_copy:
            raise ValueError("Expecting io meta provided.")
        if "sample_input_shape" not in opts_copy:
            raise ValueError("Expecting sample input shape provided.")

        dynamic_axes = {
            name: {i: v for i, v in enumerate(value) if isinstance(v, str)}
            for name, value in opts_copy.pop("io_meta").items()
            if value is not None
        }
        sample_input_shape = opts_copy.pop("sample_input_shape")
        graph_path = opts_copy.pop("graph_path", None)
        tolerance = float(opts_copy.pop("tolerance", 1e-5))

        return SerializeOptions(
            dynamic_axes=dynamic_axes,
            params_path=params_path,
            graph_path=graph_path,
            tolerance=tolerance,
            sample_input_shape=sample_input_shape,
            input_names=self.takes,
            output_names=self.provides,
            extra_opt=opts_copy.copy(),
        )

    def _update_params(self, kwargs: Dict):
        serializable_params = {}
        for k, v in kwargs.items():
            # Check if value is JSON serializable
            try:
                json.dumps(v)
                serializable_params[k] = v
            except (TypeError, OverflowError):
                # For objects like Tensors that are not JSON serializable
                serializable_params[k] = str(v)
        self._dump_params.update(serializable_params)

    def _export_params(self):
        r"""Perform stage parameters export.

        .. note::
            If this stage is not a neural net stage, better to dump used
            parameters to a json file to help further engineering.

        """

        if self._serialize_opts.params_path is None:
            logger.error("Serialization about params are not provided.")
            return None

        if not self._mkdir(self._serialize_opts.params_path):
            return None

        with open(self._serialize_opts.params_path, "w") as f:
            json.dump(self._dump_params, f, indent=2, ensure_ascii=True)

        return self._serialize_opts.params_path

    def _mkdir(self, p):
        dname = os.path.dirname(self._serialize_opts.params_path)
        try:
            os.makedirs(dname, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create directory {dname} with msg {e}")
            return False
        return True


class BaseModel(nn.Module):
    r"""Base model of model pipeline.

    This model is the base model of all customized model pipeline. Every
    customized model should override :func:`forward` and :func:`export`
    if necessary.

    Args:
        stages (List[Union[BaseStage, Callable]]): a sequence stage(or partial) unit.
        input_names (List[str]): input node names.
        output_names (List[str]): output node names.

    """

    def __init__(
            self,
            stages: List[Union[BaseStage, Callable]],
            input_names: List[str],
            output_names: List[str],
    ):
        super(BaseModel, self).__init__()
        rank_zero_deprecation(
            "`BaseModel` is deprecated from v0.2, and will be removed in v0.3."
            "Use `ModelPipeline` instead."
        )
        if not stages:
            raise ValueError("stages must be specified.")

        if all(
                [
                    isinstance(s, partial)
                    and inspect.isclass(s.func)
                    and issubclass(s.func, BaseStage)
                    for s in stages
                ]
        ):
            stages = [stage() for stage in stages]
        elif not all([isinstance(s, BaseStage) for s in stages]):
            raise TypeError(
                "Expecting `stages` to be either List[BaseStage] or "
                f"List[partial(BaseStage)], but got {stages}"
            )
        else:
            rank_zero_deprecation(
                "Stages of BaseModel's `!new` style initialization is deprecated from "
                "v0.1.1, and will be removed in v0.2. Use `!name` instead."
            )

        if not input_names:
            raise ValueError("input_names must be specified.")

        if not output_names:
            raise ValueError("output_names must be specified.")

        self.stages = nn.ModuleList(stages)
        self.input_names = input_names
        self.output_names = output_names
        self.task_type = os.getenv("SAIL_TASK_TYPE", None)
        self.model_version = os.getenv("MODEL_VERSION", "1.0")

    def forward(self, batch):
        r"""Perform forward computation.

        This method will call :func:`forward <BaseStage.forward>` function of
        :attr:`stages <self.stages>` sequentially.

        Args:
            batch (List[Tensor]): input batch which contains
             ``len(input_names)`` tensors.

        """
        if isinstance(batch, dict):
            data = batch
        else:
            if not isinstance(batch, List):
                batch = [batch]
            data = dict(zip(self.input_names, batch))
        for stage in self.stages:
            out = stage(data)
            data.update(out)
        return tuple(data[k] for k in self.output_names)

    @torch.no_grad()
    def export(self, **kwargs):
        r"""Perform export.

        This method will call :func:`export <BaseStage.export>` function of
        :attr:`stages <self.stages>` sequentially.

        """

        resource_loc = kwargs.get("resource_loc", None)
        if resource_loc is None:
            return None

        self.eval()
        export_stages = collections.OrderedDict()
        export_paths = {"resource_loc": resource_loc, "params_loc": []}
        for stage in self.stages:
            paths = stage.export(**kwargs)
            export_stages[stage] = paths
            export_paths["params_loc"].extend(paths)

        xml_str = self._generate_xml(export_stages=export_stages)
        if xml_str is None:
            return None
        with open(resource_loc, "w", encoding="utf8") as f:
            f.write(xml_str)
        return export_paths

    def _generate_xml(self, export_stages: OrderedDict[BaseStage, str]):
        from xml.dom import minidom

        def _create_text_element(tag_name, text):
            doc = minidom.Document()
            tag = doc.createElement(tag_name)
            tag.appendChild(doc.createTextNode(str(text)))
            return tag

        def _create_submodels_element(stages_mapping: OrderedDict[BaseStage, str]):
            doc = minidom.Document()
            submodels = doc.createElement("submodels")

            for stage, path in stages_mapping.items():
                submodel = doc.createElement("submodel")
                submodel.setAttribute("type", stage.__class__.__name__)

                raw_model_files = doc.createElement("raw_model_files")
                decoder_type = None
                for p in path:
                    file_node = _create_text_element("file", os.path.basename(p))
                    name = "params"
                    if p.endswith("onnx"):
                        name = "model"
                        decoder_type = "onnx"
                    file_node.setAttribute("name", name)
                    raw_model_files.appendChild(file_node)
                submodel.appendChild(raw_model_files)

                if decoder_type is not None:
                    dt = _create_text_element("decoder_type", decoder_type)
                    submodel.appendChild(dt)

                model_params = doc.createElement("model_param")
                model_params.appendChild(
                    _create_text_element("model_key", stage.SAMI_STAGE)
                )
                submodel.appendChild(model_params)

                submodels.appendChild(submodel)

            return submodels

        if self.task_type is None:
            logging.error(
                "Expect a valid task type, but got None. You could set this variable"
                " with `export SAIL_TASK_TYPE='a_task_type'`"
            )
            return None

        doc = minidom.Document()

        tag_service = doc.createElement("service")
        doc.appendChild(tag_service)
        tag_models = doc.createElement("models")
        tag_service.appendChild(tag_models)

        tag_model = doc.createElement("model")
        tag_model.setAttribute("task_type", self.task_type)
        tag_models.appendChild(tag_model)

        model_path = f"{self.task_type}_v{self.model_version}.model"
        tag_model_path = doc.createElement("model_path")
        tag_model_path.appendChild(doc.createTextNode(model_path))
        tag_model.appendChild(tag_model_path)

        tag_model_params = doc.createElement("model_param")
        tag_model_params.appendChild(
            _create_text_element("input_names", " ".join(self.input_names))
        )
        tag_model_params.appendChild(
            _create_text_element("output_names", " ".join(self.output_names))
        )
        tag_model_params.appendChild(
            _create_text_element("input_num", len(self.input_names))
        )
        tag_model_params.appendChild(
            _create_text_element("output_num", len(self.output_names))
        )
        tag_model_params.appendChild(
            _create_text_element("model_key", "UnifiedMusicModel")
        )
        tag_model_params.appendChild(
            _create_text_element(
                "stage_seq",
                " ".join([item.__class__.__name__ for item in export_stages]),
            )
        )
        tag_model.appendChild(tag_model_params)

        tag_context = doc.createElement("context")
        tag_context.appendChild(_create_text_element("context_key", "MusicBaseContext"))
        tag_model.appendChild(tag_context)

        tag_submodels = _create_submodels_element(export_stages)
        tag_model.appendChild(tag_submodels)

        xml_str = doc.toprettyxml(indent="\t")
        return xml_str


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_flash_attn=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.use_flash_attn = use_flash_attn

    def forward(self, x, rotary_emb=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout=0.0,
            use_checkpoint=True,
            use_flash_attn=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_0": RMSNorm(dim),
                        "attn": Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_flash_attn=use_flash_attn,
                        ),
                        "norm_1": RMSNorm(dim),
                        "ff": FeedForward(dim, mlp_dim, dropout=dropout),
                    }
                )
            )
        self.use_checkpoint = use_checkpoint

    def _forward_checkpoint(self, x, rotary_emb=None):
        for layer in self.layers:
            if rotary_emb is not None:
                x = (
                        checkpoint(
                            layer["attn"], checkpoint(layer["norm_0"], x), rotary_emb
                        )
                        + x
                )
            else:
                x = checkpoint(layer["attn"], x) + x

            x = checkpoint(layer["ff"], checkpoint(layer["norm_1"], x)) + x
        return x

    def _forward(self, x, rotary_emb=None):
        for layer in self.layers:
            if rotary_emb is not None:
                x = layer["attn"](layer["norm_0"](x), rotary_emb) + x
            else:
                x = layer["attn"](x) + x

            x = layer["ff"](layer["norm_1"](x)) + x
        return x

    def forward(self, x, rotary_emb=None):
        if self.use_checkpoint:
            x = self._forward_checkpoint(x, rotary_emb)
        else:
            x = self._forward(x, rotary_emb)
        return x


class BandSplit(nn.Module):
    def __init__(
            self,
            num_feature=128,
            channel_num=4,
            subspec_idxs=[
                [0, 47],
                [48, 95],
                [96, 191],
                [192, 383],
                [384, 767],
                [768, 1023],
            ],
            n_band_per_subspec=[24, 12, 8, 8, 8, 2],
    ):
        """21.5 hz per bin

        1000 / 21.5 = 46.5  2 bin per band
        2000 / 21.5 = 93   4 bin per band
        4000 / 21.5 = 186  12 bin per band
        8000 / 21.5 = 373  24 bin per band
        16000 / 21.5 = 744 48 bin per band
        other: 2 bin per band

        Input:
            x: (b, c, t, f)
        Output:
            x: (b, t, k, n)
        """
        super().__init__()
        self.channel_num = channel_num  # (real + imag) * stereo
        num_feature = num_feature
        self.subspec_idxs = subspec_idxs
        self.n_band_per_subspec = n_band_per_subspec

        band_split_transforms = nn.ModuleList([])

        for subspec_idx, n_band in zip(subspec_idxs, n_band_per_subspec):
            # Ensure the number of bands can be evenly divided by the number of bins in a subspec
            assert np.isclose((subspec_idx[1] - subspec_idx[0] + 1) % n_band, 0)
            input_dim = (
                    int((subspec_idx[1] - subspec_idx[0] + 1) / n_band) * channel_num
            )

            subspec_transforms = nn.ModuleList([])

            for i in range(n_band):
                subband_split = nn.Sequential(
                    collections.OrderedDict(
                        [
                            ("norm", RMSNorm(input_dim)),
                            ("ff", nn.Linear(input_dim, num_feature)),
                        ]
                    )
                )
                subspec_transforms.append(subband_split)
            band_split_transforms.append(subspec_transforms)

        self.band_split_transforms = band_split_transforms

    def forward(self, x):
        # x: (b, c, t, f)
        x = rearrange(x, "b c t f -> b t (f c)")
        outs = []
        for subspec_idx, n_band, band_split_transform in zip(
                self.subspec_idxs, self.n_band_per_subspec, self.band_split_transforms
        ):
            input_dim = (
                    int((subspec_idx[1] - subspec_idx[0] + 1) / n_band) * self.channel_num
            )

            for i, transform in enumerate(band_split_transform):
                sub_x = transform(x[:, :, i * input_dim: (i + 1) * input_dim])
                outs.append(sub_x)
        x = rearrange(torch.stack(outs), "k b t n -> b t k n")
        return x


class BSTransformer_block(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_heads: int = 8,
            dim_head: int = 32,
            dropout: float = 0.1,
            use_checkpoint: bool = True,
            use_flash_attn: bool = False,
    ):
        super(BSTransformer_block, self).__init__()

        self.transform_t = Transformer(
            dim=input_dim,
            depth=1,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=input_dim * 4,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn,
        )
        self.transform_k = Transformer(
            dim=input_dim,
            depth=1,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=input_dim * 4,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn,
        )

    def forward(self, x, time_pos_emb=None, freq_pos_emb=None):
        b, t, k, c = x.shape
        # x: (bs, t, k, c)

        x = rearrange(x, "b t k c -> (b k) t c")
        # ((bs * k), t, c)

        if time_pos_emb is not None:
            x = self.transform_t(x, time_pos_emb)
        else:
            x = self.transform_t(x)

        x = rearrange(x, "(b k) t c -> (b t) k c", b=b)
        # (bs * t, k, c)

        if freq_pos_emb is not None:
            x = self.transform_k(x, freq_pos_emb)
        else:
            x = self.transform_k(x)

        x = rearrange(x, " (b t) k c -> b t k c", b=b)

        return x


class MaskEstimation(nn.Module):
    def __init__(
            self,
            num_feature=128,
            channel_num=4,
            subspec_idxs=[
                [0, 47],
                [48, 95],
                [96, 191],
                [192, 383],
                [384, 767],
                [768, 1023],
            ],
            n_band_per_subspec=[24, 12, 8, 8, 8, 2],
            use_checkpoint: bool = True,
    ):
        """21.5 hz per bin

        1000 / 21.5 = 46.5  2 bin per band
        2000 / 21.5 = 93   4 bin per band
        4000 / 21.5 = 186  12 bin per band
        8000 / 21.5 = 373  24 bin per band
        16000 / 21.5 = 744 48 bin per band
        other: 2 bin per band

        Input:
            x: (b, t, k, n)
        Output:
            x: (b, c, t, f)
        """
        super().__init__()
        self.channel_num = channel_num  # (real + imag) * stereo
        num_feature = num_feature
        self.subspec_idxs = subspec_idxs
        self.n_band_per_subspec = n_band_per_subspec
        self.use_checkpoint = use_checkpoint

        mask_estimation = nn.ModuleList([])

        for i, subspec_idx, n_band in zip(
                range(len(subspec_idxs)), subspec_idxs, n_band_per_subspec
        ):
            # Ensure the number of bands can be evenly divided by the number of bins in a subspec
            assert np.isclose((subspec_idx[1] - subspec_idx[0] + 1) % n_band, 0)
            submask_estimations = nn.ModuleList([])

            for j in range(n_band):
                # special treatment for the first band in each subspec
                out_dim = (
                        int((subspec_idx[1] - subspec_idx[0] + 1) / n_band) * channel_num
                )

                submask_estimation = nn.Sequential(
                    collections.OrderedDict(
                        [
                            ("norm", RMSNorm(num_feature)),
                            ("ff_0", nn.Linear(num_feature, 4 * num_feature)),
                            ("tanh", nn.Tanh()),
                            ("ff_1", nn.Linear(4 * num_feature, 2 * out_dim)),
                            ("glu", nn.GLU()),
                        ]
                    )
                )
                submask_estimations.append(submask_estimation)
            mask_estimation.append(submask_estimations)

        self.mask_estimation = mask_estimation

    def forward(self, x):
        # x: (b, t k n)
        outs = []
        for i, submask_estimations in enumerate(self.mask_estimation):
            n_bands = self.n_band_per_subspec[i]
            for n, submask_estimation in zip(range(n_bands), submask_estimations):
                if self.use_checkpoint:
                    sub_x = checkpoint_sequential(submask_estimation, 4, x[:, :, n])
                else:
                    sub_x = submask_estimation(x[:, :, n])
                outs.append(sub_x)

        x = rearrange(
            torch.cat(outs, dim=-1), "b t (f c) -> b c t f", c=self.channel_num
        )
        # b c t f
        return x


def get_bandwidth_ranges(sample_rate=44100, n_fft=2047, n_filterbank=63, overlap=True):
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_filterbank)
    bw_idxs = [list(np.where(row > 0)[0]) for row in mel_basis]
    bw_idxs[0] = [0] + bw_idxs[0]

    if not overlap:  # will return n_filterbank+1 of bw_idxs
        new_bw_idxs = []
        split_idx = len(bw_idxs[0]) // 2
        new_bw_idxs.append(bw_idxs[0][:split_idx])
        new_bw_idxs.append(bw_idxs[0][split_idx:])
        for i in range(1, n_filterbank):
            new_bw_idxs.append([j for j in bw_idxs[i] if j not in bw_idxs[i - 1]])
        bw_idxs = new_bw_idxs

    bw_ranges = [[i[0], i[-1]] for i in bw_idxs]
    return bw_ranges


class MelBand(nn.Module):
    def __init__(
            self,
            num_feature=128,
            channel_num=4,
            subspec_idxs=[],
    ):
        """
        Input:
            x: (b, c, t, f)
        Output:
            x: (b, t, k, n)
        """
        super().__init__()
        self.channel_num = channel_num  # (real + imag) * stereo
        num_feature = num_feature
        self.subspec_idxs = subspec_idxs
        self.mel_band_transforms = nn.ModuleList([])

        for subspec_idx in subspec_idxs:
            input_dim = (subspec_idx[-1] - subspec_idx[0] + 1) * channel_num

            mel_band_transform = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("norm", RMSNorm(input_dim)),
                        ("ff", nn.Linear(input_dim, num_feature)),
                    ]
                )
            )
            self.mel_band_transforms.append(mel_band_transform)

    def forward(self, x):
        # x: (b, c, t, f)
        x = rearrange(x, "b c t f -> b t (f c)")
        outs = []
        for subspec_idx, mel_band_transform in zip(self.subspec_idxs, self.mel_band_transforms):
            sub_x = mel_band_transform(
                x[:, :, subspec_idx[0] * self.channel_num: (subspec_idx[-1] + 1) * self.channel_num])
            outs.append(sub_x)
        x = rearrange(torch.stack(outs), "k b t n -> b t k n")
        return x


class MelMaskEstimation(nn.Module):
    def __init__(
            self,
            num_feature=128,
            channel_num=4,
            subspec_idxs=[],
            band_overlap: bool = True,
            use_checkpoint: bool = True,
    ):
        """
        Input:
            x: (b, t, k, n)
        Output:
            x: (b, c, t, f)
        """
        super().__init__()
        self.channel_num = channel_num  # (real + imag) * stereo

        self.band_overlap = band_overlap
        self.subspec_idxs = subspec_idxs
        self.overlapped_subspec_idx_range = [subspec_idxs[1][0], subspec_idxs[-2][-1] + 1]  # [3, 961]
        self.frequency_dim = subspec_idxs[-1][-1] + 1  # 1024
        self.use_checkpoint = use_checkpoint

        self.mask_estimations = nn.ModuleList([])
        for subspec_idx in subspec_idxs:
            out_dim = (subspec_idx[-1] - subspec_idx[0] + 1) * channel_num

            submask_estimation = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("norm", RMSNorm(num_feature)),
                        ("ff_0", nn.Linear(num_feature, 4 * num_feature)),
                        ("tanh", nn.Tanh()),
                        ("ff_1", nn.Linear(4 * num_feature, 2 * out_dim)),
                        ("glu", nn.GLU()),
                    ]
                )
            )
            self.mask_estimations.append(submask_estimation)

    def forward(self, x):
        # x: (b, t, k n)

        out_x = torch.zeros((x.shape[0], x.shape[1], self.frequency_dim * self.channel_num)).to(x.device)
        # (b, t, 1024*4)
        for i, submask_estimation, subspec_idx in zip(range(len(self.subspec_idxs)), self.mask_estimations,
                                                      self.subspec_idxs):
            if self.use_checkpoint:
                sub_x = checkpoint_sequential(submask_estimation, 4, x[:, :, i])
            else:
                sub_x = submask_estimation(x[:, :, i])
            out_x[:, :, subspec_idx[0] * self.channel_num: (subspec_idx[-1] + 1) * self.channel_num] += sub_x

        if self.band_overlap:
            out_x[:, :, self.overlapped_subspec_idx_range[0]: self.overlapped_subspec_idx_range[1]] /= 2
        x = rearrange(out_x, "b t (f c) -> b c t f", c=self.channel_num)
        # b c t f
        return x


class BSTransformer(BaseStage):
    def __init__(
            self,
            takes,
            provides,
            input_channels: int,
            output_channels: int,
            target_sources_num: int,
            depth: int = 12,
            num_feature: int = 384,
            window_size: int = 2048,
            hop_size: int = 441,
            serialize_opts=None,
            use_checkpoint=True,
            use_flash_attn=False,
            enforce_dropout=0.1,
            mel_bands=60,
            band_overlap=True,
    ):
        BaseStage.__init__(self, takes, provides, serialize_opts)

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_sources_num = target_sources_num

        # Spec modules
        self.stft_funcs = {
            "stft": torchaudio.transforms.Spectrogram(
                n_fft=window_size,
                win_length=window_size,
                hop_length=hop_size,
                power=None,
                window_fn=torch.hann_window,
                center=True,
                pad_mode="reflect",
            ),
            "istft": torchaudio.transforms.InverseSpectrogram(
                n_fft=window_size,
                win_length=window_size,
                hop_length=hop_size,
                window_fn=torch.hann_window,
                center=True,
                pad_mode="reflect",
            ),
        }

        # Band split
        if mel_bands > 0:
            subspec_idxs = get_bandwidth_ranges(
                n_fft=window_size - 1,  # 2047
                n_filterbank=mel_bands,
                overlap=band_overlap,
            )
            self.multi_band_transform = MelBand(
                num_feature=num_feature,
                channel_num=self.input_channels * 2,  # * 2 for real and imag
                subspec_idxs=subspec_idxs,
            )
        else:
            self.multi_band_transform = BandSplit(
                num_feature=num_feature,
                channel_num=self.input_channels * 2,  # * 2 for real and imag
            )

        self.rotary_emb_t = RotaryEmbedding(dim=int(num_feature / 8))
        self.rotary_emb_k = RotaryEmbedding(dim=int(num_feature / 8))

        transformer_stack = nn.ModuleList([])
        for _ in range(depth):
            layer = BSTransformer_block(
                input_dim=num_feature,
                num_heads=8,
                dim_head=int(num_feature / 8),
                dropout=enforce_dropout,
                use_checkpoint=use_checkpoint,
                use_flash_attn=use_flash_attn,
            )
            transformer_stack.append(layer)
        self.transformer_stack = transformer_stack

        if mel_bands > 0:
            self.mask_estimation = MelMaskEstimation(
                num_feature=num_feature,
                channel_num=self.output_channels * 2,
                use_checkpoint=use_checkpoint,
                subspec_idxs=subspec_idxs,
                band_overlap=band_overlap,
            )
        else:
            self.mask_estimation = MaskEstimation(
                num_feature=num_feature,
                channel_num=self.output_channels * 2,
                use_checkpoint=use_checkpoint,
            )

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        pass

    def manually_to_device(self, device):
        for k, v in self.stft_funcs.items():
            self.stft_funcs[k] = v.to(device)

    def wav_to_complex_sp(
            self, input: torch.Tensor, eps: float = 1e-10
    ) -> List[torch.Tensor]:
        r"""Convert waveforms to magnitude, cos, and sin of STFT.

        Args:
            input: (batch_size, channels_num, segment_samples)
            eps: float

        Outputs:
            real: (batch_size, channels_num, time_steps, freq_bins)
            imag: (batch_size, channels_num, time_steps, freq_bins)
        """
        batch_size, _, _ = input.shape

        complex_spec = self.stft_funcs["stft"](input)
        real, imag = complex_spec.real, complex_spec.imag
        # real, imag: (batch_size, channels_num, freq_bins, time_steps)

        real = rearrange(real, "b c f t -> b c t f", b=batch_size)
        imag = rearrange(imag, "b c f t -> b c t f", b=batch_size)

        return real, imag

    def feature_maps_to_wav(
            self, input_real, input_imag, mask_real, mask_imag
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, _, _ = input_real.shape

        output_real = input_real * mask_real - input_imag * mask_imag
        output_imag = input_real * mask_imag + input_imag * mask_real

        output_real = rearrange(output_real, "b c t f -> b c f t")
        output_imag = rearrange(output_imag, "b c t f -> b c f t")

        # ISTFT.
        new_spec = torch.complex(output_real, output_imag)
        x = self.stft_funcs["istft"](new_spec)
        # (batch_size * target_sources_num, output_channels, segments_num)

        # Reshape.
        waveform = rearrange(
            x, "(b n) c t -> b (n c) t", b=batch_size, n=self.target_sources_num
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

    def _compute(self, mixtures: torch.Tensor):
        r"""Forward data into the module.

        Args:
            input_dict: dict, e.g., {
                waveform: (batch_size, input_channels, segment_samples),
                ...,
            }

        Outputs:
            output_dict: dict, e.g., {
                'waveform': (batch_size, output_channels, segment_samples),
                ...,
            }
        """
        # (batch_size, input_channels, segment_samples)
        with torch.autocast(device_type="cuda", enabled=False):
            real, imag = self.wav_to_complex_sp(mixtures.float())
        real, imag = real.to(mixtures.dtype), imag.to(mixtures.dtype)
        # real, imag: (b, c, t, f)
        x = torch.cat((real, imag), dim=1)
        # (b, c, t, f)

        # Remove last dim in freq
        x = x[..., 0: x.shape[-1] - 1]  # (b, c, t, f)

        # Separate into subbands
        x = self.multi_band_transform(x)
        # (b, t, k, n)

        for _, layer in enumerate(self.transformer_stack):
            x = layer(x, self.rotary_emb_t, self.rotary_emb_k)
        # (b, t, k, n)

        x = self.mask_estimation(x)
        # (b, c, t, f)

        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257.

        sep_point = self.output_channels
        mask_real = x[:, 0:sep_point, :, :]
        mask_imag = x[:, sep_point:, :, :]

        # Recover shape
        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        with torch.autocast(device_type="cuda", enabled=False):
            separated_audio = self.feature_maps_to_wav(
                input_real=real.float(),
                input_imag=imag.float(),
                mask_real=mask_real.float(),
                mask_imag=mask_imag.float(),
            )
        separated_audio = separated_audio.to(mixtures.dtype)
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        return [separated_audio]


class MssModel(BaseModel):
    def __init__(self, stages, input_names, output_names):
        super().__init__(
            stages=stages, input_names=input_names, output_names=output_names
        )


if __name__ == "__main__":
    dummy_input = torch.randn(2, 2, 44100 * 3)
    model = BSTransformer(
        takes=["waveform"],
        provides=["waveform"],
        input_channels=2,
        output_channels=2,
        target_sources_num=1,
        mel_bands=63,
        band_overlap=False,
    )
    dummy_input = torch.randn(2, 2, 44100 * 3)
    output = model({"waveform": dummy_input})
