"""BaseStage class string"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

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

    :class:`~samantha.core.BaseStage` defines functions such as pre-processor,
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
