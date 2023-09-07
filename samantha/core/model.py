import collections
import inspect
import logging
import os
from functools import partial
from typing import Callable, List, OrderedDict, Union

import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_deprecation

from samantha.core import BaseStage  # noqa


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
