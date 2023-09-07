import collections
import logging
import os
from functools import partial
from inspect import isclass
from typing import Callable, Dict, List, OrderedDict, Union

import torch
import torch.nn as nn

from samantha.core import BaseStage, Step  # noqa


class ModelPipeline(nn.Module):
    r"""Model pipeline.

    This model is an alternative implementation of :class:`BaseModel`, which
    ``stages`` is a dict instead of a list. In this version, the ``stages`` is only
    a collection of :class:`BaseStage` instead of a list which defines the execution
    order. Now we need to define the execution order in a new parameter ``pipeline``.

    There are also a few changes comparing to `BaseModel`:

    1. The ``stages`` now only support ``functools.partial`` of BaseStage. In config
    yaml, use ``!name`` instead of ``!new``.

    2. The ``pipeline``, ``input_names``, ``output_names`` are now support step-level
    configuration. You can pass a dictionary which keys are steps(train, validation,
    predict, etc.) and values are lists of stage names.

    3. The ``step`` property defines the current step the model runs in, the default
    value is ``Step.TRAIN``. You should change it in your LightningModule if you want
    to enable step-level config (e.g. ``self.model.step="test"``)

    Example::

    Assume we have two stages: ``SomeStageA`` as backbone and ``SomeStageB`` as
    classifier head. We want to train the model with the classifier head, and only use
    the backbone to predict. The config yaml should be like this:

    .. code-block:: yaml

        model_cls: !name:samantha.core.ModelPipeline
            stages:
                backbone: !name:<SomeStageA>
                    takes: [input]
                    provides: [output]
                head: !name:<SomeStageB>
                    takes: [output]
                    provides: [logits]
            pipeline:
                train: [backbone, head]
                predict: [backbone]
            input_names: [input]
            output_names:
                train: [logits]
                predict: [output]

    Args:
        stages (Dict[str, Callable[..., BaseStage]]):
            a dict of partial stages and their names.
        pipeline (Union[List[str], Dict[str, List[str]]]):
            a list of stage names, or a dict of stage names for different steps.
            defines the execution order of stages.
        input_names (Union[List[str], Dict[str, List[str]]]):
            a list of input names, or a dict of input names for different steps.
        output_names (Union[List[str], Dict[str, List[str]]]):
            a list of output names, or a dict of output names for different steps.

    """

    def __init__(
        self,
        stages: Dict[str, Callable[..., BaseStage]],
        pipeline: Union[List[str], Dict[str, List[str]]],
        input_names: Union[List[str], Dict[str, List[str]]],
        output_names: Union[List[str], Dict[str, List[str]]],
    ):
        super(ModelPipeline, self).__init__()

        # Process stages
        if not stages:  # pragma: no cover
            raise ValueError("stages must be specified.")
        if not all(
            [
                isinstance(s, partial)
                and isclass(s.func)
                and issubclass(s.func, BaseStage)
                for s in stages.values()
            ]
        ):
            raise TypeError(
                f"All `stages` must be partial of BaseStage, but got: {stages}"
            )
        # Initialize stages
        stages = {name: s() for name, s in stages.items()}
        self.stages = nn.ModuleDict(stages)

        # Process pipeline
        if isinstance(pipeline, List):
            # Check all names are valid
            self.pipeline = {Step.TRAIN: pipeline}
        elif isinstance(pipeline, Dict):
            # Check all steps are valid
            if "train" not in pipeline:
                raise KeyError(
                    "The `train` step must be defined when pipeline is a dict."
                )
            self.pipeline = {Step(k): v for k, v in pipeline.items()}

        # Check names are not empty and exist in `self.stages``
        for step, names in self.pipeline.items():
            if not names:  # pragma: no cover
                raise ValueError(f"Empty pipeline in {step} step")
            for name in names:
                if name not in self.stages:
                    raise KeyError(
                        f"Invalid stage name: {name} in {step} step. "
                        f"Available stages: {self.stages.keys()}"
                    )

        # Process input_names
        if isinstance(input_names, List):
            self.input_names = {Step.TRAIN: input_names}
        elif isinstance(input_names, Dict):
            if "train" not in input_names:
                raise KeyError(
                    "The `train` step must be defined when input_names is a dict."
                )
            self.input_names = {Step(k): v for k, v in input_names.items()}

        # Check names is not empty
        for step, names in self.input_names.items():
            if not names:  # pragma: no cover
                raise ValueError(f"Empty input_names in {step} step")

        # Process output_names
        if isinstance(output_names, List):
            self.output_names = {Step.TRAIN: output_names}
        elif isinstance(output_names, Dict):
            if "train" not in output_names:
                raise KeyError(
                    "The `train` step must be defined when output_names is a dict."
                )
            self.output_names = {Step(k): v for k, v in output_names.items()}

        # Check names is not empty
        for step, names in self.output_names.items():
            if not names:  # pragma: no cover
                raise ValueError(f"Empty output_names in {step} step")

        self.step = Step.TRAIN

        self.task_type = os.getenv("SAIL_TASK_TYPE", None)
        self.model_version = os.getenv("MODEL_VERSION", "1.0")

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step: str):
        self._step = Step(step)

    def _get_pipeline(self):
        if self.step in self.pipeline:
            return self.pipeline[self.step]
        return self.pipeline[Step.TRAIN]

    def _get_input_names(self):
        if self.step in self.input_names:
            return self.input_names[self.step]
        return self.input_names[Step.TRAIN]

    def _get_output_names(self):
        if self.step in self.output_names:
            return self.output_names[self.step]
        return self.output_names[Step.TRAIN]

    def forward(self, batch):
        r"""Perform forward computation.

        This method will call :func:`forward <BaseStage.forward>` function of
        :attr:`stages <self.stages>` based on :attr:`pipeline <self.pipeline>`.

        Args:
            batch (List[Tensor]): input batch which contains
             ``len(input_names)`` tensors.

        """
        if isinstance(batch, dict):
            data = batch
        else:
            if not isinstance(batch, List):
                batch = [batch]
            data = dict(zip(self._get_input_names(), batch))
        for stage_name in self._get_pipeline():
            stage = self.stages[stage_name]
            out = stage(data)
            data.update(out)
        return tuple(data[k] for k in self._get_output_names())

    @torch.no_grad()
    def export(self, **kwargs):
        r"""Perform export.

        This method will call :func:`export <BaseStage.export>` function of
        :attr:`stages <self.stages>` based on :attr:`pipeline <self.pipeline>`.

        """

        resource_loc = kwargs.get("resource_loc", None)
        if resource_loc is None:
            return None

        self.eval()
        self.step = Step.EXPORT
        export_stages = collections.OrderedDict()
        export_paths = {"resource_loc": resource_loc, "params_loc": []}
        for stage_name in self._get_pipeline():
            stage = self.stages[stage_name]
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
            _create_text_element("input_names", " ".join(self._get_input_names()))
        )
        tag_model_params.appendChild(
            _create_text_element("output_names", " ".join(self._get_output_names()))
        )
        tag_model_params.appendChild(
            _create_text_element("input_num", len(self._get_input_names()))
        )
        tag_model_params.appendChild(
            _create_text_element("output_num", len(self._get_output_names()))
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
