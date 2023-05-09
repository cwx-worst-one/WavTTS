r"""This module provides SAIL relevant utilities.

SAIL (`SAMI AI LAND`_) is a one-stop platform for external output and internal
efficiency built by SAMI team, including self-service, operation and maintenance
center, data platform, annotation platform, evaluation platform, training platform,
music recommendation platform, etc. We are committed to building a full chain ecological
closed loop from algorithm demand to business output.

.. _SAMI AI LAND:
   https://bytedance.feishu.cn/docx/doxcnOdcETgzDc7GXUj5mjJtTTh?bk_entity_id=enterprise_7114519401457516546

"""

import json
import logging
import os
import uuid

import command
import requests
from bytedance import easycycle

import samantha.utils.infer_dtype as infer

logger = logging.getLogger(__name__)


def get_signed_url():
    r"""Use the following format to get signed url for future model uploading.

    # request
    curl \
      --request POST 'https://sami.bytedance.net/sail/internal/experiment/callback' \
      --header 'Content-Type: application/json' \
      --header 'x-tt-env: xxxx' \ // query ENV: SAIL_PPE_ENV
      --header 'x-use-ppe: 1' \
      --data-raw '{
       "experiment_id": "xxxxx", // query ENV: SAIL_EXPERIMENT_ID
       "packed_train_param": {
           "active": true,
           "release_note": "hello, sami"
       }
      }'


    # response
    {
        "biz_resp":{"code":xx, "msg": "xxx"},
        "extra": "{'signed_url': 'xxx', 'version_id': 'xxx'}"
    }

    """

    url = "https://sami.bytedance.net/sail/internal/experiment/callback"
    exp_id = os.getenv("SAIL_EXPERIMENT_ID", None)
    if exp_id is None:
        logger.error(
            "No experiment id is provided, you may specify this by setting "
            "environment variable `SAIL_EXPERIMENT_ID`."
        )
        return None
    headers = {"Content-Type": "application/json"}

    x_tt_env = os.getenv("SAIL_PPE_ENV", None)

    if x_tt_env:
        headers["x-tt-env"] = "sail_ppe_pipeline"
        headers["x-use-ppe"] = "1"

    data = {
        "experiment_id": exp_id,
        "packed_train_param": {"active": True, "release_note": "export model"},
    }
    logger.info(
        f"Sending to {url} with header: {json.dumps(headers)},"
        f" with meta: {json.dumps(data)}"
    )

    ret = requests.post(url=url, headers=headers, json=data)
    resp = json.loads(ret.text)
    if resp["biz_resp"]["msg"] != "OK":
        raise ConnectionError(
            f"Failed to connect {url} with message: {resp['biz_resp']['msg']}"
        )
    extra = json.loads(resp["extra"])
    signed_url, version_id = extra["signed_url"], extra["version_id"]
    logger.info(f"Successfully get singed_url: {signed_url}, version_id: {version_id}.")
    return (signed_url, version_id)


def check_model_uploading(model_version_id):
    r"""Use the following format to check model uploading.
    curl \
      --request POST \
        'https://sami.bytedance.net/sail/internal/model/confirm_packed_submittal' \
      --header 'Content-Type: application/json' \
      --header 'x-use-ppe: 1' \
      --header "x-tt-env: ${ppe_env}" \
      --data-raw "{\"version_id\": \"${version_id}\",
      \"failed\":false,
      \"msg\": \"success\"
      }"

    Args:
        model_version_id(str): version id of the model
    """

    url = "https://sami.bytedance.net/sail/internal/model/confirm_packed_submittal"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    x_tt_env = os.getenv("SAIL_PPE_ENV", None)

    if x_tt_env:
        headers["x-tt-env"] = "sail_ppe_pipeline"
        headers["x-use-ppe"] = "1"
    data = {"version_id": model_version_id, "failed": False, "msg": "success"}

    ret = requests.post(url=url, headers=headers, json=data)
    resp = json.loads(ret.text)
    if resp["biz_resp"]["msg"] != "OK":
        raise ConnectionError(
            f"Failed to connect {url} with message: {resp['biz_resp']['msg']}"
        )


def upload_to_tos(model_path):
    r"""Upload model to TOS and check status.

    Args:
        model_path(str): path to model need upload.
    """

    signed_url, version_id = get_signed_url()
    model_bin = open(model_path, "rb").read()
    ret = requests.put(url=signed_url, data=model_bin)  # response
    if ret.status_code != 200:
        raise ConnectionError(
            f"Failed to connect {signed_url} with message: {ret.text}"
        )
    check_model_uploading(version_id)


def compile_model(dname: str):
    xml_path = None
    for item in os.listdir(dname):
        if item.endswith("xml"):
            xml_path = f"{dname}/{item}"
            break
    else:
        logger.error("Failed to find resource.xml")
        return None

    try:
        cmd = f"bash tools/model_compiler/compile.sh {xml_path}"
        command.run(cmd.split())
    except command.CommandException as e:
        logger.error(f"Failed to compile model with msg {e}")
        return None

    for item in os.listdir(dname):
        if item.endswith("model"):
            return f"{dname}/{item}"
    return None


def compile_and_upload_model(dname):
    model_path = compile_model(dname)
    upload_to_tos(model_path)


class SAILInferProcessor:
    DTYPE_MAPPING = {
        infer.Dtype.ID: easycycle.DataType.DATA_RECORD_ID,
        infer.Dtype.TEXT: easycycle.DataType.TEXT,
        infer.Dtype.VIDEO: easycycle.DataType.COMMON_URL,
        infer.Dtype.IMAGE: easycycle.DataType.COMMON_URL,
        infer.Dtype.AUDIO: easycycle.DataType.COMMON_URL,
    }

    @classmethod
    def process(cls, model_output: infer.Container):
        return easycycle.InferInputAndOutput(
            input=cls.process_input_output(model_output.input),
            output=cls.process_input_output(model_output.output),
            meta=cls.process_meta(model_output),
        )

    @classmethod
    def process_input_output(cls, item: infer.Element):
        dtype = cls.DTYPE_MAPPING[item.dtype]
        value = item.value

        if dtype is easycycle.DataType.COMMON_URL:
            host = easycycle.Host.CN
            username = os.getenv("ARNOLD_TRIAL_OWNER", "samantha")
            space = "samantha"
            expire_seconds = 60 * 60 * 24 * 1000
            filename = f"{uuid.uuid4().hex}.wav"
            signed_url = easycycle.upload_data_and_get_public_url(
                host=host,
                user=username,
                file_content=value,
                space=space,
                unique_file_name=filename,
                expire_seconds=expire_seconds,
            )
            value = signed_url
        return easycycle.InferElement(dtype, value)

    @classmethod
    def process_meta(cls, model_output: infer.Container):
        if model_output.meta is None:
            return "{}"
        return json.dumps(model_output.meta)
