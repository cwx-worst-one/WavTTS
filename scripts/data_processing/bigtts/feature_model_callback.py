import logging
import os

from bytedance import easycycle

from samantha.utils.envs import getenv_int

logger = logging.getLogger("scripts.data_processing.bigtts.feature_model_callback")


def callback():
    train_task_id = os.getenv("ARNOLD_TRIAL_ID", "null")
    feature_name = os.getenv("FEATURE_NAME", None)
    template_train_dataset_id = getenv_int("TEMPLATE_TRAIN_DATASET_ID", -1)
    auto_train_workflow_uuid = os.getenv("AUTO_TRAIN_WORKFLOW_UUID", None)
    if (
        feature_name is None
        or template_train_dataset_id == -1
        or auto_train_workflow_uuid is None
    ):
        logger.warning("no feature related env var provided, skipping")
        return
    req = easycycle.FeatureCallBackReq()
    req.train_task_id = train_task_id
    req.feature_name = feature_name
    req.template_train_dataset_id = template_train_dataset_id
    req.auto_train_workflow_uuid = auto_train_workflow_uuid
    try:
        easycycle.feature_call_back(req=req)
    except Exception as e:
        logger.error("Failed to send callback", exc_info=e)
        return
    logger.info("Success to send callback")


if __name__ == "__main__":
    callback()
