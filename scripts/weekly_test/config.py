'''
All kind of TOKEN, auth, CONSTANT.
'''

import os
import time
import hmac
import hashlib

URL_PREFIX = '{}://speech.{}.{}'.format('https', 'bytedance', 'net')
ARNOLD_URL_PREFIX = '{}://{}-api.{}.org'.format('https', 'arnold', 'byted')
ARNOLD_PREFIX = '{}://{}.{}.org'.format('https', 'arnold', 'byted')

# the username of who run the weekly test
WTEST_OWNER = os.getenv('WTEST_OWNER', '')
assert len(WTEST_OWNER) > 0, f"WTEST_WONER is not set."
# scm token https://cloud.bytedance.net/scm/favor
WTEST_SCM_TOKEN = os.getenv('WTEST_SCM_TOKEN', '')
assert len(WTEST_SCM_TOKEN) > 0, f"WTEST_SCM_TOKEN is not set."
# falcon token ${URL_PREFIX}/falcon/account/profile/token
WTEST_FALCON_TOKEN = os.getenv('WTEST_FALCON_TOKEN', '')
assert len(WTEST_FALCON_TOKEN) > 0, f"WTEST_FALCON_TOKEN is not set."
# arnold token ARNOLD_URL_PREFIX/api/v2/token/
WTEST_ARNOLD_TOKEN = os.getenv('WTEST_ARNOLD_TOKEN', '')
assert len(WTEST_ARNOLD_TOKEN) > 0, f"WTEST_ARNOLD_TOKEN is not set."
# to create new branch on gitlab https://code.byted.org/profile/personal_access_tokens
WTEST_GITLAB_TOKEN = os.getenv('WTEST_GITLAB_TOKEN', '')
assert len(WTEST_GITLAB_TOKEN) > 0, f"WTEST_GITLAB_TOKEN is not set."



# UUID for automated platform.
TASK_BASE_URL = ARNOLD_URL_PREFIX + "/api/v2/task"
SCM_BASE_URL = "https://scm.byted.org/api/repos/61223/versions"
FALCON_ARTIFACTS_URL = URL_PREFIX + "/falcon/apps/v1/artifacts"
FALCON_BASE_URL = URL_PREFIX + "/falcon/apps/v1/studio/workflows"
WORKFLOW_BASE_URL = URL_PREFIX + "/automated/projects/0/studio/workflows"
DOLPHIN_TREE_URL = "https://code.byted.org/lab-speech/dolphin/tree"
MODEL_TRAIN_WORKFLOW_UUID = "8d100131-64e6-4fe8-bbdf-0c6da55cfd0f"
MODEL_TRAIN_OPERATOR_UUID = "7e1a45fb-e04a-43ed-92bc-7f3270c8499a"
MODEL_DECODE_WORKFLOW_UUID = "3b5f2576-dc03-444c-8d6b-d35201cf0fd2"
MODEL_DECODE_OPERATOR_UUID = "fb357b8b-375c-455f-b224-3a71d247b474"
ONNX_EXPORT_WORKFLOW_UUID = "d115b4c1-da9e-435d-8c24-2672636524aa"
ONNX_EXPORT_OPERATOR_UUID = "964af53d-4477-4765-92e3-84063431dfee"

HDFS_BASE = "hdfs://haruna/home/byte_arnold_hl_speech_asr"
REPORT_BASE = f"{HDFS_BASE}/user/zhengyijie/report"
REPORT_JSON = f"{HDFS_BASE}/user/zhengyijie/report_json"

TRAIN_MODE = "训练"
DECODE_MODE = "解码"
UNUSED_MODELS = []
# Put here if model_name != trial_name or model_name + DECODE_MODE != trial_name.
WEEKLY_MODEL_TRAIN = {
    'RNN-T Transfomer': {TRAIN_MODE: 'tel_rnnt_transfomer_80kh', DECODE_MODE: '80kh解码'},
    'DFSMN pretrain input10K': {TRAIN_MODE: 'dfsmn_pretrain_input10K'},
    'DFSMN online finetune input10K': {TRAIN_MODE: 'dfsmn_finetune_input10K', DECODE_MODE: 'input10K解码'},
    'LSTM RNNT': {TRAIN_MODE: 'lstm_rnnt_ctc', DECODE_MODE: 'lstm_rnnt解码'},
    'SID feature': {TRAIN_MODE: "SID_feature"},
    'SID ecapa': {TRAIN_MODE: "SID_ecapa"},
    'SID wav': {TRAIN_MODE: "sid_wav"},
    'KWS 1wh': {TRAIN_MODE: 'kws_1wh'},
    'DFSMN Ce': {TRAIN_MODE: 'dfsmn_ce'},
    'aed emotion': {TRAIN_MODE: 'aed_emotion'},
    'Stress Prediction': {TRAIN_MODE: 'stress_prediction'},
}
WEEKLY_MODEL_DECODE = {}
# Some metrics are obsolete.

TRIAL_NAME_CHANGED = {
    # new : old
    'Lark DFSMN 中文流式' : 'AIOT中文流式',
    'Lark DFSMN 英文流式' : 'AIOT英文流式'
}
OBSOLETE_METRICS = {
    'Penguin Transformer torch': [
        '_DOT__DOT__DIR__DOT__DOT__DIR_tele_DIR_test_DIR_bytebot_tele_analysis_20210412_31224',
    ],
    'SID_feature': ['eer'],
    'cif san nonstream feat input10k解码': [
        'eval_meeting_refine_20201223', 'eval_lark_meeting_remove_zeros_refine_20201223',
        'eval_lark_chinese_refine_20201223', 'eval_lark_mixed_refine_20201223',
        'eval_h_test_mobile', 'eval_thchs', 'eval_ceo_external_refine_20201223',
        'eval_smart_dog_refine_20201223', 'eval_learning', 'eval_toutiao_school',
        'eval_aishell_2', 'eval_aishell_2019a', 'eval_aishell_2019c',
    ],
    'cif san stream feat input10k解码': [
        'eval_lark_mixed_refine_20201223', 'eval_h_test_mobile', 'eval_learning',
        'eval_toutiao_school', 'eval_meeting_refine_20201223',
        'eval_lark_meeting_remove_zeros_refine_20201223',
        'eval_lark_chinese_refine_20201223', 'eval_aishell_2019a',
        'eval_aishell_2019c', 'eval_thchs', 'eval_ceo_external_refine_20201223',
        'eval_smart_dog_refine_20201223', 'eval_aishell_2',
    ],
}

# The id of dolphin weekly test info sheet.
# https://bytedance.feishu.cn/wiki/wikcnstEneHEqZ7rdgjzHwqbrPd?sheet=i6WjU4
WEEKLY_TEST_LARK_SHEET_TOKEN = 'shtcnOl3eqqqKwzqqNsbSk0G43g'
WEEKLY_TEST_MODEL_PAGE_ID = 'i6WjU4'

# Agw auth.

# ak = "48b426c896793b4a44c37c2bf16b36aa"
# sk = "4c934bcf49002fe7ece0fcf5f8da4351"


class Agw:
    def __init__(self):
        self.ak = "48b426c896793b4a44c37c2bf16b36aa"
        self.sk = "4c934bcf49002fe7ece0fcf5f8da4351"
        # only support auth v1 now
        self.version = "auth-v1"
        self.expiration = 60

    def sha256_hmac(self, key, data):
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def sign(self, body_bytes):
        sign_key_info = '%s/%s/%d/%d' % (self.version, self.ak, time.time(), self.expiration)
        sign_key = self.sha256_hmac(str.encode(self.sk), str.encode(sign_key_info))
        sign_result = self.sha256_hmac(str.encode(sign_key), body_bytes)
        return '%s/%s' % (sign_key_info, sign_result)

    def __call__(self, req):
        if req.body:
            body = req.body
            body = str.encode(body)
            sign_key = self.sign(body)
        else:
            body = bytes()
            sign_key = self.sign(body)
        req.headers['Agw-Auth'] = sign_key
        return req

# who can receive the robot massage
robot_config={
    'emails':[
        WTEST_OWNER + '@bytedance.com',
        'shaoziqi.sam@bytedance.com',
        'liyong.0517@bytedance.com',
        'jiangbo.jacob@bytedance.com'
        ],
    'sleep_time':60, #60s
    'send_interval':60*30, #30min
    'stop': False, #to stop robot
}
