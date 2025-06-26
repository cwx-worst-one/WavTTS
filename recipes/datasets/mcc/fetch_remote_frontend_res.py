import re
import os
import json
import string

from typing import Dict, Tuple, List

#以下依赖为tts 在线前端依赖
import euler
import thriftpy2

base_thrift = thriftpy2.load(os.path.join(
    os.path.dirname(__file__), "server/base.thrift"))
sami_thrift = thriftpy2.load(os.path.join(
    os.path.dirname(__file__), "server/sami.thrift"))

access_key = "BSDtSRVchN"
sami_gateway_client = euler.Client(
    sami_thrift.SamiService,
    'sd://lab.sami.gateway?cluster=release_thrift',
    timeout=1200,
)

def call_tts_frontend(text, appkey='BSDtSRVchN', speaker="front_end_zh", lang="zh_en", enable_recover_puncts=True, with_tn_text=True):
    payload = {
        "speaker":speaker,
        "internal":{
            "lab_version": "V3", 
            "enable_recover_puncts": enable_recover_puncts,
            "context_language":lang,
        },
        "audio_config":{
            "lang":lang,
        },
        "text": text,
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=appkey,
        method="TTS",
        payload=json.dumps(payload),
        data=None,
    )
    result = sami_gateway_client.Invoke(req)
    #logging.debug(f"invoke tts frontend result, code: {result.BaseResp.StatusCode}, payload: {result.payload}")
    for i in range(3):
        result = sami_gateway_client.Invoke(req)
        if result is not None and result.data is not None:
            ret_payload = json.loads(result.payload)
            if with_tn_text:
                return result.data, ret_payload["tn"]
            else:
                return result.data, None
    return None, None


def remote_run(text, lang="zh_en")-> str:
    front_res= call_tts_frontend(text, lang=lang)[0].decode()
    return front_res

#验证对齐
if __name__ == '__main__':
    # sentences = [
    #     "你好，世界！",
    #     "这是一个测试。",
    #     "欢迎使用Python。",
    #     "我喜欢编程。",
    #     "人工智能是未来的趋势。",
    #     "机器学习和深度学习是人工智能的主要方法。",
    #     "Python是一种非常好的编程语言。",
    #     "数据科学是目前非常热门的领域。",
    #     "数字化转型是未来的趋势。",
    #     "云计算在很多行业都有广泛的应用。",
    #     "大数据是现代科技的重要组成部分。",
    #     "区块链技术有着广泛的应用前景。",
    #     "我喜欢看科幻电影。",
    #     "音乐对人的心理有很大的影响。",
    #     "健康的饮食习惯对身体非常重要。",
    #     "锻炼是保持健康的重要方式。",
    #     "阅读可以提高自己的知识和见识。",
    #     "旅行可以开阔人的眼界。",
    #     "学习新的语言可以提高思维能力。",
    #     "艺术可以丰富人的精神世界。"
    # ]
    
    # try:
    #     from sami_tts_api.engine import TtsEngine, generate_tts_config
    # except Exception as e:
    #     print(f"[Warning] Failed loading sami_tts_api: {e}")

    # lib_path="/home/liuyang.1314/workspace/Project/samantha/temp/flute_sami/sami_engine_cleaned/libs/libsami.so"
    # fe="/home/liuyang.1314/workspace/Project/samantha/temp/models/tts_chinese_frontend_model__86.0.model"
    # fe_task="tts_chinese_frontend_model"

    # cfg = generate_tts_config()
    # engine = TtsEngine(lib_path=lib_path, fe=fe)
    # ex = engine.create_fe_executor(task_type=fe_task)

    # offline_res = []
    # online_res = []

    # for text in sentences:
    #     off_res = ex.run(text, config=cfg)[0]
    #     offline_res.append(off_res)
    #     on_res = remote_run(text)
    #     online_res.append(on_res)

    # print(online_res == offline_res)
    
    text="音乐对人的心理有很大的影响。"
    data = remote_run(text)
    # 忽略首行后按行分割
    lines = data.split('\n')[1:]
    # 提取每行的第六列与第一列
    parsed_data = [(line.split('\t')[0], line.split('\t')[5]) for line in lines if line]
    
    merged_data = {}
    prev_value = None

    for item in parsed_data:
        if prev_value is None:
            prev_value = item
        else:
            if item[1] == prev_value[1] or not item[1] or not prev_value[1]:
                prev_value = (prev_value[0] + '\t' + item[0], item[1] if item[1] else prev_value[1])
            else:
                merged_data[prev_value[1]] = prev_value[0]
                prev_value = item

    if prev_value is not None:
        merged_data[prev_value[1]] = prev_value[0]

    for key, value in merged_data.items():
        print(f"{key}\t{value}")
