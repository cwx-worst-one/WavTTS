import logging
import json
import euler
from recipes.serving.idls import base_thrift, sami_thrift

euler.install_thrift_import_hook()

access_key = "rcykXYrtEY"
sami_gateway_client = euler.Client(sami_thrift.SamiService, 'sd://lab.sami.gateway?cluster=release_thrift', timeout=1200)


def invoke_tts(text: str):
    payload_dict = {"speaker": "front_end_zh", "enable_text_seg": True, "text": text,"extra": {"phoneme_size":80}}
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="TTS",
        payload=json.dumps(payload_dict),
    )
    result = sami_gateway_client.Invoke(req)

    logging.debug(f"invoke tts result, code: {result.BaseResp.StatusCode}, payload: {result.payload}")
    if result.BaseResp.StatusCode != 0:
        return []

    if result.payload is not None:
        payload = json.loads(result.payload)
        return payload["text_segmentation"]

    return []


def invoke_asr(binary_data):
    config_dict = {
        "lang": "en",
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="ASR",
        payload=json.dumps(config_dict),
        data=binary_data
    )
    result = sami_gateway_client.Invoke(req)
    logging.debug(f"invoke asr result, code: {result.BaseResp.StatusCode}, payload: {result.payload}")

    if result.BaseResp.StatusCode != 0:
        return ""

    if result.payload is not None:
        payload = json.loads(result.payload)
        text = ""
        for res in payload["results"]:
            text += res["text"] + " "
        text = text.strip()
        return text

    return ""


def invoke_asr_with_alignment(binary_data):
    config_dict = {
        "lang": "en",
        "enable_word_info": True,
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="ASR",
        payload=json.dumps(config_dict),
        data=binary_data
    )
    result = sami_gateway_client.Invoke(req)
    logging.debug(f"invoke asr with alignment result, code: {result.BaseResp.StatusCode}, payload: {result.payload}")

    if result.BaseResp.StatusCode != 0:
        return ""

    if result.payload is not None:
        payload = json.loads(result.payload)
        text = ""
        speech_seg_list = []

        for res in payload["results"]:
            text += res["text"] + " "
            for w in res["alternatives"][0]['words']:
                seg = {'start': int(w['start_time'] * 1000), 'end': int(w['end_time'] * 1000)}
                speech_seg_list.append(seg)

        text = text.strip()
        return text, speech_seg_list

    return ""


def invoke_vad(binary_data):
    payload_dict = {
        "extra": {
            "begin_smooth_window_ms": 300,
            "begin_smooth_voice_proportion": 0.5,
            "end_smooth_window_ms": 500,
            "end_smooth_silence_proportion": 0.5,
            "voice_max_seconds": 25,
            "likelihood_threshold": 0.5,
            "enable_dynamic_smooth": False
        },
        "model": "v2"
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="VAD",
        payload=json.dumps(payload_dict),
        data=binary_data
    )
    result = sami_gateway_client.Invoke(req)
    logging.debug(f"invoke vad result, code: {result.BaseResp.StatusCode}, payload: {result.payload}")
    if result.BaseResp.StatusCode != 0:
        return {}

    if result.payload is not None:
        payload = json.loads(result.payload)
        return payload
    return {}
