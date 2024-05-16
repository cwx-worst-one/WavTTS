import io
import json
import logging
import re
import struct
import time

import librosa
import numpy as np
import soundfile as sf

from .client import client
from .idl import base_thrift, sami_thrift

access_key = "rcykXYrtEY"


# change extra.asr_post_process_task_type to
# chinese_switch_english_june_for_algorithm_labeled_data_asr_post_process_model for english
asr_postprocess_config_zh = {
    "lang": "zh",
    "text": "",
    "enable_word_info": True,
    "enable_punctuation": True,
    "enable_itn": False,
    "enable_true_case": True,
    "enable_disfluency": False,
    "enable_phase_info": False,
    "max_sentence_length": 256,
    "extra": {
        "skip_sentence_break": True,
        "asr_post_process_task_type": "chinese_switch_english_june_asr_post_process_model",
        "output_algorithm_brushing_data": True,
        "enable_alignment_accelerate": True,
        "alignment_break_sentence_threshold_seconds": 1.0,
        "alignment_use_prior": True,
        "alignment_min_active": 1000,
    },
}
asr_postprocess_config_en = {
    "lang": "en",
    "text": "",
    "enable_word_info": True,
    "enable_punctuation": True,
    "enable_itn": False,
    "enable_true_case": True,
    "enable_disfluency": False,
    "enable_phase_info": False,
    "max_sentence_length": 256,
    "extra": {
        "skip_sentence_break": True,
        "asr_post_process_task_type": "chinese_switch_english_june_for_algorithm_labeled_data_asr_post_process_model",
        "output_algorithm_brushing_data": True,
        "enable_alignment_accelerate": True,
        "alignment_break_sentence_threshold_seconds": 1.0,
    },
}

punct_config = {
    "confidence_threshold": 0.6,
    "no_punct_threshold": 0.1,
    "no_or_short_punct_threshold": 0.2,
    "short_punct_threshold": 0.4,
    "long_punct_threshold": 0.6,
    "punct_to_tag": {
        "，": "<comma>",
        "。": "<period>",
        "？": "<question>",
        "！": "<exclamation>",
        ",": "<comma>",
        ".": "<period>",
        "?": "<question>",
        "!": "<exclamation>",
    },
}


def invoke_tts(text: str, cur_ak=access_key):
    payload_dict = {
        "speaker": "front_end_zh",
        "enable_text_seg": True,
        "text": text,
        "extra": {"phoneme_size": 80},
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=cur_ak,
        method="TTS",
        payload=json.dumps(payload_dict),
    )
    result = client.Invoke(req)

    logging.debug(
        f"invoke tts result, code: {result.BaseResp.StatusCode}, payload: {result.payload}"
    )
    if result.BaseResp.StatusCode != 0:
        return []

    if result.payload is not None:
        payload = json.loads(result.payload)
        return payload["text_segmentation"]

    return []


def invoke_asr(binary_data, cur_ak=access_key):
    config_dict = {"lang": "en"}
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=cur_ak,
        method="ASR",
        payload=json.dumps(config_dict),
        data=binary_data,
    )
    result = client.Invoke(req)
    logging.debug(
        f"invoke asr result, code: {result.BaseResp.StatusCode}, payload: {result.payload}"
    )

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


def invoke_asr_with_alignment(binary_data, cur_ak=access_key):
    config_dict = {
        "lang": "zh",
        "model": "zh_en_mix",
        "enable_word_info": True,
        "extra": {"end_smooth_window_ms": 1500},
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=cur_ak,
        method="ASR",
        payload=json.dumps(config_dict),
        data=binary_data,
    )
    result = client.Invoke(req)
    logging.debug(
        f"invoke asr with alignment result, code: {result.BaseResp.StatusCode}, payload: {result.payload}"
    )

    if result.BaseResp.StatusCode != 0:
        return None

    if result.payload is not None:
        payload = json.loads(result.payload)
        return payload["results"]
    return None


def alignment_invoke(key, audio_binary, config, access_key):
    if "extra" not in config:
        config["extra"] = {}
    config["extra"]["debug_key"] = key
    config["enable_word_info"] = True
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="SpeechAlignment",
        data=audio_binary,
        payload=json.dumps(config),
    )

    cnt = 0
    while cnt < 5:
        try:
            resp = client.Invoke(req)
        except Exception as e:
            logging.error("SpeechToArticle getting error: {} {}".format(key, e))
            return None
        if resp.BaseResp.StatusMessage == "OK":
            return json.loads(resp.payload)
        else:
            print(resp.BaseResp.StatusMessage)
        cnt += 1
        time.sleep(1)

    logging.error(
        "SpeechToArticle req {}: service error, key = {}, task_id = {}, messgae = {}".format(
            cnt, key, resp.task_id, resp.BaseResp.StatusMessage
        )
    )
    return None


def invoke_speech_alignment(binary_data, text, cur_ak=access_key):
    config = {
        "audio_info": {"sample_rate": 16000, "channel": 1, "format": "pcm"},
        "model": "big_model",
        "extra": {"text": text, "enable_phone": True},
    }
    if "extra" not in config:
        config["extra"] = {}
    config["extra"]["debug_key"] = cur_ak
    config["enable_word_info"] = True
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=cur_ak,
        method="SpeechAlignment",
        payload=json.dumps(config),
        data=binary_data,
    )
    result = client.Invoke(req)
    logging.debug(
        f"invoke speech alignment with alignment result, code: {result.BaseResp.StatusCode}, payload: {result.payload}"
    )

    if result.BaseResp.StatusCode != 0:
        return None
    return json.loads(result.payload)


def invoke_vad(binary_data):
    payload_dict = {
        "extra": {
            "begin_smooth_window_ms": 300,
            "begin_smooth_voice_proportion": 0.5,
            "end_smooth_window_ms": 500,
            "end_smooth_silence_proportion": 0.5,
            "voice_max_seconds": 25,
            "likelihood_threshold": 0.5,
            "enable_dynamic_smooth": False,
        },
        "model": "v2",
    }
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="VAD",
        payload=json.dumps(payload_dict),
        data=binary_data,
    )
    result = client.Invoke(req)
    logging.debug(
        f"invoke vad result, code: {result.BaseResp.StatusCode}, payload: {result.payload}"
    )
    if result.BaseResp.StatusCode != 0:
        return {}

    if result.payload is not None:
        payload = json.loads(result.payload)
        return payload
    return {}


def punctuation_recovery(wav_bytes, cur_ak=access_key):
    target_sr = 16000
    wav, orig_sr = librosa.load(io.BytesIO(wav_bytes), sr=None)
    if orig_sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=orig_sr, target_sr=target_sr)

    buffer = io.BytesIO()
    sf.write(buffer, wav, target_sr, format="wav")
    buffer.seek(0)

    sentences = invoke_asr_with_alignment(buffer.read(), cur_ak)
    # print(f"ASR resp payload: {sentences}")

    alignments = []
    for idx, sent in enumerate(sentences):
        text = sent["text"]
        text_wo_punc = ""
        text = text.lower()
        for i, c in enumerate(text):
            if is_chinese(c) or "a" <= c <= "z" or "0" <= c <= "9" or c == "'":
                text_wo_punc += c
            elif (
                text_wo_punc != ""
                and i < len(text) - 1
                and ("a" <= text_wo_punc[-1] <= "z" or "0" <= text_wo_punc[-1] <= "9")
                and ("a" <= text[i + 1] <= "z" or "0" <= text[i + 1] <= "9")
            ):
                text_wo_punc += " "
        text_wo_punc = text_wo_punc.strip().lower()

        args = [
            idx,
            sent["start_time"],
            wav[
                int(sent["start_time"] * target_sr) : int(sent["end_time"] * target_sr)
            ],
            text_wo_punc,
        ]
        idx, alignment_result = force_alignment(args, target_sr, cur_ak)
        alignments.append([idx, alignment_result])

    alignments.sort(key=lambda x: x[0])
    # merge alignment
    try:
        merge_alignment = {
            "words": [],
            "punct_algorithm_info": {"words": [], "y_pt_p_result": []},
        }
        for idx, res in alignments:
            merge_alignment["words"] += res["words"]
            merge_alignment["punct_algorithm_info"]["words"] += res[
                "punct_algorithm_info"
            ]["words"]
            merge_alignment["punct_algorithm_info"]["y_pt_p_result"] += res[
                "punct_algorithm_info"
            ]["y_pt_p_result"]

        res = get_punctuation_result(merge_alignment, punct_config)
    except Exception as e:
        logging.error(f"multi-modal punctuation is error:={e}")
        res = None
    return res


def is_mostly_chinese(text):
    chinese_words = re.findall(r"[\u4e00-\u9fff]+", text)
    chinese_word_count = len(chinese_words)

    # 使用正则表达式匹配英文字母单词
    # english_words = re.findall(r"[a-zA-Z]+", text)
    # english_word_count = len(english_words)

    return chinese_word_count > 0


def force_alignment(args, sample_rate, cur_ak=access_key):
    idx, start_time, wav, text = args
    if text == "":
        logging.debug(f"Skip text is empty, idx = {idx}")
        return idx, None
    if cur_ak == access_key:
        cur_ak = "SKrwIGqYYx"
    if is_mostly_chinese(text):
        asr_postprocess_config = asr_postprocess_config_zh
    else:
        asr_postprocess_config = asr_postprocess_config_en
    res = asr_postprocess_with_data(
        wav_to_audio_binary((wav * 32767).astype("int16"), sample_rate),
        text,
        asr_postprocess_config,
        access_key=cur_ak,
    )
    if res is None or "results" not in res:
        logging.warning("ASR postprocess invoke failed, text: %s" % text)
        return idx, None

    for item in res["results"]:
        item["words"] = []
        for alter in item["alternatives"]:
            for word in alter["words"]:
                word["start_time"] = round(word["start_time"] + start_time, 3)
                word["end_time"] = round(word["end_time"] + start_time, 3)
                item["words"].append(word)
        item["text"] = item["text"].strip()
        item["start_time"] = item["words"][0]["start_time"]
        item["end_time"] = item["words"][-1]["end_time"]
        del item["alternatives"]
        del item["is_interim"]

    return idx, res["results"][0]


def asr_postprocess_with_data(audio_binary, text, config, access_key):
    if "extra" not in config:
        config["extra"] = {}
    config["text"] = text

    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=access_key,
        method="ASRPostProcess",
        data=audio_binary,
        payload=json.dumps(config),
    )
    # import pdb;pdb.set_trace()

    cnt = 0
    while cnt < 5:
        try:
            resp = client.Invoke(req)
        except Exception as e:
            logging.error("ASRPostProcess getting error: {} {}".format(text, e))
            return None
        if resp.BaseResp.StatusMessage == "OK":
            logging.debug(f"ASRPostProcess resp payload: {resp.payload}")
            return json.loads(resp.payload)
        cnt += 1
        time.sleep(1)
    logging.error(
        "SpeechToArticle req {}: service error, task_id = {}, messgae = {}".format(
            cnt, resp.task_id, resp.BaseResp.StatusMessage
        )
    )
    return None


def get_punctuation_result(sample, config):
    def postprocessing_multi_modality(words, y_pt_p_result, speech_info, config):
        """
        音频信息和文本信息融合考虑标点信息
        speech_info的内容：
        [
            {"word":x1, "start_time":t1_1, "end_time":t2_1, "confidence": t3_1},
            {"word":x2, "start_time":t1_2, "end_time":t2_2, "confidence": t3_2},
            {"word":x3, "start_time":t1_3, "end_time":t2_3, "confidence": t3_3},
            {"word":x4, "start_time":t1_4, "end_time":t2_4, "confidence": t3_4},
            ...
        ]

        words: 完整token的list，例如：["hello", "大", "家", "好", "nice"]
        y_pt_p_result: 标点预测的标签概率分布，二维list，记录每一个token所有标签的概率分布，例如：
        [
            [1.144321, -1.0023, ..., -0.1193],
            [1.134321, -2.0457, ..., -1.1193],
            ...
            [1.114321, -1.1023, ..., -3.1193],
        ]
        blank: 由于一些特殊符号，可能导致空格对齐出错，因此这里用blank存储输入的空格信息，便于加上标点后可以和输入前的空格保持一致，
            第i个位置表示对应位置的token前面是否加空格，例如：["", " ", "", " "]
        """

        result = []
        y_predict_pt_label = [np.argmax(item) for item in y_pt_p_result]

        for i in range(1, len(words) + 1):
            word = words[i - 1]

            if_valid = True
            if i < len(words):
                if (
                    speech_info[i]["confidence"] >= config["confidence_threshold"]
                    and speech_info[i - 1]["confidence"]
                    >= config["confidence_threshold"]
                ):
                    time_gap = (
                        speech_info[i]["start_time"] - speech_info[i - 1]["end_time"]
                    )
                else:
                    if_valid = False
            else:
                time_gap = 1000

            if if_valid:
                if time_gap < config["no_punct_threshold"]:
                    pass
                elif (
                    config["no_punct_threshold"]
                    <= time_gap
                    < config["no_or_short_punct_threshold"]
                ):
                    short_punct = [
                        y_pt_p_result[i - 1][1],
                        y_pt_p_result[i - 1][6],
                        y_pt_p_result[i - 1][0],
                    ]
                    max_p_index = short_punct.index(max(short_punct))

                    if max_p_index == 2:
                        pass
                    else:
                        if "a" <= word[-1] <= "z":
                            word += ","
                        else:
                            word += "，"
                elif (
                    config["no_or_short_punct_threshold"]
                    <= time_gap
                    < config["short_punct_threshold"]
                ):
                    short_punct = [y_pt_p_result[i - 1][1], y_pt_p_result[i - 1][6]]
                    max_p_index = short_punct.index(max(short_punct))

                    if "a" <= word[-1] <= "z":
                        word += ","
                    else:
                        word += "，"
                elif (
                    config["short_punct_threshold"]
                    <= time_gap
                    < config["long_punct_threshold"]
                ):
                    long_punct = [
                        y_pt_p_result[i - 1][2],
                        y_pt_p_result[i - 1][3],
                        y_pt_p_result[i - 1][4],
                        y_pt_p_result[i - 1][7],
                        y_pt_p_result[i - 1][8],
                        y_pt_p_result[i - 1][9],
                        y_pt_p_result[i - 1][1],
                        y_pt_p_result[i - 1][6],
                    ]
                    max_p_index = long_punct.index(max(long_punct))

                    if max_p_index in [1, 4]:
                        if "a" <= word[-1] <= "z":
                            word += "?"
                        else:
                            word += "？"
                    elif max_p_index in [2, 5]:
                        if "a" <= word[-1] <= "z":
                            word += "!"
                        else:
                            word += "！"
                    elif max_p_index in [6, 7]:
                        if "a" <= word[-1] <= "z":
                            word += ","
                        else:
                            word += "，"
                    else:
                        if "a" <= word[-1] <= "z":
                            word += "."
                        else:
                            word += "。"
                else:
                    long_punct = [
                        y_pt_p_result[i - 1][2],
                        y_pt_p_result[i - 1][3],
                        y_pt_p_result[i - 1][4],
                        y_pt_p_result[i - 1][7],
                        y_pt_p_result[i - 1][8],
                        y_pt_p_result[i - 1][9],
                    ]
                    max_p_index = long_punct.index(max(long_punct))

                    if max_p_index in [1, 4]:
                        if "a" <= word[-1] <= "z":
                            word += "?"
                        else:
                            word += "？"
                    elif max_p_index in [2, 5]:
                        if "a" <= word[-1] <= "z":
                            word += "!"
                        else:
                            word += "！"
                    else:
                        if "a" <= word[-1] <= "z":
                            word += "."
                        else:
                            word += "。"
            else:
                if y_predict_pt_label[i - 1] == 0:
                    pass
                elif y_predict_pt_label[i - 1] in [1, 5, 6, 10, 11]:
                    if "a" <= word[-1] <= "z":
                        word += ","
                    else:
                        word += "，"
                elif y_predict_pt_label[i - 1] in [2, 7, 12]:
                    if "a" <= word[-1] <= "z":
                        word += "."
                    else:
                        word += "。"
                elif y_predict_pt_label[i - 1] in [3, 8]:
                    if "a" <= word[-1] <= "z":
                        word += "?"
                    else:
                        word += "？"
                elif y_predict_pt_label[i - 1] in [4, 9]:
                    if "a" <= word[-1] <= "z":
                        word += "!"
                    else:
                        word += "！"
                elif y_predict_pt_label[i - 1] == 13:
                    word += "·"

            result.append(word)

        for i in range(len(speech_info)):
            speech_info[i]["word"] = result[i]

        return speech_info

    def read_value(value):
        word_flag_info = []
        y_pt_p_result = value["punct_algorithm_info"]["y_pt_p_result"]

        for word in value["words"]:
            if "tag" not in word:
                word_flag_info.append(word)

        word_list = [item["word"] for item in word_flag_info]

        return word_flag_info, word_list, y_pt_p_result

    def if_chinese(c):
        return "\u4e00" <= c <= "\u9fa5"

    def reconstruction_value_dict(sample, word_flag_info):
        global_index = 0

        cache = []
        for item in sample["words"]:
            if "tag" not in item:
                cache.append(word_flag_info[global_index])
                global_index += 1

        text = ""
        word_info = []
        for item in cache:
            if not (
                (
                    text == ""
                    or if_chinese(text[-1])
                    or text[-1] in ["、", "。", "，", "？", "！"]
                )
                and if_chinese(item["word"][0])
            ):
                text += " "
            text += item["word"]

            if (
                "，" in item["word"]
                or "。" in item["word"]
                or "？" in item["word"]
                or "！" in item["word"]
                or "," in item["word"]
                or "." in item["word"]
                or "?" in item["word"]
                or "!" in item["word"]
            ):
                word = item["word"][:-1]
                punct = item["word"][-1]

                item["word"] = word
                word_info.append(item)
                word_info.append(
                    {
                        "word": punct,
                        "start_time": item["end_time"],
                        "end_time": item["end_time"],
                        "confidence": 0,
                        "tag": config["punct_to_tag"][punct],
                    }
                )
            else:
                word_info.append(item)

        sample["text"] = text.strip()
        sample["words"] = word_info
        del sample["punct_algorithm_info"]

        return sample

    word_flat_info, word_list, y_pt_p_result = read_value(sample)
    assert len(y_pt_p_result) == len(word_list)

    word_flat_info = postprocessing_multi_modality(
        word_list, y_pt_p_result, word_flat_info, config
    )
    result = reconstruction_value_dict(sample, word_flat_info)
    return result


def is_num_or_en(s):
    return re.match("[a-zA-Z0-9]", s)


def is_chinese(ch):
    if (
        not ("\u4e00" <= ch <= "\u9fef")
        and not ("\u3400" <= ch <= "\u4db5")
        and not ("\u20000" <= ch <= "\u2a6d6")
        and not ("\u2a700" <= ch <= "\u2b734")
        and not ("\u2b740" <= ch <= "\u2b81d")
        and not ("\u2b820" <= ch <= "\u2cea1")
        and not ("\u2ceb0" <= ch <= "\u2ebe0")
        and not ("\u2f00" <= ch <= "\u2fd5")
        and not ("\u2e80" <= ch <= "\u2ef3")
        and not ("\uf900" <= ch <= "\ufad9")
        and not ("\u2f800" <= ch <= "\u2fa1d")
        and not ("\ue815" <= ch <= "\ue86f")
        and not ("\ue400" <= ch <= "\ue5e8")
        and not ("\ue600" <= ch <= "\ue6cf")
        and not ("\u31c0" <= ch <= "\u31e3")
        and not ("\u2ff0" <= ch <= "\u2ffb")
        and not ("\u3105" <= ch <= "\u312f")
        and not ("\u31a0" <= ch <= "\u31ba")
    ):
        return False
    else:
        return True


def wav_to_audio_binary(data, sample_rate):
    pcm_data = data.ravel().view("b").data

    file_size = 44 + len(pcm_data)
    header_data = b""
    header_data += b"RIFF"
    header_data += struct.pack("<I", file_size - 8)
    header_data += b"WAVE"
    header_data += b"fmt "

    format_tag = 0x0001
    channels = 1
    bit_depth = data.dtype.itemsize * 8
    bytes_per_second = sample_rate * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)
    fmt_chunk_data = struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        sample_rate,
        bytes_per_second,
        block_align,
        bit_depth,
    )

    header_data += struct.pack("<I", len(fmt_chunk_data))  # 16
    header_data += fmt_chunk_data
    header_data += b"data"
    header_data += struct.pack("<I", data.nbytes)

    return header_data + pcm_data


if __name__ == "__main__":
    # en
    # wav_url = "https://tosv.byted.org/obj/ies-multimedia-audio/samicore_platform/4c6fede5-8a8a-468c-aed5-0b51aa8ddd9d_adam_002.wav"  # noqa
    # cn
    # wav_url = "https://tosv.byted.org/obj/ies-multimedia-audio/samicore_platform/d0e7c946-423e-4269-8014-d3e2e3d803c8_magic_cn.wav"  # noqa

    # wav_path = "/mnt/bn/music-ai-unified-repo-lq/liulin/bigspeech_inference_server/src/grpc/temp_prompt.wav"
    # wav_path = "/mlx_devbox/users/lixingxing.cs/playground/seed/samantha/adam_002.wav"
    # wav_path = "/mnt/bn/cjw-lq-1/shared/from_liulin/temp.wav"
    wav_path = "/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_demo_7.0/zh/IP_caocao.wav"
    with open(wav_path, "rb") as buffer:
        import pdb

        pdb.set_trace()
        result = punctuation_recovery(buffer.read(), "oFDxHeMpJn")
        print(result["text"])
        # import pdb;pdb.set_trace()
        if result is not None:
            # print(json.dumps(result, ensure_ascii=False))
            print(result["text"])
            # print(result["words"])
