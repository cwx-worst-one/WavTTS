import io
import logging
import sys

import librosa

logger = logging.getLogger(__name__)

feature_common_keys = {"uttid", "dataset_name"}


def check_time_interval(cur_item, next_item, time_interval_threshold):
    start = cur_item.get("end_time", -1)
    end = next_item.get("start_time", -1)
    return start > 0 and end > 0 and end - start <= time_interval_threshold


def get_time_interval(item):
    start = item.get("start_time", -1)
    end = item.get("end_time", -1)
    return sys.maxsize if start < 0 or end < 0 or end < start else end - start


def merge_contextual_sample(sample_items):
    uttid = sample_items["index"][-1]["uttid"]
    sample = {"__key__": uttid, "uttid": uttid}
    # construct current features
    for name, items in sample_items.items():
        cur_item = items[-1]
        if name == "data":
            audio_bin = cur_item["audio"]
            cur_item = {
                "wav": audio_bin,
                "src_sample_rate": librosa.get_samplerate(io.BytesIO(audio_bin)),
            }
        elif name == "index":
            cur_item.pop("row_group_no", None)
            cur_item.pop("data_file", None)
        sample |= cur_item

    # construct contextual features
    for name, items in sample_items.items():
        for cur_item in items:
            if name == "data":
                audio_bin = cur_item["audio"]
                sample["contextual_wav_list"] = sample.get("contextual_wav_list", [])
                sample["contextual_wav_list"].append(audio_bin)
            elif name == "index":
                sample["contextual_uttid_list"] = sample.get(
                    "contextual_uttid_list", []
                )
                sample["contextual_uttid_list"].append(cur_item["uttid"])
                sample["contextual_meta_list"] = sample.get("contextual_meta_list", [])
                sample["contextual_meta_list"].append(cur_item["meta"])
                sample["contextual_speaker_list"] = sample.get(
                    "contextual_speaker_list", []
                )
                sample["contextual_speaker_list"].append(cur_item["speaker_id"])
            else:
                for key in cur_item.keys():
                    if key in feature_common_keys:
                        continue
                    new_key = generate_contextual_key(key)
                    sample[new_key] = sample[new_key] if new_key in sample else []
                    sample[new_key].append(cur_item[key])
    return sample


def generate_contextual_key(feature_name):
    return f"contextual_{feature_name}_list"
