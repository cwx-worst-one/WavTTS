import json
import pytest
from collections import Counter

from bytedance import easycycle

from samantha.dataio.utils import parse_data_urls, uniq_data_urls

easycycle.set_region(easycycle.get_current_region())

@pytest.mark.skip(reason="ci env not support")
def test_parse_data_urls():
    data_id = 23
    data_path = parse_data_urls(data_id=data_id)
    assert data_path == [
        "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/ci/00000.tar"
    ]

    data_urls = "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/ci/*.tar"
    data_path = parse_data_urls(data_urls=data_urls)
    assert data_path == [
        "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/ci/00000.tar"
    ]

@pytest.mark.skip(reason="ci env not support")
def test_uniq_data_urls():
    dataset_id = 2200
    if easycycle.get_current_region() == easycycle.Region.I18n:
        dataset_id = 167
    data_urls_v2 = easycycle.get_dataset_collection_info_v2(dataset_id)["origin"][
        "paths"
    ]
    data_urls_v1 = easycycle.get_dataset_collection_info(dataset_id)

    frequency_v2, uniqed_data_urls_v2 = uniq_data_urls(data_urls_v2)
    frequency_v1, uniqed_data_urls_v1 = uniq_data_urls_v1(data_urls_v1)

    assert frequency_v1 == Counter(frequency_v2)
    assert json.dumps(uniqed_data_urls_v1) == json.dumps(uniqed_data_urls_v2)


@pytest.mark.skip(reason="ci env not support")
def uniq_data_urls_v1(data_urls):
    frequency = Counter(url["index"] for url in data_urls)
    uniqed_data_urls, memory = [], set()
    for url in data_urls:
        index = url["index"]
        if index in memory:
            continue
        uniqed_data_urls.append(url)
        memory.add(index)
    return frequency, uniqed_data_urls
