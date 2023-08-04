from samantha.dataio.utils import parse_data_urls


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
