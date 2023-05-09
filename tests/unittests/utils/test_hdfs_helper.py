import contextlib
import json
import os

import pytest

from tests.helpers.runif import RunIf
from samantha.utils import hdfs_helper as hh

dummy_place = "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/example/didispeech/dummy_place.json"  # noqa


@pytest.fixture
def sample_hdfs_path():
    return "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/example/didispeech/dummy.json"  # noqa


@pytest.fixture
def sample_json_path():
    this_file_dir_path = os.path.dirname(__file__)
    test_json_path = os.path.join(
        this_file_dir_path, "../../samples/json/metadata.json"
    )
    return test_json_path


def test_ishdfs(sample_hdfs_path):
    assert hh.ishdfs(sample_hdfs_path)
    sample_non_hdfs_path = "tests/samples/images/logo.jpeg"
    assert not hh.ishdfs(sample_non_hdfs_path)


def test_put_many_error():
    try:
        hh.put_many(["", ""], "not_hdfs_path")
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "not_hdfs_path is not a directory."


def test_hopen(sample_hdfs_path, sample_json_path):
    # 1: test open hdfs
    res = hh.hopen(sample_hdfs_path)
    # need to assert content once a test file on HDFS is available
    assert isinstance(res, contextlib._GeneratorContextManager)

    # 2: test open json
    with hh.hopen(sample_json_path) as f:
        out_json = json.load(f)
        assert out_json["utt1"] == {"foo": -1, "bar": 0, "text": "hello world"}
        assert out_json["utt1"]["foo"] == -1
        assert out_json["utt1"]["bar"] == 0
        assert out_json["utt1"]["text"] == "hello world"


@pytest.mark.skip(reason="no hdfs in test env")
def test_hdfs_file(sample_json_path):
    """
    We first put a sample json file to a dummy place on hdfs,
    then create an HDFS file instance based on it.
    We write (aka append) 'final words' and verify it
    Lastly, we remove the dummy place on hdfs
    """
    hh.put(sample_json_path, dummy_place)
    sample_hdfs_file_instance = hh.HdfsFile(dummy_place)
    sample_hdfs_file_instance.write("final words")
    with hh.hopen(dummy_place) as f:
        lines = f.readlines()
        last_line = lines[-1].decode("utf-8")
        last_line = last_line.strip()
        assert last_line.endswith("final words")
    assert hh.rm(dummy_place)


@RunIf(has_hdfs=True)
def test_hdfs_ls():
    assert hh.hdfs_ls("/user")
