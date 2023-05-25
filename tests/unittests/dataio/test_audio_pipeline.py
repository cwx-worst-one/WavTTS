import librosa
import numpy as np
import pytest

from samantha.dataio.audio_pipeline import AudioPipeline


@pytest.fixture
def data_iter(global_datadir):
    return [str(global_datadir / "44k_3ch.wav")]


@pytest.fixture
def expect(data_iter):
    audio, _ = librosa.load(data_iter[0], mono=False, sr=None)
    return audio


@pytest.fixture
def expect_mono(data_iter):
    audio, _ = librosa.load(data_iter[0], mono=True, sr=None)
    return audio


@pytest.fixture
def expect_resample(data_iter):
    audio, _ = librosa.load(data_iter[0], mono=False, sr=16000)
    return audio


def test_build_pipeline_from_init_args(data_iter):
    pipeline = AudioPipeline(
        data_iter,
        [
            ["read"],
            ["resample", 16000],
            ["mono"],
            ["slice_sample", {"start": 10, "end": 2000}],
            ["slice_sample", 0, {"end": 1000}],
            ["slice_sample", 0, 100],
            {"name": "numpy", "channel_first": True},
            ["norm"],
        ],
    )
    for x in pipeline():
        assert isinstance(x, np.ndarray)


def test_build_pipeline_with_dot_action(data_iter):
    def multiply(audio, factor):
        return audio * factor

    pipeline = (
        AudioPipeline(data_iter=data_iter)
        .read()
        .resample(target_sample_rate=16000)
        .mono()
        .slice_sample(10, 2000)
        .slice_duration(start=1000)
        .numpy()
        .slice_sample(10, 200)
        .norm()
        .apply("multiply", func=multiply, factor=10)
    )

    for x in pipeline():
        assert isinstance(x, np.ndarray)


def test_slice_sample(data_iter, expect):
    expect = expect.T
    # slice sample before numpy
    pipeline = (
        AudioPipeline(data_iter=data_iter)
        .read()
        .slice_sample(0, 2000)
        .numpy(channel_first=False)
        .norm()
    )

    for x in pipeline():
        assert isinstance(x, np.ndarray)
        assert x.shape == (2000, 3)
        assert np.all(expect[:2000] == x)

    # slice sample after numpy
    pipeline = pipeline.slice_sample(0, 20)
    for x in pipeline():
        assert isinstance(x, np.ndarray)
        assert x.shape == (20, 3)
        assert np.all(expect[:20] == x)


def test_read_file(data_iter, expect):
    pipeline = (
        AudioPipeline(data_iter=data_iter).read_file().numpy(channel_first=True).norm()
    )
    for x in pipeline():
        assert np.all(x == expect)


def test_read_data(expect):
    pipeline = (
        AudioPipeline([expect])
        .read_data(sample_rate=44100, num_channel=3, channel_first=True)
        .numpy(channel_first=True)
        .norm()
    )
    for x in pipeline():
        assert np.all(x == expect)


def test_read(data_iter, expect):
    pipeline = (
        AudioPipeline([expect])
        .read(sample_rate=44100, num_channel=3, channel_first=True)
        .numpy(channel_first=True)
        .norm()
    )
    for x in pipeline():
        assert np.all(x == expect)

    pipeline = (
        AudioPipeline(data_iter=data_iter).read().numpy(channel_first=True).norm()
    )
    for x in pipeline():
        assert np.all(x == expect)


def test_mono(data_iter, expect_mono):
    pipeline = AudioPipeline(data_iter=data_iter).read().mono().numpy().norm()
    for x in pipeline():
        np.testing.assert_allclose(x.reshape((-1)), expect_mono, atol=5e-5)


def test_int16(data_iter, expect):
    pipeline = (
        AudioPipeline(data_iter=[expect])
        .read(sample_rate=44100, num_channel=3, channel_first=True)
        .numpy(channel_first=True)
        .norm()
        .int16()
    )
    for x in pipeline():
        assert np.all(x == (expect * 32768).astype(np.int16))
