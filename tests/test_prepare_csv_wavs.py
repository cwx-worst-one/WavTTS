import shutil
import subprocess
import sys
from importlib.resources import files

import pytest
from datasets import Dataset


def test_default_finetune_preparation_uses_released_vocab(tmp_path):
    if shutil.which("ffprobe") is None:
        pytest.skip("Dataset preparation test requires FFmpeg/ffprobe")

    examples = files("wavtts").joinpath("infer/examples")
    input_dir = tmp_path / "input"
    (input_dir / "wavs").mkdir(parents=True)
    shutil.copyfile(examples.joinpath("basic_ref_en.wav"), input_dir / "wavs" / "reference.wav")
    (input_dir / "metadata.csv").write_text(
        "audio_file|text\nwavs/reference.wav|Some call me nature, others call me mother nature.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wavtts.train.datasets.prepare_csv_wavs",
            str(input_dir),
            str(output_dir),
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "vocab.txt").read_bytes() == examples.joinpath("vocab.txt").read_bytes()
    dataset = Dataset.from_file(str(output_dir / "raw.arrow"))
    assert len(dataset) == 1
    assert dataset[0]["duration"] > 0
    assert dataset[0]["audio_path"] == (input_dir / "wavs" / "reference.wav").as_posix()
