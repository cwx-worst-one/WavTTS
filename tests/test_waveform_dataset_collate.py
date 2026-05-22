import torch

from wavtts.model.dataset import collate_fn


def test_collate_fn_pads_waveform_only_batches():
    batch = [
        {
            "text": "hello",
            "wav": torch.ones(11),
        },
        {
            "text": "world!",
            "wav": torch.arange(7, dtype=torch.float32),
        },
    ]

    out = collate_fn(batch)

    assert out["text"] == ["hello", "world!"]
    assert out["text_lengths"].tolist() == [5, 6]
    assert out["wav"].shape == (2, 11)
    assert out["wav_lengths"].tolist() == [11, 7]
    assert torch.allclose(out["wav"][1, 7:], torch.zeros(4))


if __name__ == "__main__":
    test_collate_fn_pads_waveform_only_batches()
