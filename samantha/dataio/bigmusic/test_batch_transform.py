import unittest

import numpy as np
import torch

from samantha.dataio.bigmusic.music_batch_transform import *


# Example
class TestAudioCollate(unittest.TestCase):
    def test_audio_padding_and_mask(self):
        # Create dummy audio samples with varying lengths.
        audio1 = np.ones(100, dtype=np.float32)  # length 100
        audio2 = np.ones(150, dtype=np.float32) * 2  # length 150
        audio3 = np.ones(120, dtype=np.float32) * 3  # length 120

        batch = [{"audio": audio1}, {"audio": audio2}, {"audio": audio3}]
        batch_out = {}

        # Instantiate the transform
        transform = AudioCollate(
            in_key="audio",
            out_key="audio",
            mask_key="audio_mask",
            pad_value=0.0,
            to_tensor=True,
        )

        # Apply the transform on the batch.
        transform(batch, batch_out)
        padded_audio = batch_out["audio"]
        mask = batch_out["audio_mask"]

        # Expected max length is 150 (the longest sample)
        expected_max_length = 150
        self.assertEqual(padded_audio.shape, (3, expected_max_length))
        self.assertEqual(mask.shape, (3, expected_max_length))

        # Check that the padded regions are filled with pad_value (0.0)
        self.assertTrue(torch.all(padded_audio[0, 100:] == 0))
        self.assertTrue(torch.all(padded_audio[2, 120:] == 0))
        # For audio2, no padding is expected (its length equals max_length)
        self.assertTrue(torch.all(padded_audio[1] != 0))

        # Check the mask: True for valid positions, False for padded.
        # For sample 1 (length 100)
        self.assertTrue(torch.all(mask[0, :100]))
        self.assertTrue(torch.all(~mask[0, 100:]))
        # For sample 2 (length 150)
        self.assertTrue(torch.all(mask[1, :150]))
        # For sample 3 (length 120)
        self.assertTrue(torch.all(mask[2, :120]))
        self.assertTrue(torch.all(~mask[2, 120:]))


if __name__ == "__main__":
    unittest.main()
