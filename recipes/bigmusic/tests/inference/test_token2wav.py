import os
import numpy as np
from recipes.bigmusic.inference.token2wav import Token2Wav


def test_token2wav():
    t2w = Token2Wav()
    audio_token_path = os.path.join(
        os.environ['AI_MUSIC_DIR'],
        '../../dump/ai_music/tmp/20240403.at.npy'
    )
    tokens = np.load(audio_token_path)
    wav = t2w.run(tokens)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip