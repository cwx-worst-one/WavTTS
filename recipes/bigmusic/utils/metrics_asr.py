import json
import io
import os
import torchaudio
import subprocess
import re
from typing import List, Dict, Optional, Union
import torch

from logging import getLogger
from torchaudio.functional import resample

from dataclasses import dataclass


logger = getLogger(__name__)

PYPETREL_LIB_FOUND = False
try:
    import pypetrel

    pypetrel.set_log_level(5)
    PYPETREL_LIB_FOUND = True
except ImportError as e:
    logger.error(f"`pypetrel` library was not found in PYTHONPATH {e}")


def normalize_audio(
    sample_rate: int, url=None, audio_bytes=b"", format="wav", codec="pcm_s16le"
):
    """Original https://code.byted.org/seed/bigspeech_data/blob/asr-offline/bigspeech_process/util/wave_utils.py#L141"""
    if url:
        args_i = url
    else:
        args_i = "-"

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        args_i,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",  # 单声道
        "-f",
        format,
        "-acodec",
        codec,
        "pipe:1",
    ]
    p = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    input = None
    if audio_bytes and not url:
        input = audio_bytes
    out, err_msg = p.communicate(input)
    if p.returncode != 0:
        raise Exception("execute ffmpeg fail: %s" % err_msg.decode())
    return out

ASR_MODEL_PATHS = {
    "en": "/opt/tiger/pypetrel/models/en_us_lyric",
    "en_punc": "/opt/tiger/pypetrel/models/en_us_lyric_with_itn_v2",
    "zh": "/opt/tiger/pypetrel/models/sami_zh_cn_lyric"
}

class Wav2Lyrics:
    _sample_rate = 16000
    _gpu_available = torch.cuda.is_available()

    def __init__(
        self,
        model_name: str = "en_punc",
        device_id = None,
    ):
        self.verify_dependencies()
        self.device_id = device_id
        self.model_path = ASR_MODEL_PATHS[model_name]

        

    def verify_dependencies(self):
        if not PYPETREL_LIB_FOUND:
            raise ImportError("`pypetrel` package is not found")

        ld_library_paths = os.getenv("LD_LIBRARY_PATH", "").split(":")
        python_path = os.getenv("PYTHONPATH", "")
        if not self._gpu_available:
            raise Exception("Wav2Lyrics is only supported on GPU")

        if "/opt/tiger/pypetrel/pypetrel/lib/" not in ld_library_paths:
            raise ImportError(
                "`/opt/tiger/pypetrel/pypetrel/lib/` was not found in `LD_LIBRARY_PATH`"
            )

        if "/opt/tiger/pypetrel/pypetrel" not in python_path:
            raise ImportError(
                "`/opt/tiger/pypetrel/pypetrel` was not found in `PYTHONPATH`"
            )

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.warning(
                "Could not find env var: CUDA_VISIBLE_DEVICES. Setting default to 0"
            )

    def __call__(
        self,
        wav_batch: torch.Tensor,
        sample_rate: int,
        sample_lengths: Optional[int] = None,
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        if not pypetrel.is_engine_initialized():
            pypetrel.initialize_engine(self.model_path)

        if len(wav_batch.shape) == 2:
            wav_batch = wav_batch.unsqueeze(1)
        wav_batch = wav_batch.float().cpu()
        wav_bytes_batch = []
        for idx, wav in enumerate(wav_batch):
            if sample_lengths is not None:
                # In some cases sample_lengths may be 0 which will cause ASR to crash.
                # We use a lower-bound of 1 second here to avoid crash.
                wav = wav[..., : max(sample_lengths[idx], sample_rate)]

            byte_io = io.BytesIO()
            torchaudio.save(byte_io, wav, sample_rate, format="wav", encoding="PCM_S", bits_per_sample=16)
            byte_io.seek(0)
            wav_bytes_resampled = byte_io.read()
            wav_bytes_batch.append(wav_bytes_resampled)

        prev_flag = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id).split(':')[-1]

        if not PYPETREL_LIB_FOUND:
            return ["" for _ in range(len(wav_bytes_batch))], wav_bytes_batch

        num = len(wav_bytes_batch)
        wav_iter = iter(wav_bytes_batch)
        out = {"indices": [], "lyrics": []}
        with pypetrel.asr.OfflineRecognizer(num) as recognizer:
            recognizing_idx = 0
            num_recognized = 0
            has_next = True
            while num_recognized < recognizing_idx or has_next:
                while recognizing_idx - num_recognized < num and has_next:
                    wav = next(wav_iter, None)
                    if wav is None:
                        has_next = False
                        break

                    wav_input = pypetrel.asr.ASREngineInput()
                    wav_input.set_waveform(wav_bytes_resampled)
                    wav_input.set_finish(True)
                    wav_input.set_sample_rate(self._sample_rate) # "target sample rate". Must be set to 16k. Model will resample anyways
                    asr_input = pypetrel.asr.Input()
                    asr_input.set_asr_input(wav_input)

                    recognizer.submit(asr_input, recognizing_idx)
                    recognizing_idx += 1

                r = recognizer.poll(100)
                if r is None:
                    continue
                err, output, orig_idx = r
                num_recognized += 1

                if err:
                    print(err)
                    continue

                out["indices"].append(orig_idx)
                out["lyrics"].append(
                    json.loads(output.json_output)["result"][0]["text"]
                )
        out["wav"] = [wav_batch[idx] for idx in out["indices"]]
        lyrics_wavs = [
            (lyrics, wav)
            for _, lyrics, wav in sorted(zip(out["indices"], out["lyrics"], out["wav"]))
        ]
        lyrics, wavs = zip(*lyrics_wavs)
        if self.device_id is not None:
            if prev_flag is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_flag

        return {"lyrics": lyrics, "audio": wavs}


def init_asr(hpath, local_rank, cache_dir=None):
    wav2lyrics_module = Wav2Lyrics(
        model_name=hpath,
        device_id=local_rank
    )
    return { 'asr': wav2lyrics_module }

def asr_transcribe_lyrics(requires, wav_batch, sample_rate, sample_lengths=None):
    """
    As transcription jobs can fail and the order is not preserved,
    this function returns a dictionary contains lyrics and a corresponding wav tensor.

    wav_batch: must have shape [batch, channel, time]
    sample_lengths: length of each audio sample in batch
    sr: sample rate
    """
    return requires['asr'](wav_batch, sample_rate=sample_rate, sample_lengths=sample_lengths)['lyrics']

@dataclass
class EditOps:
    subs: int = 0
    ins: int = 0
    dels: int = 0

    def edits(self) -> int:
        return self.subs + self.ins + self.dels

    def clone(self):
        return EditOps(subs=self.subs, ins=self.ins, dels=self.dels)


def edit_distance(seq1, seq2):
    """Extension of torchaudio's edit_distance that returns detailed
    breakdown of insertions, substitutions, and deletions.
    """
    len_sent2 = len(seq2)
    dold = []
    for i in range(len_sent2 + 1):
        dold.append(EditOps(ins=i))
    dnew = [EditOps() for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = EditOps(dels=i)
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1].clone()
            else:
                x = min(dold[j - 1].edits(), dnew[j - 1].edits(), dold[j].edits())
                if x == dold[j - 1].edits():  # substitution
                    dnew[j] = dold[j - 1].clone()
                    dnew[j].subs += 1
                elif x == dnew[j - 1].edits():  # insertion
                    dnew[j] = dnew[j - 1].clone()
                    dnew[j].ins += 1
                else:  # deletion
                    dnew[j] = dold[j].clone()
                    dnew[j].dels += 1

        dnew, dold = dold, dnew

    return dold[-1]


def remove_punc_case(text):
    return re.sub("[.,!?]", "", text).lower()
