import pypetrel
import json
import io
import torchaudio

pypetrel.set_log_level(5)
import time
from dataclasses import dataclass


def wav2lyrics(wav_batch, sr=24000):
    """
    As transcription jobs can fail and the order is not preserved,
    this function returns a dictionary contains lyrics and a corresponding wav tensor.

    wav_batch: must have shape [batch, channel, time]
    sr: sample rate
    """
    if len(wav_batch.shape) == 2:
        wav_batch = wav_batch.unsqueeze(1)
    if not pypetrel.is_engine_initialized():
        pypetrel.initialize_engine("/mnt/bn/audio-diffusion/ashaw/models/asr/en_us_lyric")
    out = {"indices": [], "lyrics": []}
    num = wav_batch.size(0)
    wav_iter = iter(wav_batch)
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
          
                byte_io = io.BytesIO()
                torchaudio.save(byte_io, wav.cpu(), sr, format="wav")
                byte_io.seek(0)
                wav_bytes = byte_io.read()

                wav_input = pypetrel.asr.ASREngineInput()
                wav_input.set_waveform(wav_bytes)
                wav_input.set_finish(True)
                wav_input.set_sample_rate(sr)
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
            out["lyrics"].append(json.loads(output.json_output)["result"][0]["text"])
    out["wav"] = wav_batch[out["indices"]]
    lyrics_wavs = [(lyrics, wav) for idx,lyrics,wav in sorted(zip(out['indices'],out['lyrics'],out['wav']))]
    lyrics, wavs = zip(*lyrics_wavs)
    # pypetrel.destroy_engine() TODO: (AS) destroy engine at end of predict?
    return lyrics, wavs


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
    """ Extension of torchaudio's edit_distance that returns detailed
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
                if x == dold[j - 1].edits():    # substitution
                    dnew[j] = dold[j - 1].clone()
                    dnew[j].subs += 1
                elif x == dnew[j - 1].edits():  # insertion
                    dnew[j] = dnew[j - 1].clone()
                    dnew[j].ins += 1
                else:   # deletion
                    dnew[j] = dold[j].clone()
                    dnew[j].dels += 1

        dnew, dold = dold, dnew

    return dold[-1]