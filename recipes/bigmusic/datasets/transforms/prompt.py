import json
import glob
import os
import soundfile
import librosa
import torch
import logging
import numpy as np
from recipes.bigmusic.datasets.transforms.leadsheet import seq_offset

# def load_prompt(spkr_id):
def save_prompt(meta_song_id, speaker_id, style_audio, sliced_notes_prompt, sliced_phones_prompt):
    save_path = f"svs_prompt/{meta_song_id}_{speaker_id}"
    os.makedirs(save_path, exist_ok=True)
    audio_path = f"{save_path}/audio.wav"
    note_path = f"{save_path}/note.json"
    phones_path = f"{save_path}/phones.json"
    style_audio = style_audio.cpu().numpy().flatten()
    soundfile.write(audio_path, style_audio, samplerate=24000)
    open(note_path, "w").write(json.dumps(sliced_notes_prompt))
    open(phones_path, "w").write(json.dumps(sliced_phones_prompt))

def trim_style_audio(style_audio, sliced_notes_prompt, sliced_phones_prompt):
    last = max(sliced_notes_prompt[-1]['end'], sliced_phones_prompt[-1]['end'])
    style_audio = style_audio[:, :int(24000*last)]
    return style_audio

def load_prompt2infer(save_path):
    note_path = f"{save_path}/note.json"
    phones_path = f"{save_path}/phones.json"

    sliced_notes_prompt = json.load(open(note_path, "r"))
    sliced_phones_prompt = json.load(open(phones_path, "r"))
    return sliced_notes_prompt, sliced_phones_prompt

def pad_audio(style_audio, target_len):
    len_audio = style_audio.shape[1]
    item_pad = torch.zeros([1, int(target_len)])
    item_pad[:, :len_audio] = torch.as_tensor(style_audio[:, :len_audio])
    style_audio = item_pad
    return style_audio

def load_prompt(save_path, vocal_prompt_duration=None, sample_rate=24000):
    audio_path = f"{save_path}/audio.wav"
    note_path = f"{save_path}/note.json"
    phones_path = f"{save_path}/phones.json"


    sliced_notes_prompt = json.load(open(note_path, "r"))
    sliced_phones_prompt = json.load(open(phones_path, "r"))
    speaker_id = os.path.basename(save_path).split("-")[0]
    style_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    style_audio = torch.tensor(style_audio).view(1, -1)
    audio_duration = librosa.get_duration(filename=audio_path)

    # pad style audio to match mininum vocal_prompt_duration
    if audio_duration < vocal_prompt_duration:
        print(f"Padding audio={audio_duration}s to fit vocal_prompt_duration={vocal_prompt_duration}")
        style_audio = pad_audio(style_audio, int(vocal_prompt_duration*sample_rate))
        audio_duration = style_audio.shape[1] / sample_rate
        
    last_note = sliced_notes_prompt[-1]['end']
    last_phone = sliced_phones_prompt[-1]['end']
    leadsheet_duration = min(last_note, last_phone)
    real_duration = min(leadsheet_duration, audio_duration)
    if leadsheet_duration < audio_duration:
        print(f"Padding leadsheet={leadsheet_duration}s to fit audio={audio_duration}s")
        # Pad leadsheet accrodingly
        if audio_duration - last_note  > 0.001:
            sliced_notes_prompt.append({"start":last_note, "end":audio_duration, "phone":[], "pitch": "Rest"})
        if audio_duration - last_phone > 0.001:
            sliced_phones_prompt.append({"start":last_phone, "end":audio_duration, "phone":['sil'], "pitch": []})
        last_note = sliced_notes_prompt[-1]['end']
        last_phone = sliced_phones_prompt[-1]['end']
        leadsheet_duration = min(last_note, last_phone)
    elif leadsheet_duration > audio_duration:
        print(f"Trim audio={audio_duration}s to fit leadsheet={leadsheet_duration}s ")
        style_audio = pad_audio(style_audio, int(leadsheet_duration*sample_rate))
        audio_duration = style_audio.shape[1] / sample_rate
    else:
        pass
    return speaker_id, style_audio, sliced_notes_prompt, sliced_phones_prompt, real_duration

def load_svsinput(note_path="assets/svs_test/output/rolling_in_the_deep_7.notes.json"):
    phones_path = note_path.replace(".notes.json", ".phones.json")
    sliced_notes_prompt = json.load(open(note_path, "r"))
    sliced_phones_prompt = json.load(open(phones_path, "r"))
    return sliced_notes_prompt, sliced_phones_prompt


class Prompter(object):
    def __init__(self, leadsheet_tokenizer, style_prompt_path, sample_rate=24000, vocal_prompt_duration=10.0) -> None:
        self.leadsheet_tokenizer = leadsheet_tokenizer
        self.style_prompt_path = style_prompt_path
        self.sample_rate = sample_rate
        self.style_prompt_cache = {}
        print("style_prompt_path", self.style_prompt_path)
        self.load_prompts(vocal_prompt_duration)

    def load_prompts(self, vocal_prompt_duration):
        if self.style_prompt_path == '' or not os.path.exists(self.style_prompt_path):
            logging.warning(f"No audio prompts"+self.style_prompt_path)
            return 
        for prompt_path in glob.glob(os.path.join(self.style_prompt_path, "*")):
            if os.path.exists(prompt_path):
                speaker_name, style_audio_i, sliced_notes_prompt_i, sliced_phones_prompt_i, real_duration = load_prompt(prompt_path, vocal_prompt_duration=vocal_prompt_duration)
                start = 0
                for note in sliced_notes_prompt_i:
                    if note['pitch'] != "Rest":
                        start = note['start']
                        break
                end = 0
                for note in reversed(sliced_notes_prompt_i):
                    if note['pitch'] != "Rest":
                        end = note['end']
                        break
                if end - start < 5.0:
                    continue
                logging.info(f"Audio prompt loading {prompt_path}, style_audio={style_audio_i.shape[1]/24000}")
                if speaker_name not in self.style_prompt_cache:
                    self.style_prompt_cache[speaker_name] = []
                self.style_prompt_cache[speaker_name].append((prompt_path, style_audio_i, sliced_notes_prompt_i, sliced_phones_prompt_i, real_duration))

    def select_prompt(self, notes, speaker_name, used_prompt=[]):
        max_ovlp = 0
        result = None
        if speaker_name not in self.style_prompt_cache:
            self.style_prompt_cache[speaker_name] = []
        for prompt_path, style_audio_i, sliced_notes_prompt_i, sliced_phones_prompt_i, real_duration in self.style_prompt_cache[speaker_name]:
            if prompt_path in used_prompt:
                continue
            ovlp = self.leadsheet_tokenizer.get_pitch_ovlp(notes, sliced_notes_prompt_i)
            if ovlp * real_duration > max_ovlp:
                result = (prompt_path, style_audio_i, sliced_notes_prompt_i, sliced_phones_prompt_i)
                max_ovlp = ovlp * real_duration
        if result is None:
            result = (prompt_path, style_audio_i, sliced_notes_prompt_i, sliced_phones_prompt_i)
        return result

    def warp_prompt(self, sliced_notes, sliced_phones, target_spkr_name="", num_prompt=1):
        offset_by = 0.0
        sliced_notes_prompt = []
        sliced_phones_prompt = []
        style_audio = []
        used_prompt = []
        if self.style_prompt_path == '' or target_spkr_name=="":
            logging.info(f"No prompt for {self.style_prompt_path} {target_spkr_name}")
            return [], sliced_notes, sliced_phones
            
        for i in range(int(num_prompt)):
            prompt_path, style_audio_i, sliced_notes_prompt_i, sliced_phones_prompt_i = self.select_prompt(sliced_notes, target_spkr_name, used_prompt=used_prompt)
            
            sliced_notes_prompt += seq_offset(sliced_notes_prompt_i, offset=offset_by)
            sliced_phones_prompt += seq_offset(sliced_phones_prompt_i, offset=offset_by)
            audio_duration = style_audio_i.shape[1] / self.sample_rate
            offset_by += audio_duration
            used_prompt.append(prompt_path)
            style_audio.append(style_audio_i)
        style_audio = torch.cat(style_audio, dim=1)
        logging.info(f"Final duration of prompt = {prompt_path} duration={style_audio.shape[1]/self.sample_rate}")

        if offset_by == 0.0:
            return [], sliced_notes, sliced_phones

        sliced_notes = seq_offset(sliced_notes, offset=offset_by)
        sliced_phones = seq_offset(sliced_phones, offset=offset_by)
        sliced_notes = sliced_notes_prompt + sliced_notes
        sliced_phones = sliced_phones_prompt + sliced_phones
        return style_audio, sliced_notes, sliced_phones

if __name__ == "__main__":
    from recipes.bigmusic.datasets.tokenizers.leadsheet import LeadSheetTokenizerV2
    leadsheet_tokenizer = LeadSheetTokenizerV2()
    style_prompt_path = "assets/svs/svs_prompt"
    p = Prompter(leadsheet_tokenizer, style_prompt_path)
    