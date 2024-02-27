import os
from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import librosa

from recipes.text2semantic.datasets.sami_tacolabel import generate_tacolabels_from_textstr_punc  
from recipes.voicebox.datasets.utils import PhoneToId


class MetaBasedDataset(Dataset):
    def __init__(self, meta_lst):
        self.meta = self._parse_meta(meta_lst)

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                if len(line.strip().split("|")) == 6:
                    uttid, text, prompt_lab_path, prompt_wav_path, generate_text, generate_lab_path = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    if not os.path.isabs(prompt_lab_path):
                        prompt_lab_path = os.path.join(os.path.dirname(meta_lst), prompt_lab_path)
                    if not os.path.isabs(generate_lab_path):
                        generate_lab_path = os.path.join(os.path.dirname(meta_lst), generate_lab_path)
                    meta.append([uttid, text, prompt_lab_path, prompt_wav_path, generate_text, generate_lab_path])
                elif len(line.strip().split("|")) == 5:
                    uttid, text, prompt_lab_path, prompt_wav_path, generate_text = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    if not os.path.isabs(prompt_lab_path):
                        prompt_lab_path = os.path.join(os.path.dirname(meta_lst), prompt_lab_path)
                    meta.append([uttid, text, prompt_lab_path, prompt_wav_path, generate_text])
                else:
                    uttid, text, prompt_lab_path, prompt_wav_path = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    if not os.path.isabs(prompt_lab_path):
                        prompt_lab_path = os.path.join(os.path.dirname(meta_lst), prompt_lab_path)
                    assert os.path.exists(prompt_wav_path), prompt_wav_path
                    meta.append([uttid, text, prompt_lab_path, prompt_wav_path])
        return meta
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        return self.meta[index]


# For UMM
class PickleDataset(Dataset):
    def __init__(self, wav_dir=None, 
            text_file=None, meta_file=None,
            pickle_file=None, npy_path=None,
            prompt_lang="en", syn_lang="en"):
        assert pickle_file is not None or npy_path is not None

        if pickle_file is not None and text_file is not None:
            self.meta = self._parse_meta1(pickle_file, wav_dir, text_file)
            self.type = 1
        elif npy_path is not None and text_file is not None:
            self.meta = self._parse_meta2(npy_path, wav_dir, text_file)
            self.type = 1
        elif npy_path is not None and meta_file is not None:
            self.meta = self._parse_meta3(npy_path, wav_dir, meta_file)
            self.type = 2
        else:
            raise NotImplementedError
            
        self.phone2id = PhoneToId()

        self.prompt_frontend_version = "English_v3_punc" if prompt_lang == "en" else "Chinese_v3_punc"
        self.syn_frontend_version = "English_v3_punc" if syn_lang == "en" else "Chinese_v3_punc"


    def _parse_meta3(self, npy_path, wav_dir, meta_file):
        meta = []
        with open(meta_file, "r", encoding="utf8") as f:
            for line in f:
                uttid, prompt_text, prompt_wav_path, syn_text = line.strip().split("|")
                if not os.path.isabs(prompt_wav_path):
                    prompt_wav_path = os.path.join(os.path.dirname(wav_dir), prompt_wav_path)

                if not os.path.exists(os.path.join(npy_path, uttid+".npy")):
                    continue
                predictor_token = np.load(os.path.join(npy_path, uttid+".npy"))
                predictor_token = torch.from_numpy(predictor_token).squeeze(0)
                if predictor_token.size(0) > 1:
                    meta.append([prompt_text, syn_text, prompt_wav_path, predictor_token, uttid])
        return meta


    def _parse_meta2(self, npy_path, wav_dir, text_file):

        text_list = torch.load(text_file)
        text_dict = {}
        for line in text_list:
            text_id, prompt_taco, prompt_wav_path, target_taco = line
            text_dict[text_id] = [prompt_taco, prompt_wav_path, target_taco]

        meta = []
        npys = os.listdir(npy_path)
        for npy in npys:
            if not npy.endswith("npy"):
                continue
            uttid = npy[:-4]
            predictor_token = np.load(os.path.join(npy_path, npy))
            predictor_token = torch.from_numpy(predictor_token).squeeze(0)

            wav_file = text_dict[uttid][1]
            target_text = text_dict[uttid][2]
            prompt_text = text_dict[uttid][0]
            wav_path = os.path.join(wav_dir, wav_file)
            if predictor_token.shape[-1] > 1:
                meta.append((prompt_text, target_text, wav_path, predictor_token, uttid))

        return meta

    def _parse_meta1(self, pickle_file, wav_dir, text_file):
        with open(pickle_file, "rb") as handle:
            token_dict = pickle.load(handle)

        text_list = torch.load(text_file)
        text_dict = {}
        for line in text_list:
            text_id, prompt_taco, prompt_wav_path, target_taco = line
            text_dict[text_id] = [prompt_taco, prompt_wav_path, target_taco]

        meta = []
        empty_wav = 0
        for key, value in token_dict.items():
            if value is not None:
                # wav_file = '_'.join(key.split('_')[:-2]) + '.wav'
                # wav_file = key + '.wav'
                wav_file = text_dict[key][1]
                wav_path = os.path.join(wav_dir, wav_file)
                #wav_path = wav_file
                predict_token = value.cpu().squeeze(0)
                target_text = text_dict[key][2]
                prompt_text = text_dict[key][0]
                if predict_token.shape[-1] > 1:
                    meta.append((prompt_text, target_text, wav_path, predict_token, key))
            else:
                empty_wav += 1
                print(f"===>>> {key} has None value.")
        print(f"===>>> {empty_wav}/{len(token_dict)} generated wav empty.")
        return meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        if self.type == 1:
            prompt_text, target_text, wav_path, predictor_token, uttid = self.meta[index]
            prompt_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(prompt_text.strip().split("\n"))
            target_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(target_text.strip().split("\n"))
        elif self.type == 2:
            prompt_text, target_text, wav_path, predictor_token, uttid = self.meta[index]

            tacolab = generate_tacolabels_from_textstr_punc(prompt_text, self.prompt_frontend_version).decode()
            tacolab = tacolab.strip().split("\n")
            prompt_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)

            tacolab = generate_tacolabels_from_textstr_punc(target_text, self.syn_frontend_version).decode()
            tacolab = tacolab.strip().split("\n")
            target_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
        else:
            raise NotImplementedError

        return prompt_id, target_id, wav_path, predictor_token, uttid

    

# for unit2wav reconstrunction test.
class MetaBasedUnit2WavDataset(Dataset):
    def __init__(self, meta_lst, lang="en"):
        self.phone2id = PhoneToId()
        self.meta = self._parse_meta(meta_lst)
        self.frontend_version = "English_v3_punc" if lang == "en" else "Chinese_v3_punc"

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                uttid, prompt_text, prompt_wav_path, syn_text, syn_wav_path = line.strip().split("|")
                if not os.path.isabs(prompt_wav_path):
                    prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                if not os.path.isabs(syn_wav_path):
                    syn_wav_path = os.path.join(os.path.dirname(meta_lst), syn_wav_path)
                meta.append([uttid, prompt_wav_path, syn_wav_path, prompt_text, syn_text])
        return meta
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        uttid, prompt_wav_path, syn_wav_path, prompt_text, syn_text = self.meta[index]
        try:
            tacolab = generate_tacolabels_from_textstr_punc(prompt_text, self.frontend_version).decode()
            tacolab = tacolab.strip().split("\n")
            prompt_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
        except:
            prompt_text_id = None

        try:
            tacolab = generate_tacolabels_from_textstr_punc(syn_text, self.frontend_version).decode()
            tacolab = tacolab.strip().split("\n")
            syn_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(tacolab)
        except:
            syn_text_id = None

        return uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id
