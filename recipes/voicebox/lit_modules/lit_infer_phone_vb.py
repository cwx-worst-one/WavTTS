import logging
from multiprocessing.sharedctypes import Value
import os
from typing import Any
import yaml
from pytorch_lightning import LightningModule
import librosa
import numpy as np
import torch
import string
import random
from scipy.io.wavfile import read
from zhon.hanzi import punctuation
import string
from scipy.io.wavfile import write
from torchaudio.transforms import Resample

punctuation_all = punctuation + string.punctuation

from recipes.text2semantic.scripts.infer_utils import setup_seed
from recipes.voicebox.vocoder.BigVGAN.meldataset import mel_spectrogram
from recipes.voicebox.datasets.utils import PhoneToId, get_duration_frames, Masking
from recipes.voicebox.lit_modules.utils import ode_wrapper, plot_mel_local
from recipes.voicebox.lit_modules.lit_diffusion_voicebox import VoiceBoxModule
from recipes.voicebox.modules.speaker_encoder.utils import load_config
from recipes.voicebox.modules.speaker_encoder.processor import AudioProcessor
from recipes.text2semantic.datasets.text_converter import TextToTacolabID
from recipes.text2semantic.datasets.sami_tacolabel import generate_tacolabels_from_textstr_punc

from recipes.durator.lit_modules.lit_durator import DuratorModule

logger = logging.getLogger(__name__)


        # durator_ckpt_path='hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei/durator/AR_Discrete_en_cn/checkpoints/epoch=00-step=2000-loss=2.03.ckpt',
        #  durator_ckpt_path='hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei/durator/AR_Discrete/checkpoints/last.ckpt',

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32767.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

class VoiceBoxInfer(LightningModule):
    def __init__(
        self,
        ckpt_path,
        vocoder_ckpt_path,
        output_dir,
        text2id_path='recipes/valle/datasets/dict/metaid_to_textid.json',
        durator_type='clone',
        lang='en',
        seed=1996,
        mel_config=None,
        mel_norm_mean=-4.5,
        mel_norm_std=7,
        mel_padding=-2,
        inference_type="fake_zero_shot",
    ):
        super().__init__()
        self.phone2id = PhoneToId()
        self.mel_config = mel_config
        self.masking = Masking(p_drop_x=0.0, p_drop_audio_frames=(0.7, 0.8))
        self.model = VoiceBoxModule.load_from_checkpoint(checkpoint_path=ckpt_path).eval()
        self.seed = seed
        self.output_dir = output_dir
        self.inference_type = inference_type
        self.mel_norm_mean = mel_norm_mean
        self.mel_norm_std = mel_norm_std
        self.mel_padding = mel_padding
        self.lang = lang
        device='cuda:0'

        ####################### for durator model ########################
        self.durator_type = durator_type
        if self.durator_type == 'fastspeech':
            preprocess_config = '/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/durator/fastspeech2/config/LJSpeech/preprocess.yaml'
            model_config = '/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/durator/fastspeech2/config/LJSpeech/model.yaml'
            train_config = '/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/durator/fastspeech2/config/LJSpeech/train.yaml'
            preprocess_config = yaml.load(
                open(preprocess_config, "r"), Loader=yaml.FullLoader
            )
            # preprocess_config["preprocessing"]["text"]["language"] = args.language
            model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
            train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
            self.fs2_configs = (preprocess_config, model_config, train_config)
            self.preprocess_config = preprocess_config
            self.model_config = model_config
            self.train_config = train_config

            self.dur_model = get_model(900000, lang, self.fs2_configs, device, train=False)
        elif self.durator_type == 'durator2':
            self.durator = DuratorModule.load_from_checkpoint(checkpoint_path=durator_ckpt_path).eval()
        elif self.durator_type == 'clone':
            from recipes.voicebox.durator.clone_duration_infer.en.babble.datasets.building.text.encoding import init_tacolabel_dict
            from recipes.voicebox.durator.clone_duration_infer.en.babble.datasets.building.text.encoding import TacoLabelEncoding
            
            init_tacolabel_dict({'ZH-CN': ['default', 'C'], 'EN-GB': ['default', 'E']}, 'recipes/voicebox/durator/clone_duration_infer/en/configs/language/dictionaries', ['ZH-CN', 'EN-GB'], label_version='v2')
            self.tacolabel_encoding = TacoLabelEncoding({})
            if lang == 'en':
                self.en_dur_model = torch.jit.load('recipes/voicebox/durator/clone_duration_infer/en/en_clone_duration_model.pt').to(device).eval()
            elif lang == 'cn':
                self.cn_dur_model = torch.jit.load('recipes/voicebox/durator/clone_duration_infer/cn/cn_clone_duration_model.pt').to(device).eval()
        elif self.durator_type == 'vb_durator':
            from recipes.voicebox.lit_modules.lit_voicebox_dur import VoiceBoxModule as VoiceBoxDuratorModule
            self.durator = VoiceBoxDuratorModule.load_from_checkpoint("hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/congjian/logs/voicebox_durator/v1/checkpoints/epoch=00-step=98000-dur_loss=1.62.ckpt")
        elif self.durator_type == "zhengxi_durator":
            from ..durator.zhengxi_durator.duration_model import DurationModel_ts_prompt
            model_path = "/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/zhengxi_durator/43500.pth"
            checkpoint_dict = torch.load(model_path, map_location="cpu")  
            durator = DurationModel_ts_prompt()   
            durator.load_state_dict(checkpoint_dict["model"])  
            self.durator = durator.cuda().eval() 
        else:
            pass

        self.vocoder = self.prepare_vocoder(vocoder_ckpt_path, device)
    
        ##################################################################

    def prepare_vocoder(self, vocoder_ckpt_path, device):
        vocoder = torch.jit.load(vocoder_ckpt_path, map_location=device).eval()
        return vocoder
        

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        apply the audio model to the phone sequence and the audio context and generate the mel spectrum
        args:
            z: the phone sequence, a numpy array of int, [T]
            x_ctx: the audio context, a numpy array of float, [T, 100]
        return:
            mel: the mel spectrum, pytorch tensor, [M, 100]
        """
        setup_seed(self.seed)

        if self.inference_type == "fake_zero_shot":
            sample = self.encode_fake_zero_shot(batch)
        elif self.inference_type == "zero_shot":
            sample = self.encode_zero_shot(batch)
        else:
            sample = self.encode(batch)
        if sample is None:
            return None

        device = sample["phone"].device
        uttid = sample["uttid"]

        inputs = dict()
        inputs["frontend"] = {
            "phone": sample["phone"],
            "tone": sample["tone"],
            "word_seg": sample["word_seg"]
        }
        inputs["mel_lens"] = torch.tensor([sample["phone"].shape[1]], device=device)
        inputs["mel_mask"] = torch.ones([1, sample["phone"].shape[1]], device=device)
        inputs["mel_ctx"] = sample["mel_ctx"]
        inputs["prompt_mel"] = sample["prompt_mel"].transpose(1, 2)

        #for k, v in inputs.items():
        #    if k != "frontend":
        #        print(k, v.shape)

        with torch.no_grad():
            out_mel = self.model.inference(inputs, 50)
            out_mel = out_mel.float()
        nan_num = torch.sum(torch.isnan(out_mel))
        if nan_num > 0:
            print("{} has {} nan!!!!!".format(uttid, nan_num))
        out_mel = (out_mel * self.mel_norm_std) + self.mel_norm_mean
        out_mel = torch.clamp(out_mel, min=-8.5, max=3.5)
        out_mel = out_mel[:, :, inputs["prompt_mel"].shape[-1]:]


        output_wav = self.vocoder(out_mel)
        output_wav = output_wav.squeeze().cpu().numpy()

        #prompt_wav = sample["wav"].squeeze().cpu().numpy()
        prompt_mel = (sample["prompt_mel"] * self.mel_norm_std) + self.mel_norm_mean
        prompt_mel = torch.clamp(prompt_mel, min=-8.5, max=3.5)
        prompt_wav =self.vocoder(prompt_mel.transpose(1, 2))
        prompt_wav = prompt_wav.squeeze().cpu().numpy()


        audio = np.concatenate([prompt_wav, np.ones([10]), output_wav])
        output_path = os.path.join(self.output_dir, uttid+".wav")
        save_wav(audio, output_path)
        

    def encode_fake_zero_shot(self, sample):

        device = f"cuda:{self.trainer.local_rank}"
        data_dict = dict() 
        uttid, text, prompt_lab_path, prompt_wav_path, generate_text, generate_lab_path = sample

        duration = []
        prompt_lab = []
        duration = []
        with open(prompt_lab_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                dur = int(line.strip().split("\t")[-1])
                lab_item = "\t".join(line.strip().split("\t")[:-1])
                prompt_lab.append(lab_item)
                duration.append(dur)
        duration = np.array(duration) 

        generate_lab = []
        generate_duration = []
        with open(generate_lab_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                dur = int(line.strip().split("\t")[-1])
                lab_item = "\t".join(line.strip().split("\t")[:-1])
                generate_lab.append(lab_item)
                generate_duration.append(dur)
        generate_duration = np.array(generate_duration)

        sr, wav = read(prompt_wav_path)
        if len(wav.shape) == 2 and wav.shape[-1] == 2:
            wav = wav[:, 0]
        wav = wav / 32767.0
        if sr != self.mel_config["sampling_rate"]:
            wav = librosa.core.resample(wav, sr, self.mel_config["sampling_rate"])

        wav = torch.from_numpy(wav).float()
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        wav = wav.to(device)
        mel = mel_spectrogram(wav.unsqueeze(0), **self.mel_config).squeeze(0)
        mel = (mel - self.mel_norm_mean) / self.mel_norm_std

        assert abs(sum(duration) - mel.shape[1]) < 2, "{} {}".format(abs(sum(duration)), mel.shape)
        pad_len = sum(duration) - mel.shape[1]
        duration[-1] = duration[-1] - pad_len

        text_id, phones, tones, word_segs = self.phone2id.convert_tacolab_to_text_id_infer(prompt_lab)
        text_id = np.repeat(text_id, duration, axis=1)

        generate_text_id, _, _, _ = self.phone2id.convert_tacolab_to_text_id_infer(generate_lab)
        generate_text_id = np.repeat(generate_text_id, generate_duration, axis=1)

        text_id = np.concatenate((text_id, generate_text_id), axis=1)
        data_dict["phone"] = torch.from_numpy(text_id[0, :]).unsqueeze(0).to(device)
        data_dict["tone"] = torch.from_numpy(text_id[1, :]).unsqueeze(0).to(device)
        data_dict["word_seg"] = torch.from_numpy(text_id[2, :]).unsqueeze(0).to(device)
        data_dict['prompt_mel'] = mel.transpose(0, 1).unsqueeze(0).to(device) # [T, D]
        data_dict['mel_len'] = data_dict['prompt_mel'].shape[1]
        data_dict['uttid'] = uttid
        data_dict['wav'] = wav
        data_dict['mel_ctx'] = torch.cat((data_dict['prompt_mel'], self.mel_padding * torch.ones([1, generate_text_id.shape[1], self.mel_config["num_mels"]]).to(device)), dim=1)

        return data_dict



    
    # fake-zero-shot different text;
    def encode_zero_shot(self, sample):

        device = f"cuda:{self.trainer.local_rank}"
        data_dict = dict()
        uttid, text, prompt_lab_path, prompt_wav_path, generate_text = sample

        # get text_id
        if self.lang == 'en':
            generate_lab = generate_tacolabels_from_textstr_punc(generate_text, "English_v3_punc").decode()
        elif self.lang == 'cn':
            generate_lab = generate_tacolabels_from_textstr_punc(generate_text, "Chinese_v3_punc").decode()
        generate_lab = generate_lab.strip().split('\n')
        ret_tuple = self.phone2id.convert_tacolab_to_text_id_infer(generate_lab)
        if ret_tuple is None:
            return None
        generate_text_id, _, _, _ = ret_tuple
        generate_text_id = generate_text_id[:, 1:]

        if not os.path.exists(prompt_wav_path):
            return None

        dur_scale = 22050 / 256 / (self.mel_config["sampling_rate"] / self.mel_config["hop_size"])

        sr, wav = read(prompt_wav_path)
        prompt_lab = []
        duration = []
        with open(prompt_lab_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                dur = int(line.strip().split("\t")[-1])
                lab_item = "\t".join(line.strip().split("\t")[:-1])
                prompt_lab.append(lab_item)
                duration.append(dur)
        duration = np.array(duration)

        if len(wav.shape) == 2 and wav.shape[-1] == 2:
            wav = wav[:, 0]
        wav = wav / 32767.0
        if sr != self.mel_config["sampling_rate"]:
            wav = librosa.core.resample(wav, sr, self.mel_config["sampling_rate"])

        wav = torch.from_numpy(wav).float()
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        wav = wav.to(device)
        mel = mel_spectrogram(wav.unsqueeze(0), **self.mel_config).squeeze(0)
        mel = (mel - self.mel_norm_mean) / self.mel_norm_std


        ####################### for durator model ########################
        text_id, phones, tones, word_segs = self.phone2id.convert_tacolab_to_text_id_infer(prompt_lab)

        if self.durator_type == 'fastspeech':
            generate_duration, fs2_phon = self.fastspeech_duration(generate_text)
            # generate_duration = [5] + generate_duration
            # print(generate_duration)
        elif self.durator_type == 'durator2':
            durator_data_dict = {}
            durator_data_dict["phone"] = torch.from_numpy(np.concatenate([text_id[0, :],generate_text_id[0, :]], axis=0)).unsqueeze(0).to(device)
            durator_data_dict["tone"] = torch.from_numpy(np.concatenate([text_id[1, :],generate_text_id[1, :]], axis=0)).unsqueeze(0).to(device)
            durator_data_dict["word_seg"] = torch.from_numpy(np.concatenate([text_id[2, :],generate_text_id[2, :]], axis=0)).unsqueeze(0).to(device)
            durator_data_dict['phone_durations'] = torch.from_numpy(duration).unsqueeze(0).to(device)
            durator_data_dict['text_lens'] = torch.tensor([text_id.shape[1]+generate_text_id.shape[1]]).unsqueeze(0).to(device)
            durator_data_dict['dur_lens'] = torch.tensor([len(duration)]).unsqueeze(0).to(device)
            # 旧 generate_duration: array: shape of (t,),   
            # 重新生成 generate_duration 这样需要生成的 text 就使用了 pred duration
            generate_duration = self.durator.inference_from_text(durator_data_dict, None).cpu().numpy()
            # print('lab:', generate_lab)
            # print('dur: ',generate_duration)
        elif self.durator_type == 'clone':
            lab_path = os.path.join('/tmp', ''.join(random.choices(string.ascii_letters + string.digits, k=10))+'.lab')

            if self.lang == 'en':
                self.save_lab_for_babble_en(generate_lab, lab_path)
                ret_turple = self.tacolabel_encoding.enc_taco_label_v2_infer(lab_path)
                if ret_turple is None:
                    return None
                else:
                    phonemes, tones, word_categs, prosodies, sentence_types, label_length = ret_turple
                # print(f"phones id: ", phonemes)
                # print(f"tones id: ", tones)
                # print(f"label_length: ", label_length)

                phonemes = torch.LongTensor([phonemes]).to(self.device)
                tones = torch.LongTensor([tones]).to(self.device)
                word_categs = torch.LongTensor([word_categs]).to(self.device)
                prosodies = torch.LongTensor([prosodies]).to(self.device)
                sentence_types = torch.LongTensor([sentence_types]).to(self.device)
                label_length = torch.LongTensor([label_length]).to(self.device)

                spkid = torch.LongTensor([81]).to(self.device) # spkid固定的，不能修改

                
                # load EN model 
                generate_duration = self.en_dur_model(spkid, phonemes, tones, word_categs, prosodies, sentence_types, label_length) # 单位为帧数，每一帧12.5ms
                
            elif self.lang == 'cn':
                from recipes.voicebox.durator.clone_duration_infer.cn.babble.datasets.building.text.encoding import enc_taco_label_no_bytes
                self.save_lab_for_babble_cn(generate_lab, lab_path)
                taco_labels = enc_taco_label_no_bytes(lab_path, None, {'use_prsdword': False})
                if taco_labels is None:
                    return None

                phones = torch.stack([torch.from_numpy(taco_labels[0])]).to(self.device)
                tones = torch.stack([torch.from_numpy(taco_labels[1])]).to(self.device)
                word_categs = torch.stack([torch.from_numpy(taco_labels[2])]).to(self.device)
                prosodies = torch.stack([torch.from_numpy(taco_labels[3])]).to(self.device)
                label_length = torch.LongTensor([taco_labels[4]]).to(self.device)

                spkid = torch.LongTensor([0]).to(self.device) # spkid固定的，不能修改

                # load CN model 
                generate_duration = self.cn_dur_model(spkid, phones, tones, word_categs, prosodies, label_length)
        elif self.durator_type  == "vb_durator":
            durator_data_dict = {}
            durator_data_dict["phone"] = torch.from_numpy(np.concatenate([text_id[0, :],generate_text_id[0, :]], axis=0)).unsqueeze(0).to(device)
            durator_data_dict["tone"] = torch.from_numpy(np.concatenate([text_id[1, :],generate_text_id[1, :]], axis=0)).unsqueeze(0).to(device)
            durator_data_dict["word_seg"] = torch.from_numpy(np.concatenate([text_id[2, :],generate_text_id[2, :]], axis=0)).unsqueeze(0).to(device)
            durator_data_dict['prompt_phone_durations'] = torch.from_numpy(duration).unsqueeze(0).to(device)
            durator_data_dict['text_lens'] = torch.tensor([text_id.shape[1]+generate_text_id.shape[1]]).unsqueeze(0).to(device)
            durator_data_dict['dur_lens'] = torch.tensor([len(duration)]).unsqueeze(0).to(device)
            # 旧 generate_duration: array: shape of (t,),   
            # 重新生成 generate_duration 这样需要生成的 text 就使用了 pred duration
            generate_duration = self.durator.inference(durator_data_dict, dur_scale).cpu().numpy()
        elif self.durator_type == "zhengxi_durator":
            phones = torch.from_numpy(np.concatenate([text_id[0, :],generate_text_id[0, :]], axis=0)).unsqueeze(0).to(device)
            tones = torch.from_numpy(np.concatenate([text_id[1, :],generate_text_id[1, :]], axis=0)).unsqueeze(0).to(device)
            word_categs = torch.from_numpy(np.concatenate([text_id[2, :],generate_text_id[2, :]], axis=0)).unsqueeze(0).to(device)
            prompt_durations = torch.from_numpy(duration).unsqueeze(0).to(device)
            phones_lengths = torch.tensor([text_id.shape[1]+generate_text_id.shape[1]]).to(device)

            generate_duration = self.durator(phones, tones, word_categs, phones_lengths, prompt_durations)
            generate_duration = generate_duration.squeeze().cpu().numpy()
            generate_duration = np.round(generate_duration / dur_scale).astype(np.int32)
            generate_duration = np.clip(generate_duration, a_min=1, a_max=80)

        duration = np.round(duration / dur_scale).astype(np.int32)
        for i in range(len(duration)):
            if duration[i] <= 0:
                duration[i] = 1
        #assert abs(sum(duration) - mel.shape[1]) < 3, "{} {}".format(sum(duration), mel.shape)
        pad_len = sum(duration) - mel.shape[1]
        if pad_len < 0:
            mel = mel[:, :pad_len]
        elif pad_len > 0:
            c = pad_len
            for i in range(len(duration)):
                if duration[i] > 1:
                    duration[i] -= 1
                    c -= 1
                if c == 0:
                    break

        assert mel.shape[1] == sum(duration), "{} {}".format(mel.shape[1], sum(duration))

        if len(generate_text_id[0]) != len(generate_duration):
            print("skip {}. {} != {}".format(uttid, len(generate_text_id[0]), generate_duration.shape))
            return None

        text_id = np.repeat(text_id, duration, axis=1)
            
        # 变速：
        #prompt_dur_word = sum(duration) / len(duration)
        #target_dur_word = sum(generate_duration) / len(generate_duration)
        #generate_duration = [ int(x * min(1.25, max(0.75, prompt_dur_word / target_dur_word))) for x in generate_duration] 

        print(generate_duration)
        generate_text_id = np.repeat(generate_text_id, generate_duration, axis=1)

        text_id = np.concatenate((text_id, generate_text_id), axis=1)
        data_dict["phone"] = torch.from_numpy(text_id[0, :]).unsqueeze(0).to(device)
        data_dict["tone"] = torch.from_numpy(text_id[1, :]).unsqueeze(0).to(device)
        data_dict["word_seg"] = torch.from_numpy(text_id[2, :]).unsqueeze(0).to(device)
        #data_dict['mel'] = mel.transpose(0, 1).unsqueeze(0).to(device) # [T, D]
        #data_dict['mel_len'] = data_dict['mel'].shape[1]
        data_dict['prompt_mel'] = mel.transpose(0, 1).unsqueeze(0).to(device) # [T, D]
        data_dict['mel_len'] = data_dict['prompt_mel'].shape[1]
        data_dict['uttid'] = uttid
        data_dict['wav'] = wav
        #data_dict['mel_ctx'] = torch.cat((data_dict['mel'], torch.zeros([1, generate_text_id.shape[1], self.mel_config["num_mels"]]).to(device)), dim=1)
        data_dict['mel_ctx'] = torch.cat((data_dict['prompt_mel'], self.mel_padding * torch.ones([1, generate_text_id.shape[1], self.mel_config["num_mels"]]).to(device)), dim=1)

        return data_dict


    def save_lab_for_babble_en(self, lab, lab_path):
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write('phn\ttone\tws\tpwpp\tsentype\n')
            for i, line in enumerate(lab):
                if i != 0 and line.startswith('sil'):
                    continue
                f.write('\t'.join(line.split('\t')[:5])+'\n')


    def save_lab_for_babble_cn(self, lab, lab_path):
        with open(lab_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lab):
                if i != 0 and line.startswith('sil'):
                    continue
                line_list = line.split('\t')[:4]
                line_list = line_list[:2] + ['0.0 0.0 0.0 1.0'] + line_list[2:4]
                f.write('\t'.join(line_list)+'\n')


    def fastspeech_duration(self, text):

        ids = raw_texts = [text[:100]]
        speakers = np.array([1])
        if self.preprocess_config["preprocessing"]["text"]["language"] == "en":
            idx, phon = preprocess_english(text, self.preprocess_config)
        elif self.preprocess_config["preprocessing"]["text"]["language"] == "zh":
            idx, phon = preprocess_mandarin(text, self.preprocess_config)
            
        # phon_list = phon[1:-1].split()
        # for i in 

        texts = np.array([idx])

        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

        control_values = (1.0, 1.0, 1.0)

        duration = synthesize(self.dur_model, 900000, self.fs2_configs, None, batchs, control_values)

        return duration, phon
