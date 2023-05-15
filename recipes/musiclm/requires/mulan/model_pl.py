import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import ASTConfig, ASTModel, AutoProcessor, BertModel, BertTokenizer


# Text encoder
# class TextEncoder(pl.LightningModule):
class TextEncoder(nn.Module):
    def __init__(self, args):
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # model = BertModel.from_pretrained("bert-base-uncased")
        super(TextEncoder, self).__init__()

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir="./inference_test"
        )

        text_model = BertModel.from_pretrained(
            "bert-base-uncased", add_pooling_layer=False, cache_dir="./inference_test"
        )
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.text_model.gradient_checkpointing_enable()
        self.text_linear = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # inputs = tokenizer(x, return_tensors="pt")
        # outputs = self.text_model(**inputs)
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]

        text_output = self.text_linear(last_hidden_state[:, 0, :])

        text_output = F.normalize(text_output, p=2, dim=1)

        return text_output

    def encode(self, text_input):
        inputs = self.tokenizer(text_input, return_tensors="pt")

        device = self.text_linear.weight.device
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["token_type_ids"] = inputs["token_type_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

        outputs = self.text_model(**inputs)
        last_hidden_state = outputs["last_hidden_state"]
        # use the classification token [CLS] as the text embedding: batch_size * 768
        # text_output = self.text_linear(nn.Identity()(last_hidden_state[:,0,:]))
        text_output = self.text_linear(last_hidden_state[:, 0, :])
        text_output = F.normalize(text_output, p=2, dim=1)
        return text_output


# Music Encoder


# class MusicEncoder(pl.LightningModule):
class MusicEncoder(nn.Module):
    def __init__(self, args):
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # model = BertModel.from_pretrained("bert-base-uncased")
        super(MusicEncoder, self).__init__()
        processor = AutoProcessor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", cache_dir="./inference_test"
        )

        music_model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", cache_dir="./inference_test"
        )

        TARGET_SAMPLE_RATE = 24000
        self.feat_extract = {  # Use a dict to avoid auto convert fp16
            "mel": torchaudio.transforms.MelSpectrogram(
                sample_rate=TARGET_SAMPLE_RATE,
                n_fft=1024,
                # 400 (25ms) in AST with 16K.
                # 42.6ms if using 1024 in 24K.
                # 46.43ms in music tagging (1024, 22050Hz)
                hop_length=240,
                # 10ms in AST with 16K.
                # 21.3ms if using 512 in 24K.
                # 23.2ms in music tagging task (512, 22050Hz)
                f_min=0,
                f_max=TARGET_SAMPLE_RATE // 2,
                n_mels=128,
                window_fn=torch.hann_window,
                power=2.0,
                center=True,
                pad_mode="reflect",
            ),
            "db": torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
        }
        self.processor = processor
        self.music_model = music_model
        self.music_model.gradient_checkpointing_enable()
        self.music_linear = nn.Linear(768, 128)
        if args.spec_aug:
            self.tm = torchaudio.transforms.TimeMasking(time_mask_param=192)
            self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=48)

    def forward(self, x, spec_aug=False):
        with torch.autocast(device_type="cuda", enabled=False):
            mel_spec = self.feat_extract["mel"](x.float())
            mel_spec = self.feat_extract["db"](mel_spec)
            if spec_aug:
                mel_spec = self.fm(self.tm(mel_spec))

        mel_spec = mel_spec.to(x.dtype)
        mel_spec = F.pad(mel_spec, ((0, 1024 - mel_spec.shape[-1])))
        mel_spec = mel_spec.permute(0, 2, 1)
        # print(mel_spec.shape)
        outputs = self.music_model(input_values=mel_spec)
        last_hidden_state = outputs["last_hidden_state"]
        # use the classification token [CLS] as the text embedding: batch_size * 768
        # music_output = self.music_linear(nn.Identity()(last_hidden_state[:,0,:]))
        music_output = self.music_linear(last_hidden_state[:, 0, :])
        music_output = F.normalize(music_output, p=2, dim=1)
        return music_output

    def encode(self, x):
        inputs = self.processor(x, return_tensors="pt")
        outputs = self.music_model(**inputs)
        last_hidden_state = outputs["last_hidden_state"]
        music_output = self.music_linear(
            nn.Identity()(last_hidden_state[:, 0, :])
        )  # use the classification token [CLS] as the text embedding: batch_size * 768
        music_output = F.normalize(music_output, p=2, dim=1)
        return music_output


class MusicEncoderShort(nn.Module):
    def __init__(self, args):
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # model = BertModel.from_pretrained("bert-base-uncased")
        super(MusicEncoderShort, self).__init__()
        configuration = ASTConfig()

        processor = AutoProcessor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        configuration.max_length = 500

        music_model = ASTModel(configuration)

        self.processor = processor
        self.music_model = music_model
        self.music_linear = nn.Linear(768, 128)

    def forward(self, x):
        outputs = self.music_model(input_values=x)
        last_hidden_state = outputs["last_hidden_state"]
        music_output = self.music_linear(
            last_hidden_state[:, 0, :]
        )  # use the classification token [CLS] as the text embedding: batch_size * 768
        music_output = F.normalize(music_output, p=2, dim=1)
        return music_output


# Multimodal Encoder
# class TextMusicEncoder(pl.LightningModule):
class TextMusicEncoder(nn.Module):
    def __init__(self, args):
        super(TextMusicEncoder, self).__init__()
        self.text_encoder = TextEncoder(args)
        self.music_encoder = MusicEncoder(args)

        self.args = args

    def forward(self, input_ids, attention_mask, token_type_ids, music, spec_aug=False):
        text_output = self.text_encoder(input_ids, attention_mask, token_type_ids)
        music_output = self.music_encoder(music, spec_aug)

        return text_output, music_output
