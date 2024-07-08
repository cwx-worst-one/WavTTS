import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm
from recipes.musiclm.inference.utils import save_wav
from samantha.utils.hparams import DotDict
from torchaudio.transforms import Resample

from recipes.musiclm.inference.utils import save_wav, load_wav
from recipes.bigmusic.utils.upload import audio_tensor_to_bytes, upload_to_easycycle


class MusicDatabaseModule(pl.LightningModule):
    def __init__(self, required_modules, extra_params=None):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        if self.extra_params.sample_rate != 24000:
            self.resample = Resample(self.extra_params.sample_rate, 24000)
        else:
            self.resample = None

    def setup(self, stage):
        if not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, item in self.hparams.required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        audio = batch["target_audio"]
        if self.resample:
            audio_resample = self.resample(audio).to(audio.device)
        audio_resample = audio_resample[..., :24000 * 10]
        with torch.no_grad():
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"],
                music=audio_resample.float(),
                device=audio_resample.device,
            )
        if not os.path.exists(self.extra_params.output_dir):
            os.makedirs(self.extra_params.output_dir, exist_ok=True)
        fname = f"{self.global_rank}_{dataloader_idx}_{batch_idx}"
        mulan_embeds = mulan_embeds.cpu().float().numpy()
        if self.extra_params.save_text:
            style_category = batch['style_text']
            with open(os.path.join(self.extra_params.output_dir, f"{fname}.style_text.json"), 'w') as f:
                json.dump(style_category, f)
        if self.extra_params.save_audio:
            audio = (audio.cpu().float().numpy() * 32768.0).astype("int16")
            np.save(os.path.join(self.extra_params.output_dir, f"{fname}.audio.npy"), audio)
        np.save(os.path.join(self.extra_params.output_dir, f"{fname}.mulan_embeds.npy"), mulan_embeds)
        del audio
        del mulan_embeds
        # return mulan_embeds, audio # do not save outputs to prevent OOM

class RetrievalModule(pl.LightningModule):
    def __init__(self, required_modules, extra_params=None):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.metadatas = []

    def infer_batch_size(self, batch):
        batch_size = [len(t) for t in batch.values() if torch.is_tensor(t) or isinstance(t, list)][0]
        return batch_size

    def infer_conditions(self, batch):
        if type(batch["conditions"]) == list:
            assert (
                len(set(list(map(tuple, batch["conditions"])))) == 1
            ), "Make sure that all conditions in the batch are the same"
            conditions = batch['conditions'][0].split(',')
        else:
            conditions = batch['conditions'].split(',')
        return conditions

    def setup(self, stage):
        if not self.requires:
            self.load_required_modules()
        self.db = self.load_retrieval_db()
        print(f"Retrieval db size: {len(self.db)}")

    def load_required_modules(self):
        for name, item in self.hparams.required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_retrieval_db(self):
        db = []
        for fname in os.listdir(self.extra_params.db_dir):
            if not fname.endswith(".mulan_embeds.npy"):
                continue
            mulan_embeds = os.path.join(self.extra_params.db_dir, fname)
            audio = os.path.join(
                self.extra_params.db_dir,
                fname.replace(".mulan_embeds.npy", ".audio.npy"),
            )
            if not os.path.exists(audio):
                print(f"File not found: {audio}, skipping")
                continue
            db.append((mulan_embeds, audio))
        return db

    @torch.no_grad()
    def get_mulan_embeds(self, batch):
        conditions = self.infer_conditions(batch)
        if "style_text" in conditions:
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"],
                text=batch["style_text"],
                device=self.device,
            )
        elif "style_audio" in conditions:
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"],
                music=batch["style_audio"].to(self.device).float(),
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid conditions: {conditions}")
        return mulan_embeds

    def load_audio(self, audio_fnames):
        retrieved_audio = []
        for audio_fname, index in audio_fnames:
            audio = np.load(audio_fname)
            audio = audio[index]
            if audio.dtype == np.int16:
                audio = audio / 32768.0
            elif audio.dtype == np.int32:
                audio = audio / 2_147_483_648.0
            audio = torch.from_numpy(audio.astype(np.float32))
            retrieved_audio.append(audio)
        return retrieved_audio

    def do_retrieval(self, query_mulan_embeds):
        batch_size = len(query_mulan_embeds)
        retrieved_scores = [None for _ in range(batch_size)]
        retrieved_audio = [None for _ in range(batch_size)]
        pbar = tqdm(range(len(self.db)))
        for i in pbar:
            pbar.set_description(f"retrieval [0 - {len(self.db)}]")
            mulan_embeds_fname, audio_fname = self.db[i]
            mulan_embeds = np.load(mulan_embeds_fname)
            mulan_embeds = torch.from_numpy(mulan_embeds).float().to(self.device)
            chunk_size = len(mulan_embeds)
            # (batch_size, chunk_size)
            sim = F.cosine_similarity(
                # (batch_size, D) -> (batch_size * chunk_size, D)
                query_mulan_embeds.reshape(batch_size, 1, -1).expand(-1, chunk_size, -1).reshape(batch_size * chunk_size, -1),
                # (chunk_size, D) -> (batch_size * chunk_size, D)
                mulan_embeds.reshape(1, chunk_size, -1).expand(batch_size, -1, -1).reshape(batch_size * chunk_size, -1),
            ).reshape(batch_size, chunk_size)
            values, indices = sim.max(dim=-1)
            values = values.cpu().tolist()
            indices = indices.cpu().tolist()
            for j in range(batch_size):
                if retrieved_scores[j] is None or retrieved_scores[j] < values[j]:
                    retrieved_scores[j] = values[j]
                    retrieved_audio[j] = (audio_fname, indices[j])
        retrieved_audio = self.load_audio(retrieved_audio)
        return retrieved_scores, retrieved_audio

    def predict_step(self, batch, batch_idx):
        query_mulan_embeds = self.get_mulan_embeds(batch)
        retrieved_scores, retrieved_audio = self.do_retrieval(query_mulan_embeds)
        return retrieved_scores, retrieved_audio

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        retrieved_scores, retrieved_audio = outputs
        index = batch["index"]
        categories = batch.get('category')
        style_audio = batch.get('style_audio')
        sample_rate = self.extra_params.sample_rate
        if "duration" in batch:
            max_len = batch["duration"] * sample_rate
        else:
            max_len = self.extra_params.duration * sample_rate
        for i, wav in enumerate(retrieved_audio):
            fname = index[i]
            wav_file_name = f"{fname}.generated.wav"
            metadata = {"retrieval_score": retrieved_scores[i]}
            metadata['file_name'] = fname
            metadata['index'] = i
            if categories is not None and categories[i]:
                wav_dir = os.path.join(self.extra_params.output_dir, categories[i])
                metadata['category'] = categories[i]
            else:
                wav_dir = self.extra_params.output_dir
            os.makedirs(wav_dir, exist_ok=True)
            wav_fp = os.path.join(wav_dir, wav_file_name)
            print(f"[Saving] {wav_fp}")
            wav = wav[..., :max_len]
            output_wav_fp =save_wav(
                wav.cpu().float(),
                wav_fp,
                sr=sample_rate,
                save_mp3=self.extra_params.save_mp3,
                normalize_volume=True
            )

            save_mode = "upload"
            if save_mode == "upload":
                saved_wav = torch.from_numpy(load_wav(output_wav_fp, sr=sample_rate))
                audio_bytes = audio_tensor_to_bytes(saved_wav, sample_rate)
                metadata["audio_url"] = upload_to_easycycle(audio_bytes, wav_file_name)

            if self.extra_params.save_style_audio and style_audio is not None:
                style_wav_fp = os.path.join(wav_dir, f"{fname}.style_audio.wav")
                save_wav(
                    style_audio[i].cpu().float(),
                    style_wav_fp,
                    sr=self.extra_params.sample_rate,
                    save_mp3=self.extra_params.save_mp3,
                    normalize_volume=True
                )

            meta_fp = os.path.join(wav_dir, f"{fname}.metadata.json")
            print(f"Saving metadata: {metadata}")
            self.metadatas.append(metadata)
            with open(meta_fp, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                

    def on_predict_end(self) -> None:
        index_fname = os.path.join(self.extra_params.output_dir, "index.csv")
        with open(index_fname, "w", encoding='utf-8') as fw:
            fw.write("file_name,beam_id,audio_url\n")
            for metadata in self.metadatas:
                file_name = metadata["file_name"]
                beam_id = metadata["index"]
                audio_url = metadata["audio_url"]
                fw.write(f"{file_name},{beam_id},{audio_url}\n")