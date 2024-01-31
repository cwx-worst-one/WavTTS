import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm
from recipes.musiclm.inference.utils import save_wav
from samantha.utils.hparams import DotDict


class MusicDatabaseModule(pl.LightningModule):
    def __init__(self, required_modules, extra_params=None):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.requires = {}

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

    def predict_step(self, batch, batch_idx):
        audio = batch["target_audio"]
        with torch.no_grad():
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"],
                music=audio.float(),
                device=audio.device,
            )
        return mulan_embeds, audio

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        mulan_embeds, audio = outputs
        if not os.path.exists(self.extra_params.output_dir):
            os.makedirs(self.extra_params.output_dir, exist_ok=True)
        fname = f"{self.global_rank}_{dataloader_idx}_{batch_idx}"
        audio = (audio.cpu().float().numpy() * 32768.0).astype("int16")
        mulan_embeds = mulan_embeds.cpu().float().numpy()
        np.save(os.path.join(self.extra_params.output_dir, f"{fname}.audio.npy"), audio)
        np.save(os.path.join(self.extra_params.output_dir, f"{fname}.mulan_embeds.npy"), mulan_embeds)
        del audio
        del mulan_embeds


class RetrievalModule(pl.LightningModule):
    def __init__(self, required_modules, extra_params=None):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.requires = {}

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
        if "duration" in batch:
            max_len = batch["duration"] * self.extra_params.sample_rate
        else:
            max_len = self.extra_params.duration * self.extra_params.sample_rate
        for i, wav in enumerate(retrieved_audio):
            if categories is not None and categories[i]:
                wav_dir = os.path.join(self.extra_params.output_dir, categories[i])
            else:
                wav_dir = self.extra_params.output_dir
            os.makedirs(wav_dir, exist_ok=True)
            fname = index[i]

            wav_fp = os.path.join(wav_dir, f"{fname}.generated.wav")
            print(f"[Saving] {wav_fp}")
            wav = wav[..., :max_len]
            save_wav(
                wav.cpu().float(),
                wav_fp,
                sr=self.extra_params.sample_rate,
                save_mp3=self.extra_params.save_mp3,
            )
            if self.extra_params.save_style_audio and style_audio is not None:
                style_wav_fp = os.path.join(wav_dir, f"{fname}.style_audio.wav")
                save_wav(
                    style_audio[i].cpu().float(),
                    style_wav_fp,
                    sr=self.extra_params.sample_rate,
                    save_mp3=self.extra_params.save_mp3,
                )

            meta_fp = os.path.join(wav_dir, f"{fname}.metadata.json")
            metadata = {"retrieval_score": retrieved_scores[i]}
            print(f"Saving metadata: {metadata}")
            with open(meta_fp, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)