import os, glob, io
import pandas as pd
import json
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from torch import Tensor

from recipes.musiclm.requires.model_initializer import init_mulan
from recipes.musiclm.inference.utils import load_wav, tensor_resample
from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds, get_mulan_embeds_2
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length
from .psfad_utils import get_embedding_similarity, get_fad_from_embeddings, extend_with_moving_avg_pool


CACHE_DIR = '/opt/tiger/samantha/.module_cache/mulan'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_rate = 24000
mulan_min_duration = 10 * sample_rate

CONFIG = {
    'mulan61': {
        'version': 'sstkmae_v3',
        'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v9/mulan/mulan-step=005000-median_rank_1=61-kaggle-minimal.ckpt',
        'description': 'trained on sstk only'
    },
    'mulan110': {
        'version': 'g4',
        'hpath': 'hdfs:///home/byte_speech_sv/bigmusic/models/mulan/mulan-step=024600-median_rank_1=110-kaggle-minimal.ckpt',
        'description': '',
    },
    'mulan131': {
        'version': 'sstkmae_v3',
        'hpath': '/mnt/bn/music-llm-nas-lq/bochen/logs/mix_mulan/1108_train_mix_mulan_mix_moresstk_8w/checkpoints/mulan-step=001500-median_rank_0=131-kaggle.ckpt',
        'description': 'trained on inst, en vocal, cn vocal'
    },
    'mulan89': {
        'version': 'sstkmae_v3',
        'hpath': '/opt/tiger/samantha/mulan-step=002500-median_rank_1=89-kaggle.ckpt',
        # 'hpath': '/mnt/bn/music-llm-nas-lq/bochen/logs/mix_mulan/20250113_mix_mulan_4/checkpoints/mulan-step=002500-median_rank_1=89-kaggle.ckpt',
        'description': 'trained on inst, en vocal, cn vocal, more data, better training tricks (Bochen 20251113)'
    },
    'mulan95': {
        'version': 'sstkmae_v3',
        'hpath': '/mnt/bn/music-llm-nas-lq/bochen/logs/mix_mulan/20250411_mix_mulan_5/checkpoints/mulan-step=001500-median_rank_1=95-kaggle.ckpt',
        'description': 'trained on inst, en vocal, cn vocal, more data, also includes 300k model inference samples (Bochen 20250412)'
    },
}
VOCAB_CONFIG = {
    'musiccaps': [
        {
            'subset_name': 'default',
            'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/assets/text_vocab/text_vocab_musiccaps.txt',
            'top_k': 30,
        },
    ],
    'inhouse0326': [
        {
            'subset_name': 'genre',
            'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/assets/text_vocab/inhouse0326_genre.txt',
            'top_k': 10,
        },
        {
            'subset_name': 'instrument',
            'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/assets/text_vocab/inhouse0326_instrument.txt',
            'top_k': 10,
        },
        {
            'subset_name': 'sound_effect',
            'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/assets/text_vocab/inhouse0326_sound_effect.txt',
            'top_k': 5,
        },
        {
            'subset_name': 'music_for',
            'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/assets/text_vocab/inhouse0326_music_for.txt',
            'top_k': 10,
        },
        {
            'subset_name': 'descriptor',
            'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/assets/text_vocab/inhouse0326_descriptor.txt',
            'top_k': 10,
        },
    ],
}


def _load_audio_tensor(audio_path):
    # wav_tensor is already resampled to sample_rate via librosa
    wav_tensor = torch.tensor(
        load_wav(str(audio_path), sr=sample_rate, mono=True)
    ).to(device)
    if wav_tensor.shape[-1] < mulan_min_duration:
        wav_tensor = crop_pad_to_seq_length(wav_tensor, mulan_min_duration)
    return wav_tensor.unsqueeze(0)


def _load_audio_tensor_from_bytes(wav_bytes):
    wav_tensor, sr = torchaudio.load(io.BytesIO(wav_bytes))

    assert len(wav_tensor.shape) == 2
    if wav_tensor.shape[0] > 1:  # Convert to mono
        wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        # Now, wav_tensor should have dimension [1, seq_len]

    if sr != sample_rate:  # bochen: note that mulan only works on 24k audio
        wav_tensor = tensor_resample(
            wav_tensor, orig_freq=sr, new_freq=24000
        )

    if wav_tensor.shape[-1] < mulan_min_duration:  # Pad to at least 10 seconds
        wav_tensor = crop_pad_to_seq_length(wav_tensor, mulan_min_duration)

    return wav_tensor.unsqueeze(0)


class MulanUtil:

    def __init__(self, model_name='mulan89'):
        self.load_model(model_name)
        self.text_vocab = []
        self.audio_pool_emb = []

    def load_model(self, model_name):
        if model_name not in CONFIG.keys():
            raise Exception('Invalid model_name. (available model_name: %s)' % CONFIG.keys())
        version = CONFIG[model_name]['version']
        hpath = CONFIG[model_name]['hpath']
        local_rank = 0
        requires = init_mulan(hpath, local_rank, cache_dir=CACHE_DIR, version=version)
        self.requires = requires

    def get_text_emb(self, text: List[str]) -> Tensor:
        return get_mulan_embeds(self.requires, x=text, data_type='text').to(device)

    def get_text_emb_2(self, text: List[str]) -> Tensor:
        return get_mulan_embeds_2(self.requires, x=text, data_type='text').to(device)

    def get_audio_emb(self, wav: Tensor) -> Tensor:
        return get_mulan_embeds(self.requires, x=wav, data_type='music').to(device)

    def get_audio_emb_2(
        self,
        wav: Union[Tensor, List[Tensor], None] = None,
        shift_seconds: float = 5.
    ) -> List[Tensor]:
        embds = get_mulan_embeds_2(
            self.requires, x=wav, data_type='music',
            average_audio_embd=False,
            normalize_audio_embd=False,
            shift_seconds=shift_seconds,
        )
        return [embd.to(device) for embd in embds]

    def get_audio_file_emb(self, filename_audio):
        wav = _load_audio_tensor(filename_audio)
        emb = self.get_audio_emb(wav)
        return emb

    def get_mcs(self, text, filename_audio=None, wav_bytes=None):
        if wav_bytes:
            wav = _load_audio_tensor_from_bytes(wav_bytes)
        elif filename_audio:
            wav = _load_audio_tensor(filename_audio)
        else:
            raise Exception('Invalid input. (filename_audio or wav_byte should be provided)')

        while len(wav.shape) > 2:
            wav = wav.squeeze(0)
        wav = wav.to(device)

        audio_emb = self.get_audio_emb(wav)
        text_emb = self.get_text_emb(text)

        mcs = F.cosine_similarity(text_emb, audio_emb)
        return mcs.item()

    def get_scores_from_tensor(
        self,
        wav: List[Tensor],  # 24k mono with shape [t]
        style_text: Optional[List[str]] = None,  # If given, compute MuLan similarity
        freeform_text: Optional[List[str]] = None,  # If given, compute MuLan similarity
        ref_stats: Optional[Dict] = None,  # If given, compute per-song FAD
        ref_stats_pool_ext: Optional[Dict] = None,
        shift_seconds: float = 5.,
        return_embd: bool = False,
    ) -> Dict:
        """
        This function computes the MuLan similarity and/or MuLan Per-Song FAD.
        If text is given, compute MuLan similarity.
        If ref_stats is given, compute per-song FAD (if both are given, compute both).

        Audio batching is different from get_mcs(), using mulan_inference_2 which takes audio as a
        tensor list, allowing different pieces to have different durations without excessive padding.
        After spliting each tensor into 10-second chunks, they are then collated for efficiency.

        Args:
            wav (List[Tensor]):
                List of audio tensors with length b, each with shape [t].
            style_text (Optional[List[str]], optional):
                List of style text strings with length b. Defaults to None.
            freeform_text (Optional[List[str]], optional):
                List of freeform text strings with length b. Defaults to None.
            ref_stats (Optional[Dict], optional):
                A dictionary of reference statistics, where each key is a reference name,
                and the value is a dictionary with keys 'mean', 'cov', and 'cov_sqrt'.
                Defaults to None (skip PSFAD computation).
            ref_stats_pool_ext (Optional[Dict], optional):
                A dictionary of reference statistics for pooled embeddings, following the same format as ref_stats.
                Defaults to None (skip PoolExtPSFAD computation).
            shift_seconds (float, optional):
                Shift duration in seconds. Defaults to 5.
            fad_match_bottom_right (bool, optional):
                Whether to match the bottom right of the covariance matrices in FAD computation.
                This is useful for embeddings with appended time encoding. Defaults to False.
            return_embd (bool, optional):
                Whether to return the computed embeddings. Defaults to False.

        Returns:
            Dict:
                A dictionary containing the computed scores.
        """
        assert isinstance(wav, list) and len(wav) > 0
        for i, w in enumerate(wav):
            assert isinstance(w, Tensor), f"expected tensor, but got {type(w)} at index {i}"
            assert w.dim() == 1, f"expected 1-d tensor, but got {w.shape} at index {i}"
        wav = [w.to(device) for w in wav]

        # Get MuLan embeddings
        audio_embd = self.get_audio_emb_2(wav=wav, shift_seconds=shift_seconds)
        embd_dict = {"Audio": audio_embd}
        out_dict = {}

        # Compute MuLan similarity score from embeddings
        if style_text is not None:
            assert isinstance(style_text, list) and len(style_text) == len(wav)
            style_text_embd = self.get_text_emb_2(style_text).unsqueeze(1)  # Shape [b, 1, d]
            embd_dict["Style_Text"] = style_text_embd
            style_mcs_dict = get_embedding_similarity(audio_embd=audio_embd, text_embd=style_text_embd)
            out_dict["Style_MCS"] = style_mcs_dict

        if freeform_text is not None:
            assert isinstance(freeform_text, list) and len(freeform_text) == len(wav)
            freeform_text_embd = self.get_text_emb_2(freeform_text).unsqueeze(1)  # Shape [b, 1, d]
            embd_dict["Freeform_Text"] = freeform_text_embd
            freeform_mcs_dict = get_embedding_similarity(audio_embd=audio_embd, text_embd=freeform_text_embd)
            out_dict["Freeform_MCS"] = freeform_mcs_dict

        # Compute MuLan per-song FAD from embeddings
        if ref_stats is not None:
            psfad_dict = get_fad_from_embeddings(audio_embd=audio_embd, ref_stats=ref_stats, match_bottom_right=False)
            out_dict["PSFAD"] = psfad_dict

        # Compute MuLan pool-extended per-song FAD from embeddings
        if ref_stats_pool_ext is not None:
            audio_embd_pool_ext = [extend_with_moving_avg_pool(embd=ae, window_len=5) for ae in audio_embd]
            psfad_dict_pool_ext = get_fad_from_embeddings(
                audio_embd=audio_embd_pool_ext, ref_stats=ref_stats_pool_ext, match_bottom_right=False
            )
            out_dict["PSFAD_PoolExt"] = psfad_dict_pool_ext

        if return_embd:
            out_dict["Embeddings"] = embd_dict

        return out_dict

    def preprocess_audio_pool(self, path_audio, path_emb):
        pass
        filenames = glob.glob(os.path.join(path_audio, '*.wav'))
        filenames.sort()
        for filename in tqdm(filenames):
            emb = self.get_audio_file_emb(filename).to(device)
            filename_emb = os.path.join(path_emb, os.path.basename(filename).replace('.wav', '.pt'))
            if os.path.exists(filename_emb):
                continue
            torch.save(emb, filename_emb)

    def preprocess_text_vocab(self, vocab_name):

        # prepare the output path
        path_text_vocab = os.path.join(CACHE_DIR, 'text_vocab')
        os.makedirs(path_text_vocab, exist_ok=True)
        self.text_vocab = []
        for item in VOCAB_CONFIG[vocab_name]:
            subset_name = item['subset_name']
            hpath = item['hpath']
            top_k = item['top_k']
            print ('----- calculate embedding for text_vocab [%s], subset [%s]' % (vocab_name, subset_name))
            filename_vocab = os.path.join(path_text_vocab, os.path.basename(hpath))

            # load the text vocab 
            if not os.path.exists(filename_vocab):
                command = 'hdfs dfs -get %s %s' % (hpath, path_text_vocab)
                os.system(command)
                print ('text vocab has been downloaded at: %s' % filename_vocab)
            keywords = [x.strip() for x in open(filename_vocab, 'r').readlines()]

            # calculate embeddings 
            filename_emb = os.path.join(path_text_vocab, os.path.basename(hpath).replace('.txt', '') + '.pt')
            if os.path.exists(filename_emb):    # check if the embeddings are already calculated
                embeddings = torch.load(filename_emb)
                if len(embeddings) == len(keywords):
                    print ('Skip preprocess_text_vocab (text embeddings for %d keywords have been already calculated and saved at: %s.)' % (len(keywords), filename_emb))

            else:
                embeddings = []
                for _, text in tqdm(enumerate(keywords), desc='calculating text embeddings for %d keywords from [%s] subset...' % (len(keywords), subset_name)):
                    emb = self.get_text_emb(text).to(device)
                    embeddings.append({'text': text, 'emb': emb})
                torch.save(embeddings, filename_emb)
                print ('Text embeddings for %d keywords have been saved at: %s' % (len(keywords), filename_emb))
            item['embeddings'] = embeddings
            self.text_vocab.append(item)


    def retrieve_text_for_audio(self, filename_audio=None, wav_bytes=None):

        if not self.text_vocab:
            raise Exception('No text_vocab_emb found. Please call preprocess_text_vocab() first to pre-calculate text embeddings.')

        if wav_bytes:
            wav = _load_audio_tensor_from_bytes(wav_bytes)
        elif filename_audio:
            wav = _load_audio_tensor(filename_audio)
        else:
            raise Exception('Invalid input. (filename_audio or wav_byte should be provided)')

        audio_emb = self.get_audio_emb(wav)
        mcs_metrics = []

        for item in self.text_vocab:
            subset_name = item['subset_name']
            embeddings = item['embeddings']
            top_k = item['top_k']

            mcs_metrics_subset = []
            for embedding in embeddings:
                text, emb = embedding['text'], embedding['emb']
                mcs = torch.nn.functional.cosine_similarity(emb, audio_emb).cpu().item()
                mcs_metrics_subset.append({
                    'text': text,
                    'mcs': mcs
                })
            mcs_metrics_subset.sort(key=lambda x: x['mcs'], reverse=True)
            mcs_metrics_subset = mcs_metrics_subset[:top_k]
            mcs_metrics_subset = {x['text']: round(x['mcs'], 4) for x in mcs_metrics_subset}
            mcs_metrics.append({
                subset_name: mcs_metrics_subset
            })
        return mcs_metrics

    def retrieve_text_for_audio_batch(self, filename_links, path_audio=None):

        if not self.text_vocab:
            raise Exception('No text_vocab_emb found. Please call preprocess_text_vocab() first to pre-calculate text embeddings.')
        path_cache_audio = os.path.join(CACHE_DIR, 'audio')
        os.makedirs(path_cache_audio, exist_ok=True)
        if not path_audio:
            path_audio = os.path.join(path_cache_audio, os.path.splitext(os.path.basename(filename_links))[0])
        os.makedirs(path_audio, exist_ok=True)

        df = pd.read_csv(filename_links)
        filename_result = os.path.join(os.path.splitext(filename_links)[0] + '_result.csv')
        df['mulan_0326'] = ''
        for idx, row in tqdm( df.iterrows() ):
            index = str(row['index'])
            print (index)
            url = row['audio_url']
            filename = os.path.join(path_audio, index+'.wav')
            if not os.path.exists(filename):
                command = "curl -o %s '%s'" % (filename, url)
                print (command)
                os.system(command)

            result = self.retrieve_text_for_audio(filename)
            df.at[idx, 'mulan_0326'] = json.dumps(result, indent=4, ensure_ascii=False)
            df.to_csv(filename_result, index=False)

        df.to_csv(filename_result, index=False)


if __name__ == '__main__':
    mu = MulanUtil('mulan95')
    # emb = mu.get_text_emb('a happy song')
    # print (emb)
    # print (emb.shape)

    # mcs = mu.get_mcs('/mnt/bn/music-llm-nas-lq/bochen/results/20250325-114146_freeform_95k/default/s0062.generated.wav', 'aaaaaaaaaa')
    # print (mcs)

    mu.preprocess_text_vocab(
        vocab_name='inhouse0326',
    )

    # result = mu.retrieve_text_for_audio(filename_audio='/opt/tiger/samantha/.module_cache/mulan/audio/generated_samples/s0001.wav')
    # print ('--- result')
    # print(json.dumps(result, indent=4, ensure_ascii=False))

    mu.retrieve_text_for_audio_batch(
        filename_links = '/opt/tiger/samantha/.module_cache/mulan/audio/8302.chinese.csv',
    )
