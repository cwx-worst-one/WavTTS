import json
import os
from typing import Any, Dict, List, Union

import fsspec
import numpy as np
import torch
from torch import nn
from coqpit import Coqpit
import yaml
from recipes.voicebox.modules.speaker_encoder.utils import load_config
from recipes.voicebox.modules.speaker_encoder.generic_utils import setup_encoder_model
from recipes.voicebox.modules.speaker_encoder.processor import AudioProcessor
from recipes.voicebox.modules.speaker_encoder.model.resnet import ResNetSpeakerEncoder



class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        config_path,
        model_path=None,
        use_cuda=True
    ):
        super(SpeakerEncoder, self).__init__()
        self.use_cuda = use_cuda
        self.encoder_config = load_config(config_path)
        # with fsspec.open(config_path, "r", encoding="utf-8") as f:
        #     self.encoder_config = json.load(f)
        self.encoder = setup_encoder_model(self.encoder_config)
        # self.encoder = ResNetSpeakerEncoder(
        #     input_dim=64, use_torch_spec=True, log_input=True, proj_dim=512)
        if model_path is not None:
            self.encoder.load_checkpoint(
                self.encoder_config, model_path, eval=True, use_cuda=use_cuda, cache=True
            )
        self.encoder_ap = AudioProcessor(**self.encoder_config.audio)


    def compute_embedding_from_clip(self, wav_file):
        waveform = self.encoder_ap.load_wav(wav_file, sr=self.encoder_ap.sample_rate)
        print(waveform)
        if not self.encoder_config.model_params.get("use_torch_spec", False):
            m_input = self.encoder_ap.melspectrogram(waveform)
            m_input = torch.from_numpy(m_input)
        else:
            print('use_torch_spec')
            m_input = torch.from_numpy(waveform)

        if self.use_cuda:
            m_input = m_input.cuda()
        m_input = m_input.unsqueeze(0)
        embedding = self.encoder.compute_embedding(m_input)
        return embedding


if __name__ == "__main__":
    se = SpeakerEncoder(
            model_path='/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/modules/speaker_encoder/model_se.pth.tar',
            config_path='/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/modules/speaker_encoder/config_se.json'
        )

    wavpath1 = '/mnt/bn/cyz-lq-nas/project/bigtts_testset/icl_testset_fighting/prompt_ZH_better_studio/ICL0930_luyou_prompt_01.wav'
    wavpath2 = '/mnt/bn/cyz-lq-nas/project/bigtts_testset/icl_testset_fighting/prompt_ZH_better_studio/ICL0930_luyou_prompt_02.wav'
    wavpath3 = '/mnt/bn/cyz-lq-nas/project/bigtts_testset/icl_testset_fighting/prompt_ZH_better_studio/ICL0930_周紫瑶60s_01.wav'
    wavpath4 = '/mnt/bn/cyz-lq-nas/project/bigtts_testset/icl_testset_fighting/prompt_ZH_better_studio/SAMI_hyy_ch_01.wav'

    from torch.nn.functional import cosine_similarity
    emb1 = se.compute_embedding_from_clip(wavpath1)
    emb2 = se.compute_embedding_from_clip(wavpath2)
    emb3 = se.compute_embedding_from_clip(wavpath3)
    emb4 = se.compute_embedding_from_clip(wavpath4)

    print(cosine_similarity(emb1, emb2))
    print(cosine_similarity(emb1, emb3))
    print(cosine_similarity(emb2, emb3))
    print(cosine_similarity(emb1, emb4))
    print(cosine_similarity(emb3, emb4))





# class EmbeddingManager(BaseIDManager):
#     """Base `Embedding` Manager class. Every new `Embedding` manager must inherit this.
#     It defines common `Embedding` manager specific functions.

#     It expects embeddings files in the following format:

#     ::

#         {
#             'audio_file_key':{
#                 'name': 'category_name',
#                 'embedding'[<embedding_values>]
#             },
#             ...
#         }

#     `audio_file_key` is a unique key to the audio file in the dataset. It can be the path to the file or any other unique key.
#     `embedding` is the embedding vector of the audio file.
#     `name` can be name of the speaker of the audio file.
#     """

#     def __init__(
#         self,
#         embedding_file_path: Union[str, List[str]] = "",
#         id_file_path: str = "",
#         encoder_model_path: str = "",
#         encoder_config_path: str = "",
#         use_cuda: bool = False,
#     ):
#         super().__init__(id_file_path=id_file_path)

#         self.embeddings = {}
#         self.embeddings_by_names = {}
#         self.clip_ids = []
#         self.encoder = None
#         self.encoder_ap = None
#         self.use_cuda = use_cuda

#         if embedding_file_path:
#             if isinstance(embedding_file_path, list):
#                 self.load_embeddings_from_list_of_files(embedding_file_path)
#             else:
#                 self.load_embeddings_from_file(embedding_file_path)

#         if encoder_model_path and encoder_config_path:
#             self.init_encoder(encoder_model_path, encoder_config_path, use_cuda)

#     @property
#     def num_embeddings(self):
#         """Get number of embeddings."""
#         return len(self.embeddings)

#     @property
#     def num_names(self):
#         """Get number of embeddings."""
#         return len(self.embeddings_by_names)

#     @property
#     def embedding_dim(self):
#         """Dimensionality of embeddings. If embeddings are not loaded, returns zero."""
#         if self.embeddings:
#             return len(self.embeddings[list(self.embeddings.keys())[0]]["embedding"])
#         return 0

#     @property
#     def embedding_names(self):
#         """Get embedding names."""
#         return list(self.embeddings_by_names.keys())

#     def save_embeddings_to_file(self, file_path: str) -> None:
#         """Save embeddings to a json file.

#         Args:
#             file_path (str): Path to the output file.
#         """
#         save_file(self.embeddings, file_path)

#     @staticmethod
#     def read_embeddings_from_file(file_path: str):
#         """Load embeddings from a json file.

#         Args:
#             file_path (str): Path to the file.
#         """
#         embeddings = load_file(file_path)
#         speakers = sorted({x["name"] for x in embeddings.values()})
#         name_to_id = {name: i for i, name in enumerate(speakers)}
#         clip_ids = list(set(sorted(clip_name for clip_name in embeddings.keys())))
#         # cache embeddings_by_names for fast inference using a bigger speakers.json
#         embeddings_by_names = {}
#         for x in embeddings.values():
#             if x["name"] not in embeddings_by_names.keys():
#                 embeddings_by_names[x["name"]] = [x["embedding"]]
#             else:
#                 embeddings_by_names[x["name"]].append(x["embedding"])
#         return name_to_id, clip_ids, embeddings, embeddings_by_names

#     def load_embeddings_from_file(self, file_path: str) -> None:
#         """Load embeddings from a json file.

#         Args:
#             file_path (str): Path to the target json file.
#         """
#         self.name_to_id, self.clip_ids, self.embeddings, self.embeddings_by_names = self.read_embeddings_from_file(
#             file_path
#         )

#     def load_embeddings_from_list_of_files(self, file_paths: List[str]) -> None:
#         """Load embeddings from a list of json files and don't allow duplicate keys.

#         Args:
#             file_paths (List[str]): List of paths to the target json files.
#         """
#         self.name_to_id = {}
#         self.clip_ids = []
#         self.embeddings_by_names = {}
#         self.embeddings = {}
#         for file_path in file_paths:
#             ids, clip_ids, embeddings, embeddings_by_names = self.read_embeddings_from_file(file_path)
#             # check colliding keys
#             duplicates = set(self.embeddings.keys()) & set(embeddings.keys())
#             if duplicates:
#                 raise ValueError(f" [!] Duplicate embedding names <{duplicates}> in {file_path}")
#             # store values
#             self.name_to_id.update(ids)
#             self.clip_ids.extend(clip_ids)
#             self.embeddings_by_names.update(embeddings_by_names)
#             self.embeddings.update(embeddings)

#         # reset name_to_id to get the right speaker ids
#         self.name_to_id = {name: i for i, name in enumerate(self.name_to_id)}

#     def get_embedding_by_clip(self, clip_idx: str) -> List:
#         """Get embedding by clip ID.

#         Args:
#             clip_idx (str): Target clip ID.

#         Returns:
#             List: embedding as a list.
#         """
#         return self.embeddings[clip_idx]["embedding"]

#     def get_embeddings_by_name(self, idx: str) -> List[List]:
#         """Get all embeddings of a speaker.

#         Args:
#             idx (str): Target name.

#         Returns:
#             List[List]: all the embeddings of the given speaker.
#         """
#         return self.embeddings_by_names[idx]

#     def get_embeddings_by_names(self) -> Dict:
#         """Get all embeddings by names.

#         Returns:
#             Dict: all the embeddings of each speaker.
#         """
#         embeddings_by_names = {}
#         for x in self.embeddings.values():
#             if x["name"] not in embeddings_by_names.keys():
#                 embeddings_by_names[x["name"]] = [x["embedding"]]
#             else:
#                 embeddings_by_names[x["name"]].append(x["embedding"])
#         return embeddings_by_names

#     def get_mean_embedding(self, idx: str, num_samples: int = None, randomize: bool = False) -> np.ndarray:
#         """Get mean embedding of a idx.

#         Args:
#             idx (str): Target name.
#             num_samples (int, optional): Number of samples to be averaged. Defaults to None.
#             randomize (bool, optional): Pick random `num_samples` of embeddings. Defaults to False.

#         Returns:
#             np.ndarray: Mean embedding.
#         """
#         embeddings = self.get_embeddings_by_name(idx)
#         if num_samples is None:
#             embeddings = np.stack(embeddings).mean(0)
#         else:
#             assert len(embeddings) >= num_samples, f" [!] {idx} has number of samples < {num_samples}"
#             if randomize:
#                 embeddings = np.stack(random.choices(embeddings, k=num_samples)).mean(0)
#             else:
#                 embeddings = np.stack(embeddings[:num_samples]).mean(0)
#         return embeddings

#     def get_random_embedding(self) -> Any:
#         """Get a random embedding.

#         Args:

#         Returns:
#             np.ndarray: embedding.
#         """
#         if self.embeddings:
#             return self.embeddings[random.choices(list(self.embeddings.keys()))[0]]["embedding"]

#         return None

#     def get_clips(self) -> List:
#         return sorted(self.embeddings.keys())

#     def init_encoder(self, model_path: str, config_path: str, use_cuda=False) -> None:
#         """Initialize a speaker encoder model.

#         Args:
#             model_path (str): Model file path.
#             config_path (str): Model config file path.
#             use_cuda (bool, optional): Use CUDA. Defaults to False.
#         """
#         self.use_cuda = use_cuda
#         self.encoder_config = load_config(config_path)
#         self.encoder = setup_encoder_model(self.encoder_config)
#         self.encoder_criterion = self.encoder.load_checkpoint(
#             self.encoder_config, model_path, eval=True, use_cuda=use_cuda, cache=True
#         )
#         self.encoder_ap = AudioProcessor(**self.encoder_config.audio)

#     def compute_embedding_from_clip(self, wav_file: Union[str, List[str]]) -> list:
#         """Compute a embedding from a given audio file.

#         Args:
#             wav_file (Union[str, List[str]]): Target file path.

#         Returns:
#             list: Computed embedding.
#         """

#         def _compute(wav_file: str):
#             waveform = self.encoder_ap.load_wav(wav_file, sr=self.encoder_ap.sample_rate)
#             if not self.encoder_config.model_params.get("use_torch_spec", False):
#                 m_input = self.encoder_ap.melspectrogram(waveform)
#                 m_input = torch.from_numpy(m_input)
#             else:
#                 m_input = torch.from_numpy(waveform)

#             if self.use_cuda:
#                 m_input = m_input.cuda()
#             m_input = m_input.unsqueeze(0)
#             embedding = self.encoder.compute_embedding(m_input)
#             return embedding

#         if isinstance(wav_file, list):
#             # compute the mean embedding
#             embeddings = None
#             for wf in wav_file:
#                 embedding = _compute(wf)
#                 if embeddings is None:
#                     embeddings = embedding
#                 else:
#                     embeddings += embedding
#             return (embeddings / len(wav_file))[0].tolist()
#         embedding = _compute(wav_file)
#         return embedding[0].tolist()

#     def compute_embeddings(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
#         """Compute embedding from features.

#         Args:
#             feats (Union[torch.Tensor, np.ndarray]): Input features.

#         Returns:
#             List: computed embedding.
#         """
#         if isinstance(feats, np.ndarray):
#             feats = torch.from_numpy(feats)
#         if feats.ndim == 2:
#             feats = feats.unsqueeze(0)
#         if self.use_cuda:
#             feats = feats.cuda()
#         return self.encoder.compute_embedding(feats)


# class SpeakerManager(EmbeddingManager):
#     """Manage the speakers for multi-speaker 🐸TTS models. Load a datafile and parse the information
#     in a way that can be queried by speaker or clip.

#     There are 3 different scenarios considered:

#     1. Models using speaker embedding layers. The datafile only maps speaker names to ids used by the embedding layer.
#     2. Models using d-vectors. The datafile includes a dictionary in the following format.

#     ::

#         {
#             'clip_name.wav':{
#                 'name': 'speakerA',
#                 'embedding'[<d_vector_values>]
#             },
#             ...
#         }


#     3. Computing the d-vectors by the speaker encoder. It loads the speaker encoder model and
#     computes the d-vectors for a given clip or speaker.

#     Args:
#         d_vectors_file_path (str, optional): Path to the metafile including x vectors. Defaults to "".
#         speaker_id_file_path (str, optional): Path to the metafile that maps speaker names to ids used by
#         TTS models. Defaults to "".
#         encoder_model_path (str, optional): Path to the speaker encoder model file. Defaults to "".
#         encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".

#     Examples:
#         >>> # load audio processor and speaker encoder
#         >>> ap = AudioProcessor(**config.audio)
#         >>> manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
#         >>> # load a sample audio and compute embedding
#         >>> waveform = ap.load_wav(sample_wav_path)
#         >>> mel = ap.melspectrogram(waveform)
#         >>> d_vector = manager.compute_embeddings(mel.T)
#     """

#     def __init__(
#         self,
#         data_items: List[List[Any]] = None,
#         d_vectors_file_path: str = "",
#         speaker_id_file_path: str = "",
#         encoder_model_path: str = "",
#         encoder_config_path: str = "",
#         use_cuda: bool = False,
#     ):
#         super().__init__(
#             embedding_file_path=d_vectors_file_path,
#             id_file_path=speaker_id_file_path,
#             encoder_model_path=encoder_model_path,
#             encoder_config_path=encoder_config_path,
#             use_cuda=use_cuda,
#         )

#         if data_items:
#             self.set_ids_from_data(data_items, parse_key="speaker_name")

#     @property
#     def num_speakers(self):
#         return len(self.name_to_id)

#     @property
#     def speaker_names(self):
#         return list(self.name_to_id.keys())

#     def get_speakers(self) -> List:
#         return self.name_to_id

#     @staticmethod
#     def init_from_config(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "SpeakerManager":
#         """Initialize a speaker manager from config

#         Args:
#             config (Coqpit): Config object.
#             samples (Union[List[List], List[Dict]], optional): List of data samples to parse out the speaker names.
#                 Defaults to None.

#         Returns:
#             SpeakerEncoder: Speaker encoder object.
#         """
#         speaker_manager = None
#         if get_from_config_or_model_args_with_default(config, "use_speaker_embedding", False):
#             if samples:
#                 speaker_manager = SpeakerManager(data_items=samples)
#             if get_from_config_or_model_args_with_default(config, "speaker_file", None):
#                 speaker_manager = SpeakerManager(
#                     speaker_id_file_path=get_from_config_or_model_args_with_default(config, "speaker_file", None)
#                 )
#             if get_from_config_or_model_args_with_default(config, "speakers_file", None):
#                 speaker_manager = SpeakerManager(
#                     speaker_id_file_path=get_from_config_or_model_args_with_default(config, "speakers_file", None)
#                 )

#         if get_from_config_or_model_args_with_default(config, "use_d_vector_file", False):
#             speaker_manager = SpeakerManager()
#             if get_from_config_or_model_args_with_default(config, "d_vector_file", None):
#                 speaker_manager = SpeakerManager(
#                     d_vectors_file_path=get_from_config_or_model_args_with_default(config, "d_vector_file", None)
#                 )
#         return speaker_manager
