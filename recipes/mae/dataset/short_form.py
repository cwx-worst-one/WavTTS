# import random

# import webdataset as wds
# from torch.utils.data import IterableDataset
# from transformers import AutoTokenizer

# import recipes.mulan.dataset.utils as utils
# from recipes.mulan.dataset.wds import ReproducibleWebDataset, WebDataList


# def get_short_form_urls():
#     hdfs_base = "hdfs://harunava/home/byte_speech_sv/mulan/short_form_uio"
#     shards_by_drive = [2740, 2741, 2719, 2700, 2640, 2680, 2700, 2453]
#     result_urls = {}
#     for field in ["audio", "text", "album", "inst"]:
#         urls = [
#             f"pipe:hdfs dfs -cat {hdfs_base}/{field}_shards/
# shards_{i:01d}_{j:06d}.tar"
#             for i, shard in enumerate(shards_by_drive)
#             for j in range(shard)
#         ]
#         result_urls[field] = urls[5:]  # shard0~4 of drive use as validation
#     return result_urls


# class ShortFormDataList(IterableDataset):
#     def __init__(self, seed=2023, mode="train", **kwargs):
#         urls = get_short_form_urls()
#         datasets = [
#             ReproducibleWebDataset(urls["audio"], seed=seed, **kwargs)
#             .decode()
#             .map(utils.process_audio)
#             .to_tuple("__key__ audio"),
#             ReproducibleWebDataset(urls["text"], seed=seed, **kwargs)
#             .decode()
#             .map(self._process_text)
#             .to_tuple("__key__ text music_id data_source"),
#             ReproducibleWebDataset(urls["album"], seed=seed, **kwargs)
#             .decode()
#             .map(self._process_album)
#             .to_tuple("__key__ album_review data_source"),
#             ReproducibleWebDataset(urls["inst"], seed=seed, **kwargs)
#             .decode()
#             .map(self._process_inst)
#             .to_tuple("__key__ inst"),
#         ]
#         self.datalist = WebDataList(datasets, self._merge_fn)
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         self.datalist.append(wds.map(utils.tokenize_text(self.tokenizer, mode)))

#     def _process_text(self, data):
#         meta = data["meta.json"]
#         source = meta["data_source"]
#         text = ""
#         if source in ["both", "short_form"]:
#             target_ks = ["genre", "mood", "theme", "language"]
#             random.shuffle(target_ks)  # randomize the order of the tags
#             for field in meta:
#                 if field == "meta_song_title":
#                     text += f"'{meta[field]}' "
#                 elif field == "meta_song_author":
#                     text += f"by {meta[field]} "
#                 elif field == "meta_song_album_name":
#                     text += f"from {meta[field]}: "
#                 elif field in target_ks:
#                     if meta[field] != "":
#                         text += meta[field] + " "
#             text = text[:-1]

#         if source in ["both", "playlist"]:
#             target_ks = ["playlist_name", "playlist_description"]
#             for field in meta:
#                 if field in target_ks:
#                     text += meta[field] + " "
#             text = text[:-1]
#         data["text"] = text
#         data["music_id"] = meta["music_id"]
#         data["data_source"] = source
#         return data

#     def _process_inst(self, data):
#         data["inst"] = ""
#         return data

#     def _process_album(self, data):
#         meta = data["meta.json"]
#         source = meta["data_source"]
#         text = ""
#         target_ks = [
#             "review",
#             "genre",
#             "styles",
#             "release_date",
#             "album_moods",
#             "album_themes",
#         ]
#         if source in ["album_review"]:
#             for field in meta:
#                 if field in target_ks:
#                     text += meta[field] + " "
#         data["album_review"] = text
#         data["data_source"] = source
#         data["music_id"] = meta["music_id"]
#         return data

#     def _merge_fn(self, audio, text, album, inst):
#         return {
#             "__key__": audio[0],
#             "audio": audio[1],
#             "text": text[1] + album[1],
#             "music_id": text[2],
#             "data_source": [text[3], album[2]],
#         }

#     def __iter__(self):
#         return iter(self.datalist)
