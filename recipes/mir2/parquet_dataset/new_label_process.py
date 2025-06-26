from typing import Any, Dict, Generator, List, Optional, Tuple, Callable,Iterable
import json
import numpy as np

import recipes.bigmusic.datasets.utils.zh_vocab_dev as zh_vocab_dev
from recipes.bigmusic.datasets.utils.zh_vocab_dev import (
    Vocab2Id,
    tag_type_dict,
    VOCAB2ID_MIX_V4,
    UnifiedCategory,
    UnifiedGenre,
)
from samantha.utils.hdfs_tools import hdfs_open

from recipes.mir2.parquet_dataset.vocab_index import (
    # MOST_FREQUENT_GENRES,
    MOST_FREQUENT_SELECTED_GENRES,
    list_to_numbered_dict,
)


def get_id_to_tag_maps():
    """
    Get a map from tag id to tag name for each tag type.
    Note that the default indices of the tag map is 1-index, instead of 0-index.
    
    Returns:
        dict: a map from tag id to tag name for each tag type, for example
        ```
        {'GENRE': {1: 'Other genre',
            2: 'Pop',
            3: 'Electronic',
            4: 'Chinese Style',
            5: 'Rock',
            6: 'Jazz',
            }, ...
        }
        ```
    """
    categories = [
        UnifiedCategory.GENRE,
        UnifiedCategory.MOOD,
        UnifiedCategory.THEME,
        UnifiedCategory.GENDER,
        UnifiedCategory.TIMBRE
    ]
    id_to_tag_maps = {}
    for c in categories:
        id_to_category_map = { v.value: k for k, v in VOCAB2ID_MIX_V4.category_map[c].items()}
        sorted_map = dict(sorted(id_to_category_map.items(), key=lambda item: item[0]))
        id_to_tag_maps[c.name] = sorted_map
    
    # HACK (vibertthio): VOCAB2ID_MIX_V4 somehow doesn't have UnifiedTimbre.EMPTY
    # because CATEGORY_MAP_AUDIO_V5 doesn't include UnifiedTimbre.EMPTY
    # so we cannot find the index of UnifiedTimbre.EMPTY through VOCAB2ID_MIX_V4.category_map.
    # We keep using `zh_vocab_dev` for the training/inference consistency.
    # TODO: switch from `zh_vocab_dev` to `zh_vocab`, and remove this hack

    id_to_tag_maps[UnifiedCategory.TIMBRE.name][1] = "Empty"
    
    return id_to_tag_maps


class BaseTransform:

    def __init__(self, key=None, in_key=None, out_key=None):
        self.key = key
        self.in_key = in_key
        self.out_key = out_key
    
    def __call__(self, *args: Any, **kwgs: Any) -> Any:
        raise NotImplementedError()


class GetAllTags(BaseTransform):
    def __init__(self, out_key="tags",
                       tagging_params = None,
                       enable_filter = True,
                       skip_error = False
                       ):

        self.out_key = out_key
        self.tagging_params = tagging_params
        self.tag_list = self.tagging_params.keys()
        self.enable_filter = enable_filter
        self.skip_error = skip_error

    
    def _filter_item(self, audio_meta):
        if not audio_meta: 
            return True
        
        if audio_meta.get('satisfy_filter_standard','yes') == 'no':
            return False
        
        return True


    def _get_music_tags(self, item, metadata, tag_type_para):
       
        tag_keyword_in_data = tag_type_para['keyword']
        temp_label = metadata.get("audio_tags",{}).get(tag_keyword_in_data,[])

        if len(temp_label) == 0:
            return None
        temp_label = [label for label in temp_label if item!= '']
        return temp_label

    def _get_artist_tags(self, item, meta):

        artist_name = []
        artist = meta.get('raw', {}).get('artist', '').replace('"', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('artist_name', '').replace('"', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('artist_name', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('song_detail', {}).get('ar', [{}])[0].get('name', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('artist', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('meta_song_author_name', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('main_artist', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('歌手名', '')
        if artist :
            artist_name.append(artist)

        artist_name = list(set(artist_name))

        return artist_name

    def __call__(self, item, **_kwargs):
        if item is None:
            return item

        metadata = json.loads(item.get("meta",{}))

        if self.enable_filter:
            if not self._filter_item(metadata):
                #print(f"{item['uttid']}'s satisfy_filter_standard is No. Skip")
                return None

        tag_dict = {}
        
        for tag_type, tag_type_para in self.tagging_params.items():
            if tag_type in ['GENRE','THEME','MOOD','GENDER','TIMBRE']:
                temp = self._get_music_tags(item, metadata, tag_type_para)
            elif tag_type == "ARTIST":
                temp = self._get_artist_tags(item['uttid'], metadata)
            else:
                raise NotImplementedError(f"{tag_type} processing has not been implmented")

            if temp:
                tag_dict[tag_type] = temp

        # Not for training but for testing
        if not tag_dict:
            #print(f"{item['uttid']} in dataset {item['__dataset_name__']} has no tags. Skip!")
            return None

        item[self.out_key] = tag_dict

        return item


class GetTags(BaseTransform):
    def __init__(self, out_key="tags",
                       tagging_params = None,
                       enable_filter = True,
                       skip_error = False
                       ):

        self.out_key = out_key
        self.tagging_params = tagging_params
        self.tag_list = self.tagging_params.keys()
        self.enable_filter = enable_filter
        self.skip_error = skip_error

    
    def _filter_item(self, audio_meta):
        if not audio_meta: 
            return True
        
        if audio_meta.get('satisfy_filter_standard','yes') == 'no':
            return False
        
        return True


    def _get_music_tags(self, item, metadata, tag_type_para):
       
        tag_keyword_in_data = tag_type_para['keyword']
        temp_label = metadata.get("audio_tags",{}).get(tag_keyword_in_data,[])

        if len(temp_label) == 0:
            return None
        temp_label = [label for label in temp_label if item!= '']
        return temp_label

    def _get_artist_tags(self, item, meta):

        artist_name = []
        artist = meta.get('raw', {}).get('artist', '').replace('"', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('artist_name', '').replace('"', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('artist_name', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('song_detail', {}).get('ar', [{}])[0].get('name', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('artist', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('meta_song_author_name', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('main_artist', '')
        if artist :
            artist_name.append(artist)
        artist = meta.get('raw', {}).get('歌手名', '')
        if artist :
            artist_name.append(artist)

        artist_name = list(set(artist_name))

        return artist_name

    def __call__(self, item, **_kwargs):
        if item is None:
            return item

        metadata = json.loads(item.get("meta",{}))

        if self.enable_filter:
            if not self._filter_item(metadata):
                print(f"{item['uttid']}'s satisfy_filter_standard is No. Skip")
                return None

        tag_dict = {}
        
        for tag_type, tag_type_para in self.tagging_params.items():
            if tag_type in ['GENRE','THEME','MOOD','GENDER','TIMBRE']:
                temp = self._get_music_tags(item, metadata, tag_type_para)
            elif tag_type == "ARTIST":
                temp = self._get_artist_tags(item['uttid'], metadata)
            else:
                raise NotImplementedError(f"{tag_type} processing has not been implmented")
            
            # Not for training but for testing
            if not temp:
                #print(f"Get {tag_type} is None. Skip! {item['__dataset_name__']},{item['uttid']}")
                if self.skip_error:
                    tag_dict[tag_type] = []
                else:
                    raise ValueError(f"Get {tag_type} is None. Skip! {item['__dataset_name__']},{item['uttid']}")
                    return None
            else:
                tag_dict[tag_type] = temp

        item[self.out_key] = tag_dict

        return item


class BinalizeTags(BaseTransform):
    def __init__(self, in_key="tags", 
                       out_key="target_tags",                  
                       tagging_params=None, 
                       vocab2id_version = "VOCAB2ID_MIX_V4"):
        
        self.in_key = in_key
        self.out_key = out_key
        self.tagging_params = tagging_params
        self.vocab2id = self.parse_initialized_class(vocab2id_version)
        self.tag_type_dict = tag_type_dict
        self.vocab2id_version = vocab2id_version

        # map from genre names in the dataset to unified names in the vocab
        self.most_frequent_genres_to_id = list_to_numbered_dict([ 
            self.vocab2id.category_map[tag_type_dict["GENRE"]][g].name \
                for g in MOST_FREQUENT_SELECTED_GENRES
        ])

    def parse_initialized_class(self, instance_name):
        class_instance = getattr(zh_vocab_dev, instance_name, None)
        if class_instance:
            return class_instance
        else:
            raise ValueError(f"Class instance '{instance_name}' not found in classes module")


    def _get_unifed_tag(self, tag_type, temp_tag_name):
        meta_tag_type = self.tag_type_dict[tag_type]
        #TODO:@rui.xia this is too hacky, revise int future
        if temp_tag_name == "Sweet" and tag_type == "MOOD":
            temp_tag_name = "Sweet_SA_MOOD"
        
        if temp_tag_name == "Cute" and tag_type == "MOOD":
            temp_tag_name = "Cute_SA_MOOD"
        
        if temp_tag_name == "Chorus" and tag_type == "GENDER":
            temp_tag_name = "Chorus_AUDIO_GENDER"
        
        if temp_tag_name == "Chorus" and tag_type == "GENRE":
            temp_tag_name = "Chorus_AUDIO_GENRE"
        
        if temp_tag_name not in self.vocab2id.category_map[meta_tag_type]:
            #print(f"Error {temp_tag_name} not in Dictionary {self.vocab2id_version}:{meta_tag_type}")
            return None

        temp_tag_unifed_name = self.vocab2id.category_map[meta_tag_type][temp_tag_name]

        return temp_tag_unifed_name
    

    def _get_binary_vector_genre_only(self, unified_tag_names, tag_type):
        meta_tag_type = self.tag_type_dict[tag_type]
        meta_tag_dim = self.tagging_params[tag_type]["n_class"]
        result_ids = np.zeros(meta_tag_dim, dtype=np.float32)
        
        
        for unified_tag in unified_tag_names:
            # TODO: organize the logic into a class for genre selection
            # If unified_tag is None, it means the tag read from the data isn't in the unified vocab
            if unified_tag is None:
                continue
            # If the tag is indeed in the unified vocab, but not in the selected vocab in the task config
            if unified_tag.name not in self.most_frequent_genres_to_id:
                continue
            
            index = self.most_frequent_genres_to_id[unified_tag.name]
            
            result_ids[index] = 1.0
        
        # If result_ids is all 0, which means all tags are not valid, not included in the vocab, or totally empty
        # Then, mark last index 1, which is "others"
        if np.sum(result_ids) == 0:
            result_ids[-1] = 1.0
        
        return result_ids
    
    def _get_binary_vector(self, unified_tag_names, tag_type):
        meta_tag_type = self.tag_type_dict[tag_type]
        meta_tag_dim = self.tagging_params[tag_type]["n_class"]
        result_ids = np.zeros(meta_tag_dim, dtype=np.float32)

        for unified_tag in unified_tag_names:
            if unified_tag == None:
                result_ids[0] = 1.0
            else:
                result_ids[unified_tag.value-1] = 1.0
        
        # if unified_tag_names is a empty list, then also make it as Empty/Others/Unkonwn
        if np.sum(result_ids) == 0:
            result_ids[0] = 1.0
        
        return result_ids

    def __call__(self, item, **_kwargs):
        if item is None or self.in_key not in item:
            raise ValueError(f"item is None or {self.in_key} not in item")
        
        normalized_tags = {}
        binarizd_tags = {}

        for tag_type, tag_labels in item[self.in_key].items():
            
            unified_name = []
            for temp_tag_label in tag_labels:
                temp_unified_name = self._get_unifed_tag(tag_type, temp_tag_label)
                unified_name.append(temp_unified_name)
            
            # Assign Empty Class To Data
            # TODO: closely examine the logic here for EMPTY/Uknown/Others confusion
            unified_name_labels = [x.name if x is not None else "EMPTY" for x in unified_name]
            unifed_name_vector = self._get_binary_vector(unified_name, tag_type)  # use _get_binary_vector_genre_only when using selected genre

            normalized_tags[tag_type] = unified_name_labels
            binarizd_tags[tag_type] = unifed_name_vector
            
        item[self.out_key] = binarizd_tags
        item[f"normalized_{self.in_key}"] = normalized_tags
        
        return item

class FilterByUttid(BaseTransform):
    def __init__(self, key="uttid", filter_list_file=None):
        self.key = key
        self.black_list = self._get_list(filter_list_file)
    
    def _get_list(self, filter_list_file):
        assert filter_list_file is not None
        with hdfs_open(filter_list_file, "r") as f:
            blk_lst = [ line.rstrip() for line in f.readlines()]
            print(f"Number of black_list: {len(blk_lst)}")
        
        return set(blk_lst)


    def __call__(self, item, **_kwargs):
        if item is None or self.key not in item:
            return item
        
        if item[self.key] in self.black_list:
            raise ValueError(f"In black_list. Skip this data. {item['__dataset_name__']},{item['uttid']}")
        
        return item