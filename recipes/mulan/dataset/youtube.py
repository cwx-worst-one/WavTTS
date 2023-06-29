from samantha.dataio.webdataset.extension import IndexedWebDataset
from tqdm import tqdm
import webdataset as wds
from samantha.dataio.webdataset.pipeline import WebPipeline
from typing import Callable, List, Optional
from recipes.mulan.transforms.mulan import YMVransforms
from recipes.mulan.preprocess import WebDatasetBufferPreprocessor
from recipes.mulan.dataset.utils import collate_fn
import pdb
from transformers import AutoTokenizer
# d = IndexedWebDataset(
#     "/mnt/bn/audio-diffusion/data/disco/url2idx.txt",
#     nodesplitter=wds.split_by_node
# )
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

from torch.utils.data import DataLoader
# max_num_crops = 3
# crop_step_size = seconds * sample_rate

class YMVDataset(WebPipeline):

    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        duration: float,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = True,
        avoid_sound_effect: bool = True,
        avoid_vocal: bool = True,
        max_vocal_threshold: float = 0.25,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: Optional[int] = None,
        crop_step_size: Optional[int] = None,
        handler: Callable = wds.warn_and_continue,
        seed: int=2023,
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            handler=handler,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        audio_transforms = YMVransforms(
            n_samples=int(duration * sample_rate),
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            aed_filtered=aed_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            max_num_crops=max_num_crops,
            crop_step_size=crop_step_size,
            tokenizer=tokenizer
        )
        
        
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
        ]
        super().__init__(dataset, pipeline)





if __name__ == "__main__":
    # d = IndexedWebDataset(
    # "/mnt/bn/audio-diffusion/data/disco/url2idx.txt",
    # nodesplitter=wds.split_by_node)


    # loader = wds.WebLoader(d, batch_size=16, collate_fn=lambda x: x, num_workers=4)
    # for i, batch in enumerate(tqdm(loader)):
    #     # print(batch[0]['mp3'])
    #     # print(batch[0]['__index_data__'])
    #     # ['__key__', '__url__', '__index_data__', 'mp3']
    #     # ['search_query_youtube','video_title_youtube','video_description_youtube']
    #     if i == 10:
    #         break
        # assert ["mp3" in item for item in batch]

    dataset = YMVDataset(url2index="/mnt/bn/audio-diffusion/data/disco/url2idx.txt",
                sample_rate=24000,
                duration=10,
                min_volume_threshold=0.05,
                loudness_ratio_threshold=0.2,
                resampled=True,
                shardshuffle=True,
                seed=2023)
    from samantha.dataio.dataset import MultiIterableDataset

    d = MultiIterableDataset([dataset], weights=[1], num_samples=10000000)

    loader = wds.WebLoader(d, batch_size=4, collate_fn=collate_fn, num_workers=2)

    # loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=2)
    for i, batch in enumerate(tqdm(loader)):
        # print(batch[0]['music_id'])
        # print(batch[0]['input_ids'].shape)
        # print(batch[0]['attention_mask'].shape)
        print(batch['input_ids'].shape)
        if i == 10:
            break

    


