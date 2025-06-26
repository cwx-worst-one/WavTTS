from webdataset.pipeline import DataPipeline
import webdataset as wds
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional, Union,List,Dict,Generator
import json
from functools import partial
from recipes.bigmusic.datasets.lyrics import default_batch_fn, transform_dataset
from recipes.datasets.mcc.sami_tokenizer import init_sami_tokenizer
from recipes.bigmusic.datasets.zh_mix import PHONEME_TOKENIZERS
from recipes.bigmusic.datasets.zh_inference import ZhInferTransforms, collate_fn_zh, translate_style_text_vocab, PromptItemDispatcher
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.dataio.parquet import ParquetDataset
from recipes.datasets.mcc.sami_tokenizer import section_tags


from samantha.utils.utils import AudioLogger
logger = AudioLogger()


class WebDatasetBufferPreprocessor:
    def __init__(self, transforms):
        self.transforms = transforms

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)

class MirDataset(WebPipeline):
    def __init__(
        self,
        data_id: int = None,
        url_pattern: str = None,
        item_transform: Optional[str] = None,
        **kwargs,
    ):
        logger.info(f"[{self.__class__.__name__}] initializing...")
        
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)

        preprocessor = WebDatasetBufferPreprocessor(transforms=item_transform)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        logger.info(f"[{self.__class__.__name__}] initialized.")

class Convert2Promptentry:

    def __init__(self, 
                style_text_transform_target="9_cat_combo_v4",
                use_offline_phoneme=True
                ):
        self.text_category = "cn lyrics2song AIGC inhouse long"
        self.style_text_transform_target = style_text_transform_target
        self.style_text_format = translate_style_text_vocab(style_text_transform_target)
        self.use_offline_phoneme = use_offline_phoneme

        self.section_tags = [f'[{x}]' for x in section_tags]


    def __call__(self, item, **kwargs):

        item_name = self.__class__.__name__

        meta = json.loads(item['meta'])

        try: 
            style_info = json.loads(meta['raw']['response_style_info'])
            if self.use_offline_phoneme:
                lyrics = meta['raw'].get('response_lyrics')
                lyrics_without_processed = meta['lyrics']
    
                frontend_res = self._convert_utterances_to_frontend_results(lyrics_without_processed)

            else:
                lyrics = meta['raw'].get('response_lyrics')
                frontend_res= None
            
            temp_request ={
                "tags_music": style_info.get("style_tags"),
                "speaker_id": style_info.get("speaker_id"),
                "lyrics": lyrics,
                "duration": style_info.get('duration'),
                "freeform_text": style_info.get('freeform_text'),
                "is_full_song": True,
                "extra": {
                    "index": item['uttid']
                }
            }
            if frontend_res:
                temp_request["extra"]["frontend_results"] = frontend_res

            # temp_prompt_entry = request_dict_to_item(temp_request,self.style_text_format).transform_bypass(style_text_transform_target=self.style_text_transform_target,
            temp_prompt_entry = PromptItemDispatcher.from_request(
                request=temp_request,
                style_text_format=self.style_text_format,
                index=item['uttid'],
            )
            yield temp_prompt_entry

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(" {} transform error: {}, data_file: {}".format(
                    item_name,
                    e,
                    item['uttid']
                ))
            return

    def _convert_utterances_to_frontend_results(self, utterances_input):
        """
        Converts a list of utterances into the frontend_results format.

        Args:
            utterances_input (dict): A dictionary containing the key "utterances" which is a list of utterance dictionaries.
                                    Each utterance dictionary contains:
                                        - "text": The utterance text or section tag.
                                        - "phoneme_v86": The phoneme processing result.
                                        - "phoneme_v86_tn": (Optional) Token name, same as "text" for section tags.

        Returns:
            dict: A dictionary with the key "frontend_results" containing a list of formatted strings.
        """
        frontend_results = []

        for utterance in utterances_input.get("utterances", []):
            text = utterance.get("text", "")
            phoneme_v86 = utterance.get("phoneme_v86", "")

            # Check if the text is a section tag by looking for square brackets
            if text.startswith("[") and text.endswith("]"):
                # Append the section tag with a trailing '#'
                frontend_results.append(f"{text}#")
            else:
                # Append the utterance text and phoneme output separated by '#'
                frontend_results.append(f"{text}#{phoneme_v86}")

        return frontend_results


def inference_dataset_from_prompt_parquet(
    data_id: int,
    style_text_transform_target: str = "9_cat_combo_v4",
    conditions: str = "style_text,freeform_text,speaker_id,lyrics_tokens",
    batch_size: int = 1,
    max_phone_len: int = 4000,
    enable_cfg: bool = False,
    cfg_targets: Optional[list[str]] = None,  # default
    vocab_type: Optional[str] = None,
    lyrics_tokenizer: Optional[str] = None,
    prompt_sample_rate: int = 44100,
    frame_rate: int = 25,
    use_frontend_results: bool = False,
    item_transforms: Optional[List[Dict]] = None, #place holder for further transforms
    shuffle_buffer_size: Optional[int] = None,
    deterministic: bool = False,
) -> DataPipeline:

    def remap_cfg_targets(cfg_targets: Optional[list[str]]) -> Optional[list[str]]:
        def remap_section_tag(cfg_target: str) -> str:
            if cfg_target != "section_tag":
                return cfg_target
            logger.warning('CFG target "section_tag" is deprecated and replaced by "lyrics"')
            return "lyrics"
        if not cfg_targets:
            return None
        return [remap_section_tag(cfg_target) for cfg_target in cfg_targets]

    def prompt_path_to_json_hook(prompt_path: Union[str, Any]):
        """If prompt_path is a path to a JSON file, load the JSON file."""
        if not (isinstance(prompt_path, str) or isinstance(prompt_path, Path)):
            return prompt_path
        path = Path(prompt_path)
        if path.exists() and path.suffix == ".json":
            with open(prompt_path) as fin:
                return json.load(fin)
        return prompt_path

    item_transform_fn = Convert2Promptentry(
        style_text_transform_target=style_text_transform_target,
        use_offline_phoneme=use_frontend_results,
    )

    dataset = MirDataset(data_id=data_id, 
                        item_transform=item_transform_fn, 
                        resampled=True, 
                        shardshuffle=True, 
                        handler=wds.warn_and_continue)

    style_text_format = translate_style_text_vocab(style_text_transform_target)

    batch_fn = default_batch_fn(batch_size, collation_fn=partial(collate_fn_zh, app_type=None, conditions=conditions, enable_cfg=enable_cfg))
    assert (lyrics_tokenizer is None) ^ (vocab_type is None), "Either lyrics_tokenizer or vocab_type must be specified"
    if lyrics_tokenizer:
        tokenizer = PHONEME_TOKENIZERS[lyrics_tokenizer]()
    else:
        tokenizer = init_sami_tokenizer(vocab_type=vocab_type, use_frontend_results=use_frontend_results)
    
    return transform_dataset(
        dataset,
        # Wrap the rest of operations (PromptEntry -> DataSample conversion, including lyrics tokenization)
        # into a Transform that can then be used by transform_dataset.
        segment_transforms=[ZhInferTransforms(
            tokenizer=tokenizer,
            max_phone_len=max_phone_len,
            # prompt_sample_rate=prompt_sample_rate,
            frame_rate=frame_rate,
            style_text_format=style_text_format,
            enable_cfg=enable_cfg,
            cfg_targets=remap_cfg_targets(cfg_targets),
        )],
        # Any transform after batching (mainly tensor padding) has been included in the collate_fn
        batch_transforms=[],
        batch_fn=batch_fn,
        shuffle_buffer_size=shuffle_buffer_size,
        deterministic=deterministic,
    )
