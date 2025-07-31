import random
import re
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
from mariana.utils.audio.audio_logger import AudioLogger
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import samantha  # noqa: F401, import samantha first to solve mariana import path
from samantha.dataio.bigmusic.music_multitasks import multitask_chat_template
from samantha.dataio.bigmusic.transforms.song_slice import SongSlice

logger = AudioLogger()


# Example for
class DummyCollate:
    """reference label collate."""

    def __init__(self, key="label", out_key="label_out"):
        """init."""
        self.in_key = key
        self.out_key = out_key

    def mapping(self, label):
        return label + "CHANGE"

    def __call__(self, batch_in, batch_out):
        """
        do reference label collation.
        This function is for inference.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        ids = []
        for item in batch_in:
            label = item[self.in_key][self.label_idx]
            ids.append(self.mapping[label])
        ids = torch.tensor(ids, dtype=torch.int64)
        batch_out[self.out_key] = ids


class MixTokenCollate:
    """reference label collate."""

    # local tokenizer path: recipes/musiclish/lyrics_audio_tokenizer_hf/
    def __init__(
        self,
        in_key="input_strings",
        out_key="encoded",
        bpe_tokenizer_path="/opt/tiger/samantha/test_bpe_tokenizer",
    ):
        """init."""
        self.in_key = in_key
        self.out_key = out_key
        self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_tokenizer_path)

    def __call__(self, batch_in, batch_out):
        """
        do reference label collation.
        This function is for inference.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        input_strings = [item[self.in_key] for item in batch_in]
        batch_out[self.out_key] = self.bpe_tokenizer.batch_encode_plus(
            input_strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )


class MultiaskTokenCollatePre:
    """Multitask label collate. Only for inference"""

    # local tokenizer path: recipes/musiclish/lyrics_audio_tokenizer_hf/
    def __init__(
        self,
        in_key="input_strings",
        is_training=True,
        calculate_prompt_loss=False,
        bpe_tokenizer_path="/opt/tiger/bbpe155k-v6.4.3-ml.pret",
        audio_vocab_size=32768,
        moe_bos="<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
        moe_eos="<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
        tasks="Lyrics2Song,MusicCaption,TokenASR",
        task2systemprompt={},
    ):
        """init."""
        self.in_key = in_key
        self.is_training = is_training
        self.calculate_prompt_loss = calculate_prompt_loss
        self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_tokenizer_path)

        # add text tokens: we can use more boundary tokens, here we just simplified it
        added_text_tokens = ["<SECTION>", "</SECTION>"]
        added_text_vocab = self.bpe_tokenizer.add_tokens(added_text_tokens)
        self.audio_start_id = self.bpe_tokenizer.vocab_size + added_text_vocab
        logger.info(
            f"newly added token vocab is {added_text_vocab} with audio end token included\n"
        )

        # add audio tokens
        self.audio_start_token = "<audio>"
        self.audio_end_token = "</audio>"
        self.audio_pad_token = "<audio_pad>"
        audio_tokens = [f"<au_{idx}>" for idx in range(audio_vocab_size)] + [
            self.audio_start_token,
            self.audio_end_token,
            self.audio_pad_token,
        ]
        self.audio_vocab = set(
            list(range(self.audio_start_id, self.audio_start_id + audio_vocab_size))
        )
        added_audio_vocab = self.bpe_tokenizer.add_tokens(audio_tokens)
        self.audio_start_token_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_start_token
        )
        self.audio_end_token_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_end_token
        )
        self.audio_pad_token_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_pad_token
        )
        logger.info(
            f"newly added token vocab is {added_audio_vocab} with audio start/end/pad token included,"
            f"{self.audio_start_token_id=} {self.audio_end_token_id=} {self.audio_pad_token_id=}\n"
        )
        total_vocab = len(self.bpe_tokenizer)
        logger.info(f"the new whole vocab size is {total_vocab}")

        self.moe_bos = moe_bos
        self.moe_eos = moe_eos
        self.eos_id = self.bpe_tokenizer.convert_tokens_to_ids(self.audio_end_token)
        self.text_end_token = self.bpe_tokenizer.convert_tokens_to_ids(self.moe_eos)

        # lightweight multitask routing
        self.tasks = tasks.split(",")
        self.task2systemprompt = task2systemprompt

    def __call__(self, batch_in, batch_out):
        """
        do reference label collation.
        This function is for inference.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        input_strings = [item[self.in_key] for item in batch_in]
        input_ids_batch = []
        instructs = []

        for item in input_strings:
            prompt, lyrics = item
            for task in self.tasks:
                system_prompt = self.task2systemprompt[task]
                full_input, _ = multitask_chat_template(
                    prompt, lyrics, None, task, system_prompt
                )

                instructs.append(full_input)
                encoded = self.bpe_tokenizer(full_input)
                input_ids = encoded["input_ids"]  # just a list integers
                input_ids_batch.append(torch.tensor(input_ids))

        input_ids_batch = pad_sequence(
            input_ids_batch,
            batch_first=True,
            padding_value=self.bpe_tokenizer.convert_tokens_to_ids(
                self.audio_end_token
            ),
        )
        batch_out["tokenizer"] = self.bpe_tokenizer
        batch_out["input_ids"] = input_ids_batch

        batch_out["instructs"] = instructs
        batch_out["audio_vocab"] = self.audio_vocab
        batch_out["eos_id"] = self.text_end_token
        batch_out["task"] = task
        batch_out["text_codebook_size"] = self.audio_start_id
        batch_out["freeform_text"] = [item["prompt"] for item in batch_in]
        batch_out["lyrics"] = [item["lyrics"] for item in batch_in]
        batch_out["conditions"] = ["lyrics_tokens"]


class MixTokenCollatePre:
    """reference label collate."""

    # local tokenizer path: recipes/musiclish/lyrics_audio_tokenizer_hf/
    def __init__(
        self,
        in_key="input_strings",
        is_training=True,
        is_cfg_uncond=False,
        calculate_prompt_loss=False,
        bpe_tokenizer_path="/opt/tiger/bbpe155k-v6.4.3-ml.pret",
        audio_vocab_size=32768,
        text_end_token="|<endoftext>|",
        audio_end_token="|<endofaudio>|",
    ):
        """init."""
        self.in_key = in_key
        self.is_training = is_training
        self.is_cfg_uncond = is_cfg_uncond
        self.calculate_prompt_loss = calculate_prompt_loss
        self.bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_tokenizer_path)
        self.text_end_token = text_end_token
        self.audio_end_token = audio_end_token

        # TODO for future CoT, we need to rich our boundary tokens (Weituo)

        # add text tokens: we can use more boundary tokens, here we just simplified it
        added_text_tokens = ["<SECTION>", "</SECTION>", text_end_token]
        added_text_vocab = self.bpe_tokenizer.add_tokens(added_text_tokens)
        self.audio_start_id = self.bpe_tokenizer.vocab_size + added_text_vocab

        # add audio tokens:
        audio_tokens = [f"<au_{idx}>" for idx in range(audio_vocab_size)] + [
            audio_end_token
        ]
        added_audio_vocab = self.bpe_tokenizer.add_tokens(audio_tokens)
        print(
            f"newly added token vocab is {added_audio_vocab} with audio end token included\n"
        )
        total_vocab = len(self.bpe_tokenizer)
        print(f"the new whole vocab size is {total_vocab}")

        self.eos_id = self.bpe_tokenizer.convert_tokens_to_ids(self.audio_end_token)
        self.audio_start_id = self.bpe_tokenizer.convert_tokens_to_ids("<au_0>")

    def __call__(self, batch_in, batch_out):
        """
        do reference label collation.
        This function is for inference.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        input_strings = [item[self.in_key] for item in batch_in]
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []

        # For future CoT, we probably need key -> value batch
        for item in input_strings:
            prompt, lyrics, audio = item

            if self.is_training:
                full_input = (
                    prompt + lyrics + self.text_end_token + audio + self.audio_end_token
                )
            # elif self.is_cfg_uncond:
            #     full_input = lyrics + self.text_end_token
            elif self.is_cfg_uncond:
                full_input = "<SECTION></SECTION>" + self.text_end_token
            else:
                full_input = prompt + lyrics + self.text_end_token
            encoded = self.bpe_tokenizer(full_input)

            prompt_len = len(self.bpe_tokenizer(prompt)["input_ids"])

            input_ids = encoded["input_ids"]  # just a list integers

            if self.calculate_prompt_loss:
                token_type_ids = [
                    0 if (token_id < self.audio_start_id and idx > prompt_len) else 1
                    for idx, token_id in enumerate(input_ids)
                ]
            else:
                token_type_ids = [
                    0 if token_id < self.audio_start_id else 1 for token_id in input_ids
                ]
            attention_mask = [1] * len(input_ids)

            input_ids_batch.append(torch.tensor(input_ids))
            attention_mask_batch.append(torch.tensor(attention_mask))
            token_type_ids_batch.append(torch.tensor(token_type_ids))

        input_ids_batch = pad_sequence(
            input_ids_batch,
            batch_first=True,
            padding_value=self.bpe_tokenizer.convert_tokens_to_ids(
                self.audio_end_token
            ),
        )
        attention_mask_batch = pad_sequence(
            attention_mask_batch, batch_first=True, padding_value=0
        )
        token_type_ids_batch = pad_sequence(
            token_type_ids_batch, batch_first=True, padding_value=0
        )

        if self.is_cfg_uncond:
            batch_out["input_ids_uncond"] = input_ids_batch
            batch_out["attention_mask_uncond"] = attention_mask_batch
            batch_out["token_type_ids_uncond"] = token_type_ids_batch

            batch_out["freeform_text_uncond"] = [item["prompt"] for item in batch_in]
            batch_out["lyrics_uncond"] = [item["lyrics"] for item in batch_in]
            batch_out["conditions_uncond"] = ["lyrics_tokens"]
        else:
            batch_out["input_ids"] = input_ids_batch
            batch_out["attention_mask"] = attention_mask_batch
            batch_out["token_type_ids"] = token_type_ids_batch

            batch_out["freeform_text"] = [item["prompt"] for item in batch_in]
            batch_out["lyrics"] = [item["lyrics"] for item in batch_in]

        batch_out["conditions"] = ["lyrics_tokens"]
        batch_out["category"] = ["results"]
        batch_out["eos_id"] = self.eos_id
        batch_out["text_codebook_size"] = self.audio_start_id
        batch_out["freeform_text"] = [item["prompt"] for item in batch_in]
        batch_out["lyrics"] = [item["lyrics"] for item in batch_in]


class FullTokenCollate:
    """reference label collate."""

    # local tokenizer path: recipes/musiclish/lyrics_audio_tokenizer_hf/
    def __init__(
        self,
        in_key: List[str] = "input_strings",
        bpe_tokenizer_path: str = "/opt/tiger/bbpe155k-v6.4.3-ml.pret",
        audio_vocab_size: int = 32768,
        extra_tokens: List[str] = None,
        seq_template: Union[
            str, List[str]
        ] = "<prompt>{}</prompt><lyrics>{}</lyrics><audio>{}</audio>",
        out_key_suffix: Union[str, List[str]] = "",  # deprecated: keep it for compat
        padding_side="right",
    ):
        """init."""
        self.in_key = in_key if isinstance(in_key, list) else [in_key]
        self.bpe_tokenizer = AutoTokenizer.from_pretrained(
            bpe_tokenizer_path, padding_side=padding_side
        )
        self.seq_template = (
            seq_template if isinstance(seq_template, list) else [seq_template]
        )
        assert len(self.in_key) == len(self.seq_template)

        extra_tokens = extra_tokens or []

        # no section_spectial_tokens here, as we will use downloaded lyrics which has richer section tags
        # e.g. [Verse 2: Doechii & Luke, exciting]
        self.prompt_boundary_tokens = ["<prompt>", "</prompt>"]
        self.prompt_special_tokens = [
            "<genre>",
            "</genre>",
            "<mood>",
            "</mood>",
            "<language>",
            "</language>",
            "<tempo>",
            "</tempo>",
            "<gender>",
            "</gender>",
            "<scene>",
            "</scene>",
            "<timbre>",
            "</timbre>",
            "<instrument>",
            "</instrument>",
            "<character>",
            "</character>",
        ]
        self.lyrics_boudary_tokens = ["<lyrics>", "</lyrics>"]
        self.audio_boundary_tokens = ["<audio>", "</audio>"]
        self.default_boundary_tokens = ["<section>", "</section>"]

        # add audio tokens:
        audio_tokens = [f"<au_{idx}>" for idx in range(audio_vocab_size)]
        self.bpe_tokenizer.add_tokens(
            [self.audio_boundary_tokens[0]]
            + audio_tokens
            + [self.audio_boundary_tokens[1]]
        )

        # add all other tokens:
        self.bpe_tokenizer.add_tokens(
            self.prompt_boundary_tokens
            + self.prompt_special_tokens
            + self.lyrics_boudary_tokens
            + self.default_boundary_tokens
            + extra_tokens
        )

        total_vocab = len(self.bpe_tokenizer)
        logger.info(f"the new whole vocab size is {total_vocab}")

        self.eos_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_boundary_tokens[1]
        )
        self.audio_start_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_boundary_tokens[0]
        )

    def __call__(self, batch_in, batch_out):
        """
        do reference label collation.
        This function is for inference.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        input_strings_list = [
            [item[in_key] for item in batch_in] for in_key in self.in_key
        ]

        text_inputs = []
        for input_strings, seq_template in zip(input_strings_list, self.seq_template):
            if "{}" in seq_template:
                full_input = [seq_template.format(*item) for item in input_strings]
            else:
                full_input = [seq_template for _ in input_strings]
            text_inputs.extend(full_input)
        tk_output = self.bpe_tokenizer(text_inputs, padding=True, return_tensors="pt")
        # remodify token_type_ids

        token_type_ids = (
            (tk_output["input_ids"] >= self.audio_start_id)
            & (tk_output["input_ids"] <= self.eos_id)
        ).long()
        tk_output["token_type_ids"] = token_type_ids
        batch_out.update(tk_output)

        batch_size = len(text_inputs) // len(self.seq_template)
        batch_out["conditions"] = ["lyrics_tokens"]
        batch_out["category"] = ["results"] * batch_size
        batch_out["eos_id"] = self.eos_id
        batch_out["text_codebook_size"] = self.audio_start_id + 1  # exclude <audio>
        return


class StandardMetaToStyleTextCollate:
    """Convert standard music meta to style text (list[list[str]])
    This transform serves as a compatibility layer that connects the new
    BPE implementation with the legacy MIRTagMetricsCallback, which requires
    to have the old style text format to calculate accuracies.
    """

    def __init__(self, in_key: str = "standard_meta", out_key: str = "style_text"):
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, batch_in: list[dict], batch_out: dict):
        def format_standard_meta(std_meta: dict) -> list[list[str]]:
            order = ["genre", "mood", "gender"]
            if not std_meta:
                return [[] for _ in order]
            return [std_meta.get(category, []) for category in order]

        batch_out[self.out_key] = [
            format_standard_meta(item[self.in_key]) for item in batch_in
        ]


class ExpandedTokenizer:
    def __init__(
        self,
        in_key="text",
        out_key="token_ids",
        text_inputs_key="inputs",
        tokenizer_path="/opt/tiger/bbpe155k-v6.4.3-ml.pret",
        audio_vocab_size=32768,
    ):
        self.in_key = in_key
        self.out_key = out_key
        self.text_inputs_key = text_inputs_key

        self.bpe_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # add text tokens: we can use more boundary tokens, here we just simplified it
        added_text_tokens = ["<SECTION>", "</SECTION>"]
        added_text_vocab = self.bpe_tokenizer.add_tokens(added_text_tokens)
        self.audio_start_id = self.bpe_tokenizer.vocab_size + added_text_vocab
        logger.info(f"newly {added_text_vocab=}")

        # add audio tokens
        self.audio_start_token = "<audio>"
        self.audio_end_token = "</audio>"
        self.audio_pad_token = "<audio_pad>"
        audio_tokens = [f"<au_{idx}>" for idx in range(audio_vocab_size)] + [
            self.audio_start_token,
            self.audio_end_token,
            self.audio_pad_token,
        ]
        added_audio_vocab = self.bpe_tokenizer.add_tokens(audio_tokens)
        self.audio_start_token_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_start_token
        )
        self.audio_end_token_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_end_token
        )
        self.audio_pad_token_id = self.bpe_tokenizer.convert_tokens_to_ids(
            self.audio_pad_token
        )
        logger.info(
            f"newly added {added_audio_vocab=}, include audio start/end/pad."
            f"{self.audio_start_token_id=}"
            f"{self.audio_end_token_id=}"
            f"{self.audio_pad_token_id=}"
        )
        total_vocab = len(self.bpe_tokenizer)
        logger.info(f"the new whole vocab size is {total_vocab}")

    def __call__(self, item, **_kwargs):
        # process text_inputs
        try:
            text_inputs = item[self.text_inputs_key]
            input_ids = []
            token_type_ids = []

            for text_input in text_inputs:
                text = text_input.get(self.in_key, None)
                if len(text) <= 0:
                    continue
                input_id = self.bpe_tokenizer.encode(text)
                if "loss_flag" in text_input and text_input["loss_flag"] == 0:
                    token_type_id = [0] * len(input_id)
                else:
                    token_type_id = [1] * len(input_id)

                input_ids.extend(input_id)
                token_type_ids.extend(token_type_id)

            item[self.out_key] = np.asarray(input_ids, dtype=np.int32)
            item["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int32)
            item["num_tokens"] = len(input_ids)

        except Exception as e:
            logger.error(
                f"Error: {e}, {item.keys()=}, {self.in_key=}, {self.text_inputs_key=}."
            )
        return item


class TokenPadCollate:
    def __init__(self, in_key="token_ids", out_key="input_ids", pad_token_id=0):
        self.in_key = in_key
        self.out_key = out_key
        self.pad_token_id = pad_token_id

    def __call__(self, bucket_list, batch_out):
        """
        Do padding collation. pad input token_ids with padding token
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        # Collate inputs_ids and pad
        try:
            input_ids_batch = []
            attention_mask_batch = []
            token_type_ids_batch = []
            for item in bucket_list:
                input_ids = item[self.in_key]
                token_type_ids = item["token_type_ids"]
                attention_mask = [1] * len(input_ids)
                input_ids_batch.append(torch.tensor(input_ids).long())
                attention_mask_batch.append(torch.tensor(attention_mask).float())
                token_type_ids_batch.append(torch.tensor(token_type_ids).long())

            input_ids_batch = pad_sequence(
                input_ids_batch, batch_first=True, padding_value=self.pad_token_id
            )
            attention_mask_batch = pad_sequence(
                attention_mask_batch, batch_first=True, padding_value=0
            )
            token_type_ids_batch = pad_sequence(
                token_type_ids_batch, batch_first=True, padding_value=0
            )
            batch_out[self.out_key] = input_ids_batch
            batch_out["attention_mask"] = attention_mask_batch
            batch_out["token_type_ids"] = token_type_ids_batch
        except Exception as e:
            logger.error(f"TokenPadCollate Failed {e=}")
            breakpoint()


# Example
class AudioCollate:
    def __init__(
        self,
        in_key="audio",
        out_key=None,
        length_key="audio_length",
        pad_value=0.0,
        to_tensor=True,
    ):
        """
        Args:
            in_key (str): Key to fetch audio data from each sample.
            out_key (str): Key under which the padded audio is stored.
            mask_key (str): Key under which the audio mask is stored.
            pad_value (float): Value used for padding.
            to_tensor (bool): If True, returns torch.Tensors, otherwise numpy arrays.
        """
        self.in_key = in_key
        self.out_key = out_key
        self.length_key = length_key
        self.pad_value = pad_value
        self.to_tensor = to_tensor

        if self.out_key is None:
            self.out_key = self.in_key

    def __call__(self, batch_in, batch_out):

        audio_tensors = []
        lengths = []
        # Convert each audio sample to a tensor and record its length

        for item in batch_in:
            audio = torch.tensor(item[self.in_key], dtype=torch.float32)
            lengths.append(audio.shape[-1])
            audio_tensors.append(audio.squeeze(0))

        padded_audio = pad_sequence(
            audio_tensors, batch_first=True, padding_value=self.pad_value
        )
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        if not self.to_tensor:
            # Convert to numpy arrays if required
            padded_audio = padded_audio.numpy()
            lengths_tensor = lengths_tensor.numpy()

        batch_out[self.out_key] = padded_audio
        batch_out[self.length_key] = lengths_tensor


class NestListCollate:
    """Collate uttid."""

    def __init__(self, key_list="uttid", out_key=None, to_tensor=False):
        """Initialize list collate function, setup list key."""

        if isinstance(key_list, List):
            self.key_list = key_list
        elif isinstance(key_list, str):
            self.key_list = key_list.split(",")
        if out_key is None:
            self.out_key = self.key_list
        else:
            self.out_key = out_key.split(",")
            assert len(self.key_list) == len(self.out_key)

        self.to_tensor = to_tensor

    def _get_nested_value(self, dictionary, keys):
        """Retrieve a value from a nested dictionary using a list of keys."""
        for key in keys:
            if isinstance(dictionary, dict):
                dictionary = dictionary.get(key)
            else:
                logger.error(f"Error: Keys {keys}, {key} not found in dictionary.")
                return None  # If the current level is not a dictionary, return None
        return dictionary

    def __call__(self, bucket_list, batch_out):
        """
        do list collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """

        for key, out_key in zip(self.key_list, self.out_key):
            nested_keys = key.split(".")
            values = [self._get_nested_value(item, nested_keys) for item in bucket_list]
            if self.to_tensor:
                values = torch.tensor(values)
            batch_out[out_key] = values


class LyricsCollate:
    def __init__(self, in_key: str = "phrases", out_key: str = "lyrics"):
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, batch_list, batch_out):

        lyrics = []
        for item in batch_list:
            lyrics.append("\n".join([phrase.text for phrase in item[self.in_key]]))

        batch_out[self.out_key] = lyrics


class TokenCoffCollate:
    def __init__(
        self,
        in_key: str = "phoneme_tokens",
        token_key: str = "input_ids",
        coff_key: str = "coffs",
        pos_key: Optional[str] = None,  # "pos"
        # NOTE: According to the refactored dataloader's naming convention,
        # "token" stands for the token string, whereas the token id (int),
        # which is the input to the embedding module and should be collated
        # in this transform, has now been called "input id".
        # We still set the related argument names and the output's default key to be "token"
        # for compatibility reason.
        # TODO: Consider renaming the related arguments and their default values to be "input_id"
        # after switching to the refactored embedding module.
        token_output_key: str = "lyrics_tokens",
        coff_output_key: str = "lyrics_coffs",
        pos_output_key: Optional[str] = None,  # "lyrics_pos"
        max_phone_len: int = 0,  # Set default to 0 or negative to indicate dynamic padding
    ):
        """
        Args:
            in_key (str): Key for the dictionary containing token, coff, and pos data.
            token_key (str): Key for token IDs within the in_key dictionary.
            coff_key (str): Key for coefficients within the in_key dictionary.
            pos_key (Optional[str]): Key for position IDs within the in_key dictionary.
            token_output_key (str): Output key for padded/truncated token IDs.
            coff_output_key (str): Output key for padded/truncated coefficients.
            pos_output_key (Optional[str]): Output key for padded/truncated position IDs.
            max_phone_len (int): Maximum length for tokens/coffs/pos.
                                 If > 0, sequences will be padded or truncated to this length.
                                 If <= 0, sequences will be padded to the max length in the batch.
        """
        self.in_key = in_key
        self.max_phone_len = max_phone_len
        self.token_key = token_key
        self.coff_key = coff_key
        self.pos_key = pos_key
        self.token_output_key = token_output_key
        self.coff_output_key = coff_output_key
        self.pos_output_key = pos_output_key

    def _pad_or_truncate(self, tensor_list, max_len, padding_value):
        """Helper function to pad or truncate a list of tensors to max_len.

        Supports both 1D and 2D tensors. For 2D tensors, pads/truncates the first dimension.
        """
        if not tensor_list:
            return torch.empty(0)

        # Get sample tensor to determine shape and dtype
        sample_tensor = tensor_list[0]
        batch_size = len(tensor_list)

        # Determine output shape based on input tensor dimensions
        if sample_tensor.dim() == 1:
            output_shape = (batch_size, max_len)
        elif sample_tensor.dim() == 2:
            output_shape = (batch_size, max_len, sample_tensor.shape[1])
        else:
            raise ValueError(f"Unsupported tensor dimension: {sample_tensor.dim()}")

        # Pre-allocate output tensor filled with padding_value
        output_tensor = torch.full(
            output_shape,
            padding_value,
            dtype=sample_tensor.dtype,
            device=sample_tensor.device,
        )

        # Fill the output tensor
        for i, t in enumerate(tensor_list):
            current_len = t.shape[0]

            if current_len > max_len:
                # Truncate and copy
                logger.warning(
                    f"Truncating sequence at index {i} from length {current_len} to {max_len} "
                    f"for key '{self.token_output_key}'/'{self.coff_output_key}'"
                    f"{'/' + self.pos_output_key if self.pos_output_key else ''}."
                )
                output_tensor[i, :max_len] = t[:max_len]
            else:
                # Copy (no truncation needed, padding already handled by pre-allocation)
                output_tensor[i, :current_len] = t

        return output_tensor

    def __call__(self, batch_list, batch_out):
        lyrics_tokens = [
            torch.as_tensor(item[self.in_key][self.token_key], dtype=torch.long)
            for item in batch_list
        ]
        lyrics_coffs = [
            torch.as_tensor(item[self.in_key][self.coff_key]) for item in batch_list
        ]

        if self.max_phone_len > 0:
            # Pad or truncate to fixed max_phone_len
            padded_tokens = self._pad_or_truncate(
                lyrics_tokens, self.max_phone_len, padding_value=0
            )
            padded_coffs = self._pad_or_truncate(
                lyrics_coffs, self.max_phone_len, padding_value=1.0
            )
        else:
            # Pad dynamically to the max length in the batch
            padded_tokens = pad_sequence(
                lyrics_tokens, batch_first=True, padding_value=0
            )
            padded_coffs = pad_sequence(
                lyrics_coffs, batch_first=True, padding_value=1.0
            )

        batch_out[self.token_output_key] = padded_tokens
        batch_out[self.coff_output_key] = padded_coffs

        # pos is only available by using SamiPhonemeSeqPosTokenizer
        if self.pos_key is not None and self.pos_output_key is not None:
            lyrics_pos = [
                torch.as_tensor(item[self.in_key][self.pos_key]) for item in batch_list
            ]
            if self.max_phone_len > 0:
                padded_pos = self._pad_or_truncate(
                    lyrics_pos, self.max_phone_len, padding_value=-1
                )
            else:
                padded_pos = pad_sequence(
                    lyrics_pos, batch_first=True, padding_value=-1
                )
            batch_out[self.pos_output_key] = padded_pos


class TokenCollate:
    def __init__(
        self,
        in_key="target_token_ids",
        out_key=None,
        length_key="target_tokens_length",
        pad_value=0,
        to_tensor=True,
    ):
        """
        Args:
            in_key (str): Key to fetch target_token_ids data from each sample.
            out_key (str): Key under which the padded target_token_ids is stored.
            mask_key (str): Key under which the target_token mask is stored.
            pad_value (float): Value used for padding.
            to_tensor (bool): If True, returns torch.Tensors, otherwise numpy arrays.
        """
        self.in_key = in_key
        self.out_key = out_key
        self.length_key = length_key
        self.pad_value = pad_value
        self.to_tensor = to_tensor
        if self.out_key is None:
            self.out_key = self.in_key

    def __call__(self, batch_in, batch_out):
        """
        do token collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        """
        token_tensors = []
        lengths = []
        # Convert each token sample to a tensor and record its length
        for item in batch_in:
            token = torch.tensor(item[self.in_key], dtype=torch.long)
            lengths.append(token.shape[-1])
            token_tensors.append(token)
        padded_token = pad_sequence(
            token_tensors, batch_first=True, padding_value=self.pad_value
        )
        lengths_tensor = torch.tensor(lengths, dtype=torch.int64).unsqueeze(
            1
        )  # shape: (batch_size, 1)
        if not self.to_tensor:
            # Convert to numpy arrays if required
            padded_token = padded_token.numpy()
            lengths_tensor = lengths_tensor.numpy()
        batch_out[self.out_key] = padded_token
        batch_out[self.length_key] = lengths_tensor


_SkipNumInfo = Dict[int, Dict[str, int]]  # [pid, [name, skip_count]]


class _WokerInfoItem(TypedDict):
    pid: int
    num_workers: int
    rank: int
    world_size: int
    path_idx: str
    crs_filename: str
    uttid: str
    skip_num: Optional[Dict[str, int]]


class _WokerInfoBatch(TypedDict):
    pid: List[int]
    path_idx: List[str]
    crs_filename: List[str]
    uttid: List[str]
    skip_num: Optional[_SkipNumInfo]


class WorkerInfoCollate:
    def __init__(self, in_key="worker_info"):
        self.in_key = in_key

    def __call__(self, batch_list, batch_out):
        pids = []
        path_idxs = []
        crs_filenames = []
        uttids = []
        skip_nums = {}

        for item in batch_list:
            worker_info: _WokerInfoItem = item[self.in_key]
            pid = worker_info["pid"]
            pids.append(pid)
            path_idxs.append(worker_info["path_idx"])
            crs_filenames.append(worker_info["crs_filename"])
            uttids.append(worker_info["uttid"])
            skip_num_stat = worker_info["skip_num"]
            if pid not in skip_nums:
                skip_nums[pid] = skip_num_stat
            else:
                if sum(skip_nums[pid].values()) > sum(skip_num_stat.values()):
                    continue
                else:
                    skip_nums[pid] = skip_num_stat
        # Update skip nums status from dataloader workers
        # May miss some workers if some workers not offered data
        worker_info_batch = _WokerInfoBatch(
            pid=pids,
            path_idx=path_idxs,
            crs_filename=crs_filenames,
            uttid=uttids,
            skip_num=skip_nums,
        )
        batch_out["worker_info"] = worker_info_batch


class TokenNumPerCategory:
    def __init__(
        self,
        num_total_tokens_key="num_total_tokens",
        category_key="task",
        out_key="num_token_per_category",
    ):
        self.num_total_tokens_key = num_total_tokens_key
        self.category_key = category_key
        self.out_key = out_key

    def __call__(self, batch_in, batch_out):
        num_token_per_category = {}
        for item in batch_in:
            if self.num_total_tokens_key in item and self.category_key in item:
                category = item[self.category_key]
                num_total_tokens = item[self.num_total_tokens_key]
                if category in num_token_per_category:
                    num_token_per_category[category] += num_total_tokens
                else:
                    num_token_per_category[category] = num_total_tokens
        batch_out[self.out_key] = num_token_per_category
