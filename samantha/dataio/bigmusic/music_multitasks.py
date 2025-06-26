import random

from mariana.data.audio.multitask_s2t2u import (  # noqa: F401, used in config to ensure this is force-imported
    CTS2T2UPreSchemaParser,
    registry_pre_schema,
)
from mariana.utils.audio.audio_logger import AudioLogger

logger = AudioLogger()

# aligned with bbpe155k-v6.4.3-ml.pret
moe_bos = "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
moe_eos = "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"


def slice_umm_string(umm_string, slice_fraction):
    umm_tokens = umm_string.split("><")
    slice_index = int(len(umm_tokens) * slice_fraction)
    head_tokens, tail_tokens = umm_tokens[:slice_index], umm_tokens[slice_index:]
    head_string = "><".join(head_tokens)
    tail_string = "><".join(tail_tokens)
    return head_string, tail_string


def maybe_get_meta_as_prompt(item):
    if "prompt_prefix" in item:
        prompt_prefix = item["prompt_prefix"]
    elif "standard_music_meta" in item:
        prompt_prefix = str(item["standard_music_meta"])
    else:
        prompt_prefix = ""
    return prompt_prefix


def multitask_chat_template(
    caption=None, lyrics=None, umm_string=None, task=None, system_prompt=None
):
    target = None
    if task == "Lyrics2Song":
        instruct = ""
        instruct += f"<lyrics>{lyrics}</lyrics>\n"
        instruct += f"<caption>{caption}</caption>\n"
        inputs = "".join(
            [moe_bos, f"system\n{system_prompt}", f"user\n{instruct}", "assistant\n"]
        )
        if umm_string is not None:
            target = f"<audio>{umm_string}</audio>{moe_eos}"

    elif task == "MusicCaption":
        instruct = f"<audio>{umm_string}</audio>"
        inputs = "".join(
            [moe_bos, f"system\n{system_prompt}", f"user\n{instruct}", "assistant\n"]
        )
        if caption is not None:
            target = f"<caption>{caption}<caption>{moe_eos}"

    else:
        raise NotImplementedError
    return inputs, target


@registry_pre_schema(task="CT_MusicPureText")
def CTMusicPureTextParser(item, *_args, **_kwargs):
    # Data Schema: {"content_split": xx, "docid": xx, "chunk_id": xx}
    new_item = {}
    item["uttid"] = f"{item.get('docid', 'doc')}_{item.get('chunk_id', 'chunk')}"
    new_item["uttid"] = item["uttid"]
    if "content_split" not in item or not item["content_split"].strip():
        logger.warning(f"Find empty content_split in file:{item['crs_filename']},skip")
        return None
    new_item["inputs"] = [{"text": item["content_split"], "type": "text"}]
    return item, new_item


@registry_pre_schema(task="CT_MusicTTS")
def CTMusicTTSParser(item, *_args, **kwargs):
    """
    prompt:
        - item['text']
    response:
        - item['umm_string']
    """
    # Prepare system prompt
    system_prompt = item["crs_system_prompt"]

    # Prepare user instruct
    text = item["text"]
    if len(text) <= 10:
        return None
    instruct = f"<speech_text>{text}</speech_text>\n"

    # Prepare assistant response
    umm_string = item["umm_string"]
    response = f"<speech>{umm_string}</speech>"
    inputs = [moe_bos, f"system\n{system_prompt}", f"user\n{instruct}", "assistant\n"]
    target = [response, moe_eos]

    new_item = {"inputs": []}
    new_item["inputs"].append(
        {"text": "".join(inputs), "type": "prompt", "loss_flag": 0}
    )
    new_item["inputs"].append({"text": "".join(target), "type": "text", "loss_flag": 1})
    new_item["task"] = kwargs.get("task", "CT_MusicTTS")
    new_item["uttid"] = item.get("uttid", "null_uttid")
    return item, new_item


@registry_pre_schema(task="Lyrics2Song")
def Lyrics2Song_Parser(item, *_args, **kwargs):
    """
    prompt:
        - item['lyrics']
        - item['prompt_prefix']
    response:
        - item['umm_string']
    """
    try:
        # Prepare system prompt
        system_prompt = item["crs_system_prompt"]

        # Prepare user instruct
        prompt_prefix = maybe_get_meta_as_prompt(item)
        lyrics = item.get("lyrics_prefix", "")
        if len(lyrics) == 0:
            return None

        # Prepare assistant response
        umm_string = item["umm_string"]
        inputs, target = multitask_chat_template(
            caption=prompt_prefix,
            lyrics=lyrics,
            umm_string=umm_string,
            task="Lyrics2Song",
            system_prompt=system_prompt,
        )

        new_item = {"inputs": []}
        new_item["inputs"].append({"text": inputs, "type": "prompt", "loss_flag": 0})
        new_item["inputs"].append({"text": target, "type": "text", "loss_flag": 1})
        new_item["task"] = kwargs.get("task", "Lyrics2Song")
        new_item["uttid"] = item.get("uttid", "null_uttid")
    except Exception as e:
        logger.warning(f"{e} {item.keys()=}")
        return None
    return item, new_item


@registry_pre_schema(task="MusicCaption")
def MusicCaption_Parser(item, *_args, **kwargs):
    """
    prompt:
        - item['umm_string']
    response:
        - item['prompt_prefix']
    """
    try:
        prompt_prefix = maybe_get_meta_as_prompt(item)
        # Prepare system prompt
        system_prompt = item["crs_system_prompt"]
        # Prepare user instruct
        umm_string = item["umm_string"]

        # Random slice umm_string, partial generation will be easier to train
        slice_fraction = random.uniform(0.1, 0.5)
        head_umm, _ = slice_umm_string(umm_string, slice_fraction)

        inputs, target = multitask_chat_template(
            caption=prompt_prefix,
            lyrics=None,
            umm_string=head_umm,
            task="MusicCaption",
            system_prompt=system_prompt,
        )

        new_item = {"inputs": []}
        new_item["inputs"].append({"text": inputs, "type": "prompt", "loss_flag": 0})
        new_item["inputs"].append({"text": target, "type": "text", "loss_flag": 1})
        new_item["task"] = kwargs.get("task", "MusicCaption")
        new_item["uttid"] = item.get("uttid", "null_uttid")
    except Exception as e:
        logger.warning(f"{e} {item.keys()=}")
        return None
    return item, new_item


@registry_pre_schema(task="TokenASR")
def TokenASR_Parser(item, *_args, **kwargs):
    """
    prompt:
        - item['umm_string']
    response:
        - item['lyrics']
    """
    try:
        # Prepare system prompt
        system_prompt = item["crs_system_prompt"]
        # Prepare user instruct
        umm_string = item["umm_string"]
        instruct = f"<audio>{umm_string}</audio>"
        # Prepare assistant response
        lyrics = item.get("lyrics_prefix", "")
        if len(lyrics) == 0:
            return None
        response = f"<lyrics>{lyrics}</lyrics>"
        inputs = [
            moe_bos,
            f"system\n{system_prompt}",
            f"user\n{instruct}",
            "assistant\n",
        ]
        target = [response, moe_eos]

        new_item = {"inputs": []}
        new_item["inputs"].append(
            {"text": "".join(inputs), "type": "prompt", "loss_flag": 0}
        )
        new_item["inputs"].append(
            {"text": "".join(target), "type": "text", "loss_flag": 1}
        )
        new_item["task"] = kwargs.get("task", "TokenASR")
        new_item["uttid"] = item.get("uttid", "null_uttid")
    except Exception as e:
        logger.warning(f"{e} {item.keys()=}")
        return None
    return item, new_item


@registry_pre_schema(task="AudioContinuation")
def AudioContinuation_Parser(item, *_args, **kwargs):
    """
    prompt:
        - item['umm_string'][:prompt_length]
    response:
        - item['umm_string'][prompt_length:]
    """
    try:
        # Prepare system prompt
        system_prompt = item["crs_system_prompt"]
        umm_string = item["umm_string"]
        if len(umm_string) < 10:
            return None

        inputs = [moe_bos, f"system\n{system_prompt}"]
        target = [f"<audio>{umm_string}</audio>", moe_eos]

        new_item = {"inputs": []}
        new_item["inputs"].append(
            {"text": "".join(inputs), "type": "prompt", "loss_flag": 0}
        )
        new_item["inputs"].append(
            {"text": "".join(target), "type": "text", "loss_flag": 1}
        )
        new_item["task"] = kwargs.get("task", "AudioContinuation")
        new_item["uttid"] = item.get("uttid", "null_uttid")
    except Exception as e:
        logger.warning(f"{e} {item.keys()=}")
        return None
    return item, new_item


@registry_pre_schema(task="Prompt2Song")
def Prompt2Song_Parser(item, *_args, **kwargs):
    """
    prompt:
        - item['prompt_prefix']
    response:
        - item['umm_string']
    """
    try:
        # Prepare system prompt
        system_prompt = item["crs_system_prompt"]

        # Prepare user instruct
        prompt_prefix = maybe_get_meta_as_prompt(item)
        instruct = ""
        instruct += f"<caption>{prompt_prefix}</caption>\n"

        # Prepare assistant response
        umm_string = item["umm_string"]

        # Random slice umm_string, partial generation will be easier to train
        slice_fraction = random.uniform(0.1, 1.0)
        head_umm, _ = slice_umm_string(umm_string, slice_fraction)
        response = f"<audio>{head_umm}</audio>"
        inputs = [
            moe_bos,
            f"system\n{system_prompt}",
            f"user\n{instruct}",
            "assistant\n",
        ]
        target = [response, moe_eos]

        new_item = {"inputs": []}
        new_item["inputs"].append(
            {"text": "".join(inputs), "type": "prompt", "loss_flag": 0}
        )
        new_item["inputs"].append(
            {"text": "".join(target), "type": "text", "loss_flag": 1}
        )
        new_item["task"] = kwargs.get("task", "Prompt2Song")
        new_item["uttid"] = item.get("uttid", "null_uttid")
    except Exception as e:
        logger.warning(f"{e} {item.keys()=}")
        return None
    return item, new_item


@registry_pre_schema(task="Tag2Prompt")
def Tag2Prompt_Parser(item, *_args, **kwargs):
    """
    prompt:
        - item['standard_music_meta']['tagging_model']
    response:
        - item['standard_music_meta']['web] and item['standard_music_meta']['human_annotation']
    """
    try:
        # Prepare system prompt
        system_prompt = item["crs_system_prompt"]

        # Prepare user instruct
        standard_music_meta = item["standard_music_meta"]
        tagging_model = str(standard_music_meta.pop("tagging_model"))
        residual = str(standard_music_meta)
        instruct = f"<tag>{tagging_model}</tag>\n"

        # Prepare assistant response
        response = f"<caption>{residual}</caption>"

        inputs = [
            moe_bos,
            f"system\n{system_prompt}",
            f"user\n{instruct}",
            "assistant\n",
        ]
        target = [response, moe_eos]

        new_item = {"inputs": []}
        new_item["inputs"].append(
            {"text": "".join(inputs), "type": "prompt", "loss_flag": 0}
        )
        new_item["inputs"].append(
            {"text": "".join(target), "type": "text", "loss_flag": 1}
        )
        new_item["task"] = kwargs.get("task", "Tag2Prompt")
        new_item["uttid"] = item.get("uttid", "null_uttid")
    except Exception as e:
        logger.warning(f"{e} {item.keys()=}")
        return None
    return item, new_item
