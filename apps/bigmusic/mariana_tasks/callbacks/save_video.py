import json
import os
import re
import textwrap
from pathlib import Path

import pytorch_lightning as pl

from recipes.bigmusic.callbacks.common_callbacks import UploadToEasyCycleCallback

from .utils import sync_all_ranks


def _empty_dict() -> dict:
    return {}


class SaveVideoCallback(pl.Callback):
    def __init__(
        self,
        output_dir=None,
        index_key: str = "index",
        additional_keys: dict = _empty_dict(),
        fontfile: str = "/usr/share/fonts/truetype/arphic/ukai.ttc",
        upload: bool = True,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.index_key = index_key
        self.additional_keys = additional_keys
        self.fontfile = fontfile
        self.upload = upload

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not sync_all_ranks(trainer, self.output_dir, self.__class__.__name__):
            return
        save_video(
            input_results_dir=self.output_dir,
            output_video_dir=self.output_dir,
            index_key=self.index_key,
            additional_keys=self.additional_keys,
            upload=self.upload,
            fontfile=self.fontfile,
        )


def rm_tree(pth: Path) -> None:
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


# ---------------------------------------------
#                 Formatters
# ---------------------------------------------


def format_regular_text(key: str, text: str, max_width: int) -> list[str]:
    lines = textwrap.wrap(text, max_width, break_long_words=False)
    for i in range(1, len(lines)):
        lines[i] = "    " + lines[i]  # add indentation
    return [f"{key}: "] + lines


def format_float(key: str, text: float, _max_width: int) -> list[str]:
    return [f"{key}: {text:.2f}"]


def format_int(key: str, text: float, _max_width: int) -> list[str]:
    return [f"{key}: {int(text)}"]


def format_xml(_key: str, text: str, max_width: int) -> list[str]:
    """
    Wrap XML-like tags into lines with specified width.

    Args:
        text (str): The input text with XML tags
        width (int): Maximum width per line

    Returns:
        list: List of strings, each representing a line
    """
    # Split into individual tags
    tags = re.findall(r"<[^>]+>[^<]*</[^>]+>", text)

    lines = []
    current_line = ""

    for tag in tags:
        # Check if adding this tag would exceed width
        if current_line and len(current_line + tag) > max_width:
            # Start a new line
            lines.append(current_line.rstrip())
            current_line = tag
        else:
            # Add to current line
            current_line += tag

    # Add the last line if it has content
    if current_line:
        lines.append(current_line.rstrip())

    return lines


def format_lyrics(_key: str, lyrics: str, max_width: int) -> list[str]:
    lyrics = lyrics.split("\n")

    lyrics_list = []
    for x in lyrics:
        for wrap_idx, y in enumerate(
            textwrap.wrap(x, max_width, break_long_words=False)
        ):
            if wrap_idx > 0:
                lyrics_list.append("\t")
            lyrics_list.append(y)
            lyrics_list.append("\n")
    return "".join(lyrics_list).split("\n")


def format_style_text(_key: str, style_text: list[list[str]], _max_width: int) -> list[str]:
    if not style_text:
        return None
    if isinstance(style_text, str):
        return style_text
    return ["| ".join([", ".join(sublist) for sublist in style_text])]


TEXT_FORMATTERS = {
    "regular": format_regular_text,
    "float": format_float,
    "int": format_int,
    "lyrics": format_lyrics,
    "xml": format_xml,
    "style_text": format_style_text,
}


# ---------------------------------------------


def format_video_text(
    metadata: dict,
    index_key: str = "index",
    additional_keys: dict = _empty_dict(),
    max_width: int = 60,
) -> str:
    index = metadata[index_key]
    additional_text = {k: metadata.get(k, "") for k in additional_keys}
    lines = [str(index)]
    for k, v in additional_text.items():
        formatter = TEXT_FORMATTERS[additional_keys[k]]
        lines.extend(formatter(k, v, max_width))
    video_text = "\n".join(lines)
    return video_text


def resize_video_text(video_text: str):
    num_lines = len(video_text.split("\n"))

    if num_lines < 18:
        fontsize = 22
        line_spacing = 14
    if 18 <= num_lines < 30:
        fontsize = 20
        line_spacing = 4
    if 30 <= num_lines < 36:
        fontsize = 18
        line_spacing = 3
    if 36 <= num_lines < 40:
        fontsize = 16
        line_spacing = 2
    if 40 <= num_lines < 48:
        fontsize = 14
        line_spacing = 1
    if num_lines >= 48:
        fontsize = 12
        line_spacing = 0

    return video_text, fontsize, line_spacing


def save_video(
    input_results_dir: str,
    output_video_dir: str,
    index_key: str = "index",
    additional_keys: dict = _empty_dict(),
    remove_segments: bool = True,
    upload: bool = True,
    fontfile: str = "/usr/share/fonts/truetype/arphic/ukai.ttc",
):
    colors = ["green", "blue", "brown"]
    output_video_dir = Path(output_video_dir)
    output_video_dir_tmp = (
        output_video_dir.parent / f"{output_video_dir.stem}.video_tmp"
    )
    output_video_dir_tmp.mkdir(exist_ok=True, parents=True)

    audio_format = "wav"
    generated_output_fps = sorted(list(Path(input_results_dir).glob(f"**/*.generated.{audio_format}")))
    for idx, generated_output_fp in enumerate(generated_output_fps):
        audio_fp = generated_output_fp
        metadata_fp = str(generated_output_fp).replace(
            f"generated.{audio_format}", "metadata.json"
        )

        output_video_fp = (
            output_video_dir_tmp / generated_output_fp.with_suffix(".mp4").name
        )
        output_text_fp = (
            output_video_dir_tmp / generated_output_fp.with_suffix(".txt").name
        )
        with open(metadata_fp, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(output_text_fp, "w", encoding="utf-8") as f:
            video_text = format_video_text(
                metadata=metadata,
                index_key=index_key,
                additional_keys=additional_keys,
                max_width=60,
            )
            video_text, fontsize, line_spacing = resize_video_text(video_text)
            f.write(video_text)
        color = colors[idx % len(colors)]
        cmd = f'ffmpeg -y -v 0 -f lavfi -i color=c={color}:s=800x800:d=0.5 -i "{audio_fp}" -c:a aac -vf "drawtext=fontfile={fontfile}:fontsize={fontsize}:line_spacing={line_spacing}:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile={output_text_fp}" "{output_video_fp}" -v 0'
        os.system(cmd)

    # concat output videos
    video_output_fp = output_video_dir / f"{output_video_dir.name}.mp4"
    cmd_concat = f"cd {output_video_dir_tmp} && find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -v 0 -f concat -i fl.txt -c copy output.mp4; rm fl.txt"
    os.system(cmd_concat)
    (output_video_dir_tmp / "output.mp4").rename(video_output_fp)
    if remove_segments:
        rm_tree(Path(output_video_dir_tmp))

    if upload and Path(video_output_fp).exists():
        url = UploadToEasyCycleCallback.upload_file(video_output_fp)
        print("Saved video:", url)

        # update inference_params
        metadata = None
        for meta_fp in list(Path(output_video_dir).glob("inference_params.*.json")):
            with open(meta_fp, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
                if metadata is None:
                    metadata = _metadata
                else:
                    if "output_paths" in metadata:
                        metadata["output_paths"] += _metadata.get("output_paths", [])
                    else:
                        metadata["output_paths"] = _metadata.get("output_paths", [])
        metadata["demo_video_url"] = url
        meta_fp = os.path.join(output_video_dir, "inference_params.json")
        with open(meta_fp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    return video_output_fp
