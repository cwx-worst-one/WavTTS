import argparse
import json
import textwrap


def parse_data(input_fname):
    data = json.load(open(input_fname, "r"))
    lyrics = ""
    if data.get("lyrics") is not None:
        ly_list = []
        for x in data.get("lyrics").split("\n"):
            for y in textwrap.wrap(x, 42, break_long_words=False):
                ly_list.append(y)
                ly_list.append("\n")
            ly_list.append("\n")
        lyrics = "".join(ly_list)
    style_text = ""
    if data.get("style_text") is not None:
        style_text = '\n'.join(
            textwrap.wrap(
                data.get("style_text"), 42, break_long_words=False
            )
        )
    return lyrics, style_text, data["index"]["absolute_idx"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname")
    parser.add_argument("output_fname")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    lyrics, prompt, idx = parse_data(args.input_fname)
    # lyrics is already wrapped
    with open(args.output_fname, "w") as fw:
        title = idx if args.title is None else args.title
        to_write = f"{title}:\n{prompt}\n\n\n" + f"{lyrics}"
        fw.write(f"{to_write}\n")