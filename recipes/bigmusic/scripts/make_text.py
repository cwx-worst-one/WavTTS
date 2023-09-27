import argparse
import textwrap


def parse_data(input_fname):
    lyrics = []
    prompt = None
    wer = None
    in_lyrics = False
    with open(args.input_fname, "r") as f:
        for line in f:
            if line.startswith("Lyrics:"):
                in_lyrics = True
                line = line.replace("Lyrics:", "").strip()
            if line.startswith("Prompt:"):
                in_lyrics = False
                prompt = line.replace("Prompt:", "").strip()
            if line.startswith("WER:"):
                wer = 100 * float(line.replace("WER:", "").strip())
            if in_lyrics:
                lyrics.append(line.strip())
    return lyrics, prompt, wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("input_fname")
    parser.add_argument("output_fname")
    args = parser.parse_args()

    lyrics, prompt, wer = parse_data(args.input_fname)
    ly_list = []
    for x in lyrics:
        ly_list.extend(textwrap.wrap(x, 40, break_long_words=False))
    ly_str = "\n".join(ly_list)
    with open(args.output_fname, "w") as fw:
        to_write = f"{args.title} (CER={wer:.2f})\n\n" + f"Prompt: {prompt}\n\n" + f"{ly_str}"
        fw.write(f"{to_write}\n")