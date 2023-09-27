import argparse
import textwrap


def parse_data(input_fname):
    with open(args.input_fname, "r") as f:
        for line in f:
            prompt = line.strip()
            break
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("input_fname")
    parser.add_argument("output_fname")
    args = parser.parse_args()

    prompt = parse_data(args.input_fname)
    prompt_str = "\n".join(textwrap.wrap(prompt, 40, break_long_words=False))
    with open(args.output_fname, "w") as fw:
        to_write = f"{args.title}\n\n" + f"Prompt: {prompt_str}"
        fw.write(f"{to_write}\n")