import os
import argparse


emo_map = {
    "angry": "生气",
    "coldness": "冷漠",
    "depressed": "沮丧",
    "excited": "激动",
    "fear": "恐惧",
    "happy": "开心",
    "hate": "厌恶",
    "sad": "悲伤",
    "surprised": "惊讶",
    "neutral": "中立",
}


def get_syn_texts(text_path):
    texts = {}
    with open(text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip().split("|")

            if len(line) == 2:
                style = line[0].strip()
                text = line[1].strip()
            elif len(line) == 3:
                style = line[1].strip()
                text = line[2].strip()
            else:
                raise ValueError("no such format")

            if style in texts:
                texts[style].append(text)
            else:
                texts[style] = [text]
    return texts


def get_meta(text_path, prompt_path, out_path):
    syn_texts = get_syn_texts(text_path)

    emo_prompts = {}
    neutral_prompts = {}
    with open(prompt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            fid, emotion, prompt_text, prompt_wav_path, _ = line

            cur = None
            for emo, emo_cn in emo_map.items():
                if emo in emotion:
                    cur = emo_cn
                    break

            if cur == "中立":
                if cur in neutral_prompts:
                    neutral_prompts[cur].append((fid, prompt_text, prompt_wav_path))
                else:
                    neutral_prompts[cur] = [(fid, prompt_text, prompt_wav_path)]
            else:
                if cur in emo_prompts:
                    emo_prompts[cur].append((fid, prompt_text, prompt_wav_path))
                else:
                    emo_prompts[cur] = [(fid, prompt_text, prompt_wav_path)]

    fo = open(out_path, "w", encoding="utf-8")
    idx = 0
    print(syn_texts)
    for style_tag, text in syn_texts.items():
        for text_ in text:
            if style_tag == "中立": 
                if style_tag not in neutral_prompts :
                    continue

                if not neutral_prompts[style_tag]:
                    continue

                for pid, syn_text, syn_wav_path in neutral_prompts[style_tag]:
                    fid = f"{idx:04d}_{style_tag}_{pid}"
                    fo.write(f"{fid}|{syn_text}|{syn_wav_path}|{text_.strip()}\n")

            else:
                if style_tag not in emo_prompts :
                    continue

                if not emo_prompts[style_tag]:
                    continue
                for pid, syn_text, syn_wav_path in emo_prompts[style_tag]:
                    fid = f"{idx:04d}_{style_tag}_{pid}"
                    fo.write(f"{fid}|{syn_text}|{syn_wav_path}|{text_.strip()}\n")

                for pid, syn_text, syn_wav_path in neutral_prompts["中立"]:
                    fid = f"{idx:04d}_中立_{pid}"
                    fo.write(f"{fid}|{syn_text}|{syn_wav_path}|{text_.strip()}\n")

            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", required=True, type=str)
    parser.add_argument("--prompt_path", required=True, type=str)
    parser.add_argument("--out_path", required=True, type=str)
    args = parser.parse_args()

    get_meta(args.text_path, args.prompt_path, args.out_path)


if __name__ == "__main__":
    main()
