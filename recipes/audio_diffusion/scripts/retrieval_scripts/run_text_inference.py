import argparse
import logging

import torch
from mulan_infer import create_mulan_model
from webdataset.writer import TarWriter

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def tokenize(text, tokenizer, device):
    token_outputs = tokenizer(text, return_tensors="pt")
    token_outputs["input_ids"] = token_outputs["input_ids"].to(device)
    token_outputs["token_type_ids"] = token_outputs["token_type_ids"].to(device)
    token_outputs["attention_mask"] = token_outputs["attention_mask"].to(device)
    return token_outputs


def get_mulan_model(device):
    ckpt_path = (
        "/mnt/bn/audio-diffusion/pretrained_models/mulan/"
        "young-mulan-shortform-step=028000-median_rank_1=149-kaggle_merged.pt"
    )
    logger.info("Initializing MuLan model from %s...", ckpt_path)
    model = create_mulan_model(ckpt_path=ckpt_path, device=device)
    return model


def main(args):
    device = torch.device("cuda:0")
    mulan_model = get_mulan_model(device)

    logger.info(
        "Reading text prompts from %s, outputting to %s",
        args.text_fname,
        args.output_tar_fname,
    )
    with open(args.text_fname, "r") as fin:
        fout = TarWriter(open(args.output_tar_fname, "wb"))
        count = 0
        for line in fin:
            text = line.strip()
            logger.info("Processing: %s", text)
            token_outputs = tokenize(text, mulan_model.tokenizer, device)
            text_emb = mulan_model.text_encoder(**token_outputs)[0]
            logger.info("text_emb: %s %s", text_emb.shape, text_emb)
            text_emb = text_emb.detach().cpu().float().numpy()
            fout.write(
                {"__key__": f"text_{count}", "text.txt": text, "emb.npy": text_emb}
            )
            count += 1
        fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument("text_fname", help="Text prompt file, 1 prompt/line")
    args = parser.add_argument("output_tar_fname", help="Output tar path")
    args = parser.parse_args()
    main(args)
