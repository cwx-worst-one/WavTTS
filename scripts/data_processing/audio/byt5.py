import torch
from transformers import AutoTokenizer, T5EncoderModel


def preprocess_index(index_item, *_, **__):
    return index_item["text"], len(index_item["text"])


@torch.no_grad()
def process_batch(model, batch, device, max_length=6000, *_, **__):
    if not batch:
        yield from batch

    tokenizer = model["tokenizer"]
    encoder = model["encoder"]
    batch_tokens = tokenizer(
        batch, truncation=True, max_length=max_length, return_tensors="pt", padding=True
    )
    for k, v in batch_tokens.items():
        batch_tokens[k] = v.to(device)

    batch_embedding = encoder(**batch_tokens)["last_hidden_state"].cpu().numpy()
    for embedding, mask in zip(batch_embedding, batch_tokens["attention_mask"]):
        yield embedding[: mask.sum()]


def load_model(device, model_path, *_, **__):
    return {
        "tokenizer": AutoTokenizer.from_pretrained(model_path),
        "encoder": T5EncoderModel.from_pretrained(model_path).to(device),
    }
