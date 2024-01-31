import torch
from transformers import BloomTokenizerFast, BloomForCausalLM


def preprocess_index(index_item, *_, **__):
    return index_item["text"], len(index_item["text"])


@torch.no_grad()
def process_batch(model, batch, device, max_length=2048, *_, **__):
    if not batch:
        yield from batch

    tokenizer = model["tokenizer"]
    encoder = model["encoder"]
    batch_tokens = tokenizer(
        batch, truncation=True, max_length=max_length, return_tensors="pt", padding=True
    )
    for k, v in batch_tokens.items():
        batch_tokens[k] = v.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        batch_embedding = (
            encoder(**batch_tokens, return_dict=True, output_hidden_states=True)
            .hidden_states[-1]
            .cpu()
            .numpy()
        )
    for embedding, mask in zip(batch_embedding, batch_tokens["attention_mask"]):
        yield embedding[-mask.sum() :]


def load_model(device, model_path, *_, **__):
    return {
        "tokenizer": BloomTokenizerFast.from_pretrained(model_path),
        "encoder": BloomForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval(),
    }
