from functools import partial
from pathlib import Path

import torch
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, RandomSampler

from recipes.byteformers_example.callbacks.generate_text import (  # noqa: F401
    GenerateText,
)
from recipes.byteformers_example.data import TokenDataset
from recipes.byteformers_example.modules.language_modeling import LanguageModelingModule
from recipes.byteformers_example.tokenizer import LlamaTokenizer
from samantha.callbacks.gradient_noise_scale_logger import GradientNoiseScaleLogger
from samantha.byteformers.models.llama import LlamaBlock
from samantha.byteformers import FabricTrainer
from samantha.utils.checkpoints_utils.convert_llama import convert_meta_llama_weights

if __name__ == "__main__":
    seq_len = 2048
    batch_size = 1
    micro_batch_size = 1
    n_workers = 0
    warmup_steps = 100
    max_steps = 5000
    learning_rate = 1e-3
    gradient_clip_val = 1.0
    weight_decay = 1e-1
    betas = (0.99, 0.9999)
    dropout = 0

    input_file_path = "input.txt"
    with open(input_file_path, "r") as f:
        text = f.read()

    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9) :]

    # Train our own tokenizer:
    # LlamaTokenizer.train(input_file_path, ".", vocab_size=vocab_size)

    # Or load a pre-trained tokenizer (e.g., LLaMA's)
    tokenizer = LlamaTokenizer("tokenizer.model")

    TokenDataset.prepare(train_data, tokenizer, "train.bin")
    TokenDataset.prepare(val_data, tokenizer, "valid.bin")

    train_dataset = TokenDataset("train.bin", seq_len=seq_len)
    val_dataset = TokenDataset("valid.bin", seq_len=seq_len)

    random_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=random_sampler,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True
    )

    # convert LLaMA model:
    output_dir = "output"
    converted_model = convert_meta_llama_weights("downloads/llama", output_dir)
    torch.save(converted_model, Path(output_dir, "llama.pt"))

    vocab_size = tokenizer.vocab_size
    model_name = "7B"
    module = LanguageModelingModule(
        model_name=model_name,
        vocab_size=vocab_size,
        seq_len=seq_len,
        weight_decay=weight_decay,
        betas=betas,
        learning_rate=learning_rate,
        warmup_tokens=warmup_steps * seq_len,
        final_tokens=max_steps * seq_len,
    )

    state_dict = torch.load(Path(output_dir, model_name, "llama.pt"))
    module.model.load_state_dict(state_dict)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={LlamaBlock}
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=LlamaBlock
    )

    gns_logger = GradientNoiseScaleLogger(
        module.model,
        batch_size_small=batch_size,
        n_batches=10,
        is_pipe_parallel=False,
        fdsp_strategy=True,
    )
    logger = TensorBoardLogger("logs")
    torch.set_float32_matmul_precision("high")
    trainer = FabricTrainer(
        strategy=strategy,
        devices="auto",
        accelerator="auto",
        precision="bf16-mixed",
        max_steps=max_steps,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=micro_batch_size,
        callbacks=[gns_logger],
    )

    # module = torch.compile(module)
    trainer.fit(module, train_loader, val_loader)
