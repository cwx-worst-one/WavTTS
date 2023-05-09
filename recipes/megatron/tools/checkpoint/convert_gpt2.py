import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import shard_checkpoint, WEIGHTS_INDEX_NAME


tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
]

megatron_to_transformers_name = {
    "self_attention.dense": "attn.c_proj",
    "mlp.dense_h_to_4h": "mlp.c_fc",
    "mlp.dense_4h_to_h": "mlp.c_proj",
}


def get_by_dot_path(state_dict, dot_path):
    """
    Get the value in the state dict by the dot path.
    
    Args:
        state_dict (dict): the state dict
        dot_path (str): the dot path
    """
    keys = dot_path.split(".")
    value = state_dict
    for i, key in enumerate(keys):
        if key not in value:
            err_dotpath = ".".join(keys[:i + 1])
            possible_keys = list(value.keys())
            raise KeyError(
                f"Cannot find key `{err_dotpath}` in the state dict. "
                f"Possible keys: {possible_keys}"
            )
        value = value[key]
    return value


def get_megatron_sharded_states(ckpt_folder, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from Megatron-LM checkpoint based on the provided 
    tensor parallel size, pipeline parallel size and pipeline parallel rank.

    Args:
        ckpt_folder (str): the folder path of the Megatron-LM checkpoint
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(ckpt_folder, sub_dir_name))[0]
        checkpoint_path = os.path.join(ckpt_folder, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_megatron_args(ckpt_folder):
    """
    Get the rank 0 checkpoint from Megatron-LM checkpoint and extract the
    arguments.

    Args:
        ckpt_folder (str): the folder path of the Megatron-LM checkpoint
    """
    possible_rank0_dirnames = ["mp_rank_00", "mp_rank_00_000"]
    rank0_dir_name = None
    for sub_dir_name in possible_rank0_dirnames:
        if os.path.exists(os.path.join(ckpt_folder, sub_dir_name)):
            rank0_dir_name = sub_dir_name
            break
    if rank0_dir_name is None:
        raise ValueError("Cannot find rank 0 checkpoint in the provided folder.")
    checkpoint_name = os.listdir(os.path.join(ckpt_folder, rank0_dir_name))[0]
    checkpoint_path = os.path.join(ckpt_folder, rank0_dir_name, checkpoint_name)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in state_dict:
        raise ValueError("Cannot find key `args` in the rank 0 checkpoint.")
    megatron_args = state_dict["args"]
    return megatron_args


def main():
    args = parse_args()
    ckpt_folder = args.ckpt_folder
    output_path = args.output_folder
    max_shard_size = args.max_shard_size

    megatron_args = get_megatron_args(ckpt_folder)
    os.makedirs(output_path, exist_ok=True)

    # Get vocab size and activation function from args
    vocab_size = megatron_args.padded_vocab_size
    if megatron_args.bias_gelu_fusion:
        activation_function = "gelu_fast"
    elif megatron_args.openai_gelu:
        activation_function = "gelu_new"
    else:
        activation_function = "gelu"

    # Create GPT2 Config in transformers
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=megatron_args.max_position_embeddings,
        n_embd=megatron_args.hidden_size,
        n_layer=megatron_args.num_layers,
        n_head=megatron_args.num_attention_heads,
        n_inner=megatron_args.ffn_hidden_size,
        activation_function=activation_function,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        architectures=["GPT2LMHeadModel"],
    )
    # Check the config
    print(config)

    # Create the model state dict
    output_state_dict = {}
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    print(f"tp_size: {tp_size}, pp_size: {pp_size}")

    dtype = torch.float32
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    print("Merging and converting the Megatron-LM checkpoint...")

    # Get embedding
    print("Converting the embedding...")
    pp0_state_dicts = get_megatron_sharded_states(ckpt_folder, tp_size, pp_size, 0)

    # Get position embedding
    print("Converting the position embedding...")
    position_embeddings = get_by_dot_path(
        pp0_state_dicts[0],
        "model.language_model.embedding.position_embeddings.weight"
    )
    output_state_dict["transformer.wpe.weight"] = position_embeddings.to(dtype)
    print(f"position_embeddings shape: {position_embeddings.shape}")

    # Get word embedding
    print("Converting the word embedding...")
    word_embeddings = torch.cat(
        [
            get_by_dot_path(
                pp0_state_dicts[tp_rank],
                "model.language_model.embedding.word_embeddings.weight"
            ).to(dtype)
            for tp_rank in range(tp_size)
        ],
        dim=0
    )
    output_state_dict["transformer.wte.weight"] = word_embeddings
    print(f"word_embeddings shape: {word_embeddings.shape}")

    # Get transformer layers
    print("Converting the transformer layers...")
    n_head = config.n_head
    d = config.n_embd // n_head
    n_positions = config.n_positions
    n_layer = config.n_layer
    n_layer_per_stage = n_layer // pp_size
    # Record max layer index
    max_layer_idx = -1

    for pp_rank in range(pp_size):
        # Get the states of the current pipeline stage
        stage_state_dicts = get_megatron_sharded_states(
            ckpt_folder,
            tp_size,
            pp_size,
            pp_rank
        )

        # Extract the layer states
        tp0_state_dicts = stage_state_dicts[0]
        prefix = "model.language_model.encoder"
        model_state_dict = get_by_dot_path(tp0_state_dicts, prefix)
        
        for k, v in model_state_dict.items():
            # Match the layer names
            m = layer_re.match(k)
            if m is None:
                # Not a layer (e.g., final layer norm)
                continue

            # layer index
            layer_idx = int(m.group(1)) + pp_rank * n_layer_per_stage
            # layer operation
            layer_op = m.group(2)
            # layer is weight or bias
            weight_or_bias = m.group(3)

            # Update max layer index
            max_layer_idx = max(max_layer_idx, layer_idx)
            
            # layer name in transformers gpt2
            layer_name = f"transformer.h.{layer_idx}"

            if f"{layer_op}.{weight_or_bias}" not in tensor_parallel_params:
                # Not a tensor parallel parameter
                params = v.to(dtype)
            else:
                # Tensor parallel parameter
                dim = 1 if layer_op in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [
                        get_by_dot_path(stage_state_dicts[tp_rank], prefix)[k].to(dtype)
                        for tp_rank in range(tp_size)
                    ],
                    dim=dim
                )
            
            # For layer norm, simply store the parameters
            if layer_op.endswith("layernorm"):
                ln_index = 1 if layer_op.startswith("input") else 2
                output_state_dict[f"{layer_name}.ln_{ln_index}.{weight_or_bias}"] = params
            
            # For self-attention, we need to transpose QKV
            elif (
                layer_op == "self_attention.query_key_value"
                and weight_or_bias == "weight"
            ):
                # Add causal mask and masked bias
                casual_mask = torch.tril(torch.ones(n_positions, n_positions)).view(
                    1, 1, n_positions, n_positions
                )
                masked_bias = torch.tensor(-1e4, dtype=dtype)
                output_state_dict[f"{layer_name}.attn.bias"] = casual_mask
                output_state_dict[f"{layer_name}.attn.masked_bias"] = masked_bias

                # Transpose QKV
                # The original shape is [n_head * 3 * d, hidden_size]
                input_shape = params.size()
                # View as [n_head, 3, d, hidden_size]
                params = params.view(n_head, 3, d, -1)
                # Change to [3, n_head, d, hidden_size]
                params = params.transpose(0, 1).contiguous()
                # View as [n_head * 3 * d, hidden_size]
                params = params.view(*input_shape)
                # Change to [hidden_size, n_head * 3 * d]
                params = params.transpose(0, 1).contiguous()
                output_state_dict[f"{layer_name}.attn.c_attn.weight"] = params
            
            # For self-attention, we need to transpose bias
            elif (
                layer_op == "self_attention.query_key_value"
                and weight_or_bias == "bias"
            ):
                # Transpose bias
                # The original shape is [n_head * 3 * d]
                input_shape = params.size()
                # View as [n_head, 3, d]
                params = params.view(n_head, 3, d, -1)
                # Change to [3, n_head, d]
                params = params.transpose(0, 1).contiguous()
                # View as [n_head * 3 * d]
                params = params.view(*input_shape)
                output_state_dict[f"{layer_name}.attn.c_attn.bias"] = params
            
            # For weight, transpose the weight
            elif weight_or_bias == "weight":
                op_name = megatron_to_transformers_name[layer_op]
                name = f"{layer_name}.{op_name}.{weight_or_bias}"
                output_state_dict[name] = params.transpose(0, 1).contiguous()
            
            # For bias, simply store the bias
            elif weight_or_bias == "bias":
                op_name = megatron_to_transformers_name[layer_op]
                name = f"{layer_name}.{op_name}.{weight_or_bias}"
                output_state_dict[name] = params

        print(f"Converted stage {pp_rank} of {pp_size} pipeline stages")
        
    # Check max layer index
    assert max_layer_idx + 1 == n_layer, (
        f"Expecting {n_layer} layers, but found {max_layer_idx + 1} layers"
    )
    
    # Get final layer norm
    print("Converting the final layer norm...")
    pplast_state_dicts = get_megatron_sharded_states(
        ckpt_folder,
        tp_size,
        pp_size,
        pp_size - 1
    )
    state_dict = get_by_dot_path(pplast_state_dicts[0], prefix)
    output_state_dict["transformer.ln_f.weight"] = state_dict["final_layernorm.weight"].to(dtype)
    output_state_dict["transformer.ln_f.bias"] = state_dict["final_layernorm.bias"].to(dtype)

    # Get LM head
    print("Converting the LM head...")
    output_state_dict["lm_head.weight"] = word_embeddings.to(dtype)

    # All parameters are converted
    print("All parameters are converted!")

    # Get tokenizer class
    print("Convert the tokenizer class...")
    tokenizer_model = megatron_args.tokenizer_model
    if tokenizer_model is None:
        tokenizer_model = "gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Save the config
    print(f"Saving the config to {output_path}...")
    config.save_pretrained(output_path)

    # Save the tokenizer
    if megatron_args.tokenizer_model is not None:
        print(f"Saving the tokenizer to {output_path}...")
        tokenizer.save_pretrained(output_path)
    
    # Save the model
    print(f"Saving the model to {output_path}...")
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_path, shard_file))
    
    if index is not None:
        print(f"Saving the index to {output_path}...")
        index_file = os.path.join(output_path, WEIGHTS_INDEX_NAME)
        with open(index_file, "w") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
    
    print("All done!")
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_folder",
        type=str,
        help="The folder containing the megatron checkpoint"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="The folder to save the converted transformers checkpoint"
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="10GB",
        help="The maximum huggingface checkpoint shard size"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
