import argparse
from typing import Optional
import pprint
import torch


def gpt2_name_mapping(name, xperf_compitable):
    name = name.replace('module.gpt2.', 'gpt.')
    if '.mlp.' in name:
        name = name.replace('.mlp', '.mlp.moe')
    if name.endswith('.mlp.moe.experts.fc1.weight'):
        name = name.replace('.mlp.moe.experts.fc1.weight', '.mlp.moe.experts.fc1')
    if name.endswith('.mlp.moe.experts.fc2.weight'):
        name = name.replace('.mlp.moe.experts.fc2.weight', '.mlp.moe.experts.fc2')
    if name.endswith('.mlp.moe.share_experts.fc1.weight'):
        name = name.replace('.mlp.moe.share_experts.fc1.weight', '.mlp.moe.share_experts.fc1')
    if name.endswith('.mlp.moe.share_experts.fc2.weight'):
        name = name.replace('.mlp.moe.share_experts.fc2.weight', '.mlp.moe.share_experts.fc2')
    if xperf_compitable:
        if 'wte.text_embedding.weight' in name:
            name = name.replace('wte.text_embedding.weight', 'wte.weight')
    return name


def split(
    ckpt: str,
    text_llm_ckpt: str,
    audio_ckpt: str,
    xperf_compitable: bool = False,
    is_sanity_check: bool = True,
    moe_model: bool=False
):

    state_dict_full = torch.load(ckpt, map_location='cpu', mmap=True)
    text_llm_state_dict = dict()
    encoder_state_dict = dict()
    for k, v in state_dict_full.items():
        if k == 'from_omnistore':
            continue
        print(k, v.shape, v.dtype)
        if "gpt." in k:  # 兼容xperf split
            text_llm_state_dict[k] = v
            print('add ', k, v.shape, v.dtype, ' to llm weight')
        else:
            if 'gpt2.' in k and 'external' not in k:
                if moe_model:
                    new_k = gpt2_name_mapping(k, xperf_compitable)
                else:
                    new_k = k.replace('module.gpt2.', 'gpt.')
                   
                if torch.is_floating_point(v) and 'gate' not in new_k:
                    v = v.to(torch.bfloat16)

                text_llm_state_dict[new_k] = v
                print('add ', new_k, v.shape, v.dtype, ' to llm weight')
            elif "module.emb." in k:
                new_key = k.replace('module.emb.', '')
                encoder_state_dict[new_key] = v
                print('add ', new_key, v.shape, v.dtype, ' to encoder weight')
                if "mulan_music" in new_key:
                    mulan_key = new_key.replace("mulan_music", "mulan_text")
                    encoder_state_dict[mulan_key] = v.clone()
                    print('add ', mulan_key, v.shape, v.dtype, ' to encoder weight')
            elif "module.audio_encoder." in k:
                new_key = k.replace('module.audio_encoder.', '')
                encoder_state_dict[new_key] = v
                print('add ', new_k, v.shape, v.dtype, ' to audio encoder')
            elif "module.audio_adapter" in k:
                new_key = k.replace('module.audio_adapter.', '')
                encoder_state_dict[new_key] = v
                print('add ', new_k, v.shape, v.dtype, ' to audio encoder')
            else:
                print('drop ', k, v.shape, v.dtype)

    torch.save(text_llm_state_dict, text_llm_ckpt)
    print(f'save text llm weight to {text_llm_ckpt}')

    torch.save(encoder_state_dict, audio_ckpt)
    print(f'save encoder weight to {audio_ckpt}')

    if is_sanity_check:
        try:
            tmp_state_dict = torch.load(text_llm_ckpt, map_location='cpu', weights_only=True)
            print('load text llm weight success')
        except Exception as e:
            print(f'load text llm weight failed: {str(e)}')

        try:
            tmp_state_dict = torch.load(audio_ckpt, map_location='cpu', weights_only=True)
            print('load encoder weight success')
        except Exception as e:
            print(f'load text llm weight failed: {str(e)}')


if __name__ == '__main__':
    """
    split bigmusic ckpt to encoder and llm
    python3 apps/bigmusic/mariana_tasks/utils/split_weight.py  \
        --ckpt='epoch=0-step=14000.pt' \
        --llm_ckpt=llm.pt --audio_ckpt=emb.pt
    """
    parser = argparse.ArgumentParser(description='split bigmusic weight to encoder and llm')
    parser.add_argument('--ckpt', help='input merged ckpt', type=str, required=True)
    parser.add_argument('--text_llm_ckpt', help='text llm ckpt', type=str, required=True)
    parser.add_argument('--audio_ckpt', help='audio encoder ckpt', type=str, required=True)
    parser.add_argument('--xperf', help="xperf compatiable", action='store_true', default=False, required=False)
    parser.add_argument('--moe_model', help="Whether model is MoE (Mixture of Experts)", type=lambda x: x.lower() == 'true', required=True)
    
    print(parser.format_help())
    
    args, _ = parser.parse_known_args()

    pprint.pprint(vars(args))  # Converts to dict and pretty-prints

    split(
        args.ckpt,
        args.text_llm_ckpt,
        args.audio_ckpt,
        args.xperf,
        moe_model=args.moe_model
    )
