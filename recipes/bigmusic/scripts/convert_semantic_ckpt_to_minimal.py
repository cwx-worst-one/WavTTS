from pathlib import Path
import torch
import os
import argparse


#### USAGE #####
# python3 recipes/bigmusic/scripts/convert_semantic_ckpt_to_minimal.py --ckpt_path /mnt/bn/lyrics-to-song/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt --overwrite
# mlx worker launch -- python3 recipes/bigmusic/scripts/convert_semantic_ckpt_to_minimal.py --ckpt_path /mnt/bn/lyrics-to-song/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt --overwrite

def main(ckpt_path='/mnt/bn/lyrics-to-song/baseline_models/q4/20231106-2min_varlen/checkpoints/step=216000-tr_loss=0.0000-val_accu_0=17.27.ckpt', output_path=None, map_location=None, overwrite=True):
    assert ckpt_path.endswith('.ckpt'), 'Please provide valid .ckpt path'
    if output_path is None:
        output_path = ckpt_path.replace('.ckpt', '-minimal.ckpt')
    if overwrite:
        output_path = ckpt_path

    if map_location is None:
        assert torch.cuda.is_available(), "No GPU found. Set --map_location cpu. Or run script on GPU with 'mlx worker launch -- path/to/script'"

    state_dict = torch.load(ckpt_path, map_location=map_location)

    def check_and_convert_to_minimal_path(path):
        minimal_path = str(path).replace('.ckpt', '-minimal.ckpt')
        if path.endswith('-minimal.ckpt'): return path
        if os.path.exists(minimal_path):
            print('Updating path from\n', path, '\nto:\n', minimal_path)
            return minimal_path
        else:
            print('Could not find minimal path. Skipping:', path)
        return path

    # Convert to minimal mulan
    try:
        mulan_path = state_dict['hyper_parameters']['required_modules']['mulan']['hpath']
        if mulan_path == '/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt':
            # convert old checkpoints to new path
            mulan_path = '/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-median_rank_1=110-kaggle-minimal.ckpt'
            print('Old mulan 110 path found. Converting to new path first.', mulan_path)
        state_dict['hyper_parameters']['required_modules']['mulan']['hpath'] = check_and_convert_to_minimal_path(mulan_path)
    except: 
        pass

    try:
        umm_path = state_dict['hyper_parameters']['required_modules']['bestrq']['hpath']
        if umm_path == 'hdfs://harunava/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt':
            umm_path = '/mnt/bn/audio-diffusion/ducle/recipes/diffusion/assets/umm_stage3_music_chroma_vq32768x32/step=0030000-minimal.ckpt'
            print('UMM HDFS path found. Converting to local bytenas path first.', umm_path)
        state_dict['hyper_parameters']['required_modules']['bestrq']['hpath'] = check_and_convert_to_minimal_path(umm_path)
    except: 
        pass


    # Remove optimizer states
    if 'optimizer_states' in state_dict:
        # Diffusion case. save ema state which is needed at inference time
        if 'ema'in state_dict["optimizer_states"][0]:
            print('Diffusion/vocoder detected. Saving ema optimizer states')
            state_dict["optimizer_states"] = [{"ema": state_dict["optimizer_states"][0]["ema"]}]
        else:
            del state_dict['optimizer_states']
            print('Deleting optimizer states')

    torch.save(state_dict, output_path)
    print('Saved ckpt to:', output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None
    )
    parser.add_argument(
        '--map_location',
        type=str,
        default=None
    )
    parser.add_argument(
        '--overwrite',
        action='store_true'
    )
    args = parser.parse_args()
    assert (args.ckpt_path is not None), f'Must provide valid ckpt path: {args.ckpt_path}'
    main(args.ckpt_path, args.output_path, args.map_location, args.overwrite)

