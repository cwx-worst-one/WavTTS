# coding:utf-8
import torch

def rewrite_tag_taxonomy_lang(input_ckpt, output_ckpt, tag_taxonomy_lang):
    params = torch.load(input_ckpt)
    #print(params.keys())
    print('before: ', params['hyper_parameters'])
    params['hyper_parameters']['extra_params']['tag_taxonomy_lang'] = tag_taxonomy_lang
    print('after: ', params['hyper_parameters'])
    torch.save(params, output_ckpt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--input_ckpt", type=str, help="Path to the input ckpt")
    parser.add_argument("--output_ckpt", type=str, help="Path to the output ckpt")
    parser.add_argument("--tag_taxonomy_lang", type=str, help="tag_taxonomy_lang want to be overwrited")
    args = parser.parse_args()

    rewrite_tag_taxonomy_lang(args.input_ckpt, args.output_ckpt, args.tag_taxonomy_lang)
