import os
import sys
import glob
import argparse
from bytedance import easycycle
from prompt_toolkit import prompt
"""
先安装环境：

pip3 install bytedance.easycycle --index-url https://bytedpypi.byted.org/simple
"""

tos_url_expires = 60 * 60 * 24 * 1000
tos_bucket = "bigspeech-platform"

import re


def drop_punctuation(text):
    punc = '~`!#$%^&*()_+-=|\';"＂:/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》{《}】【\n\]\[ '
    new_text = re.sub(r"[%s]+" % punc, "", text)
    return new_text


def upload_wo_text(video_dir, prefix):

    with open("/mnt/bn/zxb-lq/workspace/samantha/apps/bigtts/audiogen/testdata/v2a.txt") as f:
        file_list = f.read().splitlines()

    res = []

    for idx, in_path in enumerate(file_list):

        file_name = os.path.basename(in_path)
        if not file_name.endswith("mp4"):
            continue

        in_dir = os.path.dirname(in_path)

        file_name = file_name[0:-4]
        out_path = os.path.join(video_dir, f"{idx}_{file_name[0:10]}.mp4")

        file_dir = out_path.replace(video_dir, "")[1:]

        print(out_path)

        if not os.path.exists(out_path):
            print("skip")
            continue
        with open(out_path, "rb") as af:
            wav_data = af.read()
            audio_url = easycycle.upload_data_and_get_public_url(
                easycycle.Host.CN,
                'wangtuo.todd',
                wav_data,
                tos_bucket,
                f"{prefix}_{file_name}",
                tos_url_expires,
            )
            print(str(idx), file_dir, audio_url)
            res.append(f"{idx},{file_dir},{file_name},{audio_url},")

        with open(f"{prefix}.csv", "w") as outf:
            outf.write("name,filename,url\n")
            for item in res:
                outf.write(item + '\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 输入
    parser.add_argument('--prefix', type=str)  # 保存的csv名称
    parser.add_argument('--video_dir', type=str)
    args = parser.parse_args()

    upload_wo_text(args.video_dir, args.prefix)  # 只有音频
