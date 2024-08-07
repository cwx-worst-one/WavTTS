#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: bingyang
# @Date: 2024/8/3 23:08
# @Last Modified by: bingyang
# @Last Modified time: 2024/8/3 23:08
# @File Name: inhouse_metafile.py

from __future__ import unicode_literals
from __future__ import division
import argparse
import logging
import re
import sys
import os
from typing import Dict, List, Tuple


def init_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # output debug to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def contains_chinese(input_texts: List[str]):
    for text in input_texts:
        if re.search('[\u4e00-\u9fff]', text):
            return True
    return False


def join_text(input_texts: List[str]):
    result = ''
    for text in input_texts:
        text = text.strip()
        if len(text) <= 0:
            continue
        if text[0].isascii() and ord(text[0]) < 128:
            result += ' '
        result += text
    return result


def load_prompt_wave(input_prompt_path: str):
    prompt_dict = {}
    path1 = os.path.join(input_prompt_path, 'text.txt')
    path2 = os.path.join(input_prompt_path, 'prompt.txt')
    prompt_text_file = ''
    if os.path.exists(path1):
        logging.info(f'found {path1}')
        prompt_text_file = path1
    elif os.path.exists(path2):
        logging.info(f'found {path2}')
        prompt_text_file = path2
    else:
        logging.fatal('found not prompt text file')

    with open(prompt_text_file, 'r', encoding='utf8') as ifs:
        for line in ifs:
            line = line.strip()
            if len(line) <= 0:
                continue
            vline = line.split('|')
            if len(vline) == 3:
                # path|text
                curr_path = vline[0].strip()
                curr_text = vline[1].strip()
                if curr_path.find('_zh') > 0:
                    curr_lang = 'zh'
                elif curr_path.find('_en') > 0:
                    curr_lang = 'en'
                else:
                    curr_lang = ''
            elif len(vline) == 4:
                # zh|path|text
                curr_lang = vline[0].strip().lower()
                curr_path = vline[1].strip()
                curr_text = vline[2].strip()
            else:
                continue

            if not os.path.isabs(curr_path):
                curr_path = os.path.join(input_prompt_path, curr_path)
            if not os.path.exists(curr_path):
                logging.warning(f'{curr_path} not exists')
                continue

            if curr_lang in ['zh', 'zh_cn', 'chinese']:
                prompt_dict['zh_prompt_text'] = curr_text
                prompt_dict['zh_prompt_wav'] = curr_path
            if curr_lang in ['en', 'en_us', 'english']:
                prompt_dict['en_prompt_text'] = curr_text
                prompt_dict['en_prompt_wav'] = curr_path
    return prompt_dict


def load_infer_text(input_infer_text_file: str):
    with_tag = False
    if input_infer_text_file.find('_with_tag') > 0:
        # sentence_id|sentece1|senetence2
        with_tag = True
    infer_line_number = 0
    results = []
    with open(input_infer_text_file, 'r', encoding='utf8') as ifs:
        for line in ifs:
            line = line.strip()
            infer_line_number += 1
            if len(line) <= 0:
                continue
            curr_tag = '{:05d}'.format(infer_line_number)
            vline = line.split('|')
            curr_texts = vline
            if with_tag:
                curr_tag = vline[0].strip()
                curr_texts = vline[1:]
            curr_texts = [item.strip() for item in curr_texts]
            results.append((curr_tag, curr_texts))
    return results


def create_inhouse_metafile(prompts: Dict[str, str], infer_texts: List[Tuple[str, List[str]]], output_dir: str):
    # 按照 是否包含中文分成2个文件，并使用不同的prompt，同时生成对应的分句文件
    zh2zh_metafile = os.path.join(output_dir, 'meta.lst.zh2zh')
    zh2zh_splitfile = os.path.join(output_dir, 'meta.spt.zh2zh')
    en2en_metafile = os.path.join(output_dir, 'meta.lst.en2en')
    en2en_splitfile = os.path.join(output_dir, 'meta.spt.en2en')
    has_english = False
    has_chinese = False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(zh2zh_metafile, 'w', encoding='utf8') as zh2zh_meta_fd, \
            open(zh2zh_splitfile, 'w', encoding='utf8') as zh2zh_split_fd, \
            open(en2en_metafile, 'w', encoding='utf8') as en2en_meta_fd, \
            open(en2en_splitfile, 'w', encoding='utf8') as en2en_split_fd:
        for item in infer_texts:
            curr_tag = item[0].strip()
            curr_texts = item[1]
            if not contains_chinese(curr_texts):
                curr_prompt_text = prompts['en_prompt_text'].strip()
                curr_prompt_wav = prompts['en_prompt_wav'].strip()
                curr_meta_fd = en2en_meta_fd
                curr_split_fd = en2en_split_fd
                has_english = True
            else:
                curr_prompt_text = prompts['zh_prompt_text'].strip()
                curr_prompt_wav = prompts['zh_prompt_wav'].strip()
                curr_meta_fd = zh2zh_meta_fd
                curr_split_fd = zh2zh_split_fd
                has_chinese = True
            curr_join_text = join_text(curr_texts).strip()
            curr_split_text = '|'.join(curr_texts).strip()
            curr_meta_fd.write(f'{curr_tag}|{curr_prompt_text}|{curr_prompt_wav}|{curr_join_text}\n')
            curr_split_fd.write(f'{curr_tag}\t{curr_split_text}\n')

    if not has_chinese:
        os.remove(zh2zh_metafile)
        os.remove(zh2zh_splitfile)
    if not has_english:
        os.remove(en2en_metafile)
        os.remove(en2en_splitfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_dir',
                        required=False,
                        type=str,
                        default='',
                        help='inhouse speaker prompt wave and text dir')
    parser.add_argument('--zh_prompt_wav',
                        required=False,
                        type=str,
                        default='',
                        help='chinese prompt wave path')
    parser.add_argument('--zh_prompt_text',
                        required=False,
                        type=str,
                        default='',
                        help='chinese prompt text')
    parser.add_argument('--en_prompt_wav',
                        required=False,
                        type=str,
                        default='',
                        help='english prompt wave path')
    parser.add_argument('--en_prompt_text',
                        required=False,
                        type=str,
                        default='',
                        help='english prompt text')
    parser.add_argument('--infer_text_file',
                        required=False,
                        type=str,
                        default='',
                        help='infer text file')
    parser.add_argument('--output',
                        required=False,
                        type=str,
                        default='output',
                        help='output dir')
    args = parser.parse_args()
    # 初始化日志
    init_logger()

    prompt_dir = args.prompt_dir
    zh_prompt_wav = args.zh_prompt_wav
    zh_prompt_text = args.zh_prompt_text
    en_prompt_wav = args.en_prompt_wav
    en_prompt_text = args.en_prompt_text

    infer_text_file = args.infer_text_file
    output = args.output

    # 检查Prompt
    prompts_dict = {}
    if len(zh_prompt_text) > 0 \
            and os.path.exists(zh_prompt_wav) \
            and len(en_prompt_text) > 0 \
            and os.path.exists(en_prompt_wav):
        prompts_dict = {'zh_prompt_text': zh_prompt_text,
                        'zh_prompt_wav': zh_prompt_wav,
                        'en_prompt_text': en_prompt_text,
                        'en_prompt_wav': en_prompt_wav}
    elif os.path.exists(prompt_dir):
        prompts_dict = load_prompt_wave(prompt_dir)
    else:
        logging.fatal('must input prompt with [*_prompt_wav&*_prompt_text] or prompt_dir')
    prompt_keys = ['zh_prompt_text', 'zh_prompt_wav', 'en_prompt_text', 'en_prompt_wav']
    for keyname in prompt_keys:
        if keyname not in prompts_dict:
            logging.fatal(f'{keyname} not in prompts_dict')

    # 检测合成文本
    if not os.path.exists(infer_text_file):
        logging.fatal(f'{infer_text_file} not exists.')

    curr_infer_texts = load_infer_text(infer_text_file)
    if len(curr_infer_texts) <= 0:
        logging.fatal(f'curr_infer_texts is empty.')
    # 开始创建文件
    create_inhouse_metafile(prompts=prompts_dict, infer_texts=curr_infer_texts, output_dir=output)
