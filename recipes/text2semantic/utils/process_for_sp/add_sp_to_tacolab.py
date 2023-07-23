import sys, os
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(ABSPATH))

from asr_tools_voice_clone import do_speech_recognition
from tqdm import tqdm
import sox
import json
import random

def is_phone(ph):
    return ph not in set(['sil', '.', ',', '?', '!', '，', '。', '？', '！'])

def is_sil_punc(ph):
    return not is_phone(ph)

def split_tacolabs_by_word(tacolabs):
    tacolabs = tacolabs.strip().split('\n')
    tacolabs_by_word = []
    pre_tacolab = None
    # print("")
    # for tacolab in tacolabs:
    #     print(tacolab)
    for tacolab in tacolabs:
        phone, tone, wordpost, wordcateg, prosody = tacolab.split('\t')
        if is_sil_punc(phone):
            temp = []
            temp.append(tacolab)
            tacolabs_by_word.append(temp)
        else:
            if prosody == '0':
                if is_sil_punc(pre_tacolab.split('\t')[0]) or pre_tacolab.split('\t')[-1] == '1':
                    temp = []
                    temp.append(tacolab)
                    tacolabs_by_word.append(temp)
                else:
                    tacolabs_by_word[-1].append(tacolab)
            else:
                if is_sil_punc(pre_tacolab.split('\t')[0]) or pre_tacolab.split('\t')[-1] == '1':
                    temp = []
                    temp.append(tacolab)
                    tacolabs_by_word.append(temp)
                else:
                    tacolabs_by_word[-1].append(tacolab)
        pre_tacolab = tacolab
    return tacolabs_by_word

def merge_punc_sil(tacolabs_by_word):
    out_tacolabs_by_word = []
    pre_tacolab_by_word = None
    for tacolab_by_word in tacolabs_by_word:
        if pre_tacolab_by_word is not None and len(pre_tacolab_by_word) == 1 and len(tacolab_by_word) == 1:
            pre_phone = pre_tacolab_by_word[0].split('\t')[0]
            phone = tacolab_by_word[0].split('\t')[0]
            if is_sil_punc(pre_phone) and is_sil_punc(phone):
                assert phone == 'sil'
                continue
        out_tacolabs_by_word.append(tacolab_by_word)
        pre_tacolab_by_word = tacolab_by_word
    return out_tacolabs_by_word
# def analysis_tacolabs(taco_path):
#     tacos = []
#     lines = open(taco_path).readlines()
#     for line in lines:
#         tacos.append(line.strip())
#     return tacos

in_taco_dir = sys.argv[1]
meta_path = sys.argv[2]
out_taco_dir = sys.argv[3]

os.makedirs(out_taco_dir, exist_ok=True)

metas = json.load(open(meta_path))

in_taco_names = os.listdir(in_taco_dir)
# random.shuffle(in_taco_names)
for in_taco_name in tqdm(in_taco_names):
    utt_name = in_taco_name[:-4]
    in_taco_path = os.path.join(in_taco_dir, in_taco_name)

    if utt_name not in metas.keys():
        continue

    meta = metas[utt_name]

    # check
    try:
        assert meta['text_new'] == meta['text_old']
    except Exception:
        print(meta['text_new'], meta['text_old'])
        continue

    check_tacolabs = ''.join(open(in_taco_path).readlines())
    assert check_tacolabs == meta['tacolabs'], (check_tacolabs, meta['tacolabs'])

    word_num_meta = len(meta['words'])
    tacolabs_by_word = split_tacolabs_by_word(meta['tacolabs'])
    word_num_tacolabs = len([x for x in tacolabs_by_word if not (len(x) == 1 and is_sil_punc(x[0].split('\t')[0]))])
    # print("in_tacolabs_by_word")
    # for x in tacolabs_by_word:
    #     print(x)
    try:
        assert word_num_meta == word_num_tacolabs
    except Exception:
        print(utt_name, word_num_meta, word_num_tacolabs)
        continue



    sil_info = []
    pre_word_end_time = None
    for word in meta['words']:
        cur_word_start_time = word['start_time']
        cur_word_end_time = word['end_time']
        if pre_word_end_time is not None:
            assert cur_word_start_time - pre_word_end_time >= 0
            sil_info.append(cur_word_start_time - pre_word_end_time)
        pre_word_end_time = cur_word_end_time
    sil_info.append(meta['end_time'] - pre_word_end_time)


    # print("utt_name: ", utt_name)
    # print([(word['word'], x) for x, word in zip(sil_info, meta['words'])])
    # print(meta['wavpath'])
    mid_sp = "，\t0\t0.0 0.0 0.0 0.0\tS\t3"

    out_tacolabs_by_word = []
    word_index = 0
    for i, tacolab_by_word in enumerate(tacolabs_by_word):
        if i == 0 or i == len(tacolabs_by_word) - 1:
            out_tacolabs_by_word.append(tacolab_by_word)
            continue

        if is_phone(tacolab_by_word[0].split('\t')[0]):
            if sil_info[word_index] > 0.1:
                if is_sil_punc(tacolabs_by_word[i + 1][0].split('\t')[0]):
                    out_tacolabs_by_word.append(tacolab_by_word)
                else:
                    out_tacolabs_by_word.append(tacolab_by_word)
                    out_tacolabs_by_word.append([mid_sp])
            else:
                out_tacolabs_by_word.append(tacolab_by_word)
            word_index += 1
        else:
            if sil_info[word_index - 1] > 0:
                out_tacolabs_by_word.append(tacolab_by_word)
                # print("utt_name: ", utt_name, word_index)
                # print("tacolab_by_word: ", tacolab_by_word)
                # print(meta['words'][word_index])
                # exit()
    out_taco_path = os.path.join(out_taco_dir, in_taco_name)
    f_w = open(out_taco_path, 'w')
    f_w.write('\n'.join(y for x in out_tacolabs_by_word for y in x))
    f_w.close()
    # print("out_tacolabs_by_word")
    # for x in out_tacolabs_by_word:
    #     print(x)
    

    # tacolabs_by_word = merge_punc_sil(tacolabs_by_word)
    # print("tacolabs_by_word: ", tacolabs_by_word)
    # print("")
    # for x in tacolabs_by_word:
    #     for y in x:
    #         print(y)

    # exit()



    



# out_info = dict()
# for wav2taco in tqdm(wav2tacos):
#     in_wav_path, in_taco_path, in_text_path = wav2taco.strip().split('|')
#     in_wav_path = in_wav_path.strip()
#     in_wav_name = in_wav_path.split('/')[-1]
#     utt_name = in_wav_name[:-4]

#     # short_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t1"
#     # mid_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t2"
#     # long_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t3"

#     ret = do_speech_recognition(in_wav_path)
#     if ret is None:
#         continue

#     try:
#         results = ret["results"]
#     # print("results: ", results)

#     # print("in_wav_name: ", in_wav_name)
#     # print("in_wav_path: ", in_wav_path)

#     # tacolabs = analysis_tacolabs(in_taco_path)
#     # print("in_taco_path: ", in_taco_path)
#     # tacolab_word_num = 0
#     # for tacolab in tacolabs:
#     #     if tacolab.split('\t')[-1] == '1':
#     #         tacolab_word_num += 1
#     # print("tacolabs: ", tacolab_word_num, tacolabs)
#     # print("in_text_path: ", in_text_path)
#     # print("in_text: ", len(open(in_text_path).readlines()[0].strip().split(' ')), open(in_text_path).readlines()[0].strip())
#     # print("words: ", len(results[0]['alternatives'][0]['words']), results[0]['alternatives'][0]['words'])

#     # pre_end_time = -1
#     # for i in range(len(results[0]['alternatives'][0]['words'])):
#     #     cur_word_info = results[0]['alternatives'][0]['words'][i]
#     #     if pre_end_time != -1:
#     #         if cur_word_info['start_time'] != pre_end_time:
#     #             print("cur_word_info: ", cur_word_info)
#     #     pre_end_time = cur_word_info['end_time']
    
#         sub_info = dict()
#         sub_info = results[0]['alternatives'][0]

#         sub_info['wavpath'] = in_wav_path
#         sub_info['text_new'] = sub_info['text']
#         sub_info['text_old'] = open(in_text_path).readlines()[0].split('\t')[1].strip()
#         sub_info['tacolabs'] = ''.join(open(in_taco_path).readlines())
#     except Exception:
#         print("results: ", ret)
#         continue

#     out_info[utt_name] = sub_info
#     # print("sub_info: ", sub_info)
#     # # sub_info['dur'] = 
    
#     # exit()
#     # sp  0       0.0 0.0 0.0 1.0 S       3

# f_w = open(meta_path, 'w')
# json.dump(out_info, f_w, ensure_ascii=False, indent=2)