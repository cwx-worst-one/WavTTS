import sys, os
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(ABSPATH))

from asr_tools_voice_clone import do_speech_recognition
from tqdm import tqdm
import sox
import json

def analysis_tacolabs(taco_path):
    tacos = []
    lines = open(taco_path).readlines()
    for line in lines:
        tacos.append(line.strip())
    return tacos

wav2taco = sys.argv[1]
meta_path = sys.argv[2]

# os.makedirs(out_taco_dir, exist_ok=True)

f = open(wav2taco)
wav2tacos = f.readlines()
f.close()

out_info = dict()
for wav2taco in tqdm(wav2tacos):
    in_wav_path, in_taco_path, in_text_path = wav2taco.strip().split('|')
    in_wav_path = in_wav_path.strip()
    in_wav_name = in_wav_path.split('/')[-1]
    utt_name = in_wav_name[:-4]

    # short_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t1"
    # mid_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t2"
    # long_sp = "sp\t0\t0.0 0.0 0.0 0.0\tS\t3"

    ret = do_speech_recognition(in_wav_path)
    if ret is None:
        continue

    try:
        results = ret["results"]
    # print("results: ", results)

    # print("in_wav_name: ", in_wav_name)
    # print("in_wav_path: ", in_wav_path)

    # tacolabs = analysis_tacolabs(in_taco_path)
    # print("in_taco_path: ", in_taco_path)
    # tacolab_word_num = 0
    # for tacolab in tacolabs:
    #     if tacolab.split('\t')[-1] == '1':
    #         tacolab_word_num += 1
    # print("tacolabs: ", tacolab_word_num, tacolabs)
    # print("in_text_path: ", in_text_path)
    # print("in_text: ", len(open(in_text_path).readlines()[0].strip().split(' ')), open(in_text_path).readlines()[0].strip())
    # print("words: ", len(results[0]['alternatives'][0]['words']), results[0]['alternatives'][0]['words'])

    # pre_end_time = -1
    # for i in range(len(results[0]['alternatives'][0]['words'])):
    #     cur_word_info = results[0]['alternatives'][0]['words'][i]
    #     if pre_end_time != -1:
    #         if cur_word_info['start_time'] != pre_end_time:
    #             print("cur_word_info: ", cur_word_info)
    #     pre_end_time = cur_word_info['end_time']
    
        sub_info = dict()
        sub_info = results[0]['alternatives'][0]

        sub_info['wavpath'] = in_wav_path
        sub_info['text_new'] = sub_info['text']
        sub_info['text_old'] = open(in_text_path).readlines()[0].split('\t')[1].strip()
        sub_info['tacolabs'] = ''.join(open(in_taco_path).readlines())
    except Exception:
        print("results: ", ret)
        continue

    out_info[utt_name] = sub_info
    # print("sub_info: ", sub_info)
    # # sub_info['dur'] = 
    
    # exit()
    # sp  0       0.0 0.0 0.0 1.0 S       3

f_w = open(meta_path, 'w')
json.dump(out_info, f_w, ensure_ascii=False, indent=2)