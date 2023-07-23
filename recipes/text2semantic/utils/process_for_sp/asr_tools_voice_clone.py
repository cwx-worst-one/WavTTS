import euler
import os, sys
home_dir = os.getcwd()
if home_dir not in sys.path:
    sys.path.append(home_dir)

ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(ABSPATH))

import json
import argparse
import editdistance
from tqdm import tqdm

import euler
euler.install_thrift_import_hook()
from server.base_thrift import *
from server.sami_thrift import *

client = euler.Client(SAMI, 'sd://lab.sami.gateway?cluster=release_thrift', timeout=1200)

def do_speech_recognition(wav_path):
    try:
        if os.path.exists(wav_path):
            ret = asr(wav_path)
            if ret is not None:
                ret_json = json.loads(ret)
                # return parse_ret(ret_json)
                return ret_json
            else:
                print('some service error !!!')    
                return None
        else:
            print(f'{wav_path} not exist !!!')
            return None
    except Exception:
        print('some service error !!!')    
        return None

def parse_ret(ret_json):
    results = ret_json["results"]
    text = ''
    for item in results:
        text += item["text"] + ' '
    return text.strip()

def asr(wav_path):
    b = Base()
    # data by url
    req = InvokeRequest(
        Base=b,
        access_key="qvzIvCRLjf",
        method="ASR",
        # if "data" is empty, will download data by url in payload
        payload='{"url":"http://tosv.byted.org/obj/tostest/lvse_20.wav", "extra": {"lang":"en", "enable_vad": true, "enable_word_info": true, "enable_punctuation": true, \
        "enable_disfluency": false} }',
        # default use binary data from "data"
        data=open(wav_path, "rb").read()
    )
    result = client.Invoke(req)
    if result.BaseResp.StatusCode == 0:
        return result.payload
    else:
        return None


def read_text(text_path):
    map_d = {}
    with open(text_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            if len(line) == 1:
                map_d['00000' + str(index+1)] = line[0]
            else:
                map_d[line[0]] = line[1]
    return map_d


def asr_reg(wav_dir, textpath):
    data = ''
    for file in os.listdir(wav_dir):
        pred_text = do_speech_recognition(os.path.join(wav_dir, file))
        data += f'{file[:-4]}\t{pred_text}\n'
    with open(textpath, 'w', encoding='utf-8') as f:
        f.write(data)

def calc_stablily(wav_dir, text_file):
    text_map = read_text(text_file)
    stop_words = [',', '.', '?', '!', ';', ':']
    wer = 0
    count = 0
    for file in tqdm(os.listdir(wav_dir)):
        if file.endswith('.wav'):
            pred_text = do_speech_recognition(os.path.join(wav_dir, file))
            if pred_text is not None:
                pred_text = ''.join([item for item in pred_text.lower() if item not in stop_words])
                base_id = file.split('/')[-1].split('.')[0]
                if base_id in text_map:
                    truth_text = ''.join([item for item in text_map[base_id].lower() if item not in stop_words])
                    p_text = pred_text.split()
                    t_text = truth_text.split()
                    wer += editdistance.eval(t_text, p_text) / len(t_text)
                    count += 1
                
    print(f'wer: {wer/count}')

def calc_stablily_(wav_path, text):
    stop_words = [',', '.', '?', '!', ';', ':', '"']
    wer = 0
    pred_text = do_speech_recognition(wav_path)
    truth_text = ''
    if pred_text is not None:
        pred_text = ''.join([item for item in pred_text.lower() if item not in stop_words])
        truth_text = ''.join([item for item in text.lower() if item not in stop_words])
        p_text = pred_text.split()
        t_text = truth_text.split()
        wer = editdistance.eval(t_text, p_text) / len(t_text)
    return wer, pred_text, truth_text

# if __name__ == '__main__':
#     """
#     python3 ./utils/asr_tools.py --w /home/fengchengli/tts/automatic_training_platform/tt_meeting/24k/tt --f /home/fengchengli/tts/automatic_training_platform/tt_meeting/24k/text.txt
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--w', type=str, required='True')
#     parser.add_argument('--f', type=str, required='True')
#     args = parser.parse_args()
#     calc_stablily(args.w, args.f)
#     #asr_reg(args.w, args.f)
