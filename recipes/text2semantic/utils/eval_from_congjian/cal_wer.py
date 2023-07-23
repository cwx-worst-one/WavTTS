import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import sys, os
import librosa
from tqdm import tqdm

device='cuda'
processor = Wav2Vec2Processor.from_pretrained("/mnt/bd/seed-bigtts-testset/code/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("/mnt/bd/seed-bigtts-testset/code/hubert-large-ls960-ft").to(device)

def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
 
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
 
    return matrix[len(str1)][len(str2)]
 

wav_res_text_path = sys.argv[1] # wav_ref wav_res text
res_path = sys.argv[2]

sampling_rate = 16000

f = open(wav_res_text_path)
lines = f.readlines()
f.close()

f_w = open(res_path, 'w')
f_w.write("utt" + '\t' + "wav_res" + '\t' + 'res_wer' + '\t' + 'text_ref' + '\t' + 'text_res' + '\n')

total_num = len(lines)
fail_num = 0
success_num = 0
wer_list = []
for line in tqdm(lines):
    line = line.strip()
    wav_res_path, _, text_ref = line.split('\t')

    if not os.path.exists(wav_res_path):
        fail_num += 1
        continue

    print(wav_res_path)
    wav, sr = librosa.load(wav_res_path, sr=None)
    if len(wav) < 100:
        continue
    wav_16k = librosa.core.resample(wav, orig_sr=sr, target_sr=sampling_rate)

    if len(wav_16k) < 16000 * 0.05:
        fail_num += 1
        continue

    input_values = processor(wav_16k, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
    input_values = input_values.to(device)

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # lower
    ref = text_ref.lower()
    res = transcription.lower()

    out_str = ref + "\t" + res

    # remove punc
    puncs = ",.?!:\'\"[]-<>~|$&*%@()"
    for x in puncs:
        ref = ref.replace(x, '')
        res = res.replace(x, '')

    # split to list
    ref_list = ref.split(' ')
    res_list = res.split(' ')

    # cal Levenshtein_Distance
    dist = Levenshtein_Distance(ref_list, res_list)
    wer_list.append(dist / len(ref_list))
    success_num += 1

    out_str = wav_res_path.split('/')[-1][:-4] + '\t' + wav_res_path + '\t' + str(dist / len(ref_list)) + '\t' + out_str
    f_w.write(out_str + '\n')
    f_w.flush()

f_w.write("avg wer score: " + str(sum(wer_list) / len(wer_list)) + '\n')
f_w.write("total_num: " + str(total_num) + ", success_num: " + str(success_num) + ", fail_num: " + str(fail_num) + '\n')
f_w.close()
