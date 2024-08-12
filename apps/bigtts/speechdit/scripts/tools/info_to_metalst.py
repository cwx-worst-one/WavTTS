import os
import shutil
import sys
import uuid

import requests


def download_audio(url, file_path):
    if url.startswith('http'):
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
    else:
        shutil.copyfile(url, file_path)


f_ast_res = open(sys.argv[1], 'r')
f_wav = open(sys.argv[2], 'r')
f_out = open(sys.argv[3], 'w')
f_out_wav_lst = open(sys.argv[4], 'w')
OUTPUT_DIR=os.environ.get('OUTPUT_DIR')
res_dict = {}
for line in f_ast_res:
    line = line.strip().replace("\t", " ").split(" ", 1)
    if len(line) != 2:
        line.append("")
    res_dict[line[0]] = line[1]
tmp_wav_path = "/opt/tiger/samantha/tmp_wav_{}".format(str(uuid.uuid4()))
os.system("mkdir -p {}".format(tmp_wav_path))
for line in f_wav:
    line = line.strip().replace("\t", " ").split(" ", 1)
    fileid = line[0]
    audio_url = line[1]
    local_file = os.path.join(tmp_wav_path, fileid + ".wav")
    output_wav_file = os.path.join(OUTPUT_DIR, fileid + ".wav")
    download_audio(audio_url, local_file)
    if res_dict[fileid].strip() == "":
        continue
    f_out.write("|".join([fileid, "None", local_file, res_dict[fileid]]) + "\n")
    f_out_wav_lst.write("{} {}\n".format(fileid, output_wav_file))
f_out.close()
f_out_wav_lst.close()
