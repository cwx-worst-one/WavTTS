import sys
from tqdm import tqdm

in_wer_paths = sys.argv[1]
out_wer_path = sys.argv[2]

out_wer_info = []
out_mos_info = []
in_wer_paths = in_wer_paths.split(',')
for in_wer_path in in_wer_paths:
    print("in_wer_path: ", in_wer_path)
    f = open(in_wer_path)
    lines = f.readlines()
    f.close()

    sub_out_wer_info = []
    sub_out_mos_info = []
    for i in range(1, 1238):
        line = lines[i].strip()
        mos = float(line.split('\t')[2])
        wer = float(line.split('\t')[3])
        if len(out_wer_info) < i:
            out_wer_info.append([])
            out_mos_info.append([])
        out_wer_info[i - 1].append(wer)
        out_mos_info[i - 1].append(mos)

# print("out_wer_info: ", out_wer_info)
# print("out_mos_info: ", out_mos_info)
min_wer = []
for i in range(len(out_mos_info)):
    # print("out_mos_info[i]: ", out_mos_info[i])
    # print("out_wer_info[i]: ", out_wer_info[i])
    min_index= out_mos_info[i].index(max(out_mos_info[i]))
    min_wer.append(out_wer_info[i][min_index])
    # print("min_index: ", min_index)
    # print("min_wer: ", min_wer)
    # if i == 5:
    #     exit()

print("min_wer: ", min_wer)
print("wer: ", sum(min_wer) / len(min_wer))
# print("min_wer: ", min_wer, len(min_wer), sum(min_wer) / len(min_wer), min_wer[0])
    
