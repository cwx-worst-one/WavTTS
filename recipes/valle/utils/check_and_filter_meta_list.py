import sys, os
from tqdm import tqdm

in_meta_file_path = sys.argv[1]
out_meta_file_path = sys.argv[2]

f = open(in_meta_file_path)
lines = f.readlines()
f.close()

f_w = open(out_meta_file_path, 'w')
# bytedrive_list = []
# data_list = []
for line in tqdm(lines):
    line = line.strip()
    part1, part2 = line.split('|')
    if not os.path.exists(part1):
        print(line, "part1 not exists, skip")
        continue

    if not os.path.exists(part2):
        print(line, "part2 not exists, skip")
        continue

    f_w.write(line + '\n')
    # if part3 == '' or int(part3) <= 0:
    #     print(line, "part3 is empty or less than 0, skip")

#    bytedrive1 = part1.split('/')[3]
#    bytedrive2 = part2.split('/')[3]
#
#    data1 = part1.split('/')[7] + '/' + part1.split('/')[8]
#    data2 = part2.split('/')[7] + '/' + part2.split('/')[8]
#
#    bytedrive_list.append(bytedrive1)
#    bytedrive_list.append(bytedrive2)
#    data_list.append(data1)
#    data_list.append(data2)
#
#print("bytedrive_list: ", list(set(bytedrive_list)))
#print("data_list: ", list(set(data_list)))
