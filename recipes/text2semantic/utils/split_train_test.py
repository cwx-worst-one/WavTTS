import sys
import random

all_meta_path = sys.argv[1]
all_meta_shuf_path = sys.argv[2]
test_meta_path = sys.argv[3]
train_meta_path = sys.argv[4]
ratio = float(sys.argv[5])

f = open(all_meta_path)
lines = f.readlines()
f.close()

random.shuffle(lines)

f_w = open(all_meta_shuf_path, 'w')
for line in lines:
    f_w.write(line)
f_w.close()

total_num = len(lines)
test_num = int(len(lines) * ratio)
train_num = total_num - test_num

test_lines = lines[:test_num]
train_lines = lines[test_num:]

f_w = open(test_meta_path, 'w')
for line in test_lines:
    f_w.write(line)
f_w.close()

f_w = open(train_meta_path, 'w')
for line in train_lines:
    f_w.write(line)
f_w.close()
