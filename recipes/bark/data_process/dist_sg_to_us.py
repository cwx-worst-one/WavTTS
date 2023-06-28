import os
from tqdm import tqdm
from multiprocessing import  Process

res = os.popen("hdfs dfs -ls hdfs://harunasg/home/byte_speech_sv/litang/bark_data/libri_light_ver_*/*.tar").read()
res = res.split('\n')
res = [r.split(' ')[-1] for r in res]


res2 = os.popen("hdfs dfs -ls hdfs://harunava/home/byte_speech_sv/litang/bark_data/libri_light_ver_*/*.tar").read()
res2 = res2.split('\n')
res2 = [r.split(' ')[-1] for r in res2]

res2_dict = {'/'.join(r.split('/')[-2:]) : r for r in res2}
new_res = []
for r in res:
    if '/'.join(r.split('/')[-2:]) not in res2_dict:
        new_res.append(r)

res = new_res
print("len(res): ", len(res))


def copy_file(command):
    msg = os.popen(command).read()
    return msg

if __name__ == '__main__':
    for i in tqdm(range(0, len(res), 100)):
        beg = i
        end = min(len(res), i + 100)
        tmp = res[beg:end]

        process_list = []
        for r in tmp:
            command = 'hdfs dfs -cp {} {}'.format(r, r.replace("harunasg", "harunava"))
            print(command)
        
            p = Process(target=copy_file, args=(command,))
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()

        print('结束测试')
        
