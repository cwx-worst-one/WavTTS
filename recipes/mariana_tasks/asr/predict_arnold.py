from cruise.utilities.distributed import DIST_ENV

import os
import json
import time
import torch
from absl import app, flags
from tqdm import tqdm
from sft import generate
from cruise.utilities.hdfs_io import hcopy, hput, hlist_files

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "predict_file_hdfs", None, "The input file to be processed."
)

flags.DEFINE_string(
    "predict_out_hdfs", None, "The input file to be processed."
)

def get_predict_file(all_file_list, rank, world_size):
    file_sub_set = []
    for index in range(len(all_file_list)):
        if index % world_size == rank:
            file_sub_set.append(all_file_list[index])
    return file_sub_set


def main(_):
    # single node, multi_gpu
    predict_world_size = DIST_ENV.world_size
    rank = DIST_ENV.rank
    local_rank = DIST_ENV.local_rank

    # when load on cpu, different process load on different time prevent oom
    time.sleep(local_rank * 120)

    print("world_size:" + str(predict_world_size))
    print("rank:" + str(rank))
    print("local_rank:" + str(local_rank))

    if not FLAGS.predict_file_hdfs or not FLAGS.predict_out_hdfs:
        raise ValueError("do not have hdfs file to predict or output")

    all_file_list = hlist_files([FLAGS.predict_file_hdfs])
    local_file_name = get_predict_file(all_file_list, rank, predict_world_size)

    print("need_process:" + str(local_file_name))

    model, tokenizer = generate.initialize(local_rank)

    for single_file in local_file_name:
        cache_file_name = "./cache_" + str(local_rank)
        hcopy(single_file, cache_file_name)
        test_data = generate.load_test_data(cache_file_name)
        responses=[]
        for data in tqdm(test_data):
            text=data['prompt']
            input_ids, prompt_len = generate.preprocess([text], tokenizer, add_special_tokens=FLAGS.add_special_tokens)
            input_ids = input_ids.to(torch.device("cuda:" + str(local_rank)))

            outputs = model.generate(input_ids)

            response = generate.postprocess(outputs.sequences[0], prompt_len, tokenizer)

            responses.append(json.dumps({"id":data['id'],"prompt":data['prompt'],"response":response},ensure_ascii=False)+"\n")
        out_put_file_name = "predict_" + single_file.split("/")[-1]
        generate.dump_response(responses, out_put_file_name)
        hput(out_put_file_name, FLAGS.predict_out_hdfs)
        os.system("rm " + out_put_file_name)
        os.system("rm " + cache_file_name)


if __name__ == '__main__':
    app.run(main)
