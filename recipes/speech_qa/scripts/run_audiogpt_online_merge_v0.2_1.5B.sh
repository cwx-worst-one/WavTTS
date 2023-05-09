!/bin/bash

export MUSICLM_DIR=/mnt/bn/cyz-lq-nas/project/samantha
cd $MUSICLM_DIR

sudo apt update && sudo apt install tmux htop ffmpeg -y && pip3 install ipython pydub -i https://bytedpypi.byted.org/simple/

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple && \
pip3 install wheel Cython numpy && \
pip3 install -q -r requirements.txt && \
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple && \
cp $MUSICLM_DIR/recipes/speech_qa/scripts/matmul.py /home/tiger/.local/lib/python3.8/site-packages/triton/ops/blocksparse/



# if [ ! -e last.ckpt ]
# then
#     hdfs dfs -get $1/checkpoints/last.ckpt last.ckpt || echo "Can't get hdfs file"
# fi

# if [ ! -d inference_test ]
# then
#     hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/inference_test .
# fi

# if [ ! -e last.ckpt ]
# then
#     bash recipes/speech_qa/bootstrap.sh fit --config recipes/speech_qa/conf/semantic_v2_3ar_sparse.yaml --run_opts.hdfs_path=$1
# else
#     bash recipes/speech_qa/bootstrap.sh fit --config recipes/speech_qa/conf/semantic_v2_3ar_sparse.yaml --run_opts.hdfs_path=$1 --ckpt_path=./last.ckpt
# fi

bash recipes/speech_qa/bootstrap.sh fit \
--config recipes/speech_qa/conf/sparse_cn/audiogpt_online_merge_v0.2_1.5B.yaml \
--run_opts.checkpointing=True \
--run_opts.num_workers=8 


# --ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/audiogpt/test0418_onfflinecd/version_0.4/checkpoints/epoch=11-step=6000-accu=54.87.ckpt


# --config recipes/speech_qa/conf/sparse_cn/audiogpt_semantic_v4_3ar_sparse.yaml \