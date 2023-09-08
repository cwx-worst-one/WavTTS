#!/bin/bash -ex
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

# Do something before lauching the main program
# e.g. Download some data, install some extra packages, etc.
pip3 install recipes/soundstream/torch-museval
pip3 install -q -r recipes/diffusion/requirements.txt

# get google prompts data
# check if it exists
if [ ! -d "recipes/diffusion/assets/google_prompts" ]
then
    echo "Download google prompts"
    mkdir -p recipes/diffusion/assets
    hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wtl/diffusion/assets/google_prompts.zip
    unzip google_prompts.zip
    mv generated_output/google_prompts recipes/diffusion/assets/
    rm google_prompts.zip
    rm -r generated_output
else
    echo "Google prompts exists, skip download"
fi


sh launch.sh $@
