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

pip3 install bytedeuler==2.0.0 -i "https://bytedpypi.byted.org/simple"

git clone git@code.byted.org:lab-audio/sami_tts_api.git ../sami_tts_api && pip3 install ../sami_tts_api/sami_tts_api/

declare -a libfiles=(
    "libfdk-aac.so.1 " "libfstlookahead.so.22" "libfst.so.22" "libtaro.so.1.13.1" "libthrax.so.134"
    "libfstfarscript.so.22" "libfstngram.so.22" "libonnxruntime.so.1.7.2" "libtfdecoder.so"
    "libfstfar.so.22" "libfstscript.so.22" "libsami.so" "libtflite_c_api.so"
)
mkdir -p /opt/tiger/sami_engine_cleaned/libs/
for file in "${libfiles[@]}"
do
   sudo cp /opt/tiger/sami_engine/libs/$file /opt/tiger/sami_engine_cleaned/libs/
done
export LD_LIBRARY_PATH=/opt/tiger/sami_engine_cleaned/libs:$LD_LIBRARY_PATH

sh launch.sh $@
