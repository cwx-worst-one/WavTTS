

#!/bin/bash -ex

cd /opt/tiger

# Download sami_engine scm package
mkdir -p sami_engine
cd sami_engine
if [ ! -f lab.sami.sami_engine_1.0.2.1559.tar.gz ]; then
    wget http://luban-source.byted.org/repository/scm/lab.sami.sami_engine_1.0.2.1559.tar.gz;
fi
tar -xf lab.sami.sami_engine*.tar.gz;

# Download sami_tts_api repo
if [ ! -d /opt/tiger/sami_tts_api ]; then
    git clone git@code.byted.org:lab-audio/sami_tts_api.git /opt/tiger/sami_tts_api
fi
pip3 install /opt/tiger/sami_tts_api/sami_tts_api

# Clean up sami_engine libs
declare -a libfiles=(
    "libfdk-aac.so.1 " "libfstlookahead.so.22" "libfst.so.22" "libtaro.so.1.13.1" "libthrax.so.134"
    "libfstfarscript.so.22" "libfstngram.so.22" "libonnxruntime.so.1.7.2" "libtfdecoder.so"
    "libfstfar.so.22" "libfstscript.so.22" "libsami.so" "libtflite_c_api.so" "libiomp5.so" "libmkl_rt.so"
)
mkdir -p /opt/tiger/sami_engine_cleaned/libs/
for file in "${libfiles[@]}"
do
   sudo cp /opt/tiger/sami_engine/libs/$file /opt/tiger/sami_engine_cleaned/libs/
done

# This line needs to run in the terminal before example script
# or add it to ~/.bashrc or ~/.zshrc
export LD_LIBRARY_PATH=/opt/tiger/sami_engine_cleaned/libs:$LD_LIBRARY_PATH


# Install Chinese fonts for making video
sudo apt-get install fonts-arphic-ukai
ls /usr/share/fonts/truetype/arphic/ukai.ttc

# Bigmusic requirements
cd /opt/tiger/samantha
git submodule update --init recipes/soundstream/torch-museval
pip3 install ./recipes/soundstream/torch-museval

bash ./recipes/bigmusic/bootstrap.sh


# bash launch.sh $@

script="
from sami_tts_api.sail import download_model
fe_version='42.0'
fe_task='tts_chinese_frontend_model'
fe = download_model(fe_task, '/opt/tiger/sami_tts_api/models', fe_version)

from recipes.datasets.mcc.sami_tokenizer import SamiTokenizer
tk = SamiTokenizer()
label = tk(['阳光彩虹小白马 <n> 滴滴哒滴滴哒'])
print(label)
"

python3 -c """$script"""
