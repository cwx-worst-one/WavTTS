#!/bin/bash -ex

cd $(dirname $0)/../../
echo "Working DIR: $(pwd)"

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

bash launch.sh $@