pip3 install bytedray[bytedance,data,default]

# Clone main data processing repo
git clone git@code.byted.org:seed/bigspeech_data.git
cd bigspeech_data && git checkout asr-offline
cd ..

# Clone other repos
git clone git@code.byted.org:seed/samantha.git
git clone git@code.byted.org:lab-audio/sami_ai_models.git
cd sami_ai_models && git checkout jcw/mss
# install torch-museval in sami_ai_models
git submodule update --init
pip3 install ./recipes/mss/torch-museval
cd ..

# Clone sami_tts_api
git clone git@code.byted.org:lab-audio/sami_tts_api.git

mkdir bigspeech_recipes

# TODO create default.yaml
cd samantha && git checkout data/preprocessing
cp scripts/data_processing/bigmusic/default.yaml ../bigspeech_recipes/
git checkout master
cd ..

