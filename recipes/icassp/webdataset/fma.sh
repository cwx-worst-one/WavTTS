sudo apt -q update && sudo apt -q install -y libsndfile1 ffmpeg
pip3 install -r requirements.2.txt
pip3 install tqdm
pip3 install librosa

python3 recipes/icassp/webdataset/fma.py $1 $2 $3