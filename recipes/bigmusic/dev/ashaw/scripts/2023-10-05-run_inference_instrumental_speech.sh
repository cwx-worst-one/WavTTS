export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH="${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel"

# video will be stored here
output_path=assets/generated_outputs_300k_instrumental_speech_v2
video_path=$output_path/video
mkdir -p $video_path

# prompt set to run on. suno100.csv - default. sanity-check-30s-50.csv - easy check
prompt_path=/mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv

# model list can be model folders or exact checkpoint paths
model_list_path=recipes/bigmusic/dev/ashaw/scripts/2023-10-05-run_inference_instrumental_speech_ckpts.txt

for model_path in `cat $model_list_path`
do

model_name=$(basename "$model_path")
extension="${model_name##*.}"

if [ $extension = "ckpt" ]; then # checkpoint path
    semantic=$model_path
    model_name=$(basename $(dirname $(dirname $model_path)))
else # folder path
    semantic=`find "${model_path}/checkpoints/" -type f -printf "%T@ %p\n" | grep step | sort -n | cut -d' ' -f 2- | tail -n 1 `
fi
timestamp=$(stat -c %Y "$semantic")
formatted_timestamp=$(date -d "@$timestamp" "+%Y-%m-%d-%H-%M")
aff=$(basename "$semantic")
affix=${model_name}__${aff}__${formatted_timestamp}
output_dir=$output_path/$affix
mkdir -p $output_dir
echo " "
echo "#####" $semantic "###### $formatted_timestamp"
echo $affix

echo " "

set -x 
launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt $semantic --extra_params.output_dir $output_dir \
    --extra_params.prompt_path $prompt_path --predict_dataset.use_mulan_v2_genres False

## Alternative method - works the same
# launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 recipes/diffusion/inference_from_text_lyrics2song.py \
#     --semantic_model_path $semantic --output_dir $output_dir \
#     --input_prompt_path $prompt_path

cp $output_dir/vocal_music_demo.mp4 $video_path/vocal_music_demo_${affix}.mp4
sleep 5
#####################

done
