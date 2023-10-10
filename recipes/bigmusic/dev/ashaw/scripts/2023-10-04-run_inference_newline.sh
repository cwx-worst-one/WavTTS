# video will be stored here
output_path=assets/generated_outputs_newline
video_path=$output_path/video
mkdir -p $video_path

# prompt set to run on. suno100.csv - default. sanity-check-30s-50.csv - easy check
prompt_path=/mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv
# prompt_path=/mnt/bn/audio-diffusion/data/mixture_prompts/goodcase_prompts.csv

# model list can be model folders or exact checkpoint paths
model_list_path=recipes/bigmusic/dev/ashaw/scripts/2023-10-02-model_list_newline_training.txt

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
    --extra_params.prompt_path $prompt_path --predict_dataset.remove_newlines False

## Alternative method - works the same
# launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 recipes/diffusion/inference_from_text_lyrics2song.py \
#     --semantic_model_path $semantic --output_dir $output_dir \
#     --input_prompt_path $prompt_path

cp $output_dir/vocal_music_demo.mp4 $video_path/vocal_music_demo_${affix}.mp4
sleep 5
#####################

done

# /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_mulan_tagging/mulan_110_mcc9m_no_newline/checkpoints/step=184000-tr_loss=3.5102-val_loss_0=5.5939.ckpt

# launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_mulan_tagging/mulan_110_mcc9m_newline/checkpoints/step=096000-tr_loss=3.6385-val_loss_0=5.5748.ckpt \
#     --extra_params.output_dir assets/generated_outputs_mulan_tag_110/mulan_110_mcc9m_newline__step=096000-tr_loss=3.6385-val_loss_0=5.5748 \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv \
#     --predict_dataset.enable_punctuation True


launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_mulan_tagging/mulan_110_style_mixed_mulan_tag_punctuation_varlen15s/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mulan_tag_110/mulan_110_style_mixed_mulan_tag_punctuation_varlen15s__step=150000 \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv \
    --predict_dataset.enable_punctuation True
