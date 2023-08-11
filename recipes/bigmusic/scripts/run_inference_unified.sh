#!/bin/bash
OUTPUT_DIR=$1
mkdir -p $OUTPUT_DIR

python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir $OUTPUT_DIR/semantic_intrumental --extra_params.inference_conditions mulan_text --extra_params.prompt_path recipes/bigmusic/datasets/inference_prompts/instrumental.json  --extra_params.is_lyrics_prompt True

# python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir $OUTPUT_DIR/semantic_lyrics_vocal --extra_params.inference_conditions mulan_text,lyrics_tokens --extra_params.prompt_path recipes/bigmusic/datasets/inference_prompts/instrumental.json --extra_params.is_lyrics_prompt True

# python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir $OUTPUT_DIR/semantic_lyrics_vocal_pop --extra_params.inference_conditions mulan_text,lyrics_tokens --extra_params.prompt_path recipes/bigmusic/datasets/inference_prompts/pop.json --extra_params.is_lyrics_prompt True

# python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir $OUTPUT_DIR/semantic_intrumental_musiclm --extra_params.inference_conditions mulan_text --extra_params.prompt_path /mnt/bn/audio-diffusion/data/google_prompts/text_prompt_collection_20230713.csv  --extra_params.is_lyrics_prompt False

# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd
