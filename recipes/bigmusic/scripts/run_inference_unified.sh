#!/bin/bash
OUTPUT_DIR=$1
mkdir -p $OUTPUT_DIR

python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir $OUTPUT_DIR/semantic_intrumental --extra_params.conditions mulan_text --extra_params.prompt_path recipes/bigmusic/datasets/inference_prompts/instrumental.json

python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir $OUTPUT_DIR/semantic_lyrics_novocal --extra_params.conditions mulan_text,lyrics_tokens --extra_params.prompt_path recipes/bigmusic/datasets/inference_prompts/instrumental.json


# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd
