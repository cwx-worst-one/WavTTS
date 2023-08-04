# Unified modeling for musiclm and lyrics_to_song (instrumental + mixture)

## Training

`python3 -m samantha.main fit --config recipes/bigmusic/conf/semantic_modeling/semantic_model.yaml`

## Inference

`python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml --extra_params.output_dir ./generated_outputs`
