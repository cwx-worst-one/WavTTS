# Unified modeling for musiclm and lyrics_to_song (instrumental + mixture)

# Firstime Setup
```bash
bash recipes/bigmusic/bootstrap.sh
```

* If you are on Merlin, add the following to your .bashrc file:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH="${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel"
```

## Training

`python3 -m samantha.main fit --config recipes/bigmusic/conf/semantic_modeling/semantic_model_token_bestrq_30s.yaml`

## Inference

See `recipes/bigmusic/scripts/run_inference_mwer.sh` for latest inference commands

`python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml --extra_params.output_dir ./generated_outputs`

### Inference GT

See `recipes/bigmusic/scripts/run_inference_gt.sh` for latest GT inference commands
