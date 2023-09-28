launch --noupdate --type a100-80g --memory 40 --cpu 8 -- \
python3 recipes/bigmusic/scripts/gradio_demo.py \
recipes/bigmusic/conf/inference_semantic_30s.yaml
