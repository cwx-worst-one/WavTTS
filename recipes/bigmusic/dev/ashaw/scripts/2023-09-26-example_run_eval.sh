
OUTPUT_DIR=results/generated_outputs/example_run
PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/multi-tag-30s.csv # latest rewrite
FOLDER_NAME=$(basename $PROMPT_PATH .csv)
mkdir -p $OUTPUT_DIR

## Note: in order to run ASR metrics, you need to first setup ASR pipeline.
# Option 1: if you are in US cluster with access to audio-diffusion bytenas... 
#           run bash recipes/bigmusic/bootstrap.sh predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml .....
# Option 2: if running from arnold, you can setup ASR by following this doc: https://bytedance.sg.feishu.cn/docx/LRBCdwjcboV9eZxETqGlN7w9gMc


# Run eval first
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR \
    --extra_params.semantic_ckpt "/mnt/bn/lyrics-to-song/qq/logs/semantic_model_mulan_text_07B/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt" \
    --extra_params.prompt_path $PROMPT_PATH  \
    --additional_callbacks [] # disables runing default metrics callbacks

# Run metrics seconds
python3 recipes/bigmusic/dev/ashaw/scripts/2023-09-26-example_run_metrics.py --results_dir $OUTPUT_DIR