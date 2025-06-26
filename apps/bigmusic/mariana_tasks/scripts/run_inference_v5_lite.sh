pip3 install websockets emoji nest_asyncio sortedcontainers --no-deps;
pip3 install confusables prettytable pretty_midi thop mido --no-deps
pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/ --no-deps
export LITE_USE_PL_MODULE=1
semantic_ckpt="hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangshuo/bigmusic/ar/2025032216/version_43131954/checkpoints/epoch=0-step=40000.pt"
tokenizer_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/hanoi.hantrakul/logs/umm_conformer_unified_tts_rope/umm_stage3_unified_tts_rope_bert-base-multilingual-uncased_EMAEntropy32768x32/checkpoints/step=220000.ckpt"

# # download
hdfs dfs -get -c 128 --ct 128 ${semantic_ckpt} ./model.pt;
# split weight
python3 ./apps/bigmusic/mariana_tasks/utils/split_weight.py \
    --ckpt model.pt \
    --text_llm_ckpt ./llm.pt \
    --audio_ckpt ./emb.pt


hdfs dfs get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/assets/ar/eval0214_lyrics2song_zh_full_v5_160_v5.json ./ ;

output_dir="./results"
current_time=$(date  "+%Y%m%d-%H%M%S%4N");
out_dir=${output_dir}/${current_time};
bash ./apps/bigmusic/zh_vocal/env_preprocess.sh;


prompt_path="./eval0214_lyrics2song_zh_full_v5_160_v5.json"


bash ./launch.sh predict \
 --config /opt/tiger/samantha/apps/bigmusic/mariana_tasks/conf/v5_inference_dev_vocal_lite.yaml \
 --extra_params.network_cfg apps/bigmusic/mariana_tasks/conf/v5_m8_680m.yaml \
 --extra_params.tokenizer_path ${tokenizer_path} \
 --extra_params.emit_eos_thresh_secs 5 \
 --extra_params.exclude_eos_thresh_secs 1 \
 --extra_params.exclude_eos_first_secs 20 \
 --extra_params.stop_eos True \
 --extra_params.emb_path "./emb.pt" \
 --extra_params.llm_path "./llm.pt" \
 --extra_params.semantic_ckpt ${semantic_ckpt} \
 --extra_params.prompt_path ${prompt_path} \
 --extra_params.output_dir ${out_dir} \
 --extra_params.repetition_penalty 1.6 \
 --extra_params.duration 240 \
 --extra_params.lyrics_max_seq_len 4000 \
 --extra_params.use_offline_preprocess False \
 --run_opts.batch_size 1 \
 --extra_params.controller_cfg_label "[genre, genre_extra, mood, speaker, lyrics, scene, voice, sinking, freeform_text, line_break, instrument]" \
 --extra_params.rewrite_target "13_cat_combo_v5"
