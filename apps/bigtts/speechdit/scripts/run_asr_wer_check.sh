#!/bin/bash
set -x

the_dir="$1"
the_lid="$2"
the_lang_code="$3"
export cur=$(pwd)

echo "1. request_whisper_given_lid"
python3 tools/request_whisper_given_lid.py "$the_dir" "$the_lid" 2>/dev/null
line_count_1=$( cat ${the_dir}/wav_lst.txt | wc -l )
line_count_2=$( cat ${the_dir}/whisper_hyp.txt | wc -l )
if [[ "$line_count_1" -ne "$line_count_2" ]]; then
    echo "line_count_1 != line_count_2"
    exit 1
fi

echo "2. speech_evals"
cd /opt/tiger
python3 speech_evals/speech_evals/run.py \
    --config configs.asr.wer.${the_lang_code} \
    --ref_file ${the_dir}/ast_res.txt \
    --hyp_file ${the_dir}/whisper_hyp.txt \
    --out_file ${the_dir}/wer.txt

tail ${the_dir}/wer.txt

echo "3. dit_cut_check_multilingual"
cd $cur
python3 tools/dit_cut_check.py \
    ${the_dir}/whisper_hyp.txt \
    ${the_dir}/ast_res.txt \
    ${the_lang_code} \
    ${the_dir}/cut_uttid.txt \
    'char+wer'
