### zh
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/ttsfe_multi_lingual_algorithm
python3 ./tts_zh_en_data/lang_zhcn/sbank/e4_prepare_for_sbank.py \
-i /mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_ZH-TTS/data/${spk}/for_tacofrontend/manual_prosodylabeling.txt \
-o /mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_ZH-TTS/data/${spk}/for_tacofrontend/manual_prosodylabeling_clean.txt

python3 ./tts_zh_en_data/lang_zhcn/sbank/split_prosody_file.py \
/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_ZH-TTS/data/${spk}/for_tacofrontend/manual_prosodylabeling_clean.txt 8

for rank in {0..7};do
    python3 ./tts_frontend/write_lab_for_sbank.py \
        --input_txt_file=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_ZH-TTS/data/${spk}/for_tacofrontend/manual_prosodylabeling_clean_$rank.txt \
        --output_dir=/mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_ZH-TTS/data/${spk}/tacofrontend_new_v3_jingbiao \
        --pwpp_formatter_hparams="do_organz_hierarchically=True" \
        --g2p_formatter_hparams="phoneme_set_file=./tts_zh_en_data/lang_zhcn/g2p/infos/sbank_add_cmu_phoneme_set.csv,phn_mapping_file=None,is_with_syll_info=True,is_syll_smallest_unit=True,phn_lang_prex=C0,phn_lang_prex_en=E0,do_lexphn_lower_case=True,do_en_stress_to_tone=True,do_pad_for_erhuayin=True,do_map_pinyin_as_online=True" \
        --enhance_formatter_hparams="" \
        --taco_lab_version='V3' \
        --language='zhcn' \
        --is_on_syll=True \
        --is_tone_stress=False \
        --use_wordseg=True \
        --use_enhance=False \
        --is_pwpp1_pred=True \
        --treat_wseg_as_pwpp1=False \
        --ignore_pwpp2=False \
        --ignore_pwpp3=False \
        --insert_sp_at_pwpp3=True \
        --map_insent_eos_to_comma=True \
        --do_punkt_pwpp_map=False \
        --gpu_ids=$rank \
        &> /mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_ZH-TTS/data/${spk}/for_tacofrontend/log.$rank &
done
wait
cd -

### en
# git clone git@code.byted.org:lab-audio/bytespeech.git -b $YOUR_DIR
# put prosody file to one dir, such as: /mnt/bn/huangzhiying-nas-bigtts-data/data/bigtts/BigSpeech_EN-TTS/data/SAMI_will/for_tacofrontend/manual_prosodylabeling.txt
# tacofrontend_new_v3_jingbiao
whose_style=ht # 海天标注用ht，标贝标注用bb，经过测试大部分都是ht
with_sentence_type=True # prosody文件中如果有sentype，那么是True，否则是False
prosody_dir=$YOUR_PROSODY_DIR
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bytespeech_fcl_dev/code/extract_features/multi_language # $YOUR_DIR/code/extract_features/multi_language
bash ./gen_lab_hzy_v3.sh $prosody_dir \
                        $prosody_dir/temp \
                        EN \
                        temp \
                        ${whose_style} \
                        ${with_sentence_type}
cd -
# 有一些tacolab的tab是缺失的（工具bug），需要执行这一步来normalize lab
python3 /mnt/bd/huangzhiying-lq-bigtts-frontend/data/bigtts/utils/add_tab_to_v3_tacolab.py \
        $prosody_dir/temp/wav_lab/labels/temp \
        $prosody_dir/../tacofrontend_new_v3_jingbiao

# new_labv3_punc + new_labv3_jingbiao => new_labv3_punc_jingbiao
python3 /mnt/bd/huangzhiying-lq-bigtts-frontend/data/bigtts/utils/merge_jingbiao_with_punc_v3.py \
        /mnt/bd/huangzhiying-lq-bigtts-frontend/data/bigtts/BigSpeech_EN-TTS/data/${spk}/tacofrontend_new_v3_jingbiao \
        /mnt/bd/huangzhiying-lq-bigtts-frontend/data/bigtts/BigSpeech_EN-TTS/data/${spk}/tacofrontend_new_v3_punc \
        /mnt/bd/huangzhiying-lq-bigtts-frontend/data/bigtts/BigSpeech_EN-TTS/data/${spk}/tacofrontend_new_v3_punc_jingbiao
