
# evaluation.
data1_path=/mnt/bn/cjw-lq-1/project/kainan/umm/llama/out_wavs/speech_umm_sami_t0.9_p0.9_ar500k_zh_v0.6.2_spkid_v3_3_continuation_3.0B_200wh_icl_testset_3.0_seed1997/diff900k/wav_res_ref_text
data2_path=/mnt/bn/jdy-lq-2/bigtts-nar/output/DIT_1.3B_64H800_lr8e-5_setting3a_CFG0.1_QKNormHead_NoOverlap_NewMask/icl_testset_3.0_zh_4x/500000/ddim_25steps_CFG4_AdaLength_post/wav_res_ref_text
figure_path=/opt/tiger/compared_figures/
#figure_path=/mnt/bn/jdy-lq-2/bigtts-nar/output/DIT_1.3B_64H800_lr8e-5_setting3a_CFG0.1_QKNormHead_NoOverlap_NewMask/icl_testset_3.0_zh_4x/500000/ddim_25steps_CFG4_AdaLength_post/compared_figures/
mkdir -p $figure_path

label1="Baseline"
label2="DiT"

# 需修改;
bigtts_eval_dir=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts-eval/
cd $bigtts_eval_dir

python3 plot_hist.py $data1_path $data2_path $figure_path $label1 $label2 wer
python3 plot_hist.py $data1_path $data2_path $figure_path $label1 $label2 asv
python3 plot_hist.py $data1_path $data2_path $figure_path $label1 $label2 pitch
python3 plot_hist.py $data1_path $data2_path $figure_path $label1 $label2 dur

cd -
