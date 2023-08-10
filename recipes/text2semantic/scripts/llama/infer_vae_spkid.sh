export CUDA_VISIBLE_DEVICES=2

# 集内
out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output3/Dacey_emotion_shortlong0804 # 改成保存路径
mkdir $out_dir

python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/text2semantic/scripts/llama/infer_text2semantic_vae_spkid.py \
--ar_ckpt_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/chenyuanzhe/text2semantic/sft_bt45000_8A100_accu1_noreg_dacey_0801_gpt_emotion/checkpoints/epoch=00-step=145000-kl_loss=0.32.ckpt \
--meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/short_long_0804/Dacey_emotion_metalist.txt \
--device=cuda \
--out_dir=$out_dir
