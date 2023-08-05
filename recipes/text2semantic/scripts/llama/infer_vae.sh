export CUDA_VISIBLE_DEVICES=1

# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output2/inner_phoneme65k # 改成保存路径
out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output2/out_nodrop_sft_80k 
mkdir $out_dir

python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/text2semantic/scripts/llama/infer_text2semantic_vae.py \
--ar_ckpt_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/chenyuanzhe/text2semantic/sft_bt45000_8A100_accu1_noreg_nospkid/checkpoints/epoch=00-step=80000-kl_loss=0.49.ckpt \
--meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_out/new_testset_out_metalist_WFVAE_LZX.txt \
--device=cuda \
--out_dir=$out_dir

