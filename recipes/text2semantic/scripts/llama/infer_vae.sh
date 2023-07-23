export CUDA_VISIBLE_DEVICES=0

out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/inner_VFVAE_LTV2_0720 # 改成保存路径
mkdir $out_dir

python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae.py \
--ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/WFVAE_LT/debug_V2/0.0.1/checkpoints/epoch=167-step=105000-kl_loss=0.94.ckpt \
--meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_inner/new_testset_inner_metalist_WFVAE_LTV2.txt \
--device=cuda \
--out_dir=$out_dir


# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/vae_dacey_will_mos
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/DaceyNew/0.0.1/checkpoints/epoch=106-step=45000-kl_loss=0.04.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/will_mos/kat_will_mos_metalist_WaveformVAE.txt \
# --device=cuda \
# --out_dir=$out_dir



# # 集内
# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/vae_out_0719_inner_after # 改成保存路径
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_from_51k_16A00/0.0.1/checkpoints/epoch=719-step=195000-kl_loss=0.16.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_inner/new_testset_inner_metalist_WaveformVAE.txt \
# --device=cuda \
# --out_dir=$out_dir


# 集外
# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/vae_out_0714_2 # 改成保存路径
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae/avg_32A100_0712/0.0.1/checkpoints/last.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_out/new_testset_out_metalist_WaveformVAE.txt \
# --device=cuda \
# --out_dir=$out_dir

