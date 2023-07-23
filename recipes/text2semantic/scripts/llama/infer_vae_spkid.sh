export CUDA_VISIBLE_DEVICES=1

# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/vae_dacey_will_mos
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/DaceyNew/0.0.1/checkpoints/epoch=106-step=45000-kl_loss=0.04.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/will_mos/kat_will_mos_metalist_WaveformVAE.txt \
# --device=cuda \
# --out_dir=$out_dir



# 集内
out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/Dacey_AR_testset_0718_2 # 改成保存路径
mkdir $out_dir

python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae_spkid.py \
--ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_spkid_from_55k/0.0.1/checkpoints/last.ckpt \
--meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/AR_testset/Dacey_metalist.txt \
--device=cuda \
--out_dir=$out_dir


# 集外
# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/vae_out_0714_2 # 改成保存路径
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic_vae.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae/avg_32A100_0712/0.0.1/checkpoints/last.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_out/new_testset_out_metalist_WaveformVAE.txt \
# --device=cuda \
# --out_dir=$out_dir

# /mnt/bn/cyz-lq-nas/input_dir/t2s_input/will_mos/Dacey_will_mos_metalist_WaveformVAE_spkid.txt