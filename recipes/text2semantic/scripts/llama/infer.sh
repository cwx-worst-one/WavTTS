
# export CUDA_VISIBLE_DEVICES=0


################################### 测试： 
# inner 
out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/inner_VFVAE_LT_0719_mae  # 改成你保存bestrq的路径
mkdir $out_dir

python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic.py \
--ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/WFVAE_LT/debug_mae/0.0.1/checkpoints/epoch=207-step=65000-kl_loss=0.00.ckpt \
--meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_inner/new_testset_inner_metalist_WFVAE_LT.txt \
--device=cuda \
--out_dir=$out_dir


# out
# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/out_avg_epoch22
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/average_model/enTTS_libritts460_librilight_32A100/0.0.1/checkpoints/epoch=22-step=47000-mae_loss=0.09.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_out/new_testset_out_metalist.txt \
# --device=cuda \
# --out_dir=$out_dir

#############################################


# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/test_delta
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/SFT/SFT_deltal12Loss_8A100/0.0.1/checkpoints/epoch=567-step=134000-mae_loss=0.29.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/will_mos/kat_will_mos_metalist.txt \
# --device=cuda \
# --out_dir=$out_dir



# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output/pretrain_kat_indomain
# mkdir $out_dir

# python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/valle/scripts/llama/infer_text2semantic.py \
# --ar_ckpt_path=/mnt/bn/cyz-lq-nas/work_dir/text2semantic/average_model/enTTS_libritts460_librilight_32A100/0.0.1/checkpoints/last.ckpt \
# --meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/indomain/indomain_metalist.txt \
# --device=cuda \
# --out_dir=$out_dir
