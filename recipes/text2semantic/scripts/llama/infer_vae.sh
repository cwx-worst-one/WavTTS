cuda_id=5
testset=out
export CUDA_VISIBLE_DEVICES=${cuda_id}


# out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output3/sft_${testset}_wd0.0_180k
out_dir=/mnt/bn/cyz-lq-nas/output_dir/t2s_output3/pretrain0.9b_${testset}_wd0.1_120k
mkdir $out_dir

python3 /mnt/bn/cyz-lq-nas/project/samantha/recipes/text2semantic/scripts/llama/infer_text2semantic_vae.py \
--ar_ckpt_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/chenyuanzhe/text2semantic_VAELLaMa/pretrain_wd0.1_0.9b/checkpoints/epoch=00-step=120000-kl_loss=0.58.ckpt \
--meta_file=/mnt/bn/cyz-lq-nas/input_dir/t2s_input/new_testset_${testset}/new_testset_${testset}_metalist_WFVAE_LZX.txt \
--device=cuda \
--out_dir=$out_dir

