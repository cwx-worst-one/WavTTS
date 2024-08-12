set -e

#exp=${exp:-V2_block32_300M_8H800_setting0-24w_40hzWVAE_Diff4VCUMMv062_CFG0.1_maskToken_newBranch}
#ckpt=${ckpt:-"epoch=00-step=305000-loss=0.54.ckpt"}
#step=${step:-305000}

#exp=${exp:-V2_block32_300M_8H800_setting0-24w_40hzWVAE_Diff4VCUMMv062_CFG0.1_maskToken_newBranch_origSetup}
#ckpt=${ckpt:-"epoch=00-step=300000-loss=0.54.ckpt"}
#step=${step:-300000}

#exp=${exp:-V2_block32_300M_8H800_setting0_40hzWVAE_FormantAugUMM_maskToken}
#ckpt=${ckpt:-"epoch=00-step=300000-loss=0.52.ckpt"}
#step=${step:-300000}

exp=${exp:-V2_block32_300M_8H800_setting0-16w_40hzWVAE_Diff2VCUMM_maskToken}
ckpt=${ckpt:-"epoch=00-step=300000-loss=0.54.ckpt"}
step=${step:-300000}

lang=${lang:-zh}
nfe=10
sampler=ddim
cfg=2

out_dir=${out_dir:-/mnt/bn/jdy-lq-2/bigtts-nar/output}
#meta_lst=${meta_lst:-/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst}
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/${sampler}_cfg${cfg}_${nfe}steps
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/non_para_reconstruct_meta.lst
out_dir=$out_dir/$exp/non_para_icl_testset_2.0_${lang}/$step/${sampler}_CFG${cfg}_${nfe}steps
mkdir -p $out_dir

umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt

function get_file_from_branch() {
	# usage: get_file_from_branch <branch_name> <file_or_folder_path>
    branch_name=$1
    target_file=$2
    git fetch origin $branch_name
    git checkout origin/$branch_name -- $target_file
}

#mv recipes/umm recipes/umm_bkp
#rm -r recipes/umm
#get_file_from_branch speech-diffusion recipes/umm_062
#ln -s -r recipes/umm_062 recipes/umm

diffusion_ckpt_path=${diffusion_ckpt_path:-hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt}

umm_type=${umm_type:-UMM}
umm_frame_rate=${umm_frame_rate:-25}
bn_frame_rate=${bn_frame_rate:-40}

bash launch.sh predict \
	-c apps/bigtts/umm/diffusion/conf/infer_reconstruction_40hzMel.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.umm_frame_rate $umm_frame_rate \
    --run_opts.bn_frame_rate $bn_frame_rate \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
    --run_opts.infer_type diffusion-vocoder \
	--pl_module.umm_type $umm_type \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe $nfe \
	--pl_module.diffusion_sampler $sampler \
	--pl_module.use_wvae_vocoder True \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--bn_config.bn_norm_std 2 \
	--bn_config.bn_padding 0 \
	--pl_module.cfg_w $cfg \
	--pl_module.only_use_global_prompt False \
	--pl_module.mask_prompt_token True \

bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $lang

# use example:
# export diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/wangbo.zero/voicebox/logdir/PrefixLDM4_300M_8H800_setting0_40hzWVAE_zvqv2_textDrop0.25_Norm2/checkpoints/epoch=00-step=65000-loss=0.55.ckpt
# export exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_zvqv2_textDrop0.25_Norm2
# export ckpt=epoch=00-step=65000-loss=0.55.ckpt
# export step=60000

# export out_dir=/opt/tiger/samantha/output
# export umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/zvq/v2/wave2quantizedz_%d.pt
# export umm_type=ZVQ
# export umm_frame_rate=10
# export lang=zh

# bash -x apps/bigtts/umm/diffusion/scripts/recons_umm_wvae.sh
