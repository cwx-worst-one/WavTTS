
# mode=$1
# offline_root=$2
# layer=$3
# exp_name=$4

mode=bn
offline_root=/mnt/bn/cyz-lq-nas/s3prl_gendir/stage3_v1.6.1
layer=0
exp_name=stage3_v1.6.1_beforevq
type=offline_bn


if [ "$mode" = "bn" ]; then
CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m train -u $type -d asr --offline_root $offline_root/LibriSpeech -l $layer -n ASR_$exp_name &
# CUDA_VISIBLE_DEVICES=1 python3 run_downstream.py -m train -u $type -d sv_voxceleb1 --offline_root $offline_root/Vox1 -l $layer -n ASV_$exp_name &
CUDA_VISIBLE_DEVICES=1 python3 run_downstream.py -m train -u $type -d emotion --offline_root $offline_root/IEMOCAP -l $layer -n EmoC_$exp_name

fi


if [ "$mode" = "token" ]; then
python3 run_downstream.py -m train -u offline_token -d sv_voxceleb1 -n yuanzhe_v0.3.2_75k
fi


# if [ "$mode" = "eval" ]; then
# python3 run_downstream.py -m evaluate -n test_ASR_token  -t "test-clean" -e /mnt/bn/cyz-lq-nas/project/samantha/recipes/s3prl/s3prl/result/downstream/test_ASR_token/dev-clean-best.ckpt
# fi
