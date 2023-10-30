
mode=$1

if [ "$mode" = "train" ]; then
# python3 run_downstream.py -m train -u offline_token -d asr -n yuanzhe_v0.3.2_75k
python3 run_downstream.py -m train -u offline_bn -d asr -n ASR_usm_sft_fix2
fi

if [ "$mode" = "eval" ]; then
python3 run_downstream.py -m evaluate -n test_ASR_token  -t "test-clean" -e /mnt/bn/cyz-lq-nas/project/samantha/recipes/s3prl/s3prl/result/downstream/test_ASR_token/dev-clean-best.ckpt
fi
