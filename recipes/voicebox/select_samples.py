import os

#exp="PrefixLDM3_400M_16A100_18wEN_10wCN_40hzMel_UMMv03/icl_testset_2.0_en/240000"
#exp="PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02_rerun/icl_testset_2.0_en/320000/DDIM_10steps_ununiform"
#exp="PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02_rerun/icl_testset_2.0_en/320000/DDIM_50steps"
#exp="PrefixLDM3a_ummdropout02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02Vector/icl_testset_2.0_en/160000/DDIM_10steps_ununiform"
exp="DualCondNet2_250M_16mixed_16H800_18wEN_10wCN_40hzMel_UMMv03/icl_testset_2.0_en/80000/DDIM_10steps_ununiform"

prompt_dir = "/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/en/prompt-wavs"
gt_dir = "/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/en/wavs"
input_dir = "/mnt/bn/jdy-lq-2/bigtts-nar/output/"
output_dir = "/mnt/bn/jdy-lq-2/bigtts-nar/selected_output/"

input_dir = input_dir + exp
output_dir = output_dir + exp
os.makedirs(output_dir, exist_ok=True)

#sort
os.system("cat {} | sort -n -k 2 > {}".format(
    os.path.join(input_dir, "wav_res_ref_text.asv"),
    os.path.join(input_dir, "wav_res_ref_text.asv.sort"),
    ))

os.system("cat {} | sort -n -k 2 > {}".format(
    os.path.join(input_dir, "wav_res_ref_text.wer"),
    os.path.join(input_dir, "wav_res_ref_text.wer.sort"),
    ))

good_n = 0
bad_n = 20

#items = ["wer_good", "wer_bad", "asv_good", "asv_bad"]
items = ["wer_good", "wer_bad"]

with open(os.path.join(input_dir, "wav_res_ref_text.wer.sort"), "r") as f:
    lines = f.readlines()
wer_good_lines = lines[:good_n]
wer_bad_lines = lines[-bad_n:]
with open(os.path.join(input_dir, "wav_res_ref_text.asv.sort"), "r") as f:
    lines = f.readlines()
asv_good_lines = lines[-good_n:]
asv_bad_lines = lines[:bad_n]
#item_lines = [wer_good_lines, wer_bad_lines, asv_good_lines, asv_bad_lines]
item_lines = [wer_good_lines, wer_bad_lines]

for item, item_line in zip(items, item_lines):
    sub_output_dir = os.path.join(output_dir, item)
    os.makedirs(sub_output_dir, exist_ok=True)
    for line in item_line:
        line = line.strip()
        splits = line.split("\t")
        if len(splits) <= 1:
            continue
        if item.startswith("asv"):
            generate_wav_path = splits[0].split("|")[0][:-5]
            prompt_wav_path = splits[0].split("|")[1][:-5]
        else:
            generate_wav_path = splits[0]
            prompt_wav_path = os.path.join(prompt_dir, os.path.basename(generate_wav_path).split("-")[0]+".wav")
            gt_wav_path = os.path.join(gt_dir, os.path.basename(generate_wav_path))
            os.system("cp {} {}".format(
                generate_wav_path,
                sub_output_dir
                ))
            os.system("cp {} {}".format(
                prompt_wav_path,
                sub_output_dir
                ))
            os.system("cp {} {}".format(
                gt_wav_path,
                os.path.join(sub_output_dir, os.path.basename(generate_wav_path).replace(".wav", "_gt.wav"))
                ))
            score = splits[-6]
            print(score)


