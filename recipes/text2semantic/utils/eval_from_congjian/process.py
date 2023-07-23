import sys
import os

tos_file=sys.argv[1]

def tos_utt2url(tos_file):
    utt2url = {}
    for line in open(tos_file, "r").readlines():
        utt, url = line.strip().split(": ")
        utt2url[os.path.splitext(utt)[0]] = url
    return utt2url


utt2url=tos_utt2url(tos_file)
gtutt2url=tos_utt2url("gt_tos.lst")

fout = open("artifact_rates.txt", "w")

for i, line in enumerate(open("meta.lst", "r").readlines()):
    utt, text = line.strip().split("|")
    url = utt2url[utt]
    if i > 5:
        domain = "asr"
    else:
        domain = "hqtts"
    fout.write(f"{domain}|{utt}|{text}|{url}\n")
fout.close()

# fout = open("similarity_mos.txt", "w")
# fout.write(f"domain|utt|text|reference_audio|gt|valle\n")
# for i, line in enumerate(open("meta.lst", "r").readlines()):
#     utt, text = line.strip().split("|")
#     url = utt2url[utt]
#     gt_url = gtutt2url[utt]
#     if i > 5:
#         domain = "asr"
#     else:
#         domain = "hqtts"
#     fout.write(f"{domain}|{utt}|{text}|{gt_url}|{gt_url}|{url}\n")
# fout.close()


# fout = open("cmos.txt", "w")
# fout.write(f"domain|utt|text|audio_a|audio_b\n")
# for i, line in enumerate(open("meta.lst", "r").readlines()):
#     utt, text = line.strip().split("|")
#     url = utt2url[utt]
#     gt_url = gtutt2url[utt]
#     if i > 5:
#         domain = "asr"
#     else:
#         domain = "hqtts"
#     fout.write(f"{domain}|{utt}|{text}|{url}|{gt_url}\n")
# fout.close()