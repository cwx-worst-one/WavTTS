scores = ""
name = "decoder_mix_base-32768_bert-base-uncased-30522"
for i in range(1, 10):
    with open(f"mir9_log_{name}_{(i * 10000):>07}", "r") as log_file:
        for line in log_file.readlines():
            scores += f'{float(line.split("] ")[-1].strip()):.3f}'
            scores += "\t"
        scores += "\n"
with open(f"scores_table_{name}", "w") as score_file:
    score_file.write(scores)
