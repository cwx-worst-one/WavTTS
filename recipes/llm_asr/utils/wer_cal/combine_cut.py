import os,sys
import logging
import argparse

logging.getLogger().setLevel(logging.INFO)


def merge_seg_hyp(hyp_file, new_hyp_file):

    hyp_dict = {}
    seg_line_count = 0
    merged_line_count = 0

    with open(hyp_file,  "r") as fp:
        for line in fp:
            line = line.rstrip().split()
            key = line[0]
            arr = key.split("_")
            audio_seg_rank = int(arr[-1])
            audio_name = '_'.join(arr[:-1])
            hyp = " ".join(line[1:])

            if audio_name not in hyp_dict:
                hyp_dict[audio_name] = []

            hyp_dict[audio_name].append((audio_seg_rank, hyp))
            seg_line_count += 1


    with open(new_hyp_file, "w") as fp:
        
        for key, value in hyp_dict.items():
            value.sort(key=lambda tup: tup[0])
            result = " ".join([v[1] for v in value])
            
            fp.write(f"{key} {result}")
            fp.write("\n")
            merged_line_count +=1

    
    logging.info("Number of segments:{}".format(seg_line_count))
    logging.info("Number of merged sentence:{}".format(merged_line_count))


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_file', required=True)
    parser.add_argument('--merged_hyp_file', required=True)
    args = parser.parse_args()
    # hyp_file = sys.argv[1]
    # merged_hyp_file = sys.argv[2]
    # hyp_file =  "./temp/text_ckpt50"
    # merged_hyp_file = "./temp/combined_text"
    merge_seg_hyp(args.hyp_file, args.merged_hyp_file)
