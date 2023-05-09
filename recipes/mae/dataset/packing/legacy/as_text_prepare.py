import os
import pickle

import pandas as pd


def process_audioset_meta():
    # Process label
    label_path = "audioset/metadata/class_labels_indices.csv"
    df = pd.read_csv(label_path)
    label_name = df["display_name"].values
    code = df["mid"].values
    label_dict = {i: j for i, j in zip(code, label_name)}

    # Process meta
    data_dict = {}
    path_0 = "audioset/metadata/unbalanced_train_segments.csv"
    path_1 = "audioset/metadata/eval_segments.csv"
    path_2 = "audioset/metadata/balanced_train_segments.csv"
    for path in [path_0, path_1, path_2]:
        # skip first 3 lines
        print("Processing: ", path)
        columns = ["YTID", "start_seconds", "end_seconds", "positive_labels"]
        df = pd.read_csv(
            path, skiprows=3, names=columns, quotechar='"', skipinitialspace=True
        )
        df["positive_labels"] = df["positive_labels"].apply(lambda x: x.split(","))
        df["positive_labels"] = df["positive_labels"].apply(
            lambda x: [label_dict[i] for i in x]
        )
        df["positive_labels"] = df["positive_labels"].apply(lambda x: ",".join(x))
        print("Converting to dict")
        df_d = df.set_index("YTID").to_dict("index")
        for k, v in df_d.items():
            data_dict[k] = v["positive_labels"]
        print("Done")
    return data_dict


if __name__ == "__main__":
    # Download from hdfs
    # meta: hdfs://harunava/home/byte_speech_sv/mulan/audioset/metadata

    os.makedirs("audioset", exist_ok=True)
    # Check if the metadata folder is empty
    if len(os.listdir("audioset/metadata")) == 0:
        os.system(
            "hdfs dfs -get hdfs://harunava/home/byte_speech_sv/"
            "mulan/audioset/metadata audioset"
        )

    # process meta
    data_dict = process_audioset_meta()
    # take a look
    for i, (key, value) in enumerate(data_dict.items()):
        print(key, value)
        if i > 10:
            break

    with open("audioset/audioset_meta.pkl", "wb") as f:
        pickle.dump(data_dict, f)
