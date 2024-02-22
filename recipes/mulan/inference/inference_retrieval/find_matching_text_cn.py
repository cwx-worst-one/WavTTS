import os
import glob
import pickle
import numpy as np
from collections import Counter
import shutil


audio_folder = "audio_samples"
audio_embed_folder = "audio_embeds_mulan_30ks"
text_embed_folder = "text_embeds_mulans"


audio_files = glob.glob(f"{audio_embed_folder}/*.npy")

# i-th row of text_embed corresponds to i-th string in text_pool
[text_pool_1, text_embeds_1] = pickle.load(
    open(f"{text_embed_folder}/text_pool0.pkl", "rb")
)
# text_embeds_1  (50, 512)


def process_one_audio(audio_emb_file):
    # Assume audio length is T seconds, audio_embeds will be a [T//10, 512] embedding array.
    audio_embeds = np.load(audio_emb_file)

    # calculate MxN similarity matrix with text pool 1
    sims_text_1 = np.matmul(audio_embeds, text_embeds_1.T)
    text_ids = np.argmax(sims_text_1, axis=1)
    labels_1 = dict(Counter([text_pool_1[x] for x in text_ids]))

    return labels_1
def find_most_similar_audio_embeddings(audio_embed_folder, text_embed_folder, text_pool_1):
    # Load text embeddings and text pool
    [text_pool, text_embeds] = pickle.load(open(f"{text_embed_folder}/text_pool0.pkl", "rb"))

    # Get a list of audio files
    audio_files = glob.glob(f"{audio_embed_folder}/*.npy")

    # Initialize a dictionary to store the most similar audio embeddings for each text embedding
    most_similar_audio_embeddings = {}
    most_similar_audio_name = []


    # Iterate through each row of text_embeds
    for i, text_embed in enumerate(text_embeds):
        # Initialize variables to keep track of the most similar audio embedding and its similarity score
        best_audio_embed = None
        best_similarity = -1  # Initialize to a very low value

        # Iterate through each audio file
        for audio_file in audio_files:
            # Load the audio embedding
            audio_embed = np.load(audio_file)
            audio_name = audio_file.split(".")[0]
            audio_name = audio_name.split("/")[-1]
            # Calculate the cosine similarity between the text embed and the audio embed
            similarity = np.dot(text_embed, audio_embed.T) / (np.linalg.norm(text_embed) * np.linalg.norm(audio_embed))

            # Update the best similarity and best audio embedding if the current audio embedding is more similar
            if similarity > best_similarity:
                best_similarity = similarity
                best_audio_embed = audio_embed
                best_audio_name = audio_name
            destination_folder = "audio_res_1214"
        best_text_name = text_pool_1[i]+ ".wav"
        best_text_name = best_text_name.replace("/", "")
        best_audio_name_2 = best_audio_name + ".wav"
        shutil.copy("audio_samples/"+best_audio_name+".wav", f"{destination_folder}/{best_audio_name_2}")
        # Store the most similar audio embedding for the current text embedding
        most_similar_audio_embeddings[text_pool[i]] = best_audio_embed
        most_similar_audio_name.append("http://tosv.byted.org/obj/cb-bucket-us/"+best_audio_name+".wav")



    return most_similar_audio_embeddings, most_similar_audio_name

audio_embed_folder = "audio_embeds_mulan_30ks"
text_embed_folder = "text_embeds_mulans"
result, most_similar_audio_name = find_most_similar_audio_embeddings(audio_embed_folder, text_embed_folder, text_pool_1)
print("most_similar_audio_name", most_similar_audio_name)
print("text_pool_1", text_pool_1)
def util_most_frequent_tag(stat):
    freq = 0
    label = None
    for k in stat:
        if stat[k] > freq:
            label = k
            freq = stat[k]
    return label
csv_filename = "res_cn_MuLan_1214.csv"
import csv
# 打开CSV文件并写入数据
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['text_pool_1', 'most_similar_audio_name'])
    
    # 将数据逐行写入CSV文件
    for text, audio_name in zip(text_pool_1, most_similar_audio_name):
        writer.writerow([text, audio_name])

print(f"数据已写入 {csv_filename}")
'''
summary = []
for audio_file in audio_files:
    row = [audio_file.split("/")[-1].split(".")[0]]
    row.append(
        "http://tosv.byted.org/obj/music2vibes/"
        + row[0]
        + ".wav"
    )
    labels = process_one_audio(audio_file)
    print(labels)
    for label in labels:
        label = util_most_frequent_tag(label)
        assert label is not None
        row.append(label)
    if "accordian" in row:
        print(f"\n{row[0]}")
        for lab in labels:
            print(lab)
    summary.append(row)

import pandas as pd

df = pd.DataFrame(
    summary,
    columns=["music_id", "music_url", "label_1"],
)

# df.to_csv("/mnt/bn/mm-data/user/xuchen.song/mulan_tag/inference_output.csv")
df.to_csv("mulan_cn_inference_output.csv")
'''