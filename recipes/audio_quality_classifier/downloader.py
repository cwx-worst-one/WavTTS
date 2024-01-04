import os
import pandas as pd

idx = 5
paths = [f'/opt/tiger/arnold_experiment/aq_label_{idx}.csv' ]

# df_0 = pd.read_csv(path_0, skiprows=1)
# df_1 = pd.read_csv(path_1, skiprows=1)
# df_2 = pd.read_csv(path_2, skiprows=1)


# df_0_urls = df_0['音频链接'].tolist()
# df_0_scores = df_0['为该音频的音质打分（参考“综合打分”对4个分数等级的说明）'].tolist()

# df_1_urls = df_1['音频链接'].tolist()
# df_1_scores = df_1['为该音频的音质打分（参考“综合打分”对4个分数等级的说明）'].tolist()

# df_2_urls = df_2['音频链接'].tolist()
# df_2_scores = df_2['为该音频的音质打分（参考“综合打分”对4个分数等级的说明）'].tolist()

# ignore the first row
df_urls_all = []
df_scores_all = []
for i, path in enumerate(paths):
    df = pd.read_csv(path, skiprows=1)
    df_urls = df['音频链接'].tolist()
    df_scores = df['为该音频的音质打分（参考“综合打分”对4个分数等级的说明）'].tolist()
    df_urls_all += df_urls
    df_scores_all += df_scores

output_dir = f'/opt/tiger/arnold_experiment/samantha/aq_mos_{idx}'
os.makedirs(output_dir, exist_ok=True)
for url, score in zip(df_urls, df_scores):
    # get num of fils in the directory
    num_files = len(os.listdir(output_dir))

    # download the file use wget
    cmd = f"wget -O {output_dir}/{num_files}_{score}.wav '{url}'"

    os.system(cmd)
