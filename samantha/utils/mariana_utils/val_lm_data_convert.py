import collections

import pandas as pd
from pandas import read_parquet

data = read_parquet("val_lm_data.gz.parquet")
data = data.sample(frac=1)
print(data.head(3))
total = len(data)
print(total)

val_num = 5000
sessions = collections.defaultdict(list)
for i in range(val_num):
    sent = data["content_split"].loc[i]
    pairs = [{"prompt": "", "response": sent}]
    if i % 100 == 0:
        print(i, pairs)
    sessions["session"].append(pairs)

df = pd.DataFrame(sessions)
df.to_parquet("val_lm_data_5k.parquet")
