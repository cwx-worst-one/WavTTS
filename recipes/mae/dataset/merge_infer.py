import numpy as np
import pandas as pd

filename = "logs/music_emb_{rank}_avg.npz"
music_ids = []
music_vecs = []
for rank in range(8):
    data = np.load(filename.format(rank=rank))
    music_ids.extend(data["music_ids"].tolist())
    music_vecs.append(data["music_vecs"])

music_vecs = np.concatenate(music_vecs, axis=0).astype(np.float32)

np.save("logs/music_emb_avg.npy", music_vecs)

meta = []
for id in music_ids:
    # music_id,music_title,music_author,genre,theme,mood,music_id_tos_url
    meta.append(
        {"music_id": id, "music_title": "", "music_author": "", "music_id_tos_url": ""}
    )
df_meta = pd.DataFrame(meta)
df_meta.to_csv("logs/music_emb_meta.csv", index=False)
