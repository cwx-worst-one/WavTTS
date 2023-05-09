import numpy as np

filename = "logs/music_emb_{rank}.npy"
all_data = []
for rank in range(8):
    data = np.load(filename.format(rank=rank))
    all_data.append(data)

np.save("logs/music_emb.npy", np.concatenate(all_data))
