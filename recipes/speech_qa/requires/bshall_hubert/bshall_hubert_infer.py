
import torch


# """
@torch.no_grad()
def bshall_hubert_tokenization(frontend, bshall_hubert_model, wavs, centers, device):
    b, t = wavs.size()
    input_wav = wavs.unsqueeze(1)
    embeds, _ = bshall_hubert_model.encode(input_wav, layer=7)

    # kmeans
    b, t, d = embeds.shape
    dataset = embeds.view([b * t, d])
    num_points = dataset.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)  # [1024, 1024]
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    inertia = 0
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
        inertia += distances[range(distances.shape[0]), min_ind].sum()
    codes = codes.view([b, t])
    return codes

