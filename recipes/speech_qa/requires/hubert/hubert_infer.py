import torch



def hubert_cal_conv_out_dim(length):
    for _, k, s in eval('[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2'):
        length = (length - k) // s + 1
    return length


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Examples: With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))
    bs = lengths.size()[0]
    maxlen = lengths.max()
    # if not isinstance(lengths, list):
    #     lengths = lengths.tolist()
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths.cpu()).unsqueeze(-1)

    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


# """
@torch.no_grad()
def hubert_tokenization(hubert_model, wavs, wav_lens, centers, device):
    b, t = wavs.size()
    
    batch_data_feat_length = hubert_cal_conv_out_dim(wav_lens)
    batch_data_feat_mask = make_pad_mask(batch_data_feat_length)

    hubert_embeds, mask = hubert_model(
        wavs.to(device), wav_lens.to(device), batch_data_feat_mask.to(device))

    # kmeans
    b, t, d = hubert_embeds.shape
    # dataset = hubert_embeds.view([b * t, d])
    dataset = hubert_embeds.reshape([b * t, d])
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


# """
"""
@torch.no_grad()
def w2v_bert_tokenization(frontend, w2v_model, wavs, centers, device):
    b, t = wavs.size()
    w2v_embeds, _ = w2v_model(wavs, torch.LongTensor([t]).to(device).repeat(b))
    b, t, d = w2v_embeds.shape
    dataset = w2v_embeds.view([b * t, d])
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
"""
