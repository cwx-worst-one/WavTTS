
def collate_1d(values, pad_idx=0, left_pad=False, max_len=None):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        crop_len = min(len(v), size)
        copy_tensor(v[:crop_len], res[i][size - len(v):] if left_pad else res[i][:crop_len])
    return res


def collate_2d(values,
               pad_idx=0,
               left_pad=False,
               shift_right=False,
               max_len=None):
    # list [[C, T1], [C, T2], [C, T3]]
    """Convert a list of 2d tensors into a padded 3d tensor."""
    # size = max(v.size(0) for v in values) if max_len is None else max_len
    size = max(v.size(1) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), values[0].shape[0], size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), "{} {}".format(src.shape, dst.shape)
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        t = v.shape[1]
        crop_len = min(t, size)
        copy_tensor(v[:, :crop_len], res[i, :, size - t:] if left_pad else res[i, :, :crop_len])
    return res
