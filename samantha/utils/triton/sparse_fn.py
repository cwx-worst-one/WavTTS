import torch
from triton.ops.blocksparse import matmul as sparse_matmul
from triton.ops.blocksparse import softmax as sparse_softmax

sparse_fns = {}


def _get_sparse_fn(size, device):
    global sparse_fns
    num_heads, train_len, _ = size

    stride = 32
    pad_len = train_len % stride
    if pad_len != 0:
        pad_len = stride - pad_len
    train_len = train_len + pad_len

    name = "head:{}_len:{}_device:{}".format(num_heads, train_len, device)
    if name not in sparse_fns:
        print('Prepare sparse function "{}"'.format(name))
        layout, block = default_layout(train_len, num_heads)
        qk_matmul = sparse_matmul(
            layout, block, mode="sdd", trans_a=False, trans_b=True, device=device
        )
        softmax_fn = sparse_softmax(layout, block, device, is_dense=False)
        wv_matmul = sparse_matmul(
            layout, block, mode="dsd", trans_a=False, trans_b=False, device=device
        )
        sparse_fns[name] = [qk_matmul, softmax_fn, wv_matmul]
    return sparse_fns[name], pad_len


def default_layout(train_len, num_heads, block=32):
    if train_len % block != 0:
        train_len = train_len + (block - train_len % block)
    assert train_len % block == 0
    tq = torch.arange(train_len // block).unsqueeze(1)
    tk = torch.arange(train_len // block).unsqueeze(0)
    mask = (tq - tk) >= 0
    mask = mask.unsqueeze(0).repeat(num_heads, 1, 1)
    return mask, block
