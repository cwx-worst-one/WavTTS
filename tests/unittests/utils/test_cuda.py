from samantha.utils.cuda import torch_allow_tf32


def test_torch_allow_tf32():
    import torch.backends.cuda
    import torch.backends.cudnn

    # this flag is switched off when running on test_suit,
    # temporally switch on to run the test
    torch.backends.__allow_nonbracketed_mutation_flag = True

    allow_matmul = torch.backends.cuda.matmul.allow_tf32
    allow_cudnn = torch.backends.cudnn.allow_tf32
    with torch_allow_tf32(enable_matmul=False, enable_cudnn=False):
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32 == allow_matmul
    assert torch.backends.cudnn.allow_tf32 == allow_cudnn

    with torch_allow_tf32(enable_matmul=True, enable_cudnn=True):
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32 == allow_matmul
    assert torch.backends.cudnn.allow_tf32 == allow_cudnn
