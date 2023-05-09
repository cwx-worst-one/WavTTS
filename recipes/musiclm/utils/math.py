def safe_divide(x, y, eps: float = 1e-8):
    return x / (y + eps)
