'''
some math function.
'''


def ceil(x, y):
    '''
    do divide and get ceil value.
    Args:
        x(int): first value.
        y(int): second value.
    Return:
        int: return value.
    '''
    return ((x + y - 1) // y) * y


def get_stream_align_size(size):
    '''
    return align state size for streaming op
    '''
    return ceil(size, 16)
