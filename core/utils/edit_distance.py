'''edit_distance'''

from collections import Counter
import numpy as np


def edit_distance(ref, hyp):
    '''
    edit distance
    '''
    # pylint:disable=no-else-break,too-many-branches
    assert isinstance(ref, list) and isinstance(hyp, list)

    dist = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint32)
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                dist[0][j] = j
            elif j == 0:
                dist[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                substitute = dist[i - 1][j - 1] + 1
                insert = dist[i][j - 1] + 1
                delete = dist[i - 1][j] + 1
                dist[i][j] = min(substitute, insert, delete)
    i = len(ref)
    j = len(hyp)
    steps = []
    while True:
        if i == 0 and j == 0:
            break
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]:
            steps.append('corr')
            i, j = i - 1, j - 1
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] + 1:
            assert ref[i - 1] != hyp[j - 1]
            steps.append('sub')
            i, j = i - 1, j - 1
        elif j >= 1 and dist[i][j] == dist[i][j - 1] + 1:
            steps.append('ins')
            j = j - 1
        else:
            assert i >= 1 and dist[i][j] == dist[i - 1][j] + 1
            steps.append('del')
            i = i - 1
    steps = steps[::-1]

    counter = Counter({'words': len(ref), 'corr': 0, 'sub': 0, 'ins': 0, 'del': 0})
    counter.update(steps)

    return dist, steps, counter
