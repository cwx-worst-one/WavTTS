import numpy as np

from tqdm import tqdm


def find_best_pair(tokens):
    pair_dict = {}
    prefix = None
    for t in tqdm(tokens):
        if t > 1e5:
            prefix = None
        elif prefix is None:
            prefix = t
        else:
            pair = "{}|{}".format(prefix, t)
            if pair not in pair_dict:
                pair_dict[pair] = 1
            else:
                pair_dict[pair] = pair_dict[pair] + 1
            prefix = t
    max_cnt = -1
    for key in pair_dict:
        if pair_dict[key] > max_cnt:
            max_cnt = pair_dict[key]
            max_pair = key
    return max_pair, max_cnt


def update_pair_and_cnt(tokens, max_pair, replace_codes):
    if len(tokens) <= 1:
        return tokens

    a, b = max_pair.split('|')
    new_tokens = [0] * len(tokens)
    a, b = int(a), int(b)
    i = 1
    cnt = 0
    while i < len(tokens):
        aa, bb = tokens[i-1], tokens[i]
        if aa == a and bb == b:
            new_tokens[cnt] = replace_codes
            i += 2
            cnt += 1
        else:
            new_tokens[cnt] = aa
            i += 1
            flag = False
            cnt += 1
    if i == len(tokens) or i == len(tokens) - 1:
        new_tokens[cnt] = tokens[-1]
        cnt += 1
        
    return new_tokens[0:cnt]


# fake data
tokens = list(np.random.randint(low=0, high=8192, size=[3600*50*1000]))
coded_pair = []
for i in tqdm(range(1024)):
    max_pair, max_cnt = find_best_pair(tokens)
    if max_cnt > 1:
        tokens = update_pair_and_cnt(tokens, max_pair, i + 8192)
        coded_pair.append([max_pair, i + 8192, max_cnt])
    else:
        break

