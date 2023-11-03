import torch


def merge_multiple_temporal_probs(probs, chord, sample_len, sample_hop):
    new_probs = torch.zeros(chord.shape).to(chord.device)
    count = torch.zeros(chord.shape).to(chord.device)
    for i in range(len(probs)):
        new_probs[i * sample_hop : i * sample_hop + sample_len] += probs[i][
            : len(new_probs[i * sample_hop : i * sample_hop + sample_len])
        ]
        count[i * sample_hop : i * sample_hop + sample_len] += 1
    new_probs = new_probs / count
    return new_probs
