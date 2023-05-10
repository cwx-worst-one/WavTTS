''' solution utils. '''
import copy
import numpy as np


def force_align(logits, label_seq):
    '''force alignment between ce logits and rnnt output'''
    # pylint:disable=too-many-branches
    if len(label_seq) == 0:
        return [], []
    logits = np.transpose(logits)
    blank_id = 0
    sequence = copy.copy(list(label_seq))
    for i in range(len(sequence)):
        sequence.insert(2 * i, blank_id)
    sequence.append(blank_id)

    time_rank = logits.shape[1]
    sequence_length = len(sequence)

    search_graph = np.zeros((time_rank, sequence_length))
    search_path = np.zeros((time_rank, sequence_length), dtype=np.int)
    search_path.fill(-1)
    bscr = np.zeros((time_rank, sequence_length))
    aligned_seq = np.zeros((time_rank), dtype=np.int)
    aligned_seq_score = np.zeros((time_rank), dtype=np.float)

    # filling S
    for i in range(sequence_length):
        search_graph[:, i] = logits[sequence[i]]

    # base case
    search_path[0, 0] = 0  # made this 0 instead of -1.
    search_path[0, 1] = 1
    bscr[0, 0] = search_graph[0, 0]
    bscr[0, 1] = search_graph[0, 1]
    bscr[0, 2:] = np.NINF

    # filling over the rest time stamps
    for t in range(1, time_rank):
        search_path[t, 0] = search_path[t - 1, 0]
        bscr[t, 0] = bscr[t - 1, 0] + search_graph[t, 0]
        search_path[t, 1] = 1 if bscr[t - 1, 1] > bscr[t - 1, 0] else 0
        bscr[t, 1] = bscr[t - 1, search_path[t, 1]] + search_graph[t, 1]
        for i in range(2, sequence_length):
            if i % 2 == 0 or sequence[i] == sequence[i - 2]:
                search_path[t, i] = i if bscr[t - 1, i] > bscr[t - 1, i - 1] else i - 1
            elif bscr[t - 1, i] > bscr[t - 1, i - 1] and bscr[t - 1, i] > bscr[t - 1, i - 2]:
                search_path[t, i] = i
            elif bscr[t - 1, i - 1] > bscr[t - 1, i] and bscr[t - 1, i - 1] > bscr[t - 1, i - 2]:
                search_path[t, i] = i - 1
            else:
                search_path[t, i] = i - 2
            bscr[t, i] = bscr[t - 1, search_path[t, i]] + search_graph[t, i]
    if bscr[time_rank - 1, sequence_length - 1] > bscr[time_rank - 1, sequence_length - 2]:
        aligned_seq[time_rank - 1] = sequence_length - 1
        aligned_seq_score[time_rank - 1] = search_graph[time_rank - 1, sequence_length - 1]
    else:
        aligned_seq[time_rank - 1] = sequence_length - 2
        aligned_seq_score[time_rank - 1] = search_graph[time_rank - 1, sequence_length - 2]
    for t in range(time_rank - 1, 0, -1):
        aligned_seq[t - 1] = search_path[t, aligned_seq[t]]
        aligned_seq_score[t - 1] = search_graph[t, aligned_seq[t - 1]]
    # get each token's start_index and end_index from alig_seq and sequence
    # sequence [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 3, 0, 0, 4, 4]
    # aligned_list [[1,3,6], [2,8,11], [3,12,15], [4, 17,19]]
    aligned_list = []
    time_frame = 0
    # use average
    if aligned_seq[0] != 0 and aligned_seq[0] != 1:
        average_frame = int(time_rank / len(label_seq))
        left_frame = time_rank % len(label_seq)
        for token_indx, token_id in enumerate(label_seq):
            start_index = time_frame
            end_index = (
                start_index + average_frame + 1
                if token_indx < left_frame
                else start_index + average_frame
            )
            aligned_list.append([token_id, start_index, end_index])
            time_frame = end_index
        return aligned_list, aligned_seq_score
    for token_indx, token_id in enumerate(sequence):
        if token_indx % 2 == 1:
            while aligned_seq[time_frame] != token_indx:
                time_frame += 1
            start_index = time_frame
            while time_frame < len(aligned_seq) and aligned_seq[time_frame] == token_indx:
                time_frame += 1
            end_index = time_frame
            aligned_list.append([token_id, start_index, end_index])
    return aligned_list, aligned_seq_score


def aligned_id_to_char(
    tgt_dict_pre, aligned_list, aligned_score, filter_list, frame_shift=10, frame_length=40
):
    '''post process for timestamp'''
    aligned_score = [np.exp(score) for score in aligned_score]
    out_list = []
    hyp_token = []
    out_list = [[tmp[0], tmp[1] * frame_length, tmp[2] * frame_length] for tmp in aligned_list]
    tmp_list = ["", 0, 0]
    for idx, token_info in enumerate(out_list):
        if token_info[0] < 4:
            continue
        token_info[0] = tgt_dict_pre[token_info[0]]
        token = token_info[0]
        if token not in filter_list:
            if tmp_list[0] == "":
                tmp_list = token_info
            else:
                tmp_list[0] += token_info[0]
                tmp_list[2] = token_info[2]
            if token[-1] != '@' or idx == len(out_list):
                hyp_token.append(tmp_list)
                tmp_list = ["", 0, 0]

    for cur_tok in hyp_token:
        cur_tok[0] = cur_tok[0].replace('@@', '')
        start_index = int(cur_tok[1] / frame_length)
        end_index = int(cur_tok[2] / frame_length)
        duration = end_index - start_index
        cur_tok.append(sum(aligned_score[start_index:end_index]) / duration)
        # process frame_shift
        refine_start = cur_tok[1] - frame_shift
        cur_tok[1] = refine_start if refine_start > 0 else 0
        refine_end = cur_tok[2] - frame_shift
        cur_tok[2] = refine_end if refine_end > 0 else 0
    return str(hyp_token)
