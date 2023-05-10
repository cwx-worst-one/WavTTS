'''
Functions for computing confidence.
'''

import json
import math
import torch


class Hyp:
    '''
    hyp
    '''

    def __init__(self, labels, frames, scores):
        '''init
        Args:
            labels: list of token ids
            frames: list of aligned frames
            scores: list of token scores
        '''
        assert len(labels) == len(frames)
        assert len(labels) == len(scores)
        self._labels = labels
        self._frames = frames
        self._scores = scores
        self._slots_idx = []

    def __len__(self):
        '''
        length of hyp
        '''
        return len(self._labels)

    def add_slot_idx(self, idx):
        '''
        record slot id of corresponding label
        '''
        self._slots_idx.append(idx)

    def add_slots_idx(self, idx):
        '''
        record slot ids of corresponding labels
        '''
        self._slots_idx.extend(idx)

    def update_slots_idx(self, idx, offset, total):
        '''
        update slot ids of hyp due to inserting sub cn
        '''
        # pylint: disable=consider-using-enumerate
        for i in range(len(self._slots_idx)):
            if self._slots_idx[i] == idx:
                self._slots_idx[i] += offset
            elif self._slots_idx[i] > idx:
                self._slots_idx[i] += total

    @property
    def labels(self):
        '''
        labels
        '''
        return self._labels

    @property
    def frames(self):
        '''
        frames
        '''
        return self._frames

    @property
    def scores(self):
        '''
        scores
        '''
        return self._scores

    @property
    def slots_idx(self):
        '''
        slots_idx
        '''
        return self._slots_idx


class Slot:
    '''
    slot
    '''

    def __init__(self, beg, end):
        '''init
        Args:
            beg: beginning frame of slot
            end: end frame of slot
        '''
        self._beg = beg
        self._end = end
        self._mid = (beg + end) / 2.0
        self._arcs = dict()

    def __len__(self):
        '''
        number of arcs of slot
        '''
        return len(self._arcs)

    def insert_one_arc(self, label, prob):
        '''
        insert one arc
        '''
        self._arcs[label] = max(self._arcs.get(label, 0.0), prob)

    def insert_many_arcs(self, arcs):
        '''
        insert many arcs
        '''
        for label, prob in arcs.items():
            self._arcs[label] = max(self._arcs.get(label, 0.0), prob)

    def time_warping(self, t):
        '''
        time warping
        '''
        self._mid = t

    def norm(self):
        '''
        normalize probabilities on arcs
        '''
        sum_prob = 0
        for label, prob in self._arcs.items():
            sum_prob += prob
        for label, _ in self._arcs.items():
            self._arcs[label] /= sum_prob

    @property
    def beg(self):
        '''
        beg
        '''
        return self._beg

    @property
    def end(self):
        '''
        end
        '''
        return self._end

    @property
    def mid(self):
        '''
        mid
        '''
        return self._mid

    @property
    def arcs(self):
        '''
        arcs
        '''
        return self._arcs


class ConfusionNet:
    '''
    confusion net
    '''

    def __init__(self, b_id, b_probs):
        '''init
        Args:
            b_id: blank id
            b_probs: blank emit probabilities of whole utterance
        '''
        self._slots = []
        self._slots_num = 0
        self._blk = b_id
        self._b_probs = b_probs
        self._hyps = []
        self._hyps_num = 0

    def insert_main_path(self, hyp, left, right):
        '''
        insert main path to cn by one hyp
        '''
        num = len(hyp)
        if num == 0:
            slot = Slot(left, right)
            slot.insert_one_arc(self._blk, 0.0)
            slot.time_warping((left + right) / 2.0)
            self._slots.append(slot)
            hyp.add_slot_idx(self._slots_num)
            self._slots_num += 1
        for i in range(num):
            beg = (hyp.frames[i - 1] + hyp.frames[i]) / 2.0 if i > 0 else left
            end = (hyp.frames[i] + hyp.frames[i + 1]) / 2.0 if i < num - 1 else right
            slot = Slot(beg, end)
            slot.insert_one_arc(hyp.labels[i], hyp.scores[i])
            slot.time_warping(hyp.frames[i])
            self._slots.append(slot)
            hyp.add_slot_idx(self._slots_num)
            self._slots_num += 1
        self._hyps.append(hyp)
        self._hyps_num += 1

    def insert_hyp(self, hyp):
        '''
        insert hyp to cn
        '''
        hit_idx = self.get_hit_idx(hyp.frames)
        to_insert_cn = []
        new_slot_num = 0
        for i, slot in enumerate(self._slots):
            hit_num = len(hit_idx[i])
            if hit_num == 0:
                # insert blank arc in slot
                slot.insert_one_arc(self._blk, 0.0)
                new_slot_num += 1
            elif hit_num == 1:
                # insert non-blank arc in slot
                slot.insert_one_arc(hyp.labels[hit_idx[i][0]], hyp.scores[hit_idx[i][0]])
                hyp.add_slot_idx(new_slot_num)
                new_slot_num += 1
            else:
                # insert sub cn to replace slot
                sub_cn = ConfusionNet(self._blk, self._b_probs)
                sub_hyp = Hyp(
                    [hyp.labels[x] for x in hit_idx[i]],
                    [hyp.frames[x] for x in hit_idx[i]],
                    [hyp.scores[x] for x in hit_idx[i]],
                )
                sub_cn.insert_main_path(sub_hyp, slot.beg, slot.end)
                to_insert_cn.append((i, sub_cn))
                hyp.add_slots_idx([new_slot_num + x for x in range(hit_num)])
                new_slot_num += hit_num
        offset = 0
        for i, sub_cn in to_insert_cn:
            self.insert_sub_cn(sub_cn, i + offset)
            offset += sub_cn.slots_num - 1
        self._hyps.append(hyp)
        self._hyps_num += 1

    def insert_sub_cn(self, cn, idx):
        '''
        insert sub cn to the main one by repalcing one slot
        '''
        old_slot = self._slots[idx]
        for i, slot in enumerate(cn.slots):
            if slot.beg <= old_slot.mid < slot.end:
                slot.insert_many_arcs(old_slot.arcs)
                slot.time_warping(old_slot.mid)
                offset = i
            else:
                slot.insert_one_arc(self._blk, 0.0)
        self._slots = self._slots[:idx] + cn.slots + self._slots[idx + 1 :]
        self._slots_num += cn.slots_num - 1
        self.update_slots_idx_of_hyps(idx, offset, cn.slots_num - 1)

    def update_slots_idx_of_hyps(self, idx, offset, total):
        '''
        update slots idx of hyps
        '''
        for hyp in self._hyps:
            hyp.update_slots_idx(idx, offset, total)

    def get_hit_idx(self, frames):
        '''
        get hit idx
        '''
        hit_idx = [[] for _ in range(self._slots_num)]
        next_slot_idx = 0
        for i, t in enumerate(frames):
            start_slot_idx = next_slot_idx
            for j in range(start_slot_idx, self._slots_num):
                if t < self.slots[j].beg or t >= self.slots[j].end:
                    start_slot_idx += 1
                else:
                    hit_idx[j].append(i)
                    break
        return hit_idx

    def get_b_prob(self, slot):
        '''
        get average blank probability on the slot
        '''
        beg, end = math.floor(slot.mid), math.ceil(slot.mid)
        prob = 0.5 * (self._b_probs[beg] + self._b_probs[end])
        return prob

    def norm_cn(self):
        '''
        normalize cn
        '''
        for slot in self._slots:
            if self._blk in slot.arcs:
                slot.arcs[self._blk] = self.get_b_prob(slot)
            slot.norm()

    def print_cn(self):
        '''
        print cn
        '''
        slots = []
        for slot in self._slots:
            slot_dict = {'beg': slot.beg, 'end': slot.end, 'mid': slot.mid, 'arcs': slot.arcs}
            slots.append(slot_dict)
        print(json.dumps(slots, indent=4, sort_keys=True))

    def get_confidence(self, idx):
        '''
        get confidence of each token in each hyp
        the confidence of hyp is the average confidence of all tokens
        '''
        if idx < 0 or idx >= self._hyps_num:
            print("idx out of range [0, {}).".format(self._hyps_num))
            return None
        hyp = self._hyps[idx]
        conf_list = []
        if len(hyp) == 0:
            conf = 0
            for slot in self._slots:
                conf += slot.arcs.get(self._blk, 0)
            conf_list.append(conf / self._slots_num)
        for sid, l in zip(hyp.slots_idx, hyp.labels):
            conf_list.append(self._slots[sid].arcs[l])
        return sum(conf_list) / len(conf_list), conf_list

    @property
    def slots_num(self):
        '''
        slots_num
        '''
        return self._slots_num

    @property
    def slots(self):
        '''
        slots
        '''
        return self._slots

    @property
    def hyps_num(self):
        '''
        hyps_num
        '''
        return self._hyps_num

    def hyps(self):
        '''
        hyps
        '''
        return self._hyps


def parse_path(path, logp):
    '''
    Get timestep and probabilities of non-blank symbols.
    e.g. path = [0,0,0,7,7,0,0,0,26] => align = [3.5, 8]
    score = [max probs[3:5, 7],  max probs[8:9, 26]]
    '''
    align = []
    score = []
    prev_l = 0
    start_idx = 0
    frame = len(path)
    ctc_score = 0
    for i, l in enumerate(path):
        ctc_score += logp[i, l].item()
        if l != prev_l:
            if prev_l != 0:
                align.append((start_idx + i - 1) / 2.0)
                score.append(math.exp(torch.max(logp[start_idx:i, prev_l]).item()))
            prev_l = l
            start_idx = i
    if prev_l != 0:
        align.append((start_idx + frame - 1) / 2.0)
        score.append(math.exp(torch.max(logp[start_idx:, prev_l]).item()))
    return align, score, ctc_score


def compute_confidence(nbest_hyps, logp, path_list, alpha=1.0, beta=1.0, gamma=1.0):
    '''
    compute confidence for each token by confusion network
    '''
    blk_prob = torch.exp(logp[:, 0]).tolist()
    rnnt_score_list = []
    label_list = []
    align_list = []
    score_list = []
    ctc_score_list = []
    # ctc alignment
    for (labels, rnnt_score), path in zip(nbest_hyps, path_list):
        align, score, ctc_score = parse_path(path, logp)
        # CTC alignment fails when len(encoder output) <= 2 * len(rnnt output) + 1
        # remove such hyps
        if len(labels) == len(align):
            label_list.append(labels)
            rnnt_score_list.append(rnnt_score)
            align_list.append(align)
            score_list.append(score)
            ctc_score_list.append(ctc_score)
    # create cn
    hyp_list = [Hyp(label_list[i], align_list[i], score_list[i]) for i in range(len(label_list))]
    cn = ConfusionNet(0, blk_prob)
    cn.insert_main_path(hyp_list[0], 0, logp.size(0))
    hyp_list.pop(0)
    for hyp in hyp_list:
        cn.insert_hyp(hyp)
    cn.norm_cn()
    # cn.print_cn()
    conf_score_list = []
    tok_conf_score_list = []
    for i in range(cn.hyps_num):
        conf_score, conf_list = cn.get_confidence(i)
        conf_score_list.append(conf_score)
        tok_conf_score_list.append(conf_list)
    best_idx = 0
    best_score = alpha * conf_score_list[0] + beta * rnnt_score_list[0] + gamma * ctc_score_list[0]
    for i in range(1, cn.hyps_num):
        cur_score = (
            alpha * conf_score_list[i] + beta * rnnt_score_list[i] + gamma * ctc_score_list[i]
        )
        if cur_score > best_score:
            best_score = cur_score
            best_idx = i
    return label_list[best_idx], tok_conf_score_list[best_idx]
