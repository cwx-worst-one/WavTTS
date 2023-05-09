#-*- coding: utf-8 -*-
# heyi.hy@bytedance.com

"""
CER calculation for CN and mixed-CN-EN
CN: CER
EN: WER
"""

#import copy
import re
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
import codecs
import copy
import multiprocessing
from tqdm import tqdm
from cer_edit_distance import EditDistanceCalculator
from text_normalize import TestsetTextFormat


def load_scp_file_to_list(filename):
    """Load scp file to list
    Args:
        filename: string
    Return:
        out_list: [[fileid content], ... ]
    """
    fp_in = open(filename)
    out_list = []
    fileid_map = {}
    for line in fp_in:
        tmp_list = line.strip().replace('\t', ' ').split(' ', 1)
        assert len(tmp_list) > 0
        if len(tmp_list) == 1:
            fileid = tmp_list[0]
            content = ''
        else:
            fileid = tmp_list[0]
            content = tmp_list[1]
        if fileid in fileid_map:
            logging.error("Duplicated id %s in %s" %(fileid, filename))
            sys.exit(1)
        fileid_map[fileid] = 1
        out_list.append([fileid, content])
    fp_in.close()
    return out_list

def load_nbest_res_file_to_map(filename, nbest=1):
    fp_in = open(filename)
    out_map = {}
    for line in fp_in:
        tmp_list = line.strip().replace('\t', ' ').split(' ', 1)
        assert len(tmp_list) > 0
        if len(tmp_list) == 1:
            fileid = tmp_list[0]
            content = ''
        else:
            fileid = tmp_list[0]
            content = tmp_list[1]
        if nbest > 1:
            tmp_list = fileid.split('-')
            fileid = '-'.join(tmp_list[:-1])
            num = int(tmp_list[-1])
            if num > nbest:
                continue
        if fileid in out_map:
            out_map[fileid].append(content)
        else:
            out_map[fileid] = [content]
    fp_in.close()
    return out_map

def load_scp_file_to_map(filename):
    """Load scp file to map
    Args:
        filename: string
    Return:
        out_map: {fileid: content, ... }
    """
    fp_in = open(filename)
    out_map = {}
    for line in fp_in:
        tmp_list = line.strip().replace('\t', ' ').split(' ', 1)
        assert len(tmp_list) > 0
        if len(tmp_list) == 1:
            fileid = tmp_list[0]
            content = ''
        else:
            fileid = tmp_list[0]
            content = tmp_list[1]
        if fileid in out_map:
            logging.error("Duplicated id %s in %s" %(fileid, filename))
            sys.exit(1)
        else:
            out_map[fileid] = content
    fp_in.close()
    return out_map

def load_list_file_to_map(filename):
    """ Load list file to map
    Args:
        filename: string
    Return:
        out_map: {}
    """
    fp_in = codecs.open(filename, encoding='utf-8')
    out_map = {}
    for line in fp_in:
        line = line.strip()
        out_map[line] = 1
    fp_in.close()
    return out_map

def load_substitute_map_file(filename):
    """ Load substitute words file
    Args:
        filename: string
    Return:
        out_map:
    """
    fp_in = codecs.open(filename, encoding='utf-8')
    out_map = {}
    for line in fp_in:
        tmp_list = line.strip().split()
        if len(tmp_list) < 2:
            continue
        val = tmp_list[0]
        for item in tmp_list:
            if item in out_map:
                logging.warn("Duplicated item %s in %s" %(item, filename))
            out_map[item] = val
    fp_in.close()
    return out_map

ENG_RE = re.compile(r"\*?[a-z'\-]+\*?")
INS_RE = re.compile(r"\*\*+")
def is_eng(w):
    return ENG_RE.match(w)

def is_ins(w):
    return ''.join(set(w)) == '*'

class ErrCounter:
    def __init__(self, ed_info = None):
        self.ins_err = int(ed_info['ins_err']) if ed_info else 0
        self.del_err = int(ed_info['del_err']) if ed_info else 0
        self.sub_err = int(ed_info['sub_err']) if ed_info else 0
        self.correct = int(ed_info['correct']) if ed_info else 0
        self.ref_word_num = int(ed_info['ref_word_num']) if ed_info else 0

    def add_ed_info(self, ed_info):
        self.ins_err += int(ed_info['ins_err'])
        self.del_err += int(ed_info['del_err'])
        self.sub_err += int(ed_info['sub_err'])
        self.correct += int(ed_info['correct'])
        self.ref_word_num += int(ed_info['ref_word_num'])
    
    def add(self, other_counter):
        self.ins_err += other_counter.ins_err
        self.del_err += other_counter.del_err
        self.sub_err += other_counter.sub_err
        self.correct += other_counter.correct
        self.ref_word_num += other_counter.ref_word_num
    
    def err_num(self):
        return self.ins_err + self.del_err + self.sub_err
    
    def wer(self):
        if self.ref_word_num == 0:
            return 0.0
        return float(self.err_num()) / self.ref_word_num
    
    def acc(self):
        if self.ref_word_num == 0:
            return 0.0
        return float(self.correct) / self.ref_word_num



def output_result(align_info_list, ed_info_list, out_file):
    total_cnt = ErrCounter()
    total_eng_cnt = ErrCounter()
    tot_sen_err = 0
    results = []
    for ed_info, align_info in zip(ed_info_list, align_info_list):
        fileid = align_info['fileid']
        ref_str = align_info['ref_str']
        res_str = align_info['res_str']

        cur_cnt = ErrCounter(ed_info)
        total_cnt.add(cur_cnt)
        if cur_cnt.err_num() > 0:
            tot_sen_err += 1

        ##### for english-part wer #####
        ref_toks = ref_str.strip().split()
        hyp_toks = res_str.strip().split()
        cur_eng_cnt = ErrCounter()
        for idx, (rtok, htok) in enumerate(zip(ref_toks, hyp_toks)):
            if is_eng(rtok):
                cur_eng_cnt.ref_word_num += 1
                if rtok[0] == '*':
                    cur_eng_cnt.sub_err += 1
                elif is_ins(htok):
                    cur_eng_cnt.del_err += 1
                else:
                    cur_eng_cnt.correct += 1
            elif is_eng(htok) and is_ins(rtok):
                cur_eng_cnt.ins_err += 1
        total_eng_cnt.add(cur_eng_cnt)

        results.append([fileid, ref_str, res_str, cur_cnt, cur_eng_cnt])
    # 每句按总错误数从多到少排序
    results.sort(key=lambda x: x[-2].err_num(), reverse=True)

    msg = ''
    if out_file:
        for fileid, ref_str, res_str, cur_cnt, cur_eng_cnt in results:
            msg += "%s %s-REF\n" %(ref_str, fileid)
            msg += "%s %s-RES\n" %(res_str, fileid)
            msg += "Words: %d Correct: %d Errors: %d Accuracy = %.2f%%\n" %(
                        cur_cnt.ref_word_num, cur_cnt.correct, cur_cnt.err_num(), cur_cnt.acc() * 100)
            msg += "Insertions: %d Deletions: %d Substitutions: %d\n" %(
                        cur_cnt.ins_err, cur_cnt.del_err,cur_cnt.sub_err)
            msg += "Eng: Words: %d Insertions: %d Deletions: %d Substitutions: %d\n" %(
                        cur_eng_cnt.ref_word_num, cur_eng_cnt.ins_err, cur_eng_cnt.del_err, cur_eng_cnt.sub_err)

    # English info
    msg += "TOTAL Eng Words: %d Correct: %d Errors(I+D+S): %d\n" %(
                total_eng_cnt.ref_word_num, total_eng_cnt.correct, total_eng_cnt.err_num())
    msg += "TOTAL Eng Insertions: %d Deletions: %d Substitutions: %d\n"%(
                    total_eng_cnt.ins_err, total_eng_cnt.del_err, total_eng_cnt.sub_err)
    msg += "Eng WER: %.2f%%, WAR: %.2f%%\n\n" %(total_eng_cnt.wer() * 100, 100 - total_eng_cnt.wer() * 100)

    # Total info
    ser = float(tot_sen_err) / len(ed_info_list)
    msg += "TOTAL Words: %d Correct: %d Errors(I+D+S): %d\n" %(
                total_cnt.ref_word_num, total_cnt.correct, total_cnt.err_num())
    msg += "TOTAL Insertions: %d Deletions: %d Substitutions: %d\n"%(
                    total_cnt.ins_err, total_cnt.del_err, total_cnt.sub_err)
    msg += "CER: %.2f%%, CAR: %.2f%%\n" %(total_cnt.wer() * 100, 100 - total_cnt.wer() * 100)
    msg += "SER: %.2f%%, SAR: %.2f%%\n" %(ser * 100, 100 - ser * 100)
    if out_file:
        with codecs.open(out_file, 'w', encoding='utf-8') as fw:
            fw.write(msg)
    else:
        print(msg)


################## Cal CER one ##################
def core(arg):
    fileid, ref_format, res_nbest_format, ed_calculator = arg
    best_err_count = sys.maxsize
    best_align_info = None
    best_ed_info = None
    for res_format in res_nbest_format:
        ed_info, align_info = ed_calculator.calculate(
                fileid, ref_format, res_format, align=True)
        curr_ins_err = int(ed_info['ins_err'])
        curr_del_err = int(ed_info['del_err'])
        curr_sub_err = int(ed_info['sub_err'])
        curr_err_count = curr_ins_err + curr_del_err + curr_sub_err
        if curr_err_count < best_err_count:
            best_ed_info = ed_info
            best_align_info = align_info
            best_err_count = curr_err_count
    return best_ed_info, best_align_info
# multi process
def get_cer_detail(ref_file, res_file, merge_words_file='', substitute_words_file='', ignore_not_recognized_sample=False, nbest=1, njobs=100):
    """ Comput CER with text format
    Args:
        ref_file: string, one true label per one key
        res_file: string, nbest label for one key
        merge_words_file: string, "哈哈哈" -> "哈"
        substitute_words_file: string, "噢 哦" -> "噢"
        nbest: int
        njobs: int
    Return:
        align_info_list
        ed_info_list
    """
    assert nbest == 1
    #assert substitute_words_file == ''
    ed_calculator = EditDistanceCalculator()
    ref_list = load_scp_file_to_list(ref_file)
    res_map = load_nbest_res_file_to_map(res_file, nbest=nbest)
    words_merge_map = None
    words_substitute_map = None
    if merge_words_file != '':
        words_merge_map = load_list_file_to_map(merge_words_file)
    if substitute_words_file != '':
        words_substitute_map = load_substitute_map_file(substitute_words_file)
        ed_calculator.set_substitute_map(words_substitute_map)
    align_info_list = []
    ed_info_list = []

    ################## Prepare Ref-Res pair ##################
    pairs = []
    for item in ref_list:
        fileid = item[0]
        ref = item[1]
        if fileid in res_map:
            res = res_map[fileid]
        else:
            if ignore_not_recognized_sample:
                continue
            else:
                res = ['']
        ref_format = TestsetTextFormat.text_format(ref)
        if words_merge_map is not None:
            ref_format = TestsetTextFormat.merge_words(words_merge_map, ref_format)
        res_nbest_format = []
        for res_item in res:
            res_item_format = TestsetTextFormat.text_format(res_item)
            if words_merge_map:
                res_item_format = TestsetTextFormat.merge_words(words_merge_map, res_item_format)
            res_nbest_format.append(res_item_format)

        pairs.append([fileid, ref_format, res_nbest_format, ed_calculator])
        

        
    ################## Cal CER all ##################
    pool = multiprocessing.Pool(processes=njobs)
    with tqdm(total=len(pairs)) as progress_bar:
        for ed_info, align_info in pool.imap_unordered(core, pairs):
            progress_bar.update(1)
            ed_info_list.append(ed_info)
            align_info_list.append(align_info)
    return align_info_list, ed_info_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ref_file', type=str, required=True, help='Reference file')
    parser.add_argument(
        '--res_file', type=str, required=True, help='Result file')
    parser.add_argument(
        '--nbest', type=int, default=1, help='nbest')
    parser.add_argument(
        '--merge_words_file', type=str, default='', help="Merge words, e.g. ha ha -> ha")
    parser.add_argument(
        '--substitute_words_file', type=str, default='', help="Substitete_words")
    parser.add_argument(
        '--out_stat_file', type=str, default='', help='Output stat file')
    parser.add_argument(
        '--ignore_not_recognized_sample', action='store_true', help='whether to ignore samples that are not recognized; If false, not recognized samples will increase deletion errors')
    flags, _ = parser.parse_known_args()
    logging.info(flags)

    ref_file = flags.ref_file
    res_file = flags.res_file
    nbest = flags.nbest
    merge_words_file = flags.merge_words_file
    substitute_words_file = flags.substitute_words_file
    out_stat_file = flags.out_stat_file

    if flags.out_stat_file == '':
        logging.info("Detail stat info will not be displayed since there is no out_stat_file")
    align_info_list, ed_info_list = get_cer_detail(ref_file,
                                                   res_file,
                                                   merge_words_file=merge_words_file,
                                                   substitute_words_file=substitute_words_file,
                                                   ignore_not_recognized_sample=flags.ignore_not_recognized_sample)
    output_result(align_info_list, ed_info_list, out_stat_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
