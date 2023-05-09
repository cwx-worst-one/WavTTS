#-*- coding: utf-8 -*-
# heyi.hy@bytedance.com

import copy
import logging

def get_character_type(char_in):
    """ indentify the character type
    Args:
        str_in: unicode
    Return:
        type_out: 'digit', 'english', 'chinese', 'other'
    """
    item = char_in
    if ord(item) >= 48 and ord(item) <= 57:
        type_out = 'digit'
    elif ((ord(item) >= 65 and ord(item) <= 90) or
            (ord(item) >= 97 and ord(item) <= 122)) or item == "'":
        type_out = 'english'
    elif item >= u'\u4e00' and item <= u'\u9fa5':
        type_out = 'chinese'
    else:
        type_out = 'other'
    return type_out

class EditDistanceCalculator(object):
    """ edit distance calculator
    """
    def __init__(self):
        """ init """
        self.__ed_info = {}
        self.__align_info = {}
        self.__width = 0
        self.__height = 0
        self.__align_matrix = []
        self.__bt_matrix = []
        self.__fileid = ''
        self.__ref_word_list = []
        self.__res_word_list = []
        self.__subtitute_map = None

    def set_substitute_map(self, subtitute_map):
        self.__subtitute_map = copy.deepcopy(subtitute_map)

    def is_word_equal(self, word1, word2):
        flag = (word1 == word2)
        if self.__subtitute_map:
            word1_val = word1
            word2_val = word2
            if word1 in self.__subtitute_map:
                word1_val = self.__subtitute_map[word1]
            if word2 in self.__subtitute_map:
                word2_val = self.__subtitute_map[word2]
            flag = (word1_val == word2_val)
        return flag

    def precompute(self, ref_word_list, res_word_list):
        """ prepare for computation """
        self.__ref_word_list = ref_word_list
        self.__res_word_list = res_word_list
        self.__width = len(res_word_list) + 1
        self.__height = len(ref_word_list) + 1
        self.__align_matrix = [[0 for _ in range(self.__width)] for _ in range(self.__height)]
        self.__bt_matrix = [['INS' for _ in range(self.__width)] for _ in range(self.__height)]
        for idx in range(self.__width):
            self.__align_matrix[0][idx] = idx
            self.__bt_matrix[0][idx] = 'INS'
        for idx in range(self.__height):
            self.__align_matrix[idx][0] = idx
            self.__bt_matrix[idx][0] = 'DEL'
        self.__ed_info['ins_err'] = 0
        self.__ed_info['del_err'] = 0
        self.__ed_info['sub_err'] = 0
        self.__ed_info['correct'] = 0
        self.__ed_info['ref_word_num'] = self.__height - 1
        self.__align_info = {}

    def compute(self):
        """ compute edit distance """
        width = self.__width
        height = self.__height
        ref_word_list = self.__ref_word_list
        res_word_list = self.__res_word_list
        for idx in range(1, height):
            for jdx in range(1, width):
                ref_word = ref_word_list[idx - 1]
                res_word = res_word_list[jdx - 1]
                # 1. cal the costs
                ins_cost = self.__align_matrix[idx][jdx - 1] + 1
                del_cost = self.__align_matrix[idx - 1][jdx] + 1
                equal_flag = self.is_word_equal(ref_word, res_word)
                curr_sub_cost = 1 - int(equal_flag)
                sub_cost = self.__align_matrix[idx - 1][jdx - 1] + curr_sub_cost
                min_cost = min(ins_cost, del_cost, sub_cost)
                self.__align_matrix[idx][jdx] = min_cost
                # 2. fill bt info
                if min_cost == ins_cost:
                    self.__bt_matrix[idx][jdx] = 'INS'
                elif min_cost == del_cost:
                    self.__bt_matrix[idx][jdx] = 'DEL'
                elif curr_sub_cost == 1:
                    self.__bt_matrix[idx][jdx] = 'SUB'
                else:
                    self.__bt_matrix[idx][jdx] = 'COR'  # Correct

    def backtrace(self):
        """ backtrace to get minimum path info """
        ref_word_list = self.__ref_word_list
        res_word_list = self.__res_word_list
        idx = len(ref_word_list)
        jdx = len(res_word_list)
        bt_info = []
        del_err = 0
        ins_err = 0
        sub_err = 0
        correct = 0
        while (idx > 0 or jdx > 0):
            action_str = self.__bt_matrix[idx][jdx]
            if action_str == "INS":
                bt_info.append([None, res_word_list[jdx - 1]])
                ins_err += 1
                jdx -= 1
            elif action_str == "DEL":
                bt_info.append([ref_word_list[idx - 1], None])
                del_err += 1
                idx -= 1
            elif action_str == "SUB":
                bt_info.append([ref_word_list[idx - 1], res_word_list[jdx - 1]])
                sub_err += 1
                idx -= 1
                jdx -= 1
            elif action_str == 'COR':
                bt_info.append([ref_word_list[idx - 1], res_word_list[jdx - 1]])
                correct += 1
                idx -= 1
                jdx -= 1
            else:
                logging.error("Unknown action string %s" %action_str)
                sys.exit(1)
        self.__ed_info['del_err'] = del_err
        self.__ed_info['ins_err'] = ins_err
        self.__ed_info['sub_err'] = sub_err
        self.__ed_info['correct'] = correct
        return bt_info

    def get_string_display_width(self, str_in):
        """ very simple version for getting string displayed width """
        out_len = 0
        for item in str_in:
            if get_character_type(item) == 'chinese':
                out_len += len(item) * 2
            else:
                out_len += len(item)
        return out_len

    def gen_align_string(self, bt_info):
        """ generate aligned ref_str and res_str for display """
        ref_word_list = []
        res_word_list = []
        for item in bt_info:
            ref_word = item[0]
            res_word = item[1]
            equal_flag = self.is_word_equal(ref_word, res_word)
            if equal_flag:
                ref_word_list.append(ref_word)
                res_word_list.append(res_word)
            else:
                if ref_word:
                    ref_word_len = self.get_string_display_width(ref_word)
                else:
                    ref_word_len = 0
                if res_word:
                    res_word_len = self.get_string_display_width(res_word)
                else:
                    res_word_len = 0
                if not ref_word:
                    # ins
                    ref_word_list.append('*' * res_word_len)
                    res_word_list.append(res_word)
                elif not res_word:
                    # del
                    ref_word_list.append(ref_word)
                    res_word_list.append('*' * ref_word_len)
                else:
                    len_diff = ref_word_len - res_word_len
                    if len_diff >= 0:
                        ref_word_list.append("*%s*" %ref_word)
                        tmp_str = '*' * len_diff
                        res_word_list.append("*%s%s*" %(res_word, tmp_str))
                    else:
                        res_word_list.append("*%s*" %res_word)
                        len_diff = -len_diff
                        tmp_str = '*' * len_diff
                        ref_word_list.append("*%s%s*" %(ref_word, tmp_str))
        self.__align_info['ref_str'] = " ".join(ref_word_list[::-1])
        self.__align_info['res_str'] = " ".join(res_word_list[::-1])
        self.__align_info['fileid'] = self.__fileid

    def calculate(self, fileid, ref_str, res_str, align=False):
        """ main entry of EditDistanceCalculator """
        self.__fileid = fileid
        ref_word_list = ref_str.strip().split()
        res_word_list = res_str.strip().split()
        self.precompute(ref_word_list, res_word_list)
        self.compute()
        bt_info = self.backtrace()
        if align:
            self.gen_align_string(bt_info)
        return self.__ed_info.copy(), self.__align_info.copy()

class PtTcEditDistanceCalculator(object):
    
    def __init__(self):
        """ init """
        self.__ed_info = {}
        self.__align_info = {}
        self.__width = 0
        self.__height = 0
        self.__align_matrix = []
        self.__bt_matrix = []
        self.__fileid = ''
        self.__ref_word_list = []
        self.__res_word_list = []
        self.__subtitute_map = None
        self.__punct = {",":0, ".":0, "?":0,"，":0, "。":0, "？":0}

    def set_substitute_map(self, subtitute_map):
        self.__subtitute_map = copy.deepcopy(subtitute_map)

    def is_word_equal(self, word1, word2):
        flag = (word1 == word2)
        if self.__subtitute_map:
            word1_val = word1
            word2_val = word2
            if word1 in self.__subtitute_map:
                word1_val = self.__subtitute_map[word1]
            if word2 in self.__subtitute_map:
                word2_val = self.__subtitute_map[word2]
            flag = (word1_val == word2_val)
        return flag

    def is_truecase(self, word1, word2):
        if word1 and word2:
            return (word1.lower() == word2.lower())
        else:
            return False

    def is_punct(self, word1, word2):
        if word1 in self.__punct or word2 in self.__punct:
            return True
        else:
            return False

    def precompute(self, ref_word_list, res_word_list):
        """ prepare for computation """
        self.__ref_word_list = ref_word_list
        self.__res_word_list = res_word_list
        self.__width = len(res_word_list) + 1
        self.__height = len(ref_word_list) + 1
        self.__align_matrix = [[0 for _ in range(self.__width)] for _ in range(self.__height)]
        self.__bt_matrix = [['INS' for _ in range(self.__width)] for _ in range(self.__height)]
        for idx in range(self.__width):
            self.__align_matrix[0][idx] = idx
            self.__bt_matrix[0][idx] = 'INS'
        for idx in range(self.__height):
            self.__align_matrix[idx][0] = idx
            self.__bt_matrix[idx][0] = 'DEL'
        self.__ed_info['ins_err'] = 0
        self.__ed_info['del_err'] = 0
        self.__ed_info['sub_err'] = 0
        self.__ed_info['correct'] = 0
        self.__ed_info['ref_word_num'] = self.__height - 1
        self.__align_info = {}

    def compute(self):
        """ compute edit distance """
        width = self.__width
        height = self.__height
        ref_word_list = self.__ref_word_list
        res_word_list = self.__res_word_list
        for idx in range(1, height):
            for jdx in range(1, width):
                ref_word = ref_word_list[idx - 1]
                res_word = res_word_list[jdx - 1]
                # 1. cal the costs
                ins_cost = self.__align_matrix[idx][jdx - 1] + 1
                del_cost = self.__align_matrix[idx - 1][jdx] + 1
                equal_flag = self.is_word_equal(ref_word, res_word)
                pt_flag = self.is_punct(ref_word, res_word)
                if not pt_flag:
                    tc_flag = self.is_truecase(ref_word, res_word)
                else:
                    tc_flag = False
                if tc_flag:
                    curr_sub_cost = 0.1
                elif pt_flag and (ref_word not in self.__punct or res_word not in self.__punct):
                    curr_sub_cost = float('inf')
                else:
                    curr_sub_cost = 1 - int(equal_flag)
                sub_cost = self.__align_matrix[idx - 1][jdx - 1] + curr_sub_cost
                min_cost = min(ins_cost, del_cost, sub_cost)
                self.__align_matrix[idx][jdx] = min_cost
                # 2. fill bt info
                if min_cost == ins_cost:
                    self.__bt_matrix[idx][jdx] = 'INS'
                elif min_cost == del_cost:
                    self.__bt_matrix[idx][jdx] = 'DEL'
                elif curr_sub_cost == 1:
                    self.__bt_matrix[idx][jdx] = 'SUB'
                else:
                    self.__bt_matrix[idx][jdx] = 'COR'  # Correct
    def backtrace(self):
        """ backtrace to get minimum path info """
        ref_word_list = self.__ref_word_list
        res_word_list = self.__res_word_list
        idx = len(ref_word_list)
        jdx = len(res_word_list)
        bt_info = []
        del_err = 0
        ins_err = 0
        sub_err = 0
        correct = 0
        while (idx > 0 or jdx > 0):
            action_str = self.__bt_matrix[idx][jdx]
            if action_str == "INS":
                bt_info.append([None, res_word_list[jdx - 1]])
                ins_err += 1
                jdx -= 1
            elif action_str == "DEL":
                bt_info.append([ref_word_list[idx - 1], None])
                del_err += 1
                idx -= 1
            elif action_str == "SUB":
                bt_info.append([ref_word_list[idx - 1], res_word_list[jdx - 1]])
                sub_err += 1
                idx -= 1
                jdx -= 1
            elif action_str == 'COR':
                bt_info.append([ref_word_list[idx - 1], res_word_list[jdx - 1]])
                correct += 1
                idx -= 1
                jdx -= 1
            else:
                logging.error("Unknown action string %s" %action_str)
                sys.exit(1)
        self.__ed_info['del_err'] = del_err
        self.__ed_info['ins_err'] = ins_err
        self.__ed_info['sub_err'] = sub_err
        self.__ed_info['correct'] = correct
        return bt_info

    def get_string_display_width(self, str_in):
        """ very simple version for getting string displayed width """
        out_len = 0
        for item in str_in:
            if get_character_type(item) == 'chinese':
                out_len += len(item) * 2
            else:
                out_len += len(item)
        return out_len

    def gen_align_string(self, bt_info):
        """ generate aligned ref_str and res_str for display """
        ref_word_list = []
        res_word_list = []
        for item in bt_info:
            ref_word = item[0]
            res_word = item[1]
            equal_flag = self.is_word_equal(ref_word, res_word)
            if equal_flag:
                ref_word_list.append(ref_word)
                res_word_list.append(res_word)
            else:
                if ref_word:
                    ref_word_len = self.get_string_display_width(ref_word)
                else:
                    ref_word_len = 0
                if res_word:
                    res_word_len = self.get_string_display_width(res_word)
                else:
                    res_word_len = 0
                if not ref_word:
                    # ins
                    ref_word_list.append('*' * res_word_len)
                    res_word_list.append(res_word)
                elif not res_word:
                    # del
                    ref_word_list.append(ref_word)
                    res_word_list.append('*' * ref_word_len)
                else:
                    len_diff = ref_word_len - res_word_len
                    if len_diff >= 0:
                        ref_word_list.append("*%s*" %ref_word)
                        tmp_str = '*' * len_diff
                        res_word_list.append("*%s%s*" %(res_word, tmp_str))
                    else:
                        res_word_list.append("*%s*" %res_word)
                        len_diff = -len_diff
                        tmp_str = '*' * len_diff
                        ref_word_list.append("*%s%s*" %(ref_word, tmp_str))
        self.__align_info['ref_str'] = " ".join(ref_word_list[::-1])
        self.__align_info['res_str'] = " ".join(res_word_list[::-1])
        self.__align_info['fileid'] = self.__fileid

    def calculate(self, fileid, ref_str, res_str, align=False):
        """ main entry of EditDistanceCalculator """
        self.__fileid = fileid
        ref_word_list = ref_str.strip().split()
        res_word_list = res_str.strip().split()
        self.precompute(ref_word_list, res_word_list)
        self.compute()
        bt_info = self.backtrace()
        if align:
            self.gen_align_string(bt_info)
        print(self.__ed_info)
        print(self.__align_info)
        return self.__ed_info.copy(), self.__align_info.copy()