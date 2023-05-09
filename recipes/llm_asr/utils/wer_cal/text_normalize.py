#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# heyi.hy@bytedance.com

"""
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
import codecs




class TestsetTextFormat(object):
    """ Text format for testset """

    NON_PROUNICATION_CASE_MAP = {
        u",": 1,
        u".": 1,
        u"!": 1,
        u"?": 1,
        u"\"": 1,
        u"'": 1,
        u";": 1,
        u"{": 1,
        u"}": 1,
        u"[": 1,
        u"]": 1,
        u"<": 1,
        u">": 1,
        u"\\": 1,
        u"/": 1,
        u"=": 1,
        u"+": 1,
        u"-": 1,
        u"*": 1,
        u"&": 1,
        u"^": 1,
        u"#": 1,
        u"$": 1,
        u"%": 1,
        u"~": 1,
        u"`": 1,
        u"|": 1,
        u"，": 1,
        u"。": 1,
        u"！": 1,
        u"？": 1,
        u"、": 1,
        u"／": 1,
        u"“": 1,
        u"”": 1,
        u"‘": 1,
        u"’": 1,
        u"【": 1,
        u"】": 1,
        u"「": 1,
        u"；": 1,
        u"《": 1,
        u"》": 1,
        u":": 1,
        u"：": 1,
    }

    @staticmethod
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
                (ord(item) >= 97 and ord(item) <= 122) or item == "'"):
            type_out = 'english'
        elif item >= u'\u4e00' and item <= u'\u9fa5':
            type_out = 'chinese'
        else:
            type_out = 'other'
        return type_out

    @staticmethod
    def convert_sbc_to_dbc(sbc_str):
        """Convert SBC case character to DBC case
        Args:
            sbc_str: string, might containing SBC case character, unicode
        Return:
            dbc_str: string, all dbc characters, unicode
        """
        dbc_str = ""
        sbc_blank_code = 12288
        dbc_blank_code = 32
        sbc_max_code = 65373
        sbc_min_code = 65281
        non_blank_code_diff = 65248
        for char in sbc_str:
            code = ord(char)
            if code == sbc_blank_code:
                code = dbc_blank_code
            elif code >= sbc_min_code and code <= sbc_max_code:
                code -= non_blank_code_diff
            dbc_str += unichr(code)
        return dbc_str

    @staticmethod
    def number_arabic_to_chinese(number_in):
        """ int to chinese, 20 to er4 ling2
        Args:
            number_in: int or string
        Return:
            out_str: string in chinese letters
        """
        number_list = [u'零', u'一', u'二', u'三', u'四', u'五',
                       u'六', u'七', u'八', u'九']
        out_str = ""
        in_str = str(number_in)
        for item in in_str:
            if TestsetTextFormat.get_character_type(item) != 'digit':
                logging.warn('Non digit character: %s' %item)
                return in_str
            num_id = int(item)
            out_str += number_list[num_id]
        return out_str

    @staticmethod
    def int_arabic_to_chinese(int_in):
        """ int to chinese, 20 to er4 shi2
            changing to number_arabic_to_chinese for some cases(fix me?)
        Args:
            int_in: int
        Return:
            out_str: string in chinese letters
        """
        number_list = [u'零', u'一', u'二', u'三', u'四', u'五',
                       u'六', u'七', u'八', u'九']
        bit_list = ['', u'十', u'百', u'千', u'万', u'十万', u'百万', u'千万']
        max_num_len = 8
        in_str = str(int_in)
        if in_str == '':
            return ''
        for item in in_str:
            if TestsetTextFormat.get_character_type(item) != 'digit':
                logging.warn("Non digit character: %s" %item)
                return in_str
        if len(in_str) > max_num_len:
            logging.warn('Too many numbers for convert: %s' %in_str)
            return TestsetTextFormat.number_arabic_to_chinese(in_str)
        if len(in_str) == 1:
            return number_list[int(in_str)]
        if in_str[0] == '0':
            logging.warn("wrong format of number for convertion: %s" %in_str)
            return TestsetTextFormat.number_arabic_to_chinese(in_str)
        out_str = ""
        bit_id = len(in_str) - 1
        for item in in_str:
            num_id = int(item)
            if num_id == 0:
                out_str += number_list[0]
            else:
                out_str += "%s%s" %(number_list[num_id], bit_list[bit_id])
            bit_id -= 1
        tmp_list = out_str.split(u'零')
        tmp_list = filter(None, tmp_list)
        out_str = u'零'.join(tmp_list)
        if out_str[:2] == u'一十':
            out_str = out_str[1:]
        return out_str

    @staticmethod
    def float_to_chinese(float_in):
        """ Float to chinese, 20.34 to er4 shi2 dian3 san1 si4
        Args:
            float_in: float or string
        Return:
            out_str: string in chinese letters
        """
        str_in = str(float_in)
        tmp_list = str_in.split('.')
        if len(tmp_list) != 2:
            logging.warn('wrong float number for convertion')
            return float_in
        out_str = (TestsetTextFormat.int_arabic_to_chinese(tmp_list[0]) +
                   u'点' + TestsetTextFormat.number_arabic_to_chinese(tmp_list[1]))
        return out_str

    @staticmethod
    def convert_math_symbol(str_in):
        """ Convert math symbol to chinese
        Args:
            str_in: unicode
        Return:
            str_out: unicode
        """
        str_out = ""
        for item in str_in:
            if item == '+':
                str_out += u"加"
            elif item == '<':
                str_out += u"小于"
            elif item == '>':
                str_out += u"大于"
            elif item == "=":
                str_out += u"等于"
            elif ord(item) == 247:
                str_out += u"除"
            elif ord(item) == 215:
                str_out += u"乘"
            else:
                str_out += item
        return str_out

    @staticmethod
    def convert_percent_symbol(str_in):
        """ find % and convert percent number
        Args:
            str_in: unicode
        Return:
            str_out: unicode
        """
        #str_len = len(str_in)
        replace_list = []
        target_pos_list = [m.start() for m in re.finditer('%', str_in)]
        if not target_pos_list:
            return str_in
        for pos in target_pos_list:
            target_str = ""
            for item in str_in[pos-1::-1]:
                if TestsetTextFormat.get_character_type(item) == 'digit' or item == '.':
                    target_str += item
                else:
                    break
            if target_str == '':
                continue
            target_str = target_str[::-1]
            if target_str.find('.') >= 0:
                chinese_str = TestsetTextFormat.float_to_chinese(target_str)
            else:
                chinese_str = TestsetTextFormat.int_arabic_to_chinese(target_str)
            target_str += '%'
            chinese_str = u'百分之' + chinese_str
            replace_list.append([target_str, chinese_str])
        str_out = str_in
        for item in replace_list:
            str_out = str_out.replace(item[0], item[1], 1)
        return str_out

    @staticmethod
    def convert_float_numer(str_in):
        """ Convert float number to chinese
        Args:
            str_in: unicode
        Return:
            str_out: unicode
        """
        #str_len = len(str_in)
        replace_list = []
        target_pos_list = [m.start() for m in re.finditer('.', str_in)]
        if not target_pos_list:
            return str_in
        for pos in target_pos_list:
            str_1 = ""
            for item in str_in[pos-1::-1]:
                if TestsetTextFormat.get_character_type(item) == 'digit':
                    str_1 += item
                else:
                    break
            str_1 = str_1[::-1]
            if str_1 == '':
                continue
            str_2 = ""
            for item in str_in[pos+1:]:
                if TestsetTextFormat.get_character_type(item) == 'digit':
                    str_2 += item
                else:
                    break
            if str_2 == '':
                continue
            target_str = str_1 + '.' + str_2
            chinese_str = TestsetTextFormat.float_to_chinese(target_str)
            replace_list.append([target_str, chinese_str])
        str_out = str_in
        for item in replace_list:
            str_out = str_out.replace(item[0], item[1], 1)
        return str_out

    @staticmethod
    def convert_int_number(str_in):
        """ Convert int number to chinese
        Args:
            str_in: unicode
        Return:
            str_out: unicode
        """
        int_str_list = []
        int_str = ''
        for item in str_in:
            if TestsetTextFormat.get_character_type(item) == 'digit':
                int_str += item
            else:
                if int_str != '':
                    int_str_list.append(int_str)
                int_str = ''
        if int_str != '':
            int_str_list.append(int_str)
        str_out = str_in
        for item in int_str_list:
            chinese_str = TestsetTextFormat.number_arabic_to_chinese(item)
            str_out = str_out.replace(item, chinese_str, 1)
        return str_out

    @staticmethod
    def delete_non_pronunciation_case(str_in, is_train=False):
        """Delete special case in delete-map
        Args:
            delete_map: {del_case: 1}
            str_in: string
        return:
            str_out: string
        """
        str_out = str_in
        ignore_list = TestsetTextFormat.NON_PROUNICATION_CASE_MAP.copy()
        if is_train and u"'" in ignore_list:
            del ignore_list[u"'"]
        for item in ignore_list:
            if item in str_out:
                str_out = str_out.replace(item, ' ')
        return str_out

    @staticmethod
    def merge_words(merge_map, str_in):
        """ Merge words for continuous
        Args:
            merge_map: {merge_case: 1}
            str_in: string
        return:
            str_out: string
        """
        if len(str_in) == 0:
            return str_in
        in_word_list = str_in.split()
        out_str = in_word_list[0]
        last_item = in_word_list[0]
        for item in in_word_list[1:]:
            if item == last_item and item in merge_map:
                last_item = item
                continue
            out_str += " " + item
            last_item = item
        return out_str

    @staticmethod
    def exclude_words(exclude_map, str_in):
        """ Exclude words which is not important
        Args:
            exclude_map: {exclude_case: 1}
            str_in: string
        Return:
            str_out: string
        """
        if len(str_in) == 0:
            return str_in
        in_word_list = str_in.split()
        out_str = in_word_list[0]
        for item in in_word_list[1:]:
            if item not in exclude_map:
                out_str += " " + item
        return out_str

    @staticmethod
    def text_format(str_in, is_train=False):
        """ Text formation for cn and mixed_cn_en
        Args:
            str_in: unicode
        Return:
            str_out: unicode
        """
        str_1 = TestsetTextFormat.convert_math_symbol(str_in)
        str_2 = TestsetTextFormat.convert_percent_symbol(str_1)
        str_3 = TestsetTextFormat.convert_float_numer(str_2)
        str_4 = TestsetTextFormat.convert_int_number(str_3)
        str_4 = str_4.replace('%', u'百分号')
        str_5 = TestsetTextFormat.delete_non_pronunciation_case(str_4, is_train)
        str_5 = ' '.join(str_5.split())  # del continous multi blank
        str_out = ''
        last_char_type = ' '
        for item in str_5:
            if item == ' ':
                str_out += item
                last_char_type = item
                continue
            char_type = TestsetTextFormat.get_character_type(item)
            if char_type == 'english':
                if last_char_type == 'english' or (is_train and last_char_type == ' '):
                    str_out += item.lower()
                else:
                    str_out += ' ' + item.lower()
            else:
                str_out += ' ' + item
            last_char_type = char_type
        str_out = ' '.join(str_out.split())
        return str_out

class TestsetTextFormatPT(TestsetTextFormat):

    NON_PROUNICATION_CASE_MAP = {
        u"!": 1,
        u"\"": 1,
        u"'": 1,
        u";": 1,
        u"{": 1,
        u"}": 1,
        u"[": 1,
        u"]": 1,
        u"<": 1,
        u">": 1,
        u"\\": 1,
        u"/": 1,
        u"=": 1,
        u"+": 1,
        u"-": 1,
        u"*": 1,
        u"&": 1,
        u"^": 1,
        u"#": 1,
        u"$": 1,
        u"%": 1,
        u"~": 1,
        u"`": 1,
        u"|": 1,
        u"！": 1,
        u"、": 1,
        u"／": 1,
        u"“": 1,
        u"”": 1,
        u"‘": 1,
        u"’": 1,
        u"【": 1,
        u"】": 1,
        u"「": 1,
        u"；": 1,
        u"《": 1,
        u"》": 1,
        u":": 1,
        u"：": 1,
    }

    PUNCT = {".": 1, ",": 1, "?": 1}
    
    @staticmethod
    def delete_non_pronunciation_case(str_in, is_train=False):
        """Delete special case in delete-map
        Args:
            delete_map: {del_case: 1}
            str_in: string
        return:
            str_out: string
        """
        str_out = str_in
        ignore_list = TestsetTextFormatPT.NON_PROUNICATION_CASE_MAP.copy()
        if is_train and u"'" in ignore_list:
            del ignore_list[u"'"]
        for item in ignore_list:
            if item in str_out:
                str_out = str_out.replace(item, ' ')
        return str_out
    
    @staticmethod
    def text_format(str_in, is_train=False):
        """ Text formation for cn and mixed_cn_en
        Args:
            str_in: unicode
        Return:
            str_out: unicode
        """
        str_1 = TestsetTextFormat.convert_math_symbol(str_in)
        str_2 = TestsetTextFormat.convert_percent_symbol(str_1)
        str_3 = TestsetTextFormat.convert_float_numer(str_2)
        str_4 = TestsetTextFormat.convert_int_number(str_3)
        str_4 = str_4.replace('%', u'百分号')
        str_5 = TestsetTextFormatPT.delete_non_pronunciation_case(str_4, is_train)
        str_5 = ' '.join(str_5.split())  # del continous multi blank
        str_out = ''
        last_char_type = ' '
        for item in str_5:
            if item == ' ':
                str_out += item
                last_char_type = item
                continue
            char_type = TestsetTextFormat.get_character_type(item)
            if char_type == 'english':
                if last_char_type == 'english' or (is_train and last_char_type == ' '):
                    str_out += item
                else:
                    str_out += ' ' + item
            else:
                str_out += ' ' + item
            last_char_type = char_type
        str_out = ' '.join(str_out.split())
        return str_out

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-text', default='-')
    parser.add_argument('-o', '--output-text', default='-')
    parser.add_argument('--no-key', action='store_true')
    args = parser.parse_args()
    
    if args.input_text == '-':
        fr = sys.stdin
    else:
        fr = codecs.open(args.input_text, 'r', 'utf-8')
    if args.output_text == '-':
        fw = sys.stdout
    else:
        fw = codecs.open(args.output_text, 'w', 'utf-8')
            
    datas = []
    for line in fr:
        line = line.strip()
        if args.no_key:
            text = TestsetTextFormat.text_format(line)
            fw.write(text+'\n')
        else:
            arr = line.split()
            key, text = arr[0], ' '.join(arr[1:])
            text = TestsetTextFormat.text_format(text)
            fw.write('%s %s\n' % (key, text))
            
        
    args.input_text != '-' and fr.close()
    args.output_text != '-' and fw.close()

    
