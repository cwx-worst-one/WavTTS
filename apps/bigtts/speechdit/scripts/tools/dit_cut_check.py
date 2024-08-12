import Levenshtein
import logging
import os
import re
import sys
import requests
import yaml
from zhconv import convert
from num2words import num2words


def download_yaml_file(lang):
    this_file_path = os.path.abspath(__file__)
    logging.info("this python path "+str(this_file_path))
    logging.info('Lang = ' + lang)
    tos_conf = {'cn': 'flamingo-trainset-asr-video-cn',
            'sg': 'flamingo-trainset-asr-sg',
            'va': 'flamingo-trainset-asr-global',
            'tx': 'lab-speech-alpaca-asr-tx/flamingo-trainset-asr-global'}
    config_path = os.path.join(os.path.dirname(this_file_path),
                                "format_config", "lang_%s.yaml" % lang)
    for idc in ['cn', 'va', 'sg', 'tx']:
        try:
            url = 'http://tosv.byted.org/obj/{}/text_format_config/lang_{}.yaml'.format(tos_conf[idc], lang)
            myfile = requests.get(url)
            myfile.raise_for_status()
            open(config_path, 'wb').write(myfile.content)
            logging.info('Successfully download lang_' + lang + '.yaml from ' + idc)
        except Exception as e:
            logging.warning('config not in {}'.format(idc))
            logging.warning(e)


def get_support_lang_map(filter="filter_keywords"):

    """ get the new support lang through the yaml file
    Args:
        None
    Return:
        lang_map: dict e.g { "tr" : "tr" }
    """
    config_dir_path = os.path.join(os.path.dirname(__file__), "format_config")
    tmp = [re.match(r"lang_(.*)\.yaml", e).group(1) for e in
           os.listdir(config_dir_path) if e.startswith("lang")]
    # tmp is all support lang_code
    lang_map = {e: e for e in tmp if e.split("-")[0] not in filter.split(',')}
    return lang_map


def get_lang_show_map(filter="filter_keywords"):

    """get the new support lang show map
    Return:
        lang_show_map : dict { "uig":"W" , "tr":"W"} for the wer/cer
    """
    config_dir_path = os.path.join(os.path.dirname(__file__), "format_config")
    tmp = []
    # pen the yaml and read the type of calculation (wer/cer)
    for e in os.listdir(config_dir_path):
        if e.startswith("lang"):
            lang_name = re.match(r"lang_(.*)\.yaml", e).group(1)
            with open(os.path.join(config_dir_path, e)) as f:
                try:
                    cal_type = yaml.load(f, Loader=yaml.FullLoader)["cal_type"]
                except Exception as e:
                    logging.warning('please run [!pip install pyyaml==5.4.1] to \
                            use the specific yaml version')
            if filter not in lang_name.split("-")[0]:
                tmp.append((lang_name, cal_type[0].upper()))
    lang_show_map = dict(tmp)
    return lang_show_map


class GeneralTextFormat(object):

    """ general text format by yaml congfigura """
    def __init__(self, lang="newEn", convert_num=True, category='speech', process_modal=False, process_abb=False, download_yaml=False):
        """
        open the lang config yaml and save it in _config
        init the pronunciation list
        """
        this_file_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(this_file_path),
                                   "format_config", "lang_%s.yaml" % lang)
        if download_yaml:
            download_yaml_file(lang)

        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        logging.info("config read successfully, config : ", self._config)
        self.CONVERT_NUM = convert_num
        self.PROCESS_MODAl = process_modal
        self.PROCESS_ABB = process_abb
        self.NON_PROUNICATION_CASE_PRE = self._config["non_prounciation_pre"]
        self.NON_PROUNICATION_CASE_POST = self._config["non_prounciation_post"]
        self.CHAR_VOCAB = self._config["char_vocab"]
        self.ORD_RANGE = self._config['ord_range']
        self.IN_WORD_HAVE_MEANING = self._config['in_word_have_meaning']
        self.DELIMITER = self._config['Delimiter']
        if self.PROCESS_MODAl:
            try:
                self.MODAL_WORDS = self._config['modal_words']
                self.MODAL_MAP = self._config['modal_map']
            except Exception as e:
                logging.warning('modal_words or modal_map not in yaml')
        if self.PROCESS_ABB:
            try:
                self.ABB_WORD_MAP = self._config['abb_word_map']
            except Exception as e:
                logging.warning('abb_word_map not in yaml')

        self.LANG = lang
        for item in self.IN_WORD_HAVE_MEANING:
            if item in self.NON_PROUNICATION_CASE_PRE:
                self.NON_PROUNICATION_CASE_PRE.remove(item)
        for item in self.IN_WORD_HAVE_MEANING:
            if item in self.NON_PROUNICATION_CASE_POST:
                self.NON_PROUNICATION_CASE_POST.remove(item)
        self.SUBTITUTE_WORDS = {}
        for key, val in self._config["sub_map"].items():
            if '\\u' in key:
                key=key.encode('utf-8').decode('unicode_escape')
            if '\\u' in val:
                val=val.encode('utf-8').decode('unicode_escape')
            self.SUBTITUTE_WORDS[key] = val
        self.CHAR_VOCAB_DICT = {}
        self.CHAR_ORD_DICT = {}
        # generate char dict with ord range
        for item in self.ORD_RANGE:
            start = int(item.split('-')[0])
            end = int(item.split('-')[1])
            for i in range(start, end+1):
                self.CHAR_ORD_DICT[i] = 1
                i += 1
        for item in self.CHAR_VOCAB:
            # support unicode liek '\u0646' in vocab_list
            if '\\u' in item:
                item=item.encode('utf-8').decode('unicode_escape')
            self.CHAR_VOCAB_DICT[item] = 1

    def delete_non_pronunciation_case_pre(self, str_in):
        """Delete special case in self.NON_PROUNICATION_CASE_PRE
        Args:
            self.NON_PROUNICATION_CASE_PRE: [del_case]
            str_in: string
        return:
            str_out: string
        """
        str_out = ""
        if self.DELIMITER == "":
            str_out = str_in
        else:
            str_in_list = str_in.split(self.DELIMITER)
            words = []
            for word in str_in_list:
                if word != "" and word[0] in self.IN_WORD_HAVE_MEANING:
                    word = word[1:]
                if word != "" and word[-1] in self.IN_WORD_HAVE_MEANING:
                    word = word[:-1]
                words.append(word)
            str_out = self.DELIMITER.join(words)
        for item in self.NON_PROUNICATION_CASE_PRE:
            str_out = str_out.replace(item, ' ')
        return str_out

    def delete_non_pronunciation_case_post(self, str_in):
        """Delete special case in self.NON_PROUNICATION_CASE_POST:
        Args:
            self.NON_PROUNICATION_CASE_POST: [del_case]
            str_in: string
        return:
            str_out: string
        """
        str_out = ""
        if self.DELIMITER == "":
            str_out = str_in
        else:
            str_in_list = str_in.split(self.DELIMITER)
            words = []
            for word in str_in_list:
                if word != "" and word[0] in self.IN_WORD_HAVE_MEANING:
                    word = word[1:]
                if word != "" and word[-1] in self.IN_WORD_HAVE_MEANING:
                    word = word[:-1]
                words.append(word)
            str_out = self.DELIMITER.join(words)
        for item in self.NON_PROUNICATION_CASE_POST:
            str_out = str_out.replace(item, ' ')
        return str_out

    def get_character_type(self, char_in):
        """ indentify the character type
        Args:
            str_in: string
        Return:
            type_out: 'digit', 'english', 'chinese', 'other'
        """
        item = char_in
        if (ord(item) >= 48 and ord(item) <= 57) or item == '.':
            type_out = 'digit'
        elif ((ord(item) >= 65 and ord(item) <= 90) or
                (ord(item) >= 97 and ord(item) <= 122) or item == "'"):
            type_out = 'english'
        elif (item >= u'\u4e00' and item <= u'\u9fa5') or \
                (item >= u'\u3400' and item <= u'\u4dbf') or \
                (item >= u'\uf900' and item <= u'\ufaff') or \
                (item >= u'\U00020000' and item <= u'\U0002A6DF') or \
                (item >= u'\U0002A700' and item <= u'\U0002EBEF') or \
                (item >= u'\U00030000' and item <= u'\U000323AF') or \
                (item >= u'\U0002F800' and item <= u'\U0002FA1F') or \
                (item == u'\u3007'):
            type_out = 'chinese'
        else:
            type_out = 'other'
        return type_out

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

    def convert_matchobj_num_word(self, matchobj):
        """ convert the string num in matchobj to word
        Args:
            matchobj : re.Match or string
        Return:
            word : string
        """
        tmp_str = ""
        if type(matchobj) == str:
            tmp_str = matchobj
        else:
            tmp_str = matchobj.group("number")
        use_comma = True
        if "," in tmp_str:
            use_comma = True
            # print("use ,")
        else:
            use_comma = False

        tmp_str = tmp_str.replace(r",", r".")  # in turkish, 1.2 can be 1,2

        if "." in tmp_str:
            num_part = tmp_str.split('.')[0]
            float_part = tmp_str.split('.')[1]
            word = "%s nokta %s" % (
                    self.convert_tr_leadzeronum_word(num_part),
                    self.convert_tr_leadzeronum_word(float_part))
        else:
            word = self.convert_tr_leadzeronum_word(tmp_str)
        # print("the number is : %s" % matchobj.group("number"))
        # print("virgül" in word)
        if use_comma:
            word = word.replace(r"nokta", r"virgül")
        else:
            word = word.replace(r"virgül", r"nokta")
        # consid the suffix
        if matchobj.group("suffix"):
            word = word.strip()+matchobj.group("suffix")
        return word

    def convert_matchobj_unit_word(self, matchobj):
        """ convet the unit shortcut to word
        Args:
            matchobj : re.Match
        return:
            word : string
        """
        return self._config["unit"][matchobj.group("unit")]

    def __convert_num_word(self, pattern, func, text):
        """ convert the string num to word
        Args:
            pattern : string
            func : function  input : matchobj return str
            text :string
        Return:
            text : string
        """
        start = 0
        text_tmp = ""
        for i in re.finditer(pattern, text):
            word = func(i)
            text_tmp += text[start:i.span()[0]] + " " + word + " "
            start = i.span()[1]
        if start < len(text):
            text = text_tmp + text[start:]
        else:
            text = text_tmp
        return text

    def convert_tr_leadzeronum_word(self, num_str):
        """ convert the leading zero number to word
        Args:
            num_str : string
        Retuen:
            word : string
        """
        # logging.info("num_str %s" % num_str)
        word = ""
        idx = 0
        try:
            for i in range(len(num_str)):
                if num_str[idx] == '0':
                    word += " sıfır "
                    # logging.info("word %s " % word )
                else:
                    break
                idx += 1
            if not idx == len(num_str):
                word = word + num2words(int(num_str[idx:]),
                                        lang=self.LANG.split("-")[0])
        except Exception:
            return num_str
        return word

    def convert_tr_num_word(self, text):
        """convet the tr num to words
        preprocessing : todo
            because turkish number expression is different with english
            e.g. tr: 100.100.100,07 => en: 100,100,100.07 the use of , and .
            is reverse therefore need to delete the . in number
        step 1 deal with the number + unit
        step 2 deal with the % + number
        step 3 deal with the fraction
        step 4 deal with the ordinal
        step 5 deal with the left number
        """
#        preprocessing: delete the . in number when . appear more than 2 times
        text = re.sub(r"\d{1,3}\.\d{3}(\.\d{3})+",
                      lambda x: x.group(0).replace(".", ""), text)
        for key in self._config["unit"]:  # step 1
            text = self.__convert_num_word(
                r"(?P<number>\d+[,.]\d+|\d+)('(?P<"
                "suffix>[abcçdefgğhıijklmnoöprsştu"
                "üvyz]+)){0,1}\s*(?P<unit>%s)" % key,
                lambda x: " %s %s " % (self.convert_matchobj_num_word(x),
                                       self.convert_matchobj_unit_word(x)),
                text)
        # step 2
        text = self.__convert_num_word(
                r"%\s*(?P<number>\d+[,.]\d+|\d+)('"
                "(?P<suffix>[abcçdefgğhıijklmnoöpr"
                "sştuüvyz]+)){0,1}",
                lambda x: r" %s %s " %
                ("yüzde", self.convert_matchobj_num_word(x)),
                text)
        # step 3
        text = re.sub(r"(\d+)/(\d+)", r"\1 bölü \2", text)
        # step 4 todo
        # step 5
        text = self.__convert_num_word(
                r"(?P<number>\d+[,.]\d+|\d+)('(?P<"
                "suffix>[abcçdefgğhıijklmnoöprsştu"
                "üvyz]+)){0,1}",
                lambda x: r" %s " % (self.convert_matchobj_num_word(x)),
                text)
        return text

    def convert_hi_num_word(self, text):
        """convert the hi num to words
        step 1 deal with the decimal
        step 2 deal with the fraction
        step 3 deal with the integer
        """
        num_list = self._config["cardinals"]
        large_num = self._config["large_num"]
        # preprocess the , in number
        text = re.sub(r"([\d,]+)",
                      lambda x: x.group(0).replace(",", ""), text)

        def nature_num_word(text):
            """return the nature num to word"""
            if text == "":
                return text
            if len(text) > 12:
                return text
            if len(text) <= 2:
                if text == "00":  # don't read zero
                    return ""
                else:
                    return num_list[int(text)]
            # read the leading zero
            # if text[0] == "0":
                # return num_list[0] + " " + nature_num_word(text[1:])
            # read the first num and the large num
            if len(text) in large_num:
                if int(text[0]) == 0:
                    return nature_num_word(text[1:])  # don't read zero
                else:
                    return num_list[int(text[0])] + " " +\
                           large_num[len(text)] + " " +\
                           nature_num_word(text[1:])
            elif len(text) - 1 in large_num:
                if int(text[:2]) == 0:
                    return nature_num_word(text[2:])  # don't read zero
                else:
                    return num_list[int(text[:2])] + " " +\
                           large_num[len(text)-1] + " " +\
                           nature_num_word(text[2:])
            else:
                return text

        def decimal_num_word(text):
            """return the decimal to word"""
            t1, t2 = text.split(".")
            t1 = nature_num_word(t1)  # int part, 10.2 => 10
            t2 = [nature_num_word(c) for c in t2]  # float part, 10.2 => 2
            t2 = " ".join(t2)
            return t1 + " " + "दशमलव" + " " + t2

        def fraction_num_word(re_obj, fraction_map):
            """return the re_obj to word"""
            fraction = re_obj.group(0)
            a = re_obj.group(1)
            b = re_obj.group(2)
            if int(b) == 2 or int(b) == 4:
                if fraction in fraction_map:
                    return fraction_map[fraction]
                elif int(b) == 2:
                    first = fraction_map["7/2"].split()[0]
                    return first + " " + nature_num_word(str(int(a)//int(b)))
                else:
                    first = ""  # the first part of words
                    second = ""  # the second part of words
                    if int(a) % int(b) == 1:
                        # the x + 1/4 fraction
                        first = fraction_map["9/4"].split()[0]
                        second = nature_num_word(str(int(a)//int(b)))
                    else:
                        # the x + 3/4 fraction
                        first = fraction_map["7/4"].split()[0]
                        second = nature_num_word(str(int(a)//int(b) + 1))
                    return first + " " + second
            else:
                return nature_num_word(a) + " " + fraction_map["/"] + " "\
                       + nature_num_word(b)
        # step 1
        text = re.sub(r"(\d+\.\d+)", lambda x: decimal_num_word(x.group(0)),
                      text)
        # step 2
        text = re.sub(
            r"(\d+)/(\d+)",
            lambda x: fraction_num_word(x, self._config["fraction_map"]),
            text
        )
        # step 3
        text = re.sub(r"(\d+)", lambda x: nature_num_word(x.group(0)), text)
        return text

    def convert_ru_num_word(self, text):
        """convert the ru num to words
        step 1 convert the 6 form of word into 1 form
        step 2 use the num2words convert the num
        """
        # step 1
        l_text = text.split(" ")
        ord_map = self._config["cardinals"]
        ret = []
        for word in l_text:
            c_flag = True
            for key in ord_map:
                for item in ord_map[key]:
                    if word == item and c_flag:
                        ret.append(ord_map[key][0])
                        c_flag = False
            if c_flag:
                ret.append(word)
        text = " ".join(ret)
        text = re.sub(
            r"\d{1,3}([ ]\d{3})+",
            lambda x: x.group(0).replace(" ", ""),
            text)  # delete the blank in number
        text = re.sub(
            r"\d+,\d+",
            lambda x: x.group(0).replace(",", "."),
            text)  # convert the , to .
        text = self.__convert_num_word(
            r"\d+([.]{0,1}\d+){0,1}",
            lambda x: num2words(x.group(0), lang=self.LANG.split("-")[0]),
            text)  # convert the num to words
        return text

    def convert_bo_num_word(self, text):
        """convert the bo num to words
        step 1 convert the cardinal to words
        """
        # step 1
        ord_map = self._config["cardinals"]
        large_map = self._config["large_num"]

        def bo_nature_read(text, ord_map, large_map):
            # use the recursion to read words
            if len(text) > 10:  # too long
                return text
            if len(text) <= 2:
                if text == "00":
                    return ""
                else:
                    return ord_map[int(text)]
            else:
                if text[0] == "0":
                    return bo_nature_read(text[1:], ord_map, large_map)
                else:
                    if len(text) == 3:  # special for hundred
                        tmp = "%s %s" % (ord_map[int(text[0])],
                                         large_map[len(text)])
                        tmp = tmp.replace("གཉིས།", "ཉིས")  # special for 2
                        tmp = tmp.replace("གསུམ།", "སུམ")  # special for 3
                    else:
                        tmp = "%s %s" % (large_map[len(text)],
                                         ord_map[int(text[0])])
                    return "%s %s" % (
                        tmp,
                        bo_nature_read(text[1:], ord_map, large_map))

        text = self.__convert_num_word(
            r'\d+',
            lambda x: bo_nature_read(x.group(0), ord_map, large_map),
            text)
        return text

    def convert_num_word(self, text):
        """use the third package num2words convert num to word
        """
        # special lang support
        if self.LANG == "tr-TR":
            return self.convert_tr_num_word(text)
        if self.LANG == "hi-IN":
            return self.convert_hi_num_word(text)
        if self.LANG == "ru-RU":
            return self.convert_ru_num_word(text)
        if self.LANG == "bo":
            return self.convert_bo_num_word(text)
        # convert the float to words
        start = 0
        text_tmp = ""
        for i in re.finditer(r'(\d+)\.(\d+)', text):
            num_str = i.group()
            try:
                word = num2words(num_str, lang=self.LANG.split("-")[0])
            except NotImplementedError:
                logging.warning("not support "+self.LANG.split("-")[0] +
                                " lang num2word")
                return text
            text_tmp += text[start:i.span()[0]] + " " + word + " "
            start = i.span()[1]
        if start < len(text):
            text = text_tmp + text[start:]
        else:
            text = text_tmp
        # todo: convert the intHz to words with unit ( 10Hz -> 十 赫 兹 )
        # todo: convert the fraction to words
        # convert the int to words
        start = 0
        text_tmp = ""
        for i in re.finditer(r'(\d+)', text):
            num_str = i.group()
            try:
                word = num2words(num_str, lang=self.LANG.split("-")[0])
            except NotImplementedError:
                logging.warning("not support "+self.LANG.split("-")[0] +
                                " lang num2word")
                return text
            text_tmp += text[start:i.span()[0]] + " " + word + " "
            start = i.span()[1]
        if start < len(text):
            text = text_tmp + text[start:]
        else:
            text = text_tmp
        return text

    def subtitute_word_in_map(self, str_in):
        for key in self.SUBTITUTE_WORDS:
            str_in = str_in.replace(key, self.SUBTITUTE_WORDS[key])
        return str_in

    def generate_list_from_string(self, str_in):
        '''split the string for wer/cer (by configuration)
        '''
        if self._config["cal_type"] == "cer":
            # for cal cer
            str_out = []
            for index, i in enumerate(str_in):
                if i == " ":
                    continue
                char_type = self.get_character_type(i)
                if (char_type in ['english', 'digit'] and len(str_out) != 0 and
                        self.get_character_type(str_out[-1][-1]) == char_type and
                        str_in[index-1] != ' '):
                    str_out[-1] = str_out[-1] + i
                else:
                    str_out.append(i)
        elif self._config["cal_type"] == "wer":
            # for cal wer
            str_out = []
            for i in str_in.split(self.DELIMITER):
                if len(i) == 0:
                    continue
                str_out.append(i)
        else:
            str_out = []
            logging.ERROR("error cal type! must be wer/cer")
        return str_out

    def generate_char_word_mixed_str(self, str_in):
        '''generate str from zh_en_mixed list
        '''
        list_out= []
        tmp_list = str_in.split()
        for item in tmp_list:
            tmp_out = []
            for index, i in enumerate(item):
                char_type = self.get_character_type(i)
                if char_type in ['english', 'other']:
                    if (len(tmp_out) == 0 or
                            self.get_character_type(tmp_out[-1][-1])
                            not in ['english', 'other']):
                        tmp_out.append(i)
                    else:
                        tmp_out[-1] = tmp_out[-1] + i
                else:
                    tmp_out.append(i)
            list_out.extend(tmp_out)

        str_out = ""
        last_char_type = ""
        for index, i in enumerate(list_out):
            if i == " ":
                continue
            char_type = self.get_character_type(i[0])
            if char_type == 'english':
                if last_char_type == '':
                    str_out += i
                else:
                    str_out += " " + i
            else:
                if last_char_type == 'english':
                    str_out += " "
                str_out += i
            last_char_type = char_type
        return str_out


    def save_char_in_vocab(self, str_in):
        '''save the char in the vocab
        '''
        str_out = ""
        for item in str_in:
            if item in self.CHAR_VOCAB_DICT or ord(item) in self.CHAR_ORD_DICT or item in self.IN_WORD_HAVE_MEANING:
                str_out += item
        return str_out

    def abbreviations_to_words(self, str_in):
        # TODO not support char-word-mix situation
        str_out = []
        str_in = self.generate_list_from_string(str_in)
        for item in str_in:
            if item in self.ABB_WORD_MAP:
                str_out.append(self.ABB_WORD_MAP[item])
            else:
                str_out.append(item)
        str_out = ' '.join(str_out)
        return str_out


    def merge_modal_words(self, str_in):
        # TODO not support char-word-mix situation
        tmp_out = []
        str_out = []
        str_in = self.generate_list_from_string(str_in)
        # sub similar modal words: haha-->ha, hahaha-->ha, uh-->em
        for item in str_in:
            if item in self.MODAL_MAP:
                tmp_out.append(self.MODAL_MAP[item])
            else:
                tmp_out.append(item)
        # merge same modal words to once: ha ha ha --> ha
        curr_modal = ''
        for i in range(len(tmp_out)):
            if tmp_out[i] in self.MODAL_WORDS:
                if curr_modal != tmp_out[i]:
                    curr_modal = tmp_out[i]
                    str_out.append(tmp_out[i])
                else:
                    continue
            else:
                str_out.append(tmp_out[i])
                curr_modal = ''
        str_out = ' '.join(str_out)
        return str_out


    def text_format(self, str_in, list_out=False):
        """ Text formation for lang which is configured in yaml
        Args:
            str_in: string
        Return:
            str_out: string list
        """
        str_out = self.delete_non_pronunciation_case_pre(str_in)
        str_out = self.subtitute_word_in_map(str_out)
        if self.CONVERT_NUM:
            try:
                str_out = self.convert_num_word(str_out)
            except Exception as e:
                logging.warning(e)
        str_out = self.delete_non_pronunciation_case_post(str_out)
        cn_char_type = self._config.get("cn_char_type","")
        str_out = convert(str_out, cn_char_type)
        str_out = self.save_char_in_vocab(str_out)
        if self.PROCESS_ABB:
            str_out = self.abbreviations_to_words(str_out)
        if self.PROCESS_MODAl:
            str_out = self.merge_modal_words(str_out)
        if 'char_word_mixed' in self._config and self._config['char_word_mixed'] is True:
            str_out = self.generate_char_word_mixed_str(str_out)
        if list_out:
            return self.generate_list_from_string(str_out)
        return " ".join(str_out.split())


# 计算编辑距离
def editDistance(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    return distance


def editOperations(str1, str2):
    operations = Levenshtein.editops(str1, str2)
    return operations


def read_in(filename):
    f = open(filename)
    out_dict = {}
    for line in f:
        line = line.strip().replace("\t", " ").split(" ", 1)
        if len(line) == 1: 
            line.append("") 
        out_dict[line[0]] = line[1]
    return out_dict


def get_cut_ids(res_dict, ref_dict, cal_type, lang):
    cut_ids = []
    process_abb = True if lang != 'zh-CN' else False
    formator = GeneralTextFormat(lang=lang, convert_num=True, category="speech", process_abb=process_abb)
    if cal_type == "char":
        for k,v in res_dict.items():
            res_char_num = len(re.sub(r'[^a-zA-Z0-9]', '', v).strip().replace(" ", ""))
            ref_char_num = len(re.sub(r'[^a-zA-Z0-9]', '', ref_dict[k]).strip().replace(" ", ""))
            if (res_char_num - ref_char_num) < -2:
                cut_ids.append(k)     
    if cal_type == "wer":
        for k,v in res_dict.items():
            res_text = formator.text_format(v, list_out=True)
            ref_text = formator.text_format(ref_dict[k], list_out=True)
            operations = editOperations(ref_text, res_text)
            tail_deletions = [x for x in operations if (x[0] == "delete" and x[2] == len(res_text))]
            if tail_deletions != []:
                cut_ids.append(k)
    if cal_type == "char+wer":
        for k,v in ref_dict.items():
            if k not in res_dict:
                cut_ids.append(k)
                print(f'Not found in res_dict: {k}')
                continue
            res_text = formator.text_format(res_dict[k], list_out=True)
            ref_text = formator.text_format(v, list_out=True)
            res_char_num = len(''.join(res_text).strip().replace(" ", ""))
            ref_char_num = len(''.join(ref_text).strip().replace(" ", ""))
            if (res_char_num - ref_char_num) < -4:
                cut_ids.append(k)
                # print()
                # print(f'CUT: (res_char_num - ref_char_num) < -4')
                # print('    => ref_text:', ref_text)
                # print('    => res_text:', res_text)
                continue
            operations = editOperations(ref_text, res_text)
            tail_deletions = [x for x in operations if (x[0] == "delete" and x[2] == len(res_text))]
            if tail_deletions != []:
                cut_ids.append(k)
                # print()
                # print(f'CUT: tail_deletions != []')
                # print('    => ref_text:', ref_text)
                # print('    => res_text:', res_text)
    return cut_ids


res = sys.argv[1]
ref = sys.argv[2]
lang = sys.argv[3]
out_file = sys.argv[4]
cal_type = sys.argv[5]

ref_dict = read_in(ref)
res_dict = read_in(res)
cut_ids = get_cut_ids(res_dict, ref_dict, cal_type, lang)
cutted_sample = 0
with open(out_file, 'w') as f_out:
    for item in cut_ids:
        f_out.write(item + "\n")
        cutted_sample += 1
res_sample = len(res_dict.items())
ref_sample = len(ref_dict.items())
print("================\n================\n")
print("ISCRONY RESULTS\n")
print("res samples: {}".format(res_sample))
print("ref samples: {}".format(ref_sample))
print("cutted off samples: {}".format(cutted_sample))
print("cutted off rate: {:.2f}%\n".format(cutted_sample/ref_sample*100))
print("================\n================\n")