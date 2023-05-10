"""cer utils
Note: the repo of asr_eval_tool is https://code.byted.org/lab-speech/asr_eval_tool/tree/master
"""
import sys
import codecs
import re
import os
import os.path as osp
import json
from core.utils import logging

try:
    from asr_eval_tool import EditDistanceCalculator, TextFormator
except ImportError:
    EditDistanceCalculator = None
    TextFormator = None

try:
    from asr_eval_tool import CalculatorCombiner
except ImportError:
    CalculatorCombiner = None

try:
    import falconclaw
except ImportError:
    falconclaw = None


def load_list_file_to_map(filename):
    """Load list file to map
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
    """Load substitute words file
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
                logging.warning("Duplicated item %s in %s" % (item, filename))
            out_map[item] = val
    fp_in.close()
    return out_map


def load_scp_file_to_map(filename):
    """Load scp file to map
    Args:
        filename: string
    Return:
        out_map: {fileid: content, ... }
    """
    fp_in = codecs.open(filename, encoding='utf-8')
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
            logging.error("Duplicated id %s in %s" % (fileid, filename))
            sys.exit(1)
        else:
            out_map[fileid] = content
    fp_in.close()
    return out_map


def load_scp_file_to_list(filename):
    """Load scp file to list
    Args:
        filename: string
    Return:
        out_list: [[fileid content], ... ]
    """
    fp_in = codecs.open(filename, encoding='utf-8')
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
            logging.error("Duplicated id %s in %s" % (fileid, filename))
            sys.exit(1)
        fileid_map[fileid] = 1
        out_list.append([fileid, content])
    fp_in.close()
    return out_list


def load_nbest_res_file_to_map(filename, nbest=1):
    '''load nbest_res_file to dict obj'''
    fp_in = codecs.open(filename, encoding='utf-8')
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


def output_wordboundary(word_boundary_list, out_file):
    '''output word boundary'''
    fp_out = codecs.open(out_file, 'w', encoding='utf-8')
    for line in word_boundary_list:
        fp_out.write(line)
    fp_out.close()


# pylint: disable=too-many-locals, too-many-branches
def output_result(align_info_list, ed_info_list, out_file, splited=False, lang='en'):
    '''output asr result'''
    if out_file != '':
        if splited:
            fp_out = codecs.open(out_file, 'a', encoding='utf-8')
        else:
            fp_out = codecs.open(out_file, 'w', encoding='utf-8')

    lang_show_map = {"en": 'W', "zh": 'C', 'zh-hant': 'C', "ja": 'C'}

    tot_ins_err = 0
    tot_del_err = 0
    tot_sub_err = 0
    tot_ref_word_num = 0
    tot_correct_word = 0
    # tot_ins_err_en = 0
    # tot_del_err_en = 0
    # tot_sub_err_en = 0
    # tot_ref_word_num_en = 0
    # tot_correct_word_en = 0
    tot_sen_err = 0
    tot_sen_num = len(ed_info_list)

    for item in zip(ed_info_list, align_info_list):
        curr_ins_err = int(item[0]['ins_err'])
        curr_del_err = int(item[0]['del_err'])
        curr_sub_err = int(item[0]['sub_err'])
        curr_correct = int(item[0]['correct'])
        curr_ref_word_num = int(item[0]['ref_word_num'])
        if curr_ref_word_num == 0:
            curr_correct_rate = 0.0
        else:
            curr_correct_rate = float(curr_correct) / curr_ref_word_num
        tot_ins_err += curr_ins_err
        tot_del_err += curr_del_err
        tot_sub_err += curr_sub_err
        tot_correct_word += curr_correct
        tot_ref_word_num += curr_ref_word_num
        curr_tot_err = curr_ins_err + curr_del_err + curr_sub_err
        if curr_ins_err + curr_del_err + curr_sub_err > 0:
            tot_sen_err += 1
        ref_str = item[1]['ref_str']
        res_str = item[1]['res_str']
        fileid = item[1]['fileid']

        if out_file != '' and not splited:
            fp_out.write("%s %s-REF\n" % (ref_str, fileid))
            fp_out.write("%s %s-RES\n" % (res_str, fileid))
            fp_out.write(
                "Words: %d Correct: %d Errors: %d Accuracy = %.2f%%\n"
                % (curr_ref_word_num, curr_correct, curr_tot_err, curr_correct_rate * 100)
            )
            fp_out.write(
                "Insertions: %d Deletions: %d Substitutions: %d\n"
                % (curr_ins_err, curr_del_err, curr_sub_err)
            )

    tot_word_err = tot_ins_err + tot_del_err + tot_sub_err

    if tot_ref_word_num == 0 or tot_sen_num == 0:
        if splited:
            warning_message = "No word or sentence found in splited mode."
        else:
            warning_message = "No word or sentence found."
        logging.info(warning_message)
        if out_file != '':
            fp_out.close()
        return

    word_err_rate = float(tot_word_err) / tot_ref_word_num
    sen_err_rate = float(tot_sen_err) / tot_sen_num
    word_info_str = "TOTAL Words: %d Correct: %d Errors: %d" % (
        tot_ref_word_num,
        tot_correct_word,
        tot_word_err,
    )
    detailed_err_str = "TOTAL Insertions: %d Deletions: %d Substitutions: %d" % (
        tot_ins_err,
        tot_del_err,
        tot_sub_err,
    )
    word_err_str = "%sER: %.2f%%, %sCR: %.2f%%" % (
        lang_show_map.get(lang, 'C'),
        word_err_rate * 100,
        lang_show_map.get(lang, 'C'),
        100 - word_err_rate * 100,
    )
    sen_err_str = "SER: %.2f%%, SCR: %.2f%%" % (sen_err_rate * 100, 100 - sen_err_rate * 100)

    if splited:
        word_info_str = "(Splitted) " + word_info_str
        detailed_err_str = "(Splitted) " + detailed_err_str
        word_err_str = "(Spliteed) " + word_err_str
        sen_err_str = "(Spliteed) " + sen_err_str

    if out_file == '':
        logging.info(word_info_str)
        logging.info(detailed_err_str)
        logging.info(word_err_str)
        logging.info(sen_err_str)
    else:
        fp_out.write("%s\n" % word_info_str)
        fp_out.write("%s\n" % detailed_err_str)
        fp_out.write("%s\n" % word_err_str)
        fp_out.write("%s\n" % sen_err_str)
        fp_out.close()


def output_json_result(json_content, out_file):
    '''output asr json detail result'''
    with open(out_file, 'w') as w:
        json.dump(json_content, w, indent=4, ensure_ascii=False)


def merge_json_result(
    world_size,
    test_name,
    falcon_report=False,
    in_file_name='rank{}_json_result_{}.json',
    out_file_name='json_result_{}.json',
):
    in_f_list = [open(in_file_name.format(rank, test_name)) for rank in range(world_size)]
    out_f = open(out_file_name.format(test_name), 'w')
    combiner = CalculatorCombiner('zh')  # fake lang
    merge_json = {}
    for f in in_f_list:
        merge_json.update(json.load(f))
        f.close()
    combiner = combiner.load_from_json(merge_json)
    json_content = combiner.to_dict()
    json.dump(json_content, out_f, indent=4, ensure_ascii=False)
    total_result = json_content['TOTAL']
    logging.info(f'TEST: {test_name}, TOTAL: {total_result}')
    out_f.close()


def output_nbest_result(align_info_list, ed_info_list, score_info_list, out_file, splited=False):
    '''output nbest result'''
    if out_file != '':
        if splited:
            fp_out = codecs.open(out_file, 'a', encoding='utf-8')
        else:
            fp_out = codecs.open(out_file, 'w', encoding='utf-8')

    for item in zip(ed_info_list, align_info_list, score_info_list):
        curr_ins_err = int(item[0]['ins_err'])
        curr_del_err = int(item[0]['del_err'])
        curr_sub_err = int(item[0]['sub_err'])
        curr_correct = int(item[0]['correct'])
        curr_ref_word_num = int(item[0]['ref_word_num'])
        if curr_ref_word_num == 0:
            curr_correct_rate = 0.0
        else:
            curr_correct_rate = float(curr_correct) / curr_ref_word_num
        curr_tot_err = curr_ins_err + curr_del_err + curr_sub_err
        ref_str = item[1]['ref_str']
        res_str = item[1]['res_str']
        fileid = item[1]['fileid'].replace(' ', '')
        scores = item[2][0]  # rnnt score and word confidence
        scores += ' ' + ' '.join(["%.6f" % score for score in item[2][1:]])

        if out_file != '' and not splited:
            fp_out.write("%s %s-REF\n" % (ref_str, fileid))
            fp_out.write("%s %s-RES\n" % (res_str, fileid))
            fp_out.write(
                "Words: %d Correct: %d Errors: %d Accuracy = %.2f%%\n"
                % (curr_ref_word_num, curr_correct, curr_tot_err, curr_correct_rate * 100)
            )
            fp_out.write("Scores: %s\n" % scores)
            fp_out.write(
                "Insertions: %d Deletions: %d Substitutions: %d\n"
                % (curr_ins_err, curr_del_err, curr_sub_err)
            )
    if out_file != '':
        fp_out.close()


# pylint: disable=no-else-continue
def merge_result(
    world_size,
    test_name,
    falcon_report=False,
    in_file_name='rank{}_cer_result_{}.txt',
    out_file_name='cer_result_{}.txt',
):
    '''merge asr result from different ranks'''
    in_f_list = [open(in_file_name.format(rank, test_name)) for rank in range(world_size)]
    out_f = open(out_file_name.format(test_name), 'w')
    total_words = 0
    total_correct = 0
    total_err = 0
    total_ins = 0
    total_del = 0
    total_sub = 0
    total_sen = 0
    total_correct_sen = 0
    for f in in_f_list:
        for line in f:
            if 'TOTAL Words:' in line:
                items = line.split(':')
                total_words += int(items[1].split()[0])
                total_correct += int(items[2].split()[0])
                total_err += int(items[3].split()[0])
                continue
            elif 'TOTAL Insertions:' in line:
                items = line.split(':')
                total_ins += int(items[1].split()[0])
                total_del += int(items[2].split()[0])
                total_sub += int(items[3].split()[0])
                continue
            elif 'Insertions: 0 Deletions: 0 Substitutions: 0' in line:
                total_correct_sen += 1
            elif '-REF' in line:
                total_sen += 1
            elif 'CER:' in line:
                continue
            elif 'SER:' in line:
                continue
            elif 'WER:' in line:
                continue
            out_f.write(line)

    cer = 0.0
    if total_words > 0:
        cer = total_err / float(total_words)
    scr = total_correct_sen / float(total_sen)
    out_f.write(
        "TOTAL Words: {} Correct: {} Errors: {}\n".format(total_words, total_correct, total_err)
    )
    out_f.write(
        "TOTAL Insertions: {} Deletions: {} Substitutions: {}\n".format(
            total_ins, total_del, total_sub
        )
    )
    out_f.write(
        "CER: %.2f%%, CCR: %.2f%%, SER: %.2f%%, SCR: %.2f%%\n"
        % (cer * 100, (1 - cer) * 100, (1 - scr) * 100, scr * 100)
    )
    logging.info(
        "TEST: %s, CER: %.2f%%, CCR: %.2f%%, SER: %.2f%%, SCR: %.2f%%"
        % (test_name, cer * 100, (1 - cer) * 100, (1 - scr) * 100, scr * 100)
    )
    logging.info(
        f'TOTAL_Words: {total_words}, Insertions: {total_ins}, '
        f'Deletions: {total_del}, Substitions: {total_sub}'
    )
    if falcon_report:
        falconclaw.log_report(
            {
                test_name: '%.4f' % (float(cer) * 100),
            }
        )
    out_f.close()


def merge_nbest_result(world_size, test_name):
    '''merge nbest result'''
    in_f_list = [
        open('rank{}_nbest_result_{}.txt'.format(rank, test_name)) for rank in range(world_size)
    ]
    out_f = open('nbest_result_{}.txt'.format(test_name), 'w')
    ed_info_list = []
    rnnt_score_list = []
    hotword_fst_score_list = []
    las_fw_score_list = []
    las_bw_score_list = []
    utt_id_set = set()
    for f in in_f_list:
        ed_info = []
        rnnt_score = []
        hotword_fst_score = []
        las_fw_score = []
        las_bw_score = []
        for i, line in enumerate(f):
            out_f.write(line)
            if i % 5 == 0:
                utt_id = line.split()[-1]
                if utt_id not in utt_id_set:
                    if len(ed_info) > 0:
                        ed_info_list.append(ed_info)
                        rnnt_score_list.append(rnnt_score)
                        hotword_fst_score_list.append(hotword_fst_score)
                        las_fw_score_list.append(las_fw_score)
                        las_bw_score_list.append(las_bw_score)
                    utt_id_set.add(utt_id)
                    ed_info = []
                    rnnt_score = []
                    hotword_fst_score = []
                    las_fw_score = []
                    las_bw_score = []
            elif i % 5 == 2:
                ed_info.append((int(line.split()[1]), int(line.split()[5])))
            elif i % 5 == 3:
                rnnt_score.append(line.split()[1])
                hotword_fst_score.append(line.split()[2])
                las_fw_score.append(float(line.split()[3]))
                if len(line.split()) > 4:
                    las_bw_score.append(float(line.split()[4]))
                else:
                    las_bw_score.append(0.0)
        # The last one should be appended for each rank
        ed_info_list.append(ed_info)
        rnnt_score_list.append(rnnt_score)
        hotword_fst_score_list.append(hotword_fst_score)
        las_fw_score_list.append(las_fw_score)
        las_bw_score_list.append(las_bw_score)
    out_f.close()
    return (
        ed_info_list,
        rnnt_score_list,
        hotword_fst_score_list,
        las_fw_score_list,
        las_bw_score_list,
    )


def get_cer_detail(ref_list, res_map, substitute_words_file='', nbest=1, lang="zh"):
    """Comput CER with text format
    Args:
        ref_file: string, one true label per one key
        res_file: string, nbest label for one key
        substitute_words_file: string, "噢 哦" -> "噢"
        nbest: int
    Return:
        align_info_list
        ed_info_list
    """
    assert nbest == 1
    ed_calculator = EditDistanceCalculator()
    formator = TextFormator(lang)
    words_substitute_map = None
    if substitute_words_file != '':
        words_substitute_map = load_substitute_map_file(substitute_words_file)
        ed_calculator.set_substitute_map(words_substitute_map)
    align_info_list = []
    ed_info_list = []

    for item in ref_list:
        fileid = item[0]
        ref = item[1]
        if fileid in res_map:
            res = res_map[fileid]
        else:
            res = ['']
        best_err_count = sys.maxsize
        best_align_info = None
        best_ed_info = None
        ref_format = formator(ref.strip())
        for res_item in res:
            res_format = formator(res_item.strip())
            ed_info, align_info = ed_calculator.show_alignment(fileid, ref_format, res_format)
            curr_ins_err = int(ed_info['ins_err'])
            curr_del_err = int(ed_info['del_err'])
            curr_sub_err = int(ed_info['sub_err'])
            curr_err_count = curr_ins_err + curr_del_err + curr_sub_err
            if curr_err_count < best_err_count:
                best_ed_info = ed_info
                best_align_info = align_info
                best_err_count = curr_err_count

        ed_info_list.append(best_ed_info)
        align_info_list.append(best_align_info)
    return align_info_list, ed_info_list


def sort_asr_metrics(asr_results_dir, testset_stats):
    '''collect asr results of multi checkpoints, sort them by cer'''
    if not asr_results_dir:
        raise RuntimeError(f'asr_results_dir does not exits: {asr_results_dir}')
    step_dir_list = [osp.join(asr_results_dir, d) for d in os.listdir(asr_results_dir)]
    res_dict = {}
    for step_dir in step_dir_list:
        cer_stats_files = [osp.join(step_dir, f) for f in testset_stats]
        testset_words_list = []
        cer_list = []
        average_cer = 0
        for cer_stats_file in cer_stats_files:
            with open(cer_stats_file, 'r', encoding='utf-8') as fp:
                for line in fp.readlines():
                    total_words = re.search(r'^TOTAL Words: ([0-9]+)', line)
                    if total_words:
                        total_words = int(total_words.group(1))
                        testset_words_list.append(total_words)
                    cer = re.search(r'^(CER|WER): ([0-9]+(\.[0-9]+)?)', line)
                    if cer:
                        cer = float(cer.group(2))
                        cer_list.append(cer / 100)
        total_words_sum = sum(testset_words_list)
        assert len(testset_words_list) == len(cer_list), 'something wrong in CER stats files'
        assert total_words_sum != 0
        average_cer = 0
        for words, cer in zip(testset_words_list, cer_list):
            average_cer += words * cer / total_words_sum
        step = osp.basename(step_dir)
        res_dict[step] = average_cer
    if len(res_dict) > 0:
        res_dict = dict(sorted(res_dict.items(), key=lambda x: x[1]))
        logging.info('Results sorted by cer: ')
        for k, v in res_dict.items():
            logging.info(f'{k}, cer: {v*100:.2f}%')
