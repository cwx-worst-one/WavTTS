"""stream test for rnnt"""
# pylint: disable=too-many-branches,no-member
import os
import time
import pickle

from streamz import Stream
from dataloader import FalconReader
from absl import logging

from core.solutions.inference.penguin.core.register import import_all_modules_for_register
from core.solutions.inference.penguin.core.factory import Factory
from core.solutions.inference.penguin.tasks.rnnt.tools.serializer import serializer
from core.solutions.inference.penguin.tasks.rnnt.tools.generate_config import PenguinConfigGenerator
from core.solutions.inference.penguin.core.processor.get_msg import GetMsg

from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    output_result,
    merge_result,
    TextFormator,
)
from core.utils import Config, get_dist_info, dist_barrier, get_rank, hdfs_mkdir, dist_hdfs_put
from core.dataset.preprocess.parser import ProtoParser, DecordRaw
from core.utils.misc import get_file_key, infer_text_format


def cal_cer(result, test_dict, dolphin_cfg):
    '''calculate CER'''
    rank = get_rank()
    edit_calculator = EditDistanceCalculator()
    default_filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@']
    filter_list = default_filter_list + dolphin_cfg.inference.get('filter_list', [])
    language = dolphin_cfg.inference.get('language', 'zh')
    formator = TextFormator(language)
    chunk_size = dolphin_cfg.data.get('chunk_size', 10)
    parrallel_chunk_num = dolphin_cfg.data.get('prefetch_chunk_num', 20)
    proto_parser = ProtoParser()
    decode_raw_tfrecord = DecordRaw(
        key2type={'frames': 'int16', 'transcript': 'bytes', 'uttid': 'bytes'}
    )

    ref = {}
    ed_info_dict = {}
    align_info_dict = {}
    for test_name, test_path in test_dict.items():
        reader = FalconReader(test_path, chunk_size)
        entry_num = len(reader.list_keys())
        chunks = [i * chunk_size for i in range(entry_num // chunk_size)]
        ref[test_name] = {}
        ed_info_dict[test_name] = []
        align_info_dict[test_name] = []
        for st in range(0, len(chunks), parrallel_chunk_num):
            raw_datas = reader.read_many(chunks[st : st + parrallel_chunk_num], True)
            for datas in raw_datas:
                for data in datas:
                    label = None
                    try:
                        # in most cases, parse TensorBundle data_item
                        item = pickle.loads(data)
                        label = item['label']
                        if isinstance(label, list):
                            label = ' '.join(label)
                    except Exception:
                        try:
                            # parse TFRecord data_item
                            item = proto_parser(data)
                            item = decode_raw_tfrecord(item)
                            label = item['transcript']
                        except Exception:
                            raise ValueError(
                                "Unsupported dataset, please use TensorBundle or TFRecord dataset"
                            )
                    if label:
                        ref[test_name][item['uttid']] = label

    for res in result:
        uttid, test_name = res[0].split('|')
        if uttid not in ref[test_name]:
            raise KeyError("No uttid found in the reference text")
        ref_format = infer_text_format(ref[test_name][uttid], filter_list, formator)
        res_format = infer_text_format(res[1].get("label_str", ""), filter_list, formator)
        ed_info, align_info = edit_calculator.show_alignment(uttid, ref_format, res_format)
        ed_info_dict[test_name].append(ed_info)
        align_info_dict[test_name].append(align_info)
    for test_name in test_dict.keys():
        output_result(
            align_info_dict[test_name],
            ed_info_dict[test_name],
            "rank{}_cer_result_{}.txt".format(rank, test_name),
            lang='zh',
        )
    dist_barrier()


def merge_cer(test_names, falcon_report):
    '''merge inference results'''
    rank, world_size = get_dist_info()
    if rank == 0:
        for test_name in test_names:
            merge_result(world_size, test_name, falcon_report)


def merge_output(output_path):
    '''merge inference results'''
    rank, world_size = get_dist_info()
    if rank == 0:
        filenames = [output_path + "_rank{}".format(i) for i in range(world_size)]
        with open(output_path, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)


def get_test_file_list(dataset_cfg, test_sets):
    '''get test file list.'''
    data_root = dataset_cfg.get('data_root', None)
    if isinstance(data_root, list):
        data_root = data_root[0]
    test_data_root = dataset_cfg.get('test_data_root', data_root)
    test_files = []
    for test_file in test_sets:
        test_files += [os.path.join(test_data_root, test_file)]
    return test_files


def save_asr_results(results, dolphin_cfg, test_dict):
    '''save asr result'''
    falcon_report = dolphin_cfg.get('falcon_report', False)
    config = dolphin_cfg.inference.penguin_config
    output_path = config.result_save_path
    if output_path.startswith("hdfs"):
        output_path = "./penguin.output"
    rank_output_path = output_path + "_rank{}".format(get_rank())
    with open(rank_output_path, 'w') as f:
        if config.output_streaming_stable_metric:
            streaming_stable_info_list = []
        for res in results:
            # wav_name, output, output_prob, prefetch_results, speed,
            # fixed_prefix_results, streaming_stable_info
            wav_name = res[0].split('|')[0]
            label_str = res[1].get("label_str")
            output_prob = res[1].get("output_prob", "")
            output_pronounce = res[1].get("output_pronounce", "")
            prefetch_results = res[1].get("prefetch_results", [])
            speed = res[1].get("speed", -1)
            fixed_prefix_results = res[1].get("fixed_prefix_results", [])
            streaming_stable_info = res[1].get("streaming_stable_info", ())
            f.write("Text: " + wav_name + ' ' + str(label_str) + '\n')
            if len(fixed_prefix_results) > 0:
                for result in fixed_prefix_results:
                    f.write(
                        "Fixed prefix: " + wav_name + ' ' + str(result[0]) + ' ' + result[1] + '\n'
                    )
            if output_prob != "":
                f.write("Confidence: " + wav_name + ' ' + output_prob + '\n')
            if output_pronounce != "":
                f.write("Pronounce: " + wav_name + ' ' + output_pronounce + '\n')
            if len(prefetch_results) > 0:
                for result in prefetch_results:
                    f.write("Prefetch: " + wav_name + ' ' + str(result[0]) + ' ' + result[1] + '\n')
            if speed >= 0:
                f.write("Speed: " + wav_name + ' ' + str(speed) + ' words/min\n')
            if config.output_streaming_stable_metric:
                upwr, upsr = streaming_stable_info[3], streaming_stable_info[4]
                f.write("{} UPWR: {:.2f} UPSR: {:.2f}\n".format(wav_name, upwr, upsr))
                streaming_stable_info_list.append(streaming_stable_info)
        if config.output_streaming_stable_metric:
            utt_num = len(streaming_stable_info_list)
            tot_unstable_word_num = sum([e[0] for e in streaming_stable_info_list])
            tot_unstable_seg_num = sum([e[1] for e in streaming_stable_info_list])
            tot_final_hyp_word_num = sum([e[2] for e in streaming_stable_info_list])
            tot_upwr = tot_unstable_word_num / tot_final_hyp_word_num
            tot_upsr = tot_unstable_seg_num / utt_num
            f.write("For this wav list, UPWR: {:.2f}, UPSR: {:.2f}\n".format(tot_upwr, tot_upsr))
    if config.cal_cer:
        cal_cer(results, test_dict, dolphin_cfg)
        merge_cer(test_dict.keys(), falcon_report)
        merge_output(output_path)
    if config.result_save_path.startswith('hdfs') and get_rank() == 0:
        hdfs_mkdir(config.result_save_path)
        os.system("hdfs dfs -put -f %s %s" % (output_path, config.result_save_path))
        cer_stats_files = []
        for test_name in test_dict.keys():
            cer_stats = "cer_result_{}.txt".format(test_name)
            if os.path.exists(cer_stats):
                cer_stats_files.append(cer_stats)
        if len(cer_stats_files) > 0:
            cer_stats_files = ' '.join(cer_stats_files)
            os.system("hdfs dfs -put -f %s %s" % (cer_stats_files, config.result_save_path))


class CacheResults:
    """Save output by every processor"""

    def __init__(self, cache, config):
        '''init'''
        self.cache = cache
        self.config = config

    def __call__(self, data):
        """
        Save the output by a processor.
        Args:
            data: A tuple with wav_name as the first element
        """
        wav_name = data[0]
        if wav_name not in self.cache.keys():
            self.cache[wav_name] = []
        row = self.cache[wav_name]
        if self.config.all_output_save_path is None:  # only save last output
            row = []
        row.append((data[1:]))
        self.cache[wav_name] = row

    def get_results(self):
        '''get results'''
        return [[wav_name] + list(self.cache[wav_name][-1]) for wav_name in self.cache.keys()]

    def get_all_output(self):
        """Get output with type of dict"""
        result = {}
        for key in self.cache.keys():
            result[key] = self.cache[key]
        return result


def asr_decoding(feature_out, processor_list, cache_result, config):
    '''asr_decoding'''
    serializer.set_file_path(config.serializer_path)
    if config.dual_channel:
        dual_channel(feature_out, cache_result, config)
        return
    pipline = Stream()
    tail = pipline
    for processor_name in processor_list:
        processor = Factory.get_processor(processor_name, config)
        tail = tail.map(processor)
        tail.sink(cache_result)
    tail.sink(print)
    pipline.emit(feature_out)
    serializer.finish()


def add_ch_suf(data, suf):
    """
    add channel suffix for wav
    "test.wav-ch1" means this is the recognition result of "channel 1"
    "test.wav-ch2" means this is the recognition result of "channel 2"
    """
    tmp_list = list(data)
    tmp_list[0] = tmp_list[0] + suf
    data = tuple(tmp_list)
    return data


def dual_channel(feature_out, cache_result, config):
    """
                     ↗ encoder     -> add_ch_suf(-ch1) ➘
    feature_input ->                                    -> decoder -> get_result
                     ➘ encoder_ch2 -> add_ch_suf(-ch2) ↗
    """
    pipline = Stream()
    feature_input = pipline.map(Factory.get_processor("feature_input", config))
    encoder = feature_input.map(Factory.get_processor("encoder", config)).map(
        add_ch_suf, suf="-ch1"
    )
    encoder_ch2 = feature_input.map(Factory.get_processor("encoder_channel2", config)).map(
        add_ch_suf, suf="-ch2"
    )
    decoder = encoder.union(encoder_ch2).map(Factory.get_processor("decoder", config))
    get_result = decoder.map(Factory.get_processor("get_result", config))
    get_result.sink(cache_result)
    get_result.sink(print)
    pipline.emit(feature_out)


def stream_test(dolphin_cfg):
    '''ASR test'''
    import_all_modules_for_register()
    penguin_cfg = dolphin_cfg.inference.penguin_config
    logging.set_verbosity(logging.INFO)
    start = time.time()
    input_generator = Factory.get_processor('input_generator', dolphin_cfg)
    processor_list = list(penguin_cfg.processor_list)

    inference_cfg = dolphin_cfg.get("inference", None)
    test_sets = inference_cfg.test_sets.strip().split('|')
    test_names = [get_file_key(test_file) for test_file in test_sets]
    test_file_path = get_test_file_list(dolphin_cfg.get('data', None), test_sets)
    test_dict = dict(zip(test_names, test_file_path))
    if processor_list[0] == "feature_input":
        stream_input = input_generator(test_dict)
    else:
        get_msg = GetMsg()
        message_deserialization = get_msg.get_msg_from_tos()
        index = processor_list.index("get_result")
        stream_input = [
            [key] + list(message_deserialization[key][2 - index])
            for key in message_deserialization.keys()
        ]

    cache_result = CacheResults({}, penguin_cfg)
    for x in stream_input:
        asr_decoding(x, processor_list, cache_result, penguin_cfg)
    save_asr_results(cache_result.get_results(), dolphin_cfg, test_dict)

    if penguin_cfg.all_output_save_path is not None:
        if penguin_cfg.all_output_save_path.endswith('.pkl'):
            message_filename = penguin_cfg.all_output_save_path
        else:
            message_filename = os.path.join(
                penguin_cfg.all_output_save_path, 'processor_output.pkl'
            )
        with open(message_filename, 'wb') as f:
            pickle.dump(cache_result.get_all_output(), f)
        logging.info("write output to %s" % message_filename)

    end = time.time()
    logging.info("Total time costs:%ss", end - start)


def generate_config(base_dir, dolphin_cfg):
    '''generate_xml_config
    Args: base_dir, current dir
          config, object of Config of dolphin config
    '''
    # generate penguin config
    dolphin_cfg.setdefault('penguin_config', Config())
    penguin_config = PenguinConfigGenerator(base_dir, dolphin_cfg).generate_config()
    dolphin_cfg['inference']['penguin_config'] = penguin_config
    local_onnx_dir = penguin_config.local_onnx_dir
    remote_onnx_dir = penguin_config.remote_onnx_dir
    # generate xml config
    if penguin_config.xml_generate_template != '':
        path = os.path.dirname(os.path.realpath(__file__))
        cmd = 'bash {} {} {} {} {} {}'.format(
            os.path.join(path, 'tools/xml_auto_gen.sh'),
            penguin_config.local_onnx_dir,
            os.path.join(penguin_config.local_onnx_dir, 'xml'),
            penguin_config.xml_generate_template,
            penguin_config.xml_generate_script_version,
            penguin_config.xml_generate_copy_resource_path,
        )
        print(cmd)
        assert os.system(cmd) == 0
    # upload config to hdfs
    if local_onnx_dir and remote_onnx_dir:
        dist_hdfs_put(local_onnx_dir, remote_onnx_dir, sync=False)
    return dolphin_cfg


def run_penguin(base_dir, dolphin_cfg):
    '''run penguin
    Args: base_dir, current dir
          config, object of Config of dolphin config
    '''
    dolphin_cfg = generate_config(base_dir, dolphin_cfg)
    stream_test(dolphin_cfg)
