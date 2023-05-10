'''writer for rnnt rich information output'''
import json
import logging


def merge_asr_rich_info(world_size, testset_name, input_file_tmp, output_file):
    """
    merge json from multiple sub jsons
    """
    input_file_list = [input_file_tmp.format(rank, testset_name) for rank in range(world_size)]
    output_dict = dict()
    output_dict['utterances'] = dict()
    for input_file in input_file_list:
        with open(input_file, encoding='utf-8') as fd:
            sub_dict = json.load(fd)
        output_dict['utterances'].update(sub_dict['utterances'])
        if len(input_file_list) == 1:
            output_dict['general'] = sub_dict['general']
    if len(input_file_list) > 1:
        logging.warning("to write general rich info, use 1 gpu mode!")
    with open(output_file, 'w', encoding='utf-8') as fd_out:
        fd_out.write(json.dumps(output_dict, ensure_ascii=False, indent=True))


class ASRModelRichOutWriter:
    """
    write model's rich info into structured output
    json structure example:
    {
    'general' : {
                'upwr' : float,
                'upsr' : float,
                'first_token_latency' : float,
                'last_token_latency' : float,
                'avg_token_latency' : float,
                'p50_token_latency' : float,
                'p90_token_latency' : float,
                }
    'utterances' : {
                   uttid_0(str) : {
                           'ref_format'   : list, # or 'ref_format_spk1'  : list
                           'infer_format' : list, # or 'infer_format_spk1': list
                           'infer_label'  : list, # or 'infer_label_spk1' : list
                           'timestamp'    : list, # or 'timestamp_spk1'   : list
                           'wordboundary' : list,
                           'confidence'   : list,
                           'speed'        : float,
                           'upwr'         : float,
                           'upsr'         : float
                           },
                   uttid_1(str) : { ... },
                }
    }
    more detail please refer to https://bytedance.feishu.cn/docs/doccnBn0pDnc4KBbylQQzgG5Tte
    """

    def __init__(self, args, dual_ch_mode=False):
        '''init.'''
        inference_cfg = args.inference
        # rich info to output
        self.output_timestamp = inference_cfg.get('output_timestamp', False)
        self.output_wordboundary_type = inference_cfg.get('output_wordboundary_type', None)
        self.output_rnnt_confidence = inference_cfg.get('output_rnnt_confidence', False)
        self.output_speed = inference_cfg.get('output_speed', False)
        self.output_streaming_stable_metric = inference_cfg.get(
            'output_streaming_stable_metric', False
        )
        self.prefetch = inference_cfg.get('enable_prefetch', False)
        self.fixed_prefix = inference_cfg.get('fixed_prefix_beam_search', False)
        self.output_latency_metric = inference_cfg.get('output_latency_metric', False)
        if self.output_latency_metric:
            self.output_timestamp = True
        if self.output_timestamp:
            assert args.get('runner', '') in [
                'RNNTRunner',
                'DualChannelRunner',
                'UniversalRNNTRunner',
            ]
        if self.output_rnnt_confidence or self.output_streaming_stable_metric:
            assert args.get('runner', '') in ['RNNTRunner', 'RNNTLASRescoreRunner']
        if (
            self.output_speed
            or self.output_wordboundary_type
            or self.fixed_prefix
            or self.prefetch
            or self.output_latency_metric
        ):
            assert args.get('runner', '') in ['RNNTRunner', 'RNNTLASRescoreRunner']

        self.rich_info = dict()
        self.rich_info['utterances'] = dict()
        self.rich_info['general'] = dict()
        self.base_info_to_collect = set()
        self.rich_info_to_collect = set()
        if dual_ch_mode:
            self.init_keys_to_collect(suffix="_spk1")
            self.init_keys_to_collect(suffix="_spk2")
        else:
            self.init_keys_to_collect()

    def init_keys_to_collect(self, suffix=""):
        """initialization for keys to be collected"""
        self.base_info_to_collect.add('ref_format' + suffix)
        self.base_info_to_collect.add('infer_format' + suffix)
        self.base_info_to_collect.add('infer_label' + suffix)

        if self.output_timestamp:
            self.rich_info_to_collect.add("timestamp" + suffix)
        if self.output_wordboundary_type:
            self.rich_info_to_collect.add("wordboundary" + suffix)
        if self.output_rnnt_confidence:
            self.rich_info_to_collect.add("confidence" + suffix)
        if self.output_speed:
            self.rich_info_to_collect.add("speed" + suffix)
        if self.output_streaming_stable_metric:
            self.rich_info_to_collect.add("upwr" + suffix)
            self.rich_info_to_collect.add("upsr" + suffix)
        if self.prefetch:
            self.rich_info_to_collect.add("prefetch" + suffix)
        if self.fixed_prefix:
            self.rich_info_to_collect.add("fixed_prefix" + suffix)

    def update_one_sample_info(self, uttid, info_key, info_val):
        """update one sample's info"""
        assert info_key in self.rich_info_to_collect or info_key in self.base_info_to_collect
        assert isinstance(uttid, str)
        if uttid not in self.rich_info['utterances']:
            self.rich_info['utterances'][uttid] = dict()
        self.rich_info['utterances'][uttid][info_key] = info_val

    def update_general_info(self, info_key, info_val):
        """update general info"""
        self.rich_info['general'][info_key] = info_val

    def get_one_sample_info(self, uttid, info_key=None):
        """get one sample's info"""
        if info_key:
            return self.rich_info['utterances'][uttid][info_key]
        return self.rich_info['utterances'][uttid]

    def if_necessary_to_write(self):
        """for output"""
        return len(self.rich_info_to_collect) > 0

    def write_to(self, file_path):
        """write to disk as json"""
        with open(file_path, 'w', encoding='utf-8') as fd_out:
            fd_out.write(json.dumps(self.rich_info, ensure_ascii=False, indent=True))
