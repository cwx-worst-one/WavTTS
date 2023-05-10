''' DualChannelRunner '''

import torch
from core.dataset import ValidHDFSDataset
from core.utils import dist_barrier, logging
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
    output_result,
    merge_result,
)
from core.utils.metric_output import (
    ASRModelRichOutWriter,
    merge_asr_rich_info,
)
from core.utils.misc import infer_text_format
from ..base_runner import RUNNERS
from .rnnt_runner import RNNTRunner


@RUNNERS.register_module()
class DualChannelRunner(RNNTRunner):
    '''2-ch RNNT for TAU'''

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transform config.'''
        for cfg in dataset_cfg.batch_transform:
            if cfg.type == 'PreCharCollate':
                cfg['args'] = self.solution_cfg

    @staticmethod
    def filter_label(tt, hh):
        '''filter label whose id < 4'''
        tt_new = []
        hh_new = []
        for t, h in zip(tt, hh):
            if h < 4:
                continue
            tt_new.append(t)
            hh_new.append(h)
        return tt_new, hh_new

    def post_process(
        self,
        hyp,
        uttid,
        ref_str,
        ed_info_list,
        align_info_list,
        ed_calculator,
        filter_list,
        formator,
        rich_info_writer,
        spk='1',
        output_timestamp=False,
    ):
        '''post process'''
        time = None
        if output_timestamp:
            time_t = [str(t) for t in hyp[1]]
            hyp_t = hyp[0]
            time, hyp = self.filter_label(time_t, hyp_t)
            rich_info_writer.update_one_sample_info(uttid, 'timestamp_spk%s' % spk, time)
        rich_info_writer.update_one_sample_info(uttid, 'infer_label_spk%s' % spk, hyp)
        if self.reorder_tgt_dict is not None:
            hyp_str = self.reorder_tgt_dict.string(hyp)
        else:
            hyp_str = self.tgt_dict.string(hyp)
        ref_format = infer_text_format(ref_str, filter_list, formator)
        rich_info_writer.update_one_sample_info(uttid, 'ref_format_spk%s' % spk, ref_format)
        res_format = infer_text_format(hyp_str, filter_list, formator)
        rich_info_writer.update_one_sample_info(uttid, 'infer_format_spk%s' % spk, res_format)
        logging.info('rank %d, uttid %s, spk%s: %s', self.rank, uttid, spk, hyp_str)
        if output_timestamp:
            logging.info('rank %d, uttid %s, time%s: %s', self.rank, uttid, spk, ' '.join(time))
        ed_info, align_info = ed_calculator.show_alignment(
            'spk%s_' % (spk) + uttid, ref_format, res_format
        )
        ed_info_list.append(ed_info)
        align_info_list.append(align_info)

    @torch.no_grad()
    def inference_once(self, test_file, test_name, language, inference_cfg):
        '''inference a test set'''
        # data set
        if hasattr(inference_cfg, 'bucket_schedule'):
            inf_bucket_schedule = inference_cfg.bucket_schedule
        else:
            inf_bucket_schedule = self.dataset_cfg.get('bucket_schedule', '') + ',100000'
        test_data_loader = ValidHDFSDataset(
            [test_file],
            inf_bucket_schedule,
            self.dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn_inference,
            split_path_list_by_rank=False,
        )
        test_data_loader.reset()
        batch_data = test_data_loader.next()
        # inference config
        beam_size = inference_cfg.get('beam_size', 0)
        output_timestamp = inference_cfg.get('output_timestamp', False)
        rich_info_writer = ASRModelRichOutWriter(self.args, dual_ch_mode=True)
        ed_calculator = EditDistanceCalculator()
        formator = TextFormator(language)
        filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>']
        # process every batch
        decode_count = 0
        ed_info_list = []
        align_info_list = []
        while batch_data is not None:
            if beam_size > 0:
                inf_res1, inf_res2 = self.solution.beam_inference(
                    batch_data, output_timestamp=output_timestamp
                )
            else:
                inf_res1, inf_res2 = self.solution.greedy_inference(batch_data)
            # post process
            for bid, (hyp1, hyp2) in enumerate(zip(inf_res1, inf_res2)):
                # every sentence
                decode_count += 1
                if decode_count % 50 == 0:
                    logging.info('rank %d decode %d sentence', self.rank, decode_count)
                self.post_process(
                    hyp1,
                    batch_data['uttid'][bid],
                    batch_data['ref1'][bid],
                    ed_info_list,
                    align_info_list,
                    ed_calculator,
                    filter_list,
                    formator,
                    rich_info_writer,
                    spk='1',
                    output_timestamp=output_timestamp,
                )
                self.post_process(
                    hyp2,
                    batch_data['uttid'][bid],
                    batch_data['ref2'][bid],
                    ed_info_list,
                    align_info_list,
                    ed_calculator,
                    filter_list,
                    formator,
                    rich_info_writer,
                    spk='2',
                    output_timestamp=output_timestamp,
                )
            # next batch
            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang='zh')
        if rich_info_writer.if_necessary_to_write():
            rich_info_writer.write_to("rank{}_rich_info_{}.json".format(self.rank, test_name))
        # wait all rank finish
        dist_barrier()
        # merge the CER
        if self.rank == 0:
            merge_result(self.world_size, test_name, falcon_report=self.falcon_report)
            if rich_info_writer.if_necessary_to_write():
                merge_asr_rich_info(
                    self.world_size,
                    test_name,
                    input_file_tmp="rank{}_rich_info_{}.json",
                    output_file="rich_info_{}.json".format(test_name),
                )
        dist_barrier()
