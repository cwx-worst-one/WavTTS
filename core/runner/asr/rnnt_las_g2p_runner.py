''' RNNTLASRescoreRunner '''
import time
import torch
from core.dataset import ValidHDFSDataset
from core.utils import dist_barrier, logging
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
    output_result,
    merge_result,
)
from core.utils.misc import infer_text_format
from ..base_runner import RUNNERS
from .rnnt_twopass_runner import RNNTLASRescoreRunner


@RUNNERS.register_module()
class RNNTLASG2PRunner(RNNTLASRescoreRunner):
    '''RNNT for ASR'''

    # pylint: disable=line-too-long

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)

        self.phone_dict = self.meta_data['phone_dict']
        self.solution_cfg.setdefault('phone_dict', self.phone_dict)
        self.setup_transform_cfg(dataset_cfg)

    def inference_once(self, test_file, test_name, language, inference_cfg):
        '''inference a test set'''
        # pylint:disable=too-many-locals,
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
        nbest = inference_cfg.get('nbest', 1)
        filter_list = inference_cfg.get('filter_list', [])
        keep_non_proun_tokens = inference_cfg.get('keep_non_proun_tokens', None)
        ed_calculator = EditDistanceCalculator()
        formator = TextFormator(language, keep_non_proun_tokens)
        phone_formator = TextFormator('en')
        # '@@ ' for a continuous word
        # '@@' for last word
        default_filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@']
        filter_list = default_filter_list + filter_list

        if self.reorder_tgt_dict:
            tgt_dict = self.reorder_tgt_dict
        else:
            tgt_dict = self.tgt_dict

        # process every batch
        decode_count = 0
        ed_info_list = []
        align_info_list = []
        pronounce_ed_info_list = []
        pronounce_align_info_list = []
        time_elapsed = 0
        utt_count = 0
        while batch_data is not None:
            st = time.time()
            if beam_size > 0:
                inf_res, _, _ = self.solution.beam_inference(batch_data, nbest=nbest)
            else:
                inf_res = self.solution.greedy_inference(batch_data)
            time_elapsed += time.time() - st
            utt_count += len(batch_data['uttid'])

            # g2p
            char, char_mask = self.prepare_g2p_inputs(inf_res)
            batch_data['char'] = char
            batch_data['char_mask'] = char_mask

            self.solution.prepare_rnnt_encoder_out(batch_data)
            pronounces = self.solution.g2p_greedy_inference(batch_data)

            # post process
            for bid, hyp in enumerate(inf_res):
                # every sentence
                decode_count += 1
                if decode_count % 50 == 0:
                    logging.info('rank %d decode %d sentence', self.rank, decode_count)
                ref_format = infer_text_format(batch_data['ref'][bid], filter_list, formator)
                pronounce_ref_format = infer_text_format(
                    batch_data['phone_ref'][bid], filter_list, phone_formator
                )

                hyp_str = tgt_dict.string(hyp)
                res_format = infer_text_format(hyp_str, filter_list, formator)

                pronounce_hyp = pronounces[bid]
                pronounce_str = self.get_pronounce_text(self.phone_dict, pronounce_hyp)
                pronounce_res_format = self.align_pronounce_text(
                    hyp_str, pronounce_str, filter_list
                )

                logging.info(
                    'rank %d, uttid %s, %s, %s',
                    self.rank,
                    batch_data['uttid'][bid],
                    ' '.join(res_format),
                    ' '.join(pronounce_res_format),
                )
                ed_info, align_info = ed_calculator.show_alignment(
                    batch_data['uttid'][bid], ref_format, res_format
                )
                pronounce_ed_info, pronounce_align_info = ed_calculator.show_alignment(
                    batch_data['uttid'][bid], pronounce_ref_format, pronounce_res_format
                )

                ed_info_list.append(ed_info)
                align_info_list.append(align_info)
                pronounce_ed_info_list.append(pronounce_ed_info)
                pronounce_align_info_list.append(pronounce_align_info)

            # next batch
            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang=language)
        out_pronounce_stat_file = 'rank{}_cer_result_{}_pronounce.txt'.format(self.rank, test_name)
        output_result(
            pronounce_align_info_list,
            pronounce_ed_info_list,
            out_pronounce_stat_file,
            lang=language,
        )

        # wait all rank finish
        dist_barrier()
        # merge the CER
        if self.rank == 0:
            merge_result(self.world_size, test_name, falcon_report=self.falcon_report)
            merge_result(
                self.world_size,
                test_name,
                falcon_report=self.falcon_report,
                in_file_name='rank{}_cer_result_{}_pronounce.txt',
                out_file_name='cer_result_{}_pronounce.txt',
            )

        dist_barrier()
        logging.info(
            'rank %d, testset %s, total num of utt: %d, total infer forward time: %.3f s',
            self.rank,
            test_file,
            utt_count,
            time_elapsed,
        )

    @staticmethod
    def prepare_g2p_inputs(hyps):
        '''prepare g2p inputs'''
        bsz = len(hyps)
        max_char_length = max(len(hyp) for hyp in hyps)
        tensor_char = torch.zeros(bsz, max_char_length, dtype=torch.int64).cuda()
        tensor_char_mask = torch.zeros(bsz, max_char_length, dtype=torch.float32).cuda()

        for i in range(bsz):
            t_char = torch.tensor(hyps[i]).long().cuda()
            char_length = len(hyps[i])
            tensor_char[i, 0:char_length] = t_char
            tensor_char_mask[i, 0:char_length] = 1
        return tensor_char, tensor_char_mask

    @staticmethod
    def get_pronounce_text(phone_dict, hyp):
        '''get pronounce text'''
        pronounce = [phone_dict[i] for i in hyp]
        return ' '.join(pronounce)

    @staticmethod
    def align_pronounce_text(chars, pronounces, filter_list):
        '''align pronounce text to asr result text'''
        char_parts = chars.split(' ')
        pronounce_parts = pronounces.split(' ')
        new_pronounces = []
        for idx, val in enumerate(char_parts):
            is_valid = True
            for filter_tag in filter_list:
                if filter_tag in val:
                    is_valid = False
                    break
            if is_valid:
                new_pronounces.append(pronounce_parts[idx])
        return new_pronounces
