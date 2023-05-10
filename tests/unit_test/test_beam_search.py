'''
test beam search.
'''
import os
import torch
from core.dataset import ValidHDFSDataset, build_item_augmentation, build_draw_batch_fn, get_meta
from core.utils import Config, logging
from core.solutions import setup_solution
from core.utils.cer.cer_metric import TextFormator
from core.utils.metric_output import StreamingStableMetricOneSample


def _generate_beam_search_input(bsz, t):
    '''generate input for test.'''
    torch.manual_seed(20220712)
    torch.cuda.manual_seed(184440)
    return dict(
        src=torch.rand([bsz, t, 80]).float().cuda(),
        src_mask=(torch.rand([bsz, t]) < 0.5).float().cuda(),
        backbone_mask=(torch.rand([bsz, 504]).float().cuda()),
    )


def _build_beam_search_dataset(cfg):
    '''build beam search dataset.'''
    # pylint:disable=too-many-branches
    # base_runner pre_build_dataset
    if cfg.data.get('meta_file'):
        meta_data_root = cfg.data.get("meta_data_root", cfg.data.get("data_root", None))
        if isinstance(meta_data_root, (list, tuple)):
            meta_data_root = meta_data_root[0]
        meta_file = os.path.join(meta_data_root, cfg.data.meta_file)
        meta_data = get_meta(meta_file)
    else:
        meta_data = {}
    cfg.solution.setdefault('fbank_dim', cfg.data.get("fbank_dim"))
    cfg.solution.setdefault('use_eos', cfg.data.get("use_eos"))
    if meta_data:
        cfg.solution.setdefault('tgt_dict', meta_data.get('tgt_dict'))
    cfg.solution.setdefault('iters_per_epoch', cfg.train.get("iters_per_epoch"))

    # base_asr_runner pre_build_dataset
    tgt_dict = meta_data.get('tgt_dict')
    is_reorder_dict = cfg.solution.get('reorder_dict_by_freq', 1)
    if is_reorder_dict:
        reorder_tgt_dict = meta_data.get('reorder_tgt_dict')
    else:
        reorder_tgt_dict = None
    if tgt_dict:
        tgt_vocab_size = len(tgt_dict)
        # align to 8
        tgt_vocab_size = ((tgt_vocab_size + 8 - 1) // 8) * 8
        cfg.solution.setdefault('tgt_vocab_size', tgt_vocab_size)
    cfg.solution.setdefault('reorder_tgt_dict', reorder_tgt_dict)
    in_out_ratio = cfg.data.get('in_out_ratio', 4)
    if in_out_ratio < cfg.solution.get('downsampling_size', 4):
        in_out_ratio = cfg.solution.get('downsampling_size', 4)
    for transform in cfg.data.batch_transform:
        if transform.type == 'PreCharCollate':
            transform['args'] = cfg.solution
    chunk_size = cfg.data.get('chunk_size', 40)
    if chunk_size > 100:
        cfg.data.chunk_size = 40

    # build_dataset
    data_root = cfg.data.get("data_root", None)
    valid_data_root = cfg.data.get("valid_data_root", data_root)
    valid_file_list = cfg.data.get("valid_file_list", None)
    if isinstance(valid_data_root, str):
        valid_data_root = [valid_data_root]
        valid_file_list = [valid_file_list]
    _valid_file_list = []
    assert len(valid_data_root) == len(valid_file_list)
    for valid_root, valid_file in zip(valid_data_root, valid_file_list):
        valid_dataset_now = [os.path.join(valid_root, p) for p in eval(valid_file)]
        _valid_file_list += valid_dataset_now

    if hasattr(cfg.data, 'bucket_schedule_val'):
        val_bucket_schedule = cfg.data.bucket_schedule_val
    else:
        val_bucket_schedule = cfg.data.bucket_schedule
    parse_fn_eval = build_item_augmentation(cfg.data.valid_item_transform, meta_data)
    draw_batch_fn = build_draw_batch_fn(cfg.data.batch_transform, meta_data)

    valid_split_each_dataset = cfg.solution.get('valid_multi_cer', False)
    valid_data_loader = ValidHDFSDataset(
        _valid_file_list,
        val_bucket_schedule,
        cfg.data,
        parse_fn_eval,
        draw_batch_fn,
        split_path_list_by_rank=False,
        split_each_dataset=valid_split_each_dataset,
    )
    return valid_data_loader


def _run_beam_search(cfg_file, greedy=False, use_batch=True):
    '''run beam search.'''
    # pylint:disable=too-many-branches
    cfg = Config.fromfile(cfg_file)
    data_loader = _build_beam_search_dataset(cfg)
    data_loader.reset()
    if greedy:
        cfg.inference.setdefault('beam_size', 0)
        print('Running greedy beam search test')
    else:
        cfg.inference.setdefault('use_batch_beam', use_batch)
        print('Running ' + 'non-' if not use_batch else '' + 'batch beam search test')

    cfg.solution.setdefault('fbank_dim', 80)
    tgt_vocab_size = (
        cfg.solution.adaptive_head_size
        + cfg.solution.adaptive_tail_size * cfg.solution.adaptive_tail_groups
    )
    cfg.solution.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution = setup_solution(cfg.solution)
    solution.cuda()
    solution.init_beam_search(cfg.inference, lm_solution=None)
    if 'twopass_infer_cfg' in cfg.inference:
        twopass_infer_cfg = cfg.inference.get('twopass_infer_cfg', dict())
        nbest = twopass_infer_cfg.get('rnnt_beam_size', 10)
        cfg.inference.beam_size = nbest
    else:
        twopass_infer_cfg = None
        nbest = cfg.inference.get('nbest', 1)
    filter_list = cfg.inference.get('filter_list', [])
    language = cfg.inference.get('language', 'zh')
    keep_non_proun_tokens = cfg.inference.get('keep_non_proun_tokens', None)
    output_timestamp = cfg.inference.get('output_timestamp', False)
    output_rnnt_confidence = cfg.inference.get('output_rnnt_confidence', False)
    output_streaming_stable_metric = cfg.inference.get('output_streaming_stable_metric', False)
    prefetch = cfg.inference.get('enable_prefetch', False)
    fixed_prefix = cfg.inference.get('fixed_prefix_beam_search', False)
    enable_endpoint = cfg.inference.get('enable_endpoint', False)
    output_latency_metric = cfg.inference.get('output_latency_metric', False)
    if output_latency_metric:
        output_timestamp = True
    if output_streaming_stable_metric:
        ms_per_packet = cfg.inference.get('ms_per_packet', 200)
        ms_per_frame = cfg.solution.downsampling_size * 10
    if output_timestamp or output_rnnt_confidence:
        if nbest > 1:
            logging.warning("nbest not support > 1 for output timestamp or confidence")
            nbest = 1
    formator = TextFormator(language, keep_non_proun_tokens)
    # '@@ ' for a continuous word
    # '@@' for last word
    default_filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@']
    filter_list = default_filter_list + filter_list
    if cfg.solution.get('reorder_dict_by_freq', 1):
        tgt_dict = cfg.solution.get('reorder_tgt_dict')

    solution.init_beam_search(cfg.inference, lm_solution=None)

    if greedy:
        for i in range(1, 4):
            torch.cuda.nvtx.range_push('inference {}'.format(i))
            solution.greedy_inference(data_loader.next())
            torch.cuda.nvtx.range_pop()
    else:
        for i in range(1, 4):
            torch.cuda.nvtx.range_push('inference {}'.format(i))
            stable_metric_batch_list = None
            batch_data = data_loader.next()
            if output_streaming_stable_metric:
                bsz = batch_data['src'].shape[0]
                stable_metric_batch_list = [
                    StreamingStableMetricOneSample(
                        ms_per_packet, ms_per_frame, tgt_dict, filter_list, formator
                    )
                    for _ in range(bsz)
                ]
            solution.beam_inference(
                batch_data,
                prefetch=prefetch,
                fixed_prefix=fixed_prefix,
                nbest=nbest,
                output_timestamp=output_timestamp,
                output_rnnt_confidence=output_rnnt_confidence,
                endpoint=enable_endpoint,
                stable_metric_list=stable_metric_batch_list,
                twopass_infer_cfg=twopass_infer_cfg,
            )
            torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    CFG_FILE_NAME = 'conformer_rnnt_offline_ce_align_en.py'
    torch.cuda.nvtx.range_push('greedy')
    _run_beam_search(CFG_FILE_NAME, greedy=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push('batch')
    _run_beam_search(CFG_FILE_NAME, greedy=False, use_batch=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push('non-batch')
    _run_beam_search(CFG_FILE_NAME, greedy=False, use_batch=False)
    torch.cuda.nvtx.range_pop()
