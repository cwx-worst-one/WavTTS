'''begin batch test'''
import os
import sys
import getopt
import math
import torch
from core.runner import RUNNERS
from core.utils import Config, get_world_size
from core.extensions import clip_grads_
from core.runner import OPTIMIZERS

STANDARD_LIST = {
    'configs/asr/child_zh_time_reduce_online_prosody_baseline.py': {
        'loss': float(os.getenv('CHILD_ZH_LOSS', '91.49190521240234')),
        'cer': float(os.getenv('CHILD_ZH_CER', '25.0')),
        'eps': 1e-8,
        'uttid': [
            '21537980011',
            '21537980121',
            '21537980291',
        ],
    },
    'configs/asr/dfsmn_rnnt.py': {
        'loss': float(os.getenv('DFSMN_RNNT_LOSS', '69.80582427978516')),
        'cer': float(os.getenv('DFSMN_RNNT_CER', '45.0')),
        'eps': 1e-8,
        'uttid': [
            '00092392-646a0121-1bca-4f36-b2de-6108c8456351',
            'toscebe42d01968473fae9f015c8d916db2-out__7',
        ],
    },
    'configs/asr/librispeech_rnnt_lstmp.py': {
        'loss': float(os.getenv('LIBRISPEECH_RNNT_LSTMP_LOSS', '1218.9129638671875')),
        'cer': float(os.getenv('LIBRISPEECH_RNNT_LSTMP_CER', '49.0')),
        'eps': 1e-8,
        'uttid': ['8097-108005-0041'],
    },
    'configs/asr/librispeech_rnnt_hmm_free.py': {
        'loss': float(os.getenv('LIBRISPEECH_RNNT_HMM_FREE_LOSS', '6.491137504577637')),
        'cer': float(os.getenv('LIBRISPEECH_RNNT_HMM_FREE_CER', '0.0')),
        'eps': 1e-8,
        'uttid': ['7683-103530-0066'],
    },
    'configs/asr/librispeech_rnnt.py': {
        'loss': float(os.getenv('LIBRISPEECH_RNNT_LOSS', '16.631465911865234')),
        'cer': float(os.getenv('LIBRISPEECH_RNNT_CER', '60.0')),
        'eps': 1e-8,
        'uttid': ['7683-103530-0066'],
    },
    'configs/asr/tel_rnnt_transfomer_80kh.py': {
        'loss': float(os.getenv('TEL_RNNT_TRANSFOMER_80KH_LOSS', '10.300310134887695')),
        'cer': float(os.getenv('TEL_RNNT_TRANSFOMER_80KH_CER', '59.0')),
        'eps': 1e-8,
        'uttid': [
            'tos481050d6779d4bbda6921d8bf0dfa123-in__69_2',
            'tos486613c7b59e436cad1a93345d08e5fa-out__31',
        ],
    },
}


def check(config, loss, cer, uttid):
    '''check your output with std output'''
    stdout = STANDARD_LIST[config]
    eps = stdout['eps']
    stdutt = stdout['uttid']

    # check uttid
    assert len(stdutt) == len(uttid)
    for i, val in enumerate(uttid):
        assert stdutt[i] == val

    failed = False
    # check loss
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    if math.fabs(loss - stdout['loss']) > eps:
        print(
            "Error!\n In config file: {}.Your loss is {}, std loss is {}.\n".format(
                config, loss, stdout['loss']
            )
        )
        failed = True

    # check cer
    if isinstance(cer, torch.Tensor):
        cer = cer.item()
    if math.fabs(cer - stdout['cer']) > eps:
        print(
            "Error!\n In config file: {}.Your cer is {}, std cer is {}.\n".format(
                config, cer, stdout['cer']
            )
        )
        failed = True

    if failed:
        raise RuntimeError("Cer or Loss accuracy error is too large!")


def clip_grads(params, runner):
    '''clip_grads.'''
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    grad_norm = clip_grads_(
        params, runner.opt_util_cfg.max_grad_clip, **runner.opt_util_cfg.grad_clip
    )
    return grad_norm


def batch_test(args):
    '''dolphin single_batch_test'''
    _opt, config_file = getopt.getopt(args, 'c:', ['config'])
    config_file = config_file[0]
    cfg = Config.fromfile(config_file)
    cfg.data.deterministic = True
    cfg.train.deterministic = True
    cfg.data.max_batch_size = 1000
    cfg.data.max_batch_scale = 0
    cfg.solution.cer_update_freq = 10
    cfg.train.calc_flops = False

    # get runner
    runner_cls = RUNNERS.get(cfg.runner)
    runner = runner_cls(cfg)

    # load batch data
    batch_data = runner.train_data_loader.next()
    if runner.train_data_loader is not None:
        runner.train_data_loader.terminate()
    if runner.valid_data_loader is not None:
        runner.valid_data_loader.terminate()

    if runner.opt_util_cfg.bmuf_config and get_world_size() > 1:
        bmuf_type = runner.opt_util_cfg.bmuf_config.pop('type', 'FusedBMUF')
        bmuf = OPTIMIZERS.get(bmuf_type)(runner.opt_util_cfg.bmuf_config)
    else:
        bmuf = None

    if bmuf:
        bmuf.before_run(runner.solution, runner.optimizer, runner.iter, runner.train_cfg)

    # enter train loop
    runner.call_hook('before_epoch')
    for _ in range(30):
        runner.call_hook('before_train_iter')
        grad_accum_step = runner.train_cfg.get('grad_accum_step', 1)
        for i in range(grad_accum_step):
            runner.solution.train()
            solution_out = runner.solution(batch_data)
            runner.loss = solution_out['backward_loss']
            runner.dist_handler.backward(runner.loss, unscale=(i + 1 == grad_accum_step))
        if runner.opt_util_cfg.grad_clip:
            gnorm = runner.dist_handler.clip_grad_norm(
                max_grad_clip=runner.opt_util_cfg.max_grad_clip, **runner.opt_util_cfg.grad_clip
            )
            runner.train_log_buffer.update({'gnorm': gnorm})
        # optimizer step
        runner.dist_handler.step(iters=runner.iter)
        runner.call_hook('after_train_iter')
        runner.log_metric(runner.train_log_buffer)
    runner.call_hook('after_epoch')
    if bmuf:
        bmuf.after_run()

    # check your model
    loss = solution_out['loss']
    cer = solution_out['cer']
    uttid = batch_data['uttid']
    check(config_file, loss, cer, uttid)


if __name__ == '__main__':
    batch_test(sys.argv[1:])
