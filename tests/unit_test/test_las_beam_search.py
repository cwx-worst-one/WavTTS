'''
test beam search.
Author: Xianzhao Chen (chenxianzhao@bytedance.com)
Date: 20230309
'''
# pylint:disable=too-many-lines
import torch
import numpy as np
from core.utils import Config, hdfs_get
from core.utils.config import ConfigDict
from core.solutions.inference.las_beam_search import BaseBeamSearch
from core.dataset.dictionary import ScpDictionary


def assert_eq(x, y, eps=1e-6):
    '''assert eq'''
    x = np.array(x)
    y = np.array(y)
    # print(x-y)
    assert (np.abs(x - y) <= eps).all()


class FakeDecoderModule:
    '''fake decoder'''

    def __init__(self, probs):
        '''init'''
        self.probs = probs
        self.step = 0

    # pylint: disable=unused-argument
    def forward_step(self, tokens, encoder_out, encoder_mask, temperature=None, streaming=None):
        '''forward_step'''
        ret = self.probs[self.step]
        self.step += 1
        return ret, None

    def reorder_incremental_state(self, new_order):
        '''reorder_incremental_state'''


def _defalult_config(**kwargs):
    '''defalut cfg'''
    cfg = Config()
    cfg.tgt_dict = ScpDictionary()
    inference_cfg = ConfigDict(**kwargs)
    return cfg, inference_cfg


def test_single(
    batch_size=1,
    src_len=10,
    tar_len=200,
    vocab_size=8888,
    hyps_per_beam=4,
    seed=9527,
    len_penalty=0.0,
    lingvo_beamsearch=True,
    lingvo_beam_size=3.0,
    lingvo_valid_eos_max_logit_delta=5.0,
    # pylint: disable=invalid-name
    except_res=None,
    sub_eos=0.0,
    early_stop=0,
):
    # pylint:disable=too-many-locals
    cfg, inference_cfg = _defalult_config(
        las_beam_size=hyps_per_beam,
        len_penalty=len_penalty,
        lingvo_beamsearch=lingvo_beamsearch,
        beam_width=lingvo_beam_size,
        valid_eos_max_logit_delta=lingvo_valid_eos_max_logit_delta,
        max_len=tar_len - 1,
    )
    np.random.seed(seed)
    probs = torch.tensor(np.log(np.random.rand(tar_len, batch_size * hyps_per_beam, vocab_size)))
    if early_stop != 0:
        probs[early_stop, :hyps_per_beam, 2] += 100
        probs[early_stop + 1, hyps_per_beam:, 2] += 100
    probs[:, :, 2] -= sub_eos
    encoder_out = torch.rand([batch_size, src_len, 10])
    encoder_mask = torch.ones([batch_size, src_len])
    basebeamsearch = BaseBeamSearch(cfg, inference_cfg, FakeDecoderModule(probs))
    finalized = basebeamsearch(encoder_out, encoder_mask)
    log_res = [[], [], []]
    for idx, x in enumerate(finalized):
        for idy, y in enumerate(x):
            y.pop('positional_scores')
            y.pop('attention')
            y.pop('alignment')
            hyp_tokens = y['tokens']
            hyp_scores = y['score']
            if except_res is None:
                log_res[0].append(hyp_tokens)
                log_res[1].append(len(hyp_tokens))
                log_res[2].append(hyp_scores)
                continue
            (except_tokens_, except_lens_, except_scores_) = except_res
            tmp_idx = idx * hyps_per_beam + idy
            except_hyp_tokens = except_tokens_[tmp_idx][: except_lens_[tmp_idx]]
            except_hyp_scores = except_scores_[tmp_idx]
            assert_eq(hyp_tokens, except_hyp_tokens)
            assert_eq(hyp_scores, except_hyp_scores)
    if except_res is None:
        print(log_res)
        print(finalized)


if __name__ == '__main__':
    # datas = (except_tokens, except_lens, except_scores)
    hdfs_get(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/dolphin/unit_test/las_beam_search_testcase0.pth'
    )
    datas = torch.load('las_beam_search_testcase0.pth')
    test_single(
        batch_size=1,
        src_len=10,
        tar_len=5,
        vocab_size=8,
        hyps_per_beam=4,
        seed=9527,
        len_penalty=0.0,
        lingvo_beamsearch=True,
        lingvo_beam_size=3.0,
        lingvo_valid_eos_max_logit_delta=5.0,
        except_res=datas,
    )

    hdfs_get(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/dolphin/unit_test/las_beam_search_testcase1.pth'
    )
    datas = torch.load('las_beam_search_testcase1.pth')
    test_single(
        batch_size=3,
        src_len=10,
        tar_len=5,
        vocab_size=8,
        hyps_per_beam=4,
        seed=9527,
        len_penalty=0.0,
        lingvo_beamsearch=True,
        lingvo_beam_size=3.0,
        lingvo_valid_eos_max_logit_delta=5.0,
        except_res=datas,
    )

    hdfs_get(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/dolphin/unit_test/las_beam_search_testcase2.pth'
    )
    datas = torch.load('las_beam_search_testcase2.pth')
    test_single(
        batch_size=3,
        src_len=10,
        tar_len=200,
        vocab_size=8888,
        hyps_per_beam=4,
        seed=9527,
        len_penalty=0.8,
        lingvo_beamsearch=True,
        lingvo_beam_size=3.0,
        lingvo_valid_eos_max_logit_delta=5.0,
        except_res=datas,
    )

    # lingvo_beam_size small will finish early
    hdfs_get(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/dolphin/unit_test/las_beam_search_testcase3.pth'
    )
    datas = torch.load('las_beam_search_testcase3.pth')
    test_single(
        batch_size=3,
        src_len=10,
        tar_len=10,
        vocab_size=8,
        hyps_per_beam=4,
        seed=9527,
        len_penalty=0.8,
        lingvo_beamsearch=True,
        lingvo_beam_size=30000.0,
        lingvo_valid_eos_max_logit_delta=5.0,
        except_res=datas,
        early_stop=4,
    )

    hdfs_get(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/dolphin/unit_test/las_beam_search_testcase4.pth'
    )
    datas = torch.load('las_beam_search_testcase4.pth')
    test_single(
        batch_size=3,
        src_len=10,
        tar_len=10,
        vocab_size=8,
        hyps_per_beam=4,
        seed=9527,
        len_penalty=0.8,
        lingvo_beamsearch=True,
        lingvo_beam_size=0.01,
        lingvo_valid_eos_max_logit_delta=5.0,
        except_res=datas,
        early_stop=4,
    )
