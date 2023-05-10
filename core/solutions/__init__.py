''' solutions '''
from core.utils import logging
from core.models.utils import xavier_init
from .base_solution import SOLUTION_REGISTRY
from .aed.base_overlap_detection import *
from .aed.base_emotion_solution import *
from .asr.base_las_solution import *
from .asr.base_rnnt_model import *
from .asr.base_cif_model import *
from .aed.classification_solution import *
from .asr.cif_ce_pretrain_solution import *
from .asr.cif_ce_align_solution import *
from .asr.context_aware_rnnt_model import *
from .asr.rnnt_ce_pretrain_solution import *
from .asr.rnnt_ce_align_solution import *
from .asr.rnnt_dual_channel_model import *
from .asr.rnnt_hmm_free_model import *
from .asr.rnnt_las_g2p_model import *
from .asr.rnnt_twopass_model import *
from .asr.rnnt_tog_model import *
from .asr.rnnt_deep_bias_model import *
from .asr.rnnt_universal_model import *
from .asr.rnnt_lang_aware_model import *
from .kws.ce_solution import *
from .kws.rnnt_solution import *
from .vad.ce_solution import *
from .lm.base_nnlm_solution import *
from .lm.rnnt_ilm_solution import *
from .mdd.base_mdd_solution import *
from .sid.base_sid_solution import *
from .sid.wavlm_sid_solution import *
from .se.base_se_solution import *
from .se.base_mc_rnnt_model import *
from .pretrain.wav2vec_pretrain_solution import *
from .pretrain.wav2vec_ctc_solution import *
from .pretrain.bert_pretrain_solution import *
from .pretrain.data2vec_pretrain_solution import *
from .pretrain.data2vec_ctc_solution import *
from .pretrain.bestrq_pretrain_solution import *
from .pretrain.aullm_pretrain_solution import *
from .pretrain.hubert_pretrain_solution import *
from .pretrain.spokenlm_pretrain_solution import *
from .pretrain.tokenization_solution import *
from .pretrain.usm_most_solution import *


def get_model_type(solution_cfg):
    '''get_model_type'''
    model_type = {
        'base_emotion_recognition_solution': 'model_type',
        'base_overlap_detection_solution': 'model_type',
        'base_las_solution': 'las_type',
        'base_rnnt_solution': 'rnnt_type',
        'rnnt_ce_align_solution': 'rnnt_type',
        'rnnt_ce_pretrain_solution': 'rnnt_type',
        'cif_ce_pretrain_solution': 'model_type',
        'kws_rnnt_solution': 'rnnt_type',
        'kws_ce_solution': 'ce_type',
        'vad_ce_solution': 'ce_type',
        'base_nnlm_solution': 'nnlm_type',
        'base_mdd_solution': 'mdd_model_type',
        'base_se_solution': 'se_model_type',
        'base_mc_rnnt_solution': 'rnnt_type',
        'base_wav2vec_pretrain_solution': 'model_type',
        'base_wav2vec_ctc_solution': 'model_type',
        'base_data2vec_pretrain_solution': 'model_type',
        'base_data2vec_ctc_solution': 'model_type',
        'bert_pretrain_solution': 'model_type',
    }
    try:
        return solution_cfg.model_type
    except Exception as e:
        logging.warning(
            "It is recommended to use --solution.model_type to define the type of model!"
        )
        if solution_cfg.type in model_type:
            return getattr(solution_cfg, model_type[solution_cfg.type])
        if solution_cfg.type == 'base_sid_solution':
            return 'BaseSidModel'
        if solution_cfg.type == 'wavlm_sid_solution':
            return 'WavLMSidModel'
        raise Exception(f"can't recognized solution type {solution_cfg.type}!") from e


def setup_solution(solution_cfg):
    '''setup solution'''
    solution = SOLUTION_REGISTRY[get_model_type(solution_cfg)](solution_cfg)
    if solution_cfg.get('xavier_init', False):
        xavier_init(solution)
    return solution
