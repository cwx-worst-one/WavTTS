"""penguin config generator"""
import os
import re
import json
import logging
import configparser
from core.utils import Config, dist_hdfs_get, get_rank
from core.dataset import get_meta


def soft_link(path1, path2):
    '''make soft link of files'''
    cmd = "ln -bs {} {}".format(path1, path2)
    os.system(cmd)


class PenguinConfigGenerator:
    """Penguin Config Generator"""

    def __init__(self, base_dir, config):
        '''init'''
        self.dolphin_cfg = config
        self.solution_cfg = config.solution
        self.inference_cfg = config.inference
        self.train_cfg = config.train
        self.data_cfg = config.data
        self.lm_cfg = config.get("lm", Config())
        self.penguin_cfg_backup = config.inference.get('penguin_config', Config())
        self.penguin_cfg = Config()
        self.base_dir = base_dir
        onnx_dir = self.solution_cfg.get('onnx_dir', 'onnx')
        self.model_dir = self.model_dir = os.path.join(self.base_dir, onnx_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.solution_type = self.solution_cfg.get(
            "type", self.solution_cfg.get("model_type", "")
        ).lower()

    def set_default_config(self):
        '''set default config of penguin'''
        self.penguin_cfg.setdefault('audio_sample_rate', 16000)
        self.penguin_cfg.setdefault('vocab_shift', 4)
        self.penguin_cfg.setdefault('dual_channel', False)
        self.penguin_cfg.setdefault('u_steps', 1)
        self.penguin_cfg.setdefault('beam_scale', 1)
        self.penguin_cfg.setdefault('hop_frames', 1)
        self.penguin_cfg.setdefault('padding_of_first_packet', 0)
        self.penguin_cfg.setdefault('tail_seg_padding', True)
        self.penguin_cfg.setdefault('max_state_len', -1)
        self.penguin_cfg.setdefault('output_confidence', False)
        self.penguin_cfg.setdefault('output_speed', False)
        self.penguin_cfg.setdefault('output_streaming_stable_metric', False)
        self.penguin_cfg.setdefault('result_save_path', './penguin.output')
        self.penguin_cfg.setdefault('use_ce_timestamp_conf', False)
        self.penguin_cfg.setdefault('serializer_path', '')
        self.penguin_cfg.setdefault('all_output_save_path', None)
        self.penguin_cfg.setdefault('local_onnx_dir', '')
        self.penguin_cfg.setdefault('remote_onnx_dir', '')
        self.penguin_cfg.setdefault('cal_cer', True)
        # las rescore
        self.penguin_cfg.setdefault('use_las_rescore', False)
        self.penguin_cfg.setdefault('use_las_fst', False)
        self.penguin_cfg.setdefault('las_fst_scale', 0.0)
        self.penguin_cfg.setdefault('fw_decoder_scale', 0.0)
        self.penguin_cfg.setdefault('bw_decoder_scale', 0.0)
        self.penguin_cfg.setdefault('las_bos_idx', 0)
        self.penguin_cfg.setdefault('las_eos_idx', 2)
        # lm
        self.penguin_cfg.setdefault('use_lm_solution', False)
        self.penguin_cfg.setdefault('use_domain_lm_fst', False)
        self.penguin_cfg.setdefault('use_domain_lm_nnlm', False)
        self.penguin_cfg.setdefault('ilme', False)
        self.penguin_cfg.setdefault('ngram_fst_minus_internal_lm', False)
        self.penguin_cfg.setdefault('domain_lm_scale', 0.0)
        self.penguin_cfg.setdefault('domain_nnlm_scale', 0.0)
        self.penguin_cfg.setdefault('internal_nnlm_scale', 0.0)
        self.penguin_cfg.setdefault('domain_lm_fst_path', '')
        # word fst
        self.penguin_cfg.setdefault('use_hotword_fst', False)
        self.penguin_cfg.setdefault('hotword_fst_path', '')
        self.penguin_cfg.setdefault('hotword_fst_weight', 0)
        self.penguin_cfg.setdefault('use_fst_skip', False)
        self.penguin_cfg.setdefault('fst_skip_length', 5)
        # class_lm
        self.penguin_cfg.setdefault('use_class_lm_fst', False)
        self.penguin_cfg.setdefault('class_lm_fst_fusion_weight', 0.0)
        self.penguin_cfg.setdefault('class_lm_fst_word_weight', 0.0)
        # use domain id
        self.penguin_cfg.setdefault('use_domain_id', False)
        self.penguin_cfg.setdefault('domain_num', 10)
        self.penguin_cfg.setdefault('domain_id', 2)
        # prefetch
        self.penguin_cfg.setdefault('prefetch', False)
        self.penguin_cfg.setdefault('prefetch_thresh', 1e-4)
        # fix prefix search
        self.penguin_cfg.setdefault('fixed_prefix', False)
        self.penguin_cfg.setdefault('fixed_prefix_time_freq', 10)
        self.penguin_cfg.setdefault('fixed_prefix_changeable_token', 0)
        # las g2p
        self.penguin_cfg.setdefault('use_las_g2p', False)
        # cascaded encoders
        self.penguin_cfg.setdefault('use_cascaded_encoders', False)
        self.penguin_cfg.setdefault('nc_first_seg_size', 0)
        self.penguin_cfg.setdefault('nc_seg_frames', 0)
        # reduce embed predictor
        self.penguin_cfg.setdefault('reduced_embed_predictor', False)
        self.penguin_cfg.setdefault('embed_txt_path', '')

        if "cif" in self.solution_type:
            self.penguin_cfg.setdefault('cif_threshold', 1.0)

        # xml generate
        self.penguin_cfg.setdefault('xml_generate_template', '')
        self.penguin_cfg.setdefault('xml_generate_copy_resource_path', '')
        self.penguin_cfg.setdefault('xml_generate_script_version', 'online')

    def dump_config_ini(self):
        '''save config'''
        ini_file = os.path.join(self.model_dir, "config.ini")
        if ini_file:
            config = configparser.ConfigParser()
            section = "auto_config"
            config.add_section(section)
            for k, v in self.penguin_cfg.items():
                if k in ('lm_cfg', 'inference_cfg'):
                    continue
                config[section][k] = json.dumps(v)
            with open(ini_file, 'w') as fp:
                config.write(fp)
                logging.info("Config generated successfully!")

    def generate_cmvn_vocab(self, cmvn_file, vocab_file):
        '''generate cmvn stat and vocab form meta'''
        meta_data_root = self.data_cfg.get("meta_data_root", self.data_cfg.get("data_root", None))
        if isinstance(meta_data_root, (list, tuple)):
            meta_data_root = meta_data_root[0]
        meta_file = os.path.join(meta_data_root, self.data_cfg.meta_file)
        meta_data = get_meta(meta_file)
        with open(cmvn_file, 'w') as f:
            for i in range(len(meta_data["cmvn_mean"])):
                f.write("%.18e %.18e\n" % (meta_data["cmvn_mean"][i], meta_data["cmvn_var"][i]))

        vocab = meta_data["reorder_tgt_dict"]
        with open(vocab_file, 'w') as f:
            for key in vocab.symbols[vocab.nspecial :]:
                f.write('{}\n'.format(key))

    def set_onnx_model(self):
        '''set_onnx_model'''
        if "onnx_dir" in self.solution_cfg:
            onnx_dir = self.solution_cfg.onnx_dir
            self.penguin_cfg["local_onnx_dir"] = self.model_dir
            resume_hdfs_chkpt = self.train_cfg.get('resume_hdfs_chkpt', '')
            if resume_hdfs_chkpt.startswith('hdfs'):
                remote_ckpt_dir = os.path.dirname(resume_hdfs_chkpt)
                remote_ckpt_dir = re.sub(r'checkpoints$', '', remote_ckpt_dir)
                self.penguin_cfg["remote_onnx_dir"] = os.path.join(remote_ckpt_dir, onnx_dir)
        if not self.model_dir:
            raise Exception("set onnx model_dir error!")
        self.generate_cmvn_vocab(
            os.path.join(self.model_dir, "fbank_cmvn.stat"), os.path.join(self.model_dir, "vocab")
        )
        self.penguin_cfg["vocab_path"] = os.path.join(self.model_dir, "vocab")
        self.penguin_cfg["global_cmvn_file_path"] = os.path.join(self.model_dir, "fbank_cmvn.stat")

    def set_encoder_type(self):
        '''set_encoder_type'''
        pad_flag = False
        for func in self.data_cfg.valid_item_transform:
            if func['type'] == "AppendFrames":
                self.penguin_cfg["extra_frames"] = func.get("frame", 0)
                self.penguin_cfg["padding_value"] = func.get("value", -15)
                if func.get("seg_frames", 0) == 0:
                    self.penguin_cfg["tail_seg_padding"] = False
                pad_flag = True
                break
        if not pad_flag:
            self.penguin_cfg["extra_frames"] = 0
            self.penguin_cfg["padding_value"] = 0
            self.penguin_cfg["tail_seg_padding"] = False

        backbone_type = self.solution_cfg.acoustic_backbone_type.lower()
        if "lstm" in backbone_type:
            self.penguin_cfg["encoder_type"] = "lstm"
            self.penguin_cfg["seg_frames"] = 20
            self.penguin_cfg["first_seg_size"] = 20
        elif "transformer" in backbone_type:
            self.penguin_cfg["encoder_type"] = "transformer"
            self.penguin_cfg["seg_frames"] = 1000
            self.penguin_cfg["first_seg_size"] = 20
        elif "dfsmn" in backbone_type:
            self.penguin_cfg["encoder_type"] = "dfsmn"
            self.penguin_cfg["seg_frames"] = 20
            self.penguin_cfg["first_seg_size"] = 48
        elif "conformer" in backbone_type:
            self.penguin_cfg["encoder_type"] = "conformer"
            self.penguin_cfg["seg_frames"] = 1000
            self.penguin_cfg["first_seg_size"] = 20

        frontend_type = self.solution_cfg.front_end_type
        if frontend_type == "TimeReduceLSTMP":
            self.penguin_cfg["stack_frames"] = self.solution_cfg.get("input_concat_size", 1)
        else:
            self.penguin_cfg["stack_frames"] = 1

    def set_las_rescore(self):
        '''set las_rescore'''
        twopass_infer_cfg = self.inference_cfg.get("twopass_infer_cfg", None)
        if twopass_infer_cfg and twopass_infer_cfg.enable_twopass_rescore:
            self.penguin_cfg["use_las_rescore"] = True
            self.penguin_cfg["fw_decoder_scale"] = twopass_infer_cfg.get(
                "las_forward_score_scale", 0.0
            )
            self.penguin_cfg["bw_decoder_scale"] = twopass_infer_cfg.get(
                "las_backward_score_scale", 0.0
            )

            self.penguin_cfg["las_fst_scale"] = twopass_infer_cfg.get("las_fst_score_scale", 0.0)

    def set_prefetch(self):
        '''set prefetch'''
        if self.inference_cfg.get("enable_prefetch", False):
            self.penguin_cfg["prefetch"] = True
            self.penguin_cfg["prefetch_thresh"] = self.inference_cfg.get("prefetch_thresh", 1e-4)

    def set_hotword(self):
        '''set hotword_fst'''
        if self.inference_cfg.get("hotword_fst_path", None):
            self.penguin_cfg["use_hotword_fst"] = True
            fst_path = self.inference_cfg.hotword_fst_path
            fst_name = os.path.basename(fst_path)
            if fst_path.startswith("hdfs"):
                dist_hdfs_get(fst_path, self.model_dir)
            else:
                if get_rank() == 0:
                    soft_link(fst_path, self.model_dir)
            self.penguin_cfg["hotword_fst_path"] = os.path.join(self.model_dir, fst_name)
            self.penguin_cfg["hotword_fst_weight"] = self.inference_cfg.get('hotword_fst_weight')

    def set_ngram(self):
        '''set ngram'''
        if self.inference_cfg.get("ngram_fst_path", None):
            self.penguin_cfg["use_domain_lm_fst"] = True
            self.penguin_cfg["domain_lm_scale"] = self.inference_cfg.get("ngram_fst_weight", 0)
            self.penguin_cfg["ngram_vocab"] = self.inference_cfg.get("ngram_fst", {}).get(
                "vocab", ""
            )
            self.penguin_cfg["ngram_fst_minus_internal_lm"] = self.inference_cfg.get(
                "ngram_fst_minus_internal_lm", True
            )
            ngram_fst_path = self.inference_cfg.ngram_fst_path
            ngram_fst_name = os.path.basename(self.inference_cfg.ngram_fst_path)
            if ngram_fst_path.startswith("hdfs"):
                dist_hdfs_get(ngram_fst_path, self.model_dir)
            else:
                if get_rank() == 0:
                    soft_link(ngram_fst_path, self.model_dir)
            self.penguin_cfg["domain_lm_fst_path"] = os.path.join(self.model_dir, ngram_fst_name)

    def set_nnlm(self):
        '''set nnlm config'''
        if self.inference_cfg.get("nnlm_path", None):
            self.penguin_cfg["use_domain_lm_nnlm"] = True
            self.penguin_cfg["domain_nnlm_scale"] = self.inference_cfg.nnlm_weight
            if self.inference_cfg.get("nnlm_ilme", None):
                self.penguin_cfg["ilme"] = True
                self.penguin_cfg["internal_nnlm_scale"] = self.inference_cfg.internal_lm_weight

    def set_class_lm(self):
        '''set class_lm config'''
        if self.inference_cfg.get("class_lm_fst_path", None):
            self.penguin_cfg["use_class_lm_fst"] = True
            self.penguin_cfg[
                "class_lm_fst_fusion_weight"
            ] = self.inference_cfg.class_lm_fst_fusion_weight
            self.penguin_cfg[
                "class_lm_fst_word_weight"
            ] = self.inference_cfg.class_lm_fst_word_weight

    def set_lm(self):
        '''set configs of LM'''
        self.set_hotword()
        self.set_ngram()
        self.set_nnlm()
        self.set_class_lm()
        self.lm_cfg.setdefault('hotword_fst', Config())
        self.lm_cfg.setdefault('coldword_fst', Config())
        self.lm_cfg.setdefault('ngram_fst', Config())
        self.lm_cfg.setdefault('nnlm', Config())
        self.lm_cfg.setdefault('class_lm_fst', Config())
        self.lm_cfg.hotword_fst.setdefault('lm_token_beam_size', 10)
        self.lm_cfg.hotword_fst.setdefault('max_active_lm_token_num', 5000)
        self.lm_cfg.hotword_fst.setdefault('enable_multi_step_search', True)
        self.lm_cfg.hotword_fst.hotword_weight = self.inference_cfg.get('hotword_fst_weight', '')
        self.penguin_cfg['lm_cfg'] = self.lm_cfg

    def set_ce_timestamp(self):
        '''set ce timestamp'''
        if self.inference_cfg.get(
            "output_wordboundary_type", None
        ) == "ce" or self.inference_cfg.get("output_ce_wordboundary", False):
            self.penguin_cfg["use_ce_timestamp_conf"] = True

    def set_cascaded_encoders(self):
        if self.solution_type == "RnntCaseModel".lower() and not self.inference_cfg.get(
            "causal_mode", True
        ):
            self.penguin_cfg['use_cascaded_encoders'] = True
            self.penguin_cfg["processor_list"] = [
                "feature_input",
                "encoder",
                "nc_encoder",
                "decoder",
                "get_result",
            ]

    def set_reduce_predictor(self):
        if self.solution_cfg.predictor_type == "ReducedEmbeddingPredictor":
            self.penguin_cfg['reduced_embed_predictor'] = True
            self.penguin_cfg['embed_txt_path'] = os.path.join(self.model_dir, "embed.txt")

    def generate_config(self):
        '''generator penguin config'''
        if "rnnt" in self.solution_type:
            generator = RnntConfigGenerator(self.base_dir, self.dolphin_cfg)
        elif "cif" in self.solution_type:
            generator = CifConfigGenerator(self.base_dir, self.dolphin_cfg)
        else:
            raise Exception("unsupported solution type!")
        self.penguin_cfg = generator.generate_config()
        remote_stat_dir = self.inference_cfg.get('remote_stat_dir', '')
        if remote_stat_dir:
            self.penguin_cfg.setdefault('result_save_path', remote_stat_dir)
        self.penguin_cfg['inference_cfg'] = self.inference_cfg
        # the config of penguin from CMDLline has the highest priority
        for k, v in self.penguin_cfg_backup.items():
            self.penguin_cfg[k] = v
        self.dolphin_cfg['inference']['penguin_cfg'] = self.penguin_cfg
        self.dump_config_ini()
        self.set_default_config()
        return self.penguin_cfg


class RnntConfigGenerator(PenguinConfigGenerator):
    """RNNT config generator"""

    def generate_config(self):
        '''generator penguin config'''
        self.set_onnx_model()
        self.set_encoder_type()
        self.penguin_cfg["beam_size"] = self.inference_cfg.beam_size
        self.penguin_cfg["stack_frames"] = self.solution_cfg.get("input_concat_size", 1)
        self.penguin_cfg["down_sample"] = self.solution_cfg.downsampling_size
        self.penguin_cfg["len_penalty_scale"] = self.inference_cfg.len_penalty_scale
        self.penguin_cfg["blank_scale"] = self.inference_cfg.blk_scale
        self.penguin_cfg["fbank_dim"] = self.data_cfg.fbank_dim
        self.penguin_cfg["use_gpu"] = self.solution_cfg.get('panther_use_gpu', True)
        self.penguin_cfg["processor_list"] = ["feature_input", "encoder", "decoder", "get_result"]

        data_root = self.data_cfg.data_root
        if isinstance(data_root, list):
            data_root = data_root[0]
        data_root = self.data_cfg.get('test_data_root', data_root)
        self.penguin_cfg["wav_list"] = os.path.join(
            data_root, self.inference_cfg.test_sets.strip().split('|')[0]
        )
        self.set_las_rescore()
        self.set_prefetch()
        self.set_lm()
        self.set_ce_timestamp()
        self.set_cascaded_encoders()
        self.set_reduce_predictor()
        return self.penguin_cfg


class CifConfigGenerator(PenguinConfigGenerator):
    '''CIF Penguin Config Generator'''

    def set_encoder_type(self):
        '''set encoder_type for CIF'''
        model_type = self.solution_cfg.get("acoustic_backbone_type", '').lower()
        self.penguin_cfg["encoder_type"] = "transformer_cif"
        self.penguin_cfg["seg_frames"] = 1
        self.penguin_cfg["first_seg_size"] = 1
        self.penguin_cfg["extra_frames"] = 0
        self.penguin_cfg["padding_value"] = -15.0
        self.penguin_cfg["hop_frames"] = 40
        self.penguin_cfg["stack_frames"] = 80
        self.penguin_cfg["down_sample"] = 8
        self.penguin_cfg["vocab_shift"] = 2
        if "conformer" in model_type:
            self.penguin_cfg["encoder_type"] = "conformer_cif"
            self.penguin_cfg["seg_frames"] = 1000
            self.penguin_cfg["hop_frames"] = 1
            self.penguin_cfg["vocab_shift"] = 4

    def set_bias(self):
        '''set bias for CIF'''
        dist_hdfs_get(
            "hdfs://haruna/home/byte_ailab_speech_service/user/lixuwei/cif_resource/bias",
            self.model_dir,
        )
        self.penguin_cfg["bias_path"] = os.path.join(self.model_dir, "bias")

    def generate_config(self):
        '''generator penguin config'''
        self.set_onnx_model()
        self.set_bias()
        self.set_encoder_type()
        self.set_ce_timestamp()
        self.penguin_cfg["beam_size"] = self.inference_cfg.beam_size
        self.penguin_cfg["cif_threshold"] = self.solution_cfg.cif_weight_threshold
        self.penguin_cfg["max_state_len"] = self.solution_cfg.get("limited_decoder_states", -1)
        self.penguin_cfg["fbank_dim"] = self.data_cfg.fbank_dim
        self.penguin_cfg["dense_cif_units"] = self.solution_cfg.get('dense_cif_units', 0)
        self.penguin_cfg["use_gpu"] = self.solution_cfg.get('panther_use_gpu', True)
        data_root = self.data_cfg.data_root
        if isinstance(data_root, list):
            data_root = data_root[0]
        data_root = self.data_cfg.get('test_data_root', data_root)
        self.penguin_cfg["wav_list"] = os.path.join(
            data_root, self.inference_cfg.test_sets.strip().split('|')[0]
        )

        self.penguin_cfg["processor_list"] = [
            "feature_input",
            "encoder",
            "cif_decoder",
            "cif_get_result",
        ]

        # useless parameter
        self.penguin_cfg["len_penalty_scale"] = 0
        self.penguin_cfg["blank_scale"] = 0.1
        return self.penguin_cfg
