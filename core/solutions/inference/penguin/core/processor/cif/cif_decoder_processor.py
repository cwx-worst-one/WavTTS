"""penguin CIF decoder processor"""
# pylint: disable=super-init-not-called
import copy
import numpy as np
from core.solutions.inference.penguin.core.processor.message import Message
from core.solutions.inference.penguin.core.register import Registers
from core.solutions.inference.penguin.core.factory import Factory
from core.solutions.inference.penguin.core.processor.decoder_processor import Decoder

# from core.solutions.inference.penguin.core.processor.lm.lm_solution import LmSolution


@Registers.processor.register('cif_decoder')
class CifDecoder(Decoder):
    """CIF decoder"""

    def __init__(self, config):
        '''init'''
        self.config = config
        self.bias_path = config.bias_path
        self.beam_scale = int(config.beam_scale)
        self.max_state_len = config.max_state_len
        self.use_ce_timestamp_conf = bool(config.use_ce_timestamp_conf)
        self.loadbias()
        self.beam_size = int(config.beam_size)
        self.t_hyps_ = []
        self.encoder_type = config.encoder_type
        self.inference = Factory.get_inference('decoder', config)

        # self.use_lm_solution = bool(config.use_lm_solution)
        self.use_hotword_fst = bool(config.use_hotword_fst)
        self.init_lm_state = 0
        # self.hotword_fst_weight = float(config.hotword_fst_weight)
        self.fbank_dim_ = 80
        self.output_dim_ = self.get_output_all()[0].shape[1]
        self.embedding_dim_ = self.get_input_all()[1].shape[1]
        self.state_dim_ = self.get_input_all()[-1].shape[2]
        self.reset()

    def __call__(self, decoder_data):
        '''call'''
        # self.fst_path = fst_path
        # if self.use_lm_solution and self.use_hotword_fst:
        #     self.lm_solution = LmSolution()
        #     self.lm_solution.load(self.fst_path)
        #     self.init_lm_state = self.lm_solution.start()
        # else:
        #     self.lm_solution = None
        #     self.init_lm_state = 0
        return self.process(decoder_data)

    def process(self, decoder_data):
        '''process'''
        wav_name = decoder_data[0]
        decoder_ins = decoder_data[1]
        # timestamp = decoder_data[2]
        self.reset()
        sum_len = 0
        for decoder_in in decoder_ins:
            sum_len += len(decoder_in)
            for data in decoder_in:
                self.num_ = self.num_ + 1
                self.cif_output_ = data
                self.beam_search()
        label_seq = self.get_best_label_seq()
        extra = {}
        if self.use_ce_timestamp_conf:
            ce_encoder_in = decoder_data[3]
            ce_logits = self.ce_encoder(ce_encoder_in)
            if label_seq[0] == 1:
                label_seq = label_seq[1:]
            out_list = self.apply_ce_timestamp(ce_logits.ce_output, label_seq)
            extra['ce_out_list'] = out_list
        return (wav_name, label_seq, extra)

    def loadbias(self):
        '''load bias'''
        self.bias_ = []
        with open(self.bias_path, 'r') as f:
            for line in f:
                for word in line.split():
                    self.bias_.append(float(word))

    def get_best_label_seq(self):
        '''get best label_seq of hyps'''
        self.t_hyps_.sort(key=lambda Take: Take['score'], reverse=True)
        return self.t_hyps_[0]['label_seq']

    def reset(self):
        '''reset'''
        self.num_ = 0
        self.kv_ = np.zeros([self.beam_size, 0, self.state_dim_])

        self.t_hyps_ = []
        self.t_hyps_.append(
            {'score': 0, 'label_seq': [1], 'ptr': None, 'lm_state': self.init_lm_state}
        )
        for _ in range(self.beam_size - 1):
            self.t_hyps_.append(
                {'score': -99999, 'label_seq': [1], 'ptr': None, 'lm_state': self.init_lm_state}
            )
        self.pre_cif_output_ = np.zeros([self.beam_size, self.embedding_dim_])
        self.prev_ids_ = np.zeros(self.beam_size)
        self.decoder_bias_ = np.zeros((self.beam_size, 1, 1, 1))

    def get_result(self, message):
        '''get result'''
        input_name = self.inference.get_input()
        input_data = []
        for name in input_name:
            input_data.append(message[name])
        decoder_embedding = self.inference.run(input_data=input_data)
        return decoder_embedding

    def calc_logits(self):
        '''calculate logits'''
        for i in range(len(self.t_hyps_)):
            node = self.t_hyps_[i]
            self.prev_ids_[i] = node['label_seq'][-1]
        self.prev_ids_ = np.array(self.prev_ids_).astype(np.int32)
        self.pre_cif_output_ = np.array(self.pre_cif_output_).astype(np.float32)
        tot_bias_num = vld_bias_num = self.num_
        if self.max_state_len >= 0:
            tot_bias_num = self.max_state_len + 1
            vld_bias_num = min(self.num_, tot_bias_num)

        bias_init_val = -10000.0 if self.max_state_len >= 0 else 0.0
        self.decoder_bias_ = np.full((self.beam_size, 1, 1, tot_bias_num), bias_init_val)
        for i in range(self.beam_size):
            for j in range(vld_bias_num):
                self.decoder_bias_[i][0][0][tot_bias_num - vld_bias_num + j] = self.bias_[
                    len(self.bias_) - vld_bias_num + j
                ]

        self.decoder_bias_ = np.array(self.decoder_bias_).astype(np.float32)
        self.cif_output_ = np.array(self.cif_output_).astype(np.float32)

        if self.max_state_len >= 0:
            state_shape = [self.beam_size, self.max_state_len, self.state_dim_]
        else:
            state_shape = [self.beam_size, self.num_ - 1, self.state_dim_]

        if self.num_ > 1 or self.max_state_len >= 0:
            if self.num_ > 1:
                ptrs = []
                for hyps in self.t_hyps_:
                    if self.max_state_len >= 0:
                        ptrs.append(hyps['ptr'][1:])
                    else:
                        ptrs.append(hyps['ptr'])
                # TODO: splice Key&Value global
                self.kv_ = np.concatenate(ptrs)
                self.kv_ = self.kv_.reshape(state_shape)
            else:
                self.kv_ = np.zeros(state_shape)

        self.kv_ = np.array(self.kv_).astype(np.float32)
        input_name = self.get_input()
        decoder_input = {
            input_name[0]: self.prev_ids_,
            input_name[1]: self.pre_cif_output_,
            input_name[2]: self.decoder_bias_,
            input_name[3]: self.cif_output_,
            input_name[4]: self.kv_,
        }
        decoder_out = self.get_result(Message(decoder_input))
        output_name = self.get_output()
        self.logits_ = decoder_out[output_name[0]]
        self.kv_ = decoder_out[output_name[1]]

    def beam_search(self):
        '''beam search'''
        self.calc_logits()
        h_score = []
        for i in range(len(self.t_hyps_)):
            h_score.append(self.t_hyps_[i]['score'])
        h_score = np.array(h_score).astype(np.float32)
        score = np.expand_dims(h_score, -1) + self.logits_
        score = score.reshape(-1)
        token_index = np.argpartition(score, -self.beam_scale * self.beam_size)[
            -self.beam_scale * self.beam_size :
        ]
        token_score = score[token_index]

        u_hyps = []
        for idx in range(self.beam_size * self.beam_scale):
            index = token_index[idx]
            node_idx = int(index / self.output_dim_)
            node = self.t_hyps_[node_idx]
            j = int(index % self.output_dim_)
            kv_ptr = self.kv_[node_idx]
            u_hyps.append(
                {
                    'score': token_score[idx],
                    'label_seq': copy.deepcopy(node['label_seq']),
                    'ptr': kv_ptr,
                }
            )
            u_hyps[-1]['label_seq'].append(j)
            # if self.lm_solution and self.use_hotword_fst:
            #     lm_label = j
            #     next_lm_state, lm_score = \
            #         self.lm_solution.step_fsts(lm_label, node['lm_state'])
            #     u_hyps[-1]['score'] -= self.hotword_fst_weight * lm_score
            #     u_hyps[-1]['lm_state'] = copy.deepcopy(next_lm_state)
        u_hyps.sort(key=lambda ele: ele['score'], reverse=True)
        u_hyps = u_hyps[: self.beam_size]

        self.t_hyps_ = u_hyps
        self.pre_cif_output_ = self.cif_output_
